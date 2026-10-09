from dataclasses import replace

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.memory.store import BudgetExceeded, MemoryRecord, MemoryStore
from dsrsi.model.streaming import StreamExecutor


def leaves():
    return [MemoryRecord(f"leaf-{i}", "life", torch.randn(3, 8), torch.randn(2, 4),
                         i * 4, i * 4 + 3, i * 4 + 4, wall_time=100.0 + i * 10)
            for i in range(4)]


def test_child_metadata_conditions_compression_and_future_commit_time_cannot_change_content():
    model = tiny().eval()
    children = leaves()
    with torch.no_grad():
        parent = model.consolidate_records(children, record_id="parent", visible_after=20)
        later = model.consolidate_records(children, record_id="later", visible_after=2000)
        changed = model.consolidate_records([replace(r, source_role="tool", source_kind="tool", wall_time=50.0)
                                            for r in children], record_id="changed", visible_after=20)
    assert torch.equal(parent.slots, later.slots) and torch.equal(parent.keys, later.keys)
    assert not torch.equal(parent.slots, changed.slots)
    assert parent.wall_time == 130.0 and parent.child_refs == tuple(r.record_id for r in children)
    assert parent.source_begin == 0 and parent.source_end == 15 and parent.level == 1
    with pytest.raises(ValueError, match="visibility"):
        model.consolidate_records(children, record_id="future", visible_after=15)
    with pytest.raises(ValueError, match="streams"):
        model.consolidate_records([children[0], replace(children[1], stream_id="another")], record_id="other")


def test_partly_unknown_child_wall_time_does_not_become_known_in_parent():
    model = tiny().eval()
    children = leaves()
    parent = model.consolidate_records([children[0], replace(children[1], wall_time=None)], record_id="parent")
    assert parent.wall_time is None


def test_tight_budget_keeps_accepting_incoming_writes_after_only_one_summary_remains():
    torch.manual_seed(12)
    executor = StreamExecutor(tiny().eval())
    state = executor.new_life("life", window=5, rebase_interval=8, writer_stride=4,
                              retrieval_interval=2, memory_tensor_budget=200)
    with torch.no_grad():
        for number in range(25):
            result = executor.forward(torch.tensor([[2, 3, 4, 5]]), state, wall_time=100.0 + number)
            state = result.state
            assert state.memory.tensor_bytes <= 200
    assert state.cache.clock == 100 and len(state.memory.records) == 1
    summary = next(iter(state.memory.records.values()))
    assert summary.level == 24 and summary.source_begin == 0 and summary.source_end == 99
    assert summary.wall_time == 124.0


def test_incoming_compaction_is_atomic_when_summary_cannot_fit():
    a, b = leaves()[:2]
    store = MemoryStore(memory_dim=8, key_dim=4, max_tensor_bytes=140)
    store.append(a)
    parent = replace(a, record_id="parent", source_end=b.source_end, visible_after=b.visible_after,
                     slots=torch.randn(4, 8), level=1, source_kind="summary", child_refs=(a.record_id, b.record_id))
    with pytest.raises(BudgetExceeded):
        store.compact(parent, list(parent.child_refs), incoming=b)
    assert list(store.records) == [a.record_id] and store.records[a.record_id] is a


def test_summary_answer_gradient_updates_consolidator_after_frozen_leaf_writer():
    model = tiny().train()
    model.requires_grad_(False)
    model.consolidator.requires_grad_(True)
    children = leaves()
    parent = model.consolidate_records(children, record_id="parent")
    inputs = torch.tensor([[7, 8, 9, 10, 1]])
    out = model(inputs, records=[[parent]], query_indices=[2], logical_begins=[16], stream_ids=["life"],
                labels=torch.tensor([[-100, -100, -100, 10, 1]]))
    out.loss.mean().backward()
    assert sum(float(p.grad.square().sum()) for p in model.consolidator.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("consolidator."))
