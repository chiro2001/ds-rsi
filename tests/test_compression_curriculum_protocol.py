# Protocol invariants for D compression primitives and content curriculum.
# These pin the mechanisms the D design in docs/semantic-compression.md builds
# on: the 3d+1 leaf chain reaches real depth, only the Consolidator receives
# gradients, child order is not an ordering channel, and retrieval keys are
# frozen at write time.

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.memory.store import MemoryRecord, MemoryStore


def leaf(model, index):
    slots = torch.randn(model.memory_config.slots, model.memory_config.memory_dim)
    keys = model.address.encode_keys(slots[None])[0]
    return MemoryRecord(f"leaf-{index}", "life", slots, keys, index * 4, index * 4 + 3, index * 4 + 4,
                        wall_time=100.0 + index)


def chain(model, depth):
    consumed = []
    for _ in range(4):
        consumed.append(leaf(model, len(consumed)))
    summary = model.consolidate_records(consumed, record_id="summary-1")
    for level in range(2, depth + 1):
        fresh = [leaf(model, len(consumed) + offset) for offset in range(3)]
        consumed.extend(fresh)
        summary = model.consolidate_records([summary] + fresh, record_id=f"summary-{level}")
    return consumed, summary


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_three_d_plus_one_chain_reaches_requested_depth_and_reduces_bytes(depth):
    model = tiny().eval()
    with torch.no_grad():
        consumed, summary = chain(model, depth)
    assert len(consumed) == 3 * depth + 1
    assert summary.level == depth
    assert (summary.source_begin, summary.source_end) == (0, 4 * len(consumed) - 1)
    assert summary.tensor_bytes < sum(record.tensor_bytes for record in consumed)
    assert len(summary.child_refs) == 4
    assert summary.visible_after == max(record.visible_after for record in consumed)


def test_summary_answer_ce_updates_only_the_consolidator_at_depth_two():
    model = tiny().train()
    model.requires_grad_(False)
    model.consolidator.requires_grad_(True)
    _, summary = chain(model, 2)
    out = model(torch.tensor([[7, 8, 9, 10, 1]]), records=[[summary]], query_indices=[2],
                logical_begins=[summary.source_end + 1], stream_ids=["life"],
                labels=torch.tensor([[-100, -100, -100, 10, 1]]))
    out.loss.mean().backward()
    assert sum(float(p.grad.square().sum()) for p in model.consolidator.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("consolidator."))


def test_child_order_is_not_available_as_an_ordering_channel():
    # Attention over slots is permutation invariant up to float summation
    # order, so a curriculum that needs "the first four leaves" to stay
    # distinguishable must convey it through the time/source features.
    torch.manual_seed(3)
    model = tiny().eval()
    children = [leaf(model, index) for index in range(4)]
    with torch.no_grad():
        forward = model.consolidate_records(children, record_id="forward")
        reversed_ = model.consolidate_records(list(reversed(children)), record_id="reversed")
    relative = float((forward.slots - reversed_.slots).abs().max() / forward.slots.abs().max().clamp_min(1e-6))
    assert relative < 1e-3


def test_retrieval_uses_write_time_keys_not_a_reencoded_address():
    torch.manual_seed(4)
    model = tiny().eval()
    records = [leaf(model, index) for index in range(4)]
    store = MemoryStore(memory_dim=model.memory_config.memory_dim, key_dim=model.memory_config.key_dim,
                        max_tensor_bytes=1 << 20)
    for record in records:
        store.append(record)
    query = F.normalize(torch.randn(model.memory_config.key_dim), dim=-1)
    before = [r.record_id for r in store.retrieve(query, stream_id="life", query_position=100, topk=2)]
    with torch.no_grad():
        model.address.key.weight.zero_()
    after = [r.record_id for r in store.retrieve(query, stream_id="life", query_position=100, topk=2)]
    assert before == after
