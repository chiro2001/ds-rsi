import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.model.streaming import StreamExecutor
from dsrsi.state.life_state import LifeState


def run(executor, tokens, pieces, *, write=True, budget=1 << 20):
    state = executor.new_life("one-life", window=5, rebase_interval=8,
                              writer_stride=4, retrieval_interval=3, memory_tensor_budget=budget)
    outputs = []
    offset = 0
    for length in pieces:
        result = executor.forward(tokens[:, offset:offset + length], state, write=write)
        outputs.append(result.hidden)
        state = result.state
        offset += length
    return torch.cat(outputs, 1), state


def test_canonical_schedule_and_writes_do_not_depend_on_external_chunks():
    torch.manual_seed(11)
    executor = StreamExecutor(tiny().eval())
    tokens = torch.randint(2, 31, (1, 20))
    with torch.no_grad():
        whole, a = run(executor, tokens, [20])
        chunks, b = run(executor, tokens, [3, 7, 2, 8])
        steps, c = run(executor, tokens, [1] * 20)
    torch.testing.assert_close(whole, chunks, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(whole, steps, atol=3e-5, rtol=3e-5)
    assert a.cache.clock == b.cache.clock == c.cache.clock == 20
    assert torch.equal(whole, chunks) and torch.equal(whole, steps)
    assert list(a.memory.records) == list(b.memory.records) == list(c.memory.records)
    assert a.next_record == b.next_record == c.next_record == 5
    for key in a.memory.records:
        torch.testing.assert_close(a.memory.records[key].slots, b.memory.records[key].slots, atol=1e-5, rtol=1e-5)


def test_snapshot_includes_pending_writer_and_active_memories():
    executor = StreamExecutor(tiny().eval())
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        _, state = run(executor, tokens[:, :7], [7])
        assert state.pending.shape[1] == 3
        snapshot = state.state_dict()
        a = executor.forward(tokens[:, 7:], state)
        b = executor.forward(tokens[:, 7:], LifeState.from_state_dict(snapshot))
    assert torch.equal(a.hidden, b.hidden)
    assert a.state.cache.clock == b.state.cache.clock
    assert list(a.state.memory.records) == list(b.state.memory.records)
    assert state.cache.clock == 7  # parent unchanged


def test_streaming_future_loss_trains_earlier_writer():
    executor = StreamExecutor(tiny().train())
    state = executor.new_life("life", window=5, rebase_interval=8, writer_stride=4, retrieval_interval=2)
    source = executor.forward(torch.tensor([[2, 3, 4, 5]]), state)
    assert len(source.state.memory.records) == 1
    query = executor.forward(torch.tensor([[7, 8, 9]]), source.state, write=False, force_read_at=[0])
    query.hidden[..., 0].sum().backward()
    assert sum(float(p.grad.square().sum()) for p in executor.model.writer.parameters() if p.grad is not None) > 0


def test_runtime_compaction_keeps_store_under_budget():
    executor = StreamExecutor(tiny().eval())
    with torch.no_grad():
        _, state = run(executor, torch.randint(2, 31, (1, 40)), [40], budget=512)
    assert state.memory.tensor_bytes <= 512
    assert any(record.level > 0 for record in state.memory.records.values())
    assert len(state.active_records) <= 8
    assert state.cache.layers[3].keys.shape[-2] == 5


def test_future_stream_input_cannot_change_outputs_before_it_occurs():
    executor = StreamExecutor(tiny().eval())
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    other = tokens.clone()
    other[:, 4:] = torch.tensor([12, 13, 14, 15])
    with torch.no_grad():
        a, _ = run(executor, tokens, [8])
        b, _ = run(executor, other, [8])
    assert torch.equal(a[:, :4], b[:, :4])


def test_explicit_flush_and_detach_preserve_numerical_state():
    executor = StreamExecutor(tiny().eval())
    state = executor.new_life("life", writer_stride=256)
    with torch.no_grad():
        output = executor.forward(torch.tensor([[2, 3, 4]]), state, flush=True)
    assert output.state.pending is None and len(output.state.memory.records) == 1
    detached = output.state.detached()
    record = next(iter(detached.memory.records.values()))
    assert not record.slots.requires_grad
    assert detached.cache.clock == 3
