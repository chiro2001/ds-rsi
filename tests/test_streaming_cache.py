import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.state.streaming_cache import StreamingCache


def execute(model, inputs, pieces, *, window=None, interval=None):
    cache = StreamingCache(model.base.config.layer_types, window=window, rebase_interval=interval)
    result = []
    offset = 0
    for size in pieces:
        remaining = size
        while remaining:
            limit = cache.until_rebase()
            n = min(remaining, limit) if limit is not None else remaining
            output = model.forward_cached(inputs[:, offset:offset + n], cache, memory_enabled=False)
            cache = output.cache
            result.append(output.hidden)
            offset += n
            remaining -= n
    assert offset == inputs.shape[1]
    return torch.cat(result, 1), cache


def test_unbounded_cold_cache_matches_native_qwen():
    model = tiny().eval()
    inputs = torch.tensor([[2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        native = model.base(inputs, use_cache=False).last_hidden_state
        streamed, cache = execute(model, inputs, [6])
    torch.testing.assert_close(streamed, native, atol=1e-6, rtol=1e-6)
    assert cache.clock == 6
    assert cache.layers[3].keys.shape[-2] == 6


def test_chunked_continuation_matches_token_steps_with_bounded_rebased_positions():
    torch.manual_seed(77)
    model = tiny().eval()
    inputs = torch.randint(2, 31, (1, 29))
    with torch.no_grad():
        tokens, a = execute(model, inputs, [1] * 29, window=5, interval=8)
        chunks, b = execute(model, inputs, [7, 3, 9, 10], window=5, interval=8)
    torch.testing.assert_close(tokens, chunks, atol=2e-5, rtol=2e-5)
    assert a.clock == b.clock == 29
    assert a.position_origin == b.position_origin == 24
    assert a.layers[3].keys.shape[-2] == b.layers[3].keys.shape[-2] == 5
    assert a.layers[3].kv_start == b.layers[3].kv_start == 24
    torch.testing.assert_close(a.layers[0].recurrent_states[0], b.layers[0].recurrent_states[0], atol=1e-6, rtol=1e-5)


def test_late_future_tokens_do_not_change_earlier_hidden_states():
    model = tiny().eval()
    inputs = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    changed = inputs.clone()
    changed[:, 4:] = torch.tensor([13, 14, 15, 16])
    with torch.no_grad():
        a, _ = execute(model, inputs, [8], window=4, interval=8)
        b, _ = execute(model, changed, [8], window=4, interval=8)
    assert torch.equal(a[:, :4], b[:, :4])


def test_functional_cache_does_not_mutate_parent_and_snapshot_resume_matches():
    model = tiny().eval()
    inputs = torch.tensor([[2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        _, cache = execute(model, inputs[:, :4], [4], window=3, interval=5)
        before = cache.state_dict()
        first = model.forward_cached(inputs[:, 4:5], cache, memory_enabled=False)
        # Native single-token conv update must only touch the fork's clone.
        for index, layer in enumerate(cache.layers):
            if layer.conv_states[0] is not None:
                assert torch.equal(layer.conv_states[0], before["layers"][index]["conv_states"][0])
        restored = StreamingCache.from_state_dict(before)
        second = model.forward_cached(inputs[:, 4:5], restored, memory_enabled=False)
    assert torch.equal(first.hidden, second.hidden)
    for a, b in zip(first.cache.layers, second.cache.layers):
        if a.keys is not None:
            assert torch.equal(a.keys, b.keys)
        if a.recurrent_states[0] is not None:
            assert torch.equal(a.recurrent_states[0], b.recurrent_states[0])


def test_future_loss_can_cross_functional_gdn_state_to_old_memory_write():
    model = tiny().train()
    record = model.write_records(torch.tensor([[2, 3, 4, 5]]), stream_ids=["life"],
                                 logical_begins=[0], record_ids=["source"])
    cache = StreamingCache(model.base.config.layer_types, window=5, rebase_interval=10)
    cache.clock = 4
    first = model.forward_cached(torch.tensor([[7, 8]]), cache, records=record, stream_id="life")
    # Next span does not read M directly; only the prior numerical states carry it.
    second = model.forward_cached(torch.tensor([[9, 10]]), first.cache, memory_enabled=False)
    second.hidden[..., 0].sum().backward()
    norm = sum(float(p.grad.square().sum()) for p in model.writer.parameters() if p.grad is not None)
    assert norm > 0


def test_invalid_snapshot_and_rebase_crossing_are_rejected():
    model = tiny().eval()
    cache = StreamingCache(model.base.config.layer_types, window=4, rebase_interval=3)
    with pytest.raises(ValueError, match="split"):
        model.forward_cached(torch.tensor([[2, 3, 4, 5]]), cache, memory_enabled=False)
    data = cache.state_dict()
    data["position_origin"] = 1
    with pytest.raises(ValueError, match="snapshot"):
        StreamingCache.from_state_dict(data)
