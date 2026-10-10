from dataclasses import replace
import copy

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.state.streaming_cache import StreamingCache


def model():
    torch.manual_seed(29)
    m = tiny().eval()
    m.memory_config = replace(m.memory_config, reader_gate_mode="lookup", writer_anchor_weight=1)
    m.writer.anchor_weight = 1
    with torch.no_grad():
        for r in m.readers.values():
            r.gate.weight.normal_(std=0.2)
    return m


def source(m):
    return m.write_records(torch.tensor([[2, 3, 4, 5]]), stream_ids=["life"], logical_begins=[0], record_ids=["r"])


def prefix(m):
    cache = StreamingCache(m.base.config.layer_types, window=None, rebase_interval=None)
    cache.clock = cache.position_origin = 4
    for token in (7, 8):
        cache = m.forward_cached(torch.tensor([[token]]), cache, memory_enabled=False).cache
    return cache


def test_lookup_gate_native_cached_consistency_and_no_future_answer_effect():
    m = model()
    ids = torch.tensor([[7, 8, 9, 10, 11, 12]])
    with torch.no_grad():
        records = source(m)
        native = m(ids, records=[records], query_indices=[2], logical_begins=[4], stream_ids=["life"])
        cache = prefix(m)
        hidden, gates = [], None
        for token in (9, 10, 11, 12):
            result = m.forward_cached(torch.tensor([[token]]), cache, records=records,
                                      stream_id="life", lookup_position=6)
            hidden.append(result.hidden)
            cache = result.cache
            if gates is None:
                gates = dict(cache.read_gates)
            else:
                assert all(torch.equal(gates[k], cache.read_gates[k]) for k in gates)
        torch.testing.assert_close(torch.cat(hidden, 1), native.hidden[:, 2:], atol=3e-5, rtol=3e-5)
        trace = []
        handles = [r.register_forward_hook(lambda _m, _a, kw, _o: trace.append(kw["read_gate"].clone()),
                                           with_kwargs=True) for r in m.readers.values()]
        m(ids, records=[records], query_indices=[2], logical_begins=[4], stream_ids=["life"])
        altered = ids.clone()
        altered[:, 3:] = torch.tensor([[20, 21, 22]])
        other = m(altered, records=[records], query_indices=[2], logical_begins=[4], stream_ids=["life"])
        for h in handles:
            h.remove()
        assert all(torch.equal(a, b) for a, b in zip(trace[:2], trace[2:]))
        assert torch.equal(native.hidden[:, :3], other.hidden[:, :3])


def test_lookup_snapshot_reuse_refresh_detach_and_missing_decision_rejection():
    m = model()
    with torch.no_grad():
        records = source(m)
        before = prefix(m)
        first = m.forward_cached(torch.tensor([[9]]), before, records=records, stream_id="life", lookup_position=6)
        assert not before.read_gates and first.cache.read_gate_position == 6
        snapshot = first.cache.state_dict()
        restored = StreamingCache.from_state_dict(snapshot)
        a = m.forward_cached(torch.tensor([[10]]), first.cache, records=records, stream_id="life", lookup_position=6)
        b = m.forward_cached(torch.tensor([[10]]), restored, records=records, stream_id="life", lookup_position=6)
        assert torch.equal(a.hidden, b.hidden)
        newer = m.forward_cached(torch.tensor([[11]]), a.cache, records=records, stream_id="life", lookup_position=8)
        assert newer.cache.read_gate_position == 8
        assert a.cache.read_gate_position == 6
        assert any(not torch.equal(newer.cache.read_gates[k], a.cache.read_gates[k]) for k in a.cache.read_gates)
        corrupt = restored.fork()
        corrupt.read_gates.pop(next(iter(corrupt.read_gates)))
        with pytest.raises(ValueError, match="missing"):
            m.forward_cached(torch.tensor([[10]]), corrupt, records=records, stream_id="life", lookup_position=6)
        with pytest.raises(ValueError, match="future"):
            m.forward_cached(torch.tensor([[10]]), restored, records=records, stream_id="life", lookup_position=100)
    fresh = m.forward_cached(torch.tensor([[9]]), prefix(m), records=source(m), stream_id="life", lookup_position=6)
    assert any(g.grad_fn is not None for g in fresh.cache.read_gates.values())
    detached = fresh.cache.detached()
    assert all(g.grad_fn is None for g in detached.read_gates.values())
    fresh.hidden[..., 0].sum().backward()
    assert any(r.gate.weight.grad is not None and r.gate.weight.grad.abs().sum() > 0 for r in m.readers.values())


@pytest.mark.parametrize("kind", ["nan", "range", "position", "partial"])
def test_lookup_cache_refuses_corrupt_decisions(kind):
    m = model()
    with torch.no_grad():
        saved = m.forward_cached(torch.tensor([[9]]), prefix(m), records=source(m),
                                  stream_id="life", lookup_position=6).cache.state_dict()
    if kind == "partial":
        saved.pop("read_gates")
    elif kind == "position":
        saved["read_gate_position"] = saved["clock"]
    else:
        saved["read_gates"][next(iter(saved["read_gates"]))].fill_(float("nan") if kind == "nan" else 2)
    with pytest.raises(ValueError, match="lookup-gate"):
        StreamingCache.from_state_dict(saved)


def test_lookup_life_external_chunks_and_pending_snapshot_match():
    from test_life_stream import run
    from test_distributed_trainer import equal
    from dsrsi.model.streaming import StreamExecutor
    from dsrsi.state.life_state import LifeState
    executor = StreamExecutor(model())
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    with torch.no_grad():
        a, life_a = run(executor, tokens, [12])
        b, life_b = run(executor, tokens, [3, 2, 7])
        assert torch.equal(a, b)
        equal(life_a.state_dict(), life_b.state_dict())
        _, partial = run(executor, tokens[:, :7], [7])
        assert partial.pending.shape[1] == 3 and partial.cache.read_gates
        continued = executor.forward(tokens[:, 7:], partial)
        resumed = executor.forward(tokens[:, 7:], LifeState.from_state_dict(partial.state_dict()))
        assert torch.equal(continued.hidden, resumed.hidden)
        equal(continued.state.state_dict(), resumed.state.state_dict())


def test_gate_transition_requires_explicit_new_run_and_keeps_partial_resume_exact(tmp_path):
    from test_episode_objective import SmallVocabularyTokenizer
    from test_distributed_trainer import equal
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.checkpoint import CheckpointError, capture_training_state
    from dsrsi.evaluation.checkpoint import load_ab_weights
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    config = TrainConfig(student_token_budget=100000, distractors=2, seed=17,
                         curriculum="facts", memory_candidates=1, counterfactual_weight=1)
    source_trainer = Trainer(tiny(), compiler, tmp_path / "source", config, base_identity="lookup-tiny")
    source_trainer.run(max_micro_steps=4, install_signals=False)
    with pytest.raises(CheckpointError, match="architecture"):
        load_ab_weights(model(), source_trainer.store, base_identity="lookup-tiny", allow_writer_anchor_change=True)
    def make(name, resume=False):
        return Trainer(model(), compiler, tmp_path / name, config, base_identity="lookup-tiny", resume=resume,
            initialize_from=None if resume else source_trainer.run_dir,
            allow_reader_gate_change=not resume, allow_writer_anchor_change=not resume)
    full = make("full")
    assert "reader_gate_change" in full.identity["initialization"]
    full.run(max_micro_steps=5, install_signals=False)
    partial = make("resumed")
    partial.run(max_micro_steps=3, install_signals=False)
    continued = make("resumed", resume=True)
    continued.run(max_micro_steps=5, install_signals=False)
    def state(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress, data_cursor=t.cursor, life_state=t.life)
    equal(state(full), state(continued))
