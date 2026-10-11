from dataclasses import replace

import pytest
import torch

from dsrsi.memory.modules import MemoryConfig, MemoryWriter


def saturated_writer(weight):
    cfg = MemoryConfig(hidden_size=4, memory_dim=4, slots=2, heads=1,
                       intermediate=8, key_dim=2, writer_anchor_weight=weight)
    writer = MemoryWriter(cfg)
    with torch.no_grad():
        writer.queries.zero_()
        writer.queries[:, 0] = 1
        for layer in (writer.q, writer.v, writer.o):
            layer.weight.copy_(torch.eye(4))
        writer.k.weight.copy_(torch.eye(4) * 100)
        writer.ffn.down.weight.zero_()
    return writer


def test_late_content_and_gradients_survive_saturated_learned_attention():
    source = torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0], [0, 1., 0, 0], [0, 1., 0, 0]]])
    changed = source.clone()
    changed[0, -1] = torch.tensor([0, 0, 1., 0])
    old = saturated_writer(0)
    assert torch.equal(old(source), old(changed))
    writer = saturated_writer(1)
    assert not torch.allclose(writer(source), writer(changed))
    source.requires_grad_(True)
    writer(source)[0, -1, 0].backward()
    assert source.grad[0, -1].abs().sum() > 0


@pytest.mark.parametrize("length", [1, 2, 5, 11])
def test_anchor_pool_handles_short_sources_and_ignores_nan_padding(length):
    torch.manual_seed(21)
    writer = MemoryWriter(MemoryConfig(hidden_size=4, memory_dim=4, heads=1,
                                      slots=8, intermediate=8, writer_anchor_weight=1))
    source = torch.randn(1, length, 4)
    expected = writer(source)
    padded = torch.cat([source, torch.full((1, 3, 4), float("nan"))], dim=1)
    valid = torch.tensor([[True] * length + [False] * 3])
    got = writer(padded, valid)
    # Different sequence lengths can select GEMM kernels with a few FP32 ULPs
    # of rounding difference. NaN padding must still have no semantic effect.
    torch.testing.assert_close(expected, got, rtol=1e-6, atol=5e-7)
    assert torch.isfinite(got).all()


def test_anchor_transition_is_explicit_and_new_checkpoint_restores_exactly(tmp_path):
    pytest.importorskip("transformers")
    from test_qwen_bridge import tiny
    from test_episode_objective import SmallVocabularyTokenizer
    from test_distributed_trainer import equal
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.training.checkpoint import CheckpointError, capture_training_state
    from dsrsi.evaluation.checkpoint import load_ab_weights
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    cfg = TrainConfig(student_token_budget=100000, distractors=2, seed=17)
    torch.manual_seed(15)
    source = Trainer(tiny(), compiler, tmp_path / "source", cfg, base_identity="anchor-tiny")
    source.run(max_micro_steps=4, install_signals=False)
    assert "writer_anchor_weight" not in source.model.memory_architecture_identity()["memory_config"]
    def model():
        torch.manual_seed(15)
        m = tiny()
        m.memory_config = replace(m.memory_config, writer_anchor_weight=1)
        m.writer.anchor_weight = 1
        return m
    with pytest.raises(CheckpointError, match="architecture"):
        load_ab_weights(model(), source.store, base_identity="anchor-tiny")
    with pytest.raises(ValueError, match="data revision or curriculum"):
        Trainer(model(), compiler, tmp_path / "implicit", cfg, base_identity="anchor-tiny", initialize_from=source.run_dir)
    def make(name, resume=False):
        return Trainer(model(), compiler, tmp_path / name, cfg, base_identity="anchor-tiny", resume=resume,
            initialize_from=None if resume else source.run_dir, allow_writer_anchor_change=not resume)
    full, part = make("full"), make("part")
    assert full.progress.micro_step == 0 and not full.optimizer.state
    assert full.identity["initialization"]["writer_anchor_change"]["target"]["writer_anchor_version"] == "position_pool_v1"
    full.run(max_micro_steps=5, install_signals=False)
    part.run(max_micro_steps=3, install_signals=False)
    resumed = make("part", resume=True)
    resumed.run(max_micro_steps=5, install_signals=False)
    def state(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress, data_cursor=t.cursor, life_state=t.life)
    equal(state(full), state(resumed))


def test_anchored_writer_stream_chunks_pending_snapshot_and_gradient_remain_causal():
    pytest.importorskip("transformers")
    from test_qwen_bridge import tiny
    from test_life_stream import run
    from test_distributed_trainer import equal
    from dsrsi.model.streaming import StreamExecutor
    from dsrsi.state.life_state import LifeState
    model = tiny().eval()
    model.memory_config = replace(model.memory_config, writer_anchor_weight=1)
    model.writer.anchor_weight = 1
    executor = StreamExecutor(model)
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    with torch.no_grad():
        whole, state = run(executor, tokens, [12])
        chunks, chunk_state = run(executor, tokens, [3, 7, 2])
        assert torch.equal(whole, chunks)
        equal(state.state_dict(), chunk_state.state_dict())
        _, partial = run(executor, tokens[:, :7], [7])
        assert partial.pending.shape[1] == 3
        continued = executor.forward(tokens[:, 7:], partial)
        resumed = executor.forward(tokens[:, 7:], LifeState.from_state_dict(partial.state_dict()))
        assert torch.equal(continued.hidden, resumed.hidden)
        equal(continued.state.state_dict(), resumed.state.state_dict())
    model.train()
    source = executor.forward(tokens[:, :4], executor.new_life("gradient", writer_stride=4))
    query = executor.forward(tokens[:, 4:7], source.state, write=False, force_read_at=[0])
    query.hidden[..., 0].sum().backward()
    assert sum(float(p.grad.square().sum()) for p in model.writer.parameters() if p.grad is not None) > 0
