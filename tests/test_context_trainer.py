import json

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from test_qwen_bridge import tiny
from test_tool_training import CompactTokenizer
from test_distributed_trainer import equal
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.harness.memory_tools import HarnessStore
from dsrsi.model.qwen35_memory import MemoryQwen
from dsrsi.training.checkpoint import capture_training_state
from dsrsi.training.context_runner import ContextTrainer, ContextTrainConfig


def deeper():
    sample = tiny()
    config = sample.base.config
    config.num_hidden_layers = 8
    config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    config.layer_types *= 2
    base = Qwen3_5TextModel(config)
    head = nn.Linear(16, 32, bias=False)
    head.weight = base.embed_tokens.weight
    return MemoryQwen(base, head, memory_config=sample.memory_config, writer_after=1, reader_before=(2, 6))


def build(path, *, resume=False, unroll=7):
    torch.manual_seed(83)
    config = ContextTrainConfig(student_token_budget=100000, gradient_accumulation=3, seed=17, lora_rank=2,
        unroll_tokens=unroll, window=5, rebase_interval=8, writer_stride=64, retrieval_interval=3,
        memory_tensor_budget=1 << 20, interference_tokens=0, include_tools=False, allow_untrained_memory=True)
    return ContextTrainer(deeper(), EpisodeCompiler(CompactTokenizer()), path, config, base_identity="deep-tiny-context", resume=resume)


def state(trainer):
    payload = capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
        data_cursor=trainer.cursor, life_state=trainer.life, device="cpu")
    # Paths can differ across runs; visible database/life/revision and planned
    # input data must match. IDs and messages are never stripped.
    payload["life_state"]["harness_refs"]["visible"]["path"] = "<journal>"
    case = payload["data_cursor"]["case"]
    if case:
        for frame in case["frames"]:
            frame["harness_after"]["path"] = "<journal>"
    return payload


def test_observation_chunks_advance_bounded_state_without_fake_optimizer_steps(tmp_path):
    trainer = build(tmp_path)
    assert trainer.run(max_micro_steps=2, install_signals=False) == "stopped"
    assert trainer.stream.cache.clock == trainer.progress.student_tokens
    assert trainer.teacher.cache.clock == trainer.stream.cache.clock
    assert trainer.progress.optimizer_step == 0 and trainer.progress.accumulation_index == 0
    assert all(p.grad is None for p in trainer.model.parameters())
    assert trainer.stream.cache.layers[3].keys.shape[-2] <= 5
    journal = HarnessStore.from_state_dict(trainer.harness_reference)
    assert journal.last_clock <= trainer.stream.cache.clock
    journal.close()


def test_continuous_resume_mid_message_matches_full_numeric_state_and_gradients(tmp_path):
    full = build(tmp_path / "full")
    assert full.run(max_micro_steps=35, install_signals=False) == "stopped"
    part = build(tmp_path / "resumed")
    assert part.run(max_micro_steps=17, install_signals=False) == "stopped"
    assert part.cursor["frame_offset"] > 0
    resumed = build(tmp_path / "resumed", resume=True)
    assert resumed.run(max_micro_steps=35, install_signals=False) == "stopped"
    equal(state(full), state(resumed))
    assert resumed.stream.cache.clock > resumed.config.window
    assert resumed.stream.memory.records and resumed.progress.optimizer_step > 0
    assert all(p.grad is None for p in resumed.model.writer.parameters())
    assert all(p.grad is None for name, p in resumed.model.address.named_parameters() if not name.startswith("query."))
    assert not any(s.recurrent_states[0].requires_grad for s in resumed.stream.cache.layers if s.recurrent_states[0] is not None)
    events = [json.loads(line) for line in (resumed.run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "micro" and e["queries"] == 0 for e in events)
    assert any(e["event"] == "micro" and e["queries"] > 0 for e in events)


def test_single_token_unroll_supervises_every_answer_token_once_including_first_and_end(tmp_path):
    trainer = build(tmp_path, unroll=1)
    trainer._ensure_case()
    case = trainer.cursor["case"]
    count = sum(len(frame["input_ids"]) for frame in case["frames"])
    expected = sum(frame["total_targets"] for frame in case["frames"])
    assert expected > 0
    assert trainer.run(max_micro_steps=count, install_signals=False) == "stopped"
    rows = [json.loads(line) for line in (trainer.run_dir / "events.jsonl").read_text().splitlines()]
    assert sum(row["supervised_tokens"] for row in rows if row["event"] == "micro") == expected
    assert trainer.stream.cache.clock == count and trainer.cursor["case"] is None


def test_continuous_export_restores_frozen_writer_and_keys_without_restoring_training_state(tmp_path):
    from dsrsi.evaluation.checkpoint import load_context_weights
    trainer = build(tmp_path)
    trainer.run(max_micro_steps=17, install_signals=False)
    torch.manual_seed(993)
    model = deeper()
    _, manifest = load_context_weights(model, trainer.store, base_identity="deep-tiny-context")
    assert manifest["progress"]["stage"] == "C_context"
    for name, value in trainer.model.state_dict().items():
        if not name.startswith(("base.", "head.")) or ".lora_" in name:
            assert torch.equal(value, model.state_dict()[name]), name
