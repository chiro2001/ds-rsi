import hashlib
import json
from pathlib import Path
import re

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_compiler import BoundaryTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.tool_generate import ToolEpisodeGenerator, TOOL_FAMILIES
from dsrsi.harness.memory_tools import HarnessStore
from dsrsi.model.lora import LoRAConfig, LoRALinear, install_lora
from dsrsi.training.tool_objective import ToolObjective
from dsrsi.training.tool_runner import ToolTrainer, ToolTrainConfig


class CompactTokenizer(BoundaryTokenizer):
    def encode(self, text, **kwargs):
        return [1 if part == "\x01" else int.from_bytes(hashlib.sha256(part.encode()).digest()[:2], "big") % 29 + 2
                for part in re.findall(r"\w+|[^\w]", text)]


def build(path, *, resume=False):
    torch.manual_seed(83)
    config = ToolTrainConfig(student_token_budget=100000, seed=17, lora_rank=2,
                             max_query_tokens=4096, allow_untrained_memory=True, memory_replay_fraction=0)
    return ToolTrainer(tiny(), EpisodeCompiler(CompactTokenizer(), max_query_tokens=config.max_query_tokens),
                       path, config, base_identity="tiny-tools", resume=resume)


@pytest.mark.parametrize("family", TOOL_FAMILIES)
def test_generated_tool_events_are_executed_and_deterministic(family, tmp_path):
    a = ToolEpisodeGenerator(seed=23, split="dev").episode(4, family=family, language="zh")
    b = ToolEpisodeGenerator(seed=23, split="dev").episode(4, family=family, language="zh")
    assert a == b
    compiler = EpisodeCompiler(CompactTokenizer())
    compiled = compiler.compile(a)
    assert all(response.labels[:response.prompt_length] == [-100] * response.prompt_length for response in compiled.responses)
    if family == "history":
        reply = a["trajectory"][-1]["content"]
        search = next(m["content"] for m in a["trajectory"] if m.get("name") == "history.search")
        assert reply not in json.dumps(search) and reply in a["source_events"][0]["content"]
        assert next(m for m in a["trajectory"] if m.get("name") == "history.open")["content"]["found"]
    if family == "delete":
        assert a["annotations"]["expected_final_explicit_store"] == {}
        assert a["annotations"]["deletion_scope"] == "explicit_memory_only"


def test_tool_ce_reaches_real_lora_with_no_latent_memory():
    model = tiny()
    install_lora(model, LoRAConfig(rank=2))
    row = ToolEpisodeGenerator().episode(1, family="write", language="en")
    row["training_response_index"] = 0
    result = ToolObjective(model, EpisodeCompiler(CompactTokenizer()))(row)
    result.loss.backward()
    assert result.tool_events == 1 and result.reply_events == 0 and result.teacher_tokens == 0
    assert result.student_tokens > result.supervised_tokens
    assert sum(float(m.lora_b.grad.square().sum()) for m in model.modules() if isinstance(m, LoRALinear)) > 0
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(m.original.weight.grad is None for m in model.modules() if isinstance(m, LoRALinear))


def test_keep_uses_original_backbone_teacher_and_still_trains_adapters():
    model = tiny()
    install_lora(model, LoRAConfig(rank=2))
    row = ToolEpisodeGenerator().episode(2, family="language_preservation", language="en")
    result = ToolObjective(model, EpisodeCompiler(CompactTokenizer()))(row)
    assert 0 < result.teacher_tokens < result.student_tokens and result.keep_kl >= -1e-6
    result.loss.backward()
    assert any(m.lora_b.grad is not None for m in model.modules() if isinstance(m, LoRALinear))
    assert any(p.grad is not None for p in model.readers.parameters())
    assert all(p.grad is None for p in model.writer.parameters())


def test_tool_resume_mid_accumulation_matches_continuous_and_restores_frozen_memory(tmp_path):
    full = build(tmp_path / "full")
    assert full.run(max_micro_steps=6, install_signals=False) == "stopped"
    part = build(tmp_path / "resumed")
    assert part.run(max_micro_steps=3, install_signals=False) == "stopped"
    resumed = build(tmp_path / "resumed", resume=True)
    assert resumed.progress.accumulation_index == 3
    assert resumed.run(max_micro_steps=6, install_signals=False) == "stopped"
    assert full.progress == resumed.progress and full.cursor == resumed.cursor
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    for a, b in zip(full.model.parameters(), resumed.model.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    reference = resumed.life["harness_refs"]["last_teacher_episode"]
    store = HarnessStore.from_state_dict(reference)
    assert store.life_id == reference["life_id"]
    store.close()
    journal = Path(reference["path"])
    journal.rename(journal.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError, match="durable"):
        build(tmp_path / "resumed", resume=True)


def test_pause_before_first_tool_episode_still_has_a_resumable_journal(tmp_path):
    from dsrsi.training.control import ControlChannel
    trainer = build(tmp_path)
    ControlChannel(tmp_path).request("pause")
    assert trainer.run(install_signals=False) == "paused" and trainer.progress.micro_step == 0
    resumed = build(tmp_path, resume=True)
    assert resumed.run(max_micro_steps=1, install_signals=False) == "stopped"


def test_c_memory_replay_updates_lora_and_writer_without_using_tools(tmp_path):
    torch.manual_seed(83)
    config = ToolTrainConfig(student_token_budget=100000, seed=17, lora_rank=2, replay_distractors=2,
                             max_query_tokens=4096, allow_untrained_memory=True, memory_replay_fraction=1)
    trainer = ToolTrainer(tiny(), EpisodeCompiler(CompactTokenizer(), max_query_tokens=config.max_query_tokens),
                          tmp_path, config, base_identity="tiny-replay")
    assert trainer.run(max_micro_steps=1, install_signals=False) == "stopped"
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in trainer.model.writer.parameters())
    assert any(m.lora_b.grad is not None and bool(m.lora_b.grad.abs().sum() > 0)
               for m in trainer.model.modules() if isinstance(m, LoRALinear))
    assert trainer.life["harness_refs"] == {}
