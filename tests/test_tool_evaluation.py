import copy
import re

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_tool_training import CompactTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.tool_generate import ToolEpisodeGenerator
from dsrsi.evaluation.tools import ToolEvaluator
from dsrsi.evaluation.checkpoint import load_c_tool_weights
from dsrsi.training.tool_runner import ToolTrainConfig, ToolTrainer


def native(name, arguments):
    return "<tool_call>\n<function=" + name + ">\n" + "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()) + "</function>\n</tool_call>"


def test_tool_rollout_executes_generated_calls_and_never_reads_teacher_future(tmp_path):
    generator = ToolEpisodeGenerator(split="dev", journal_path=tmp_path / "dev.sqlite")
    row = generator.episode(2, family="write", language="zh", select_one=False)
    evaluator = ToolEvaluator(tiny(), EpisodeCompiler(CompactTokenizer()))
    def policy(messages):
        key, value = re.fullmatch(r"请把(.+)记为：(.+)。", messages[0]["content"]).groups()
        if len(messages) == 1:
            return native("memory.write", {"key": key, "text": value}), True, 1
        assert messages[-1]["role"] == "tool" and messages[-1]["content"]["ok"]
        return f"已保存：{value}。", True, 1
    evaluator.generate = policy
    a = evaluator.episode(row)
    assert a["end_to_end_success"] and a["explicit_state_match"]
    poisoned = copy.deepcopy(row)
    poisoned["trajectory"][1]["tool_calls"][0]["function"]["arguments"]["text"] = "forbidden-future-label"
    poisoned["trajectory"][-1]["content"] = "different expected answer"
    b = evaluator.episode(poisoned)
    assert a["predicted_calls"] == b["predicted_calls"] and a["generations"] == b["generations"]
    assert a["observations"] == b["observations"] and not b["end_to_end_success"]


def test_update_copies_record_id_from_real_read_result_and_changes_explicit_state(tmp_path):
    row = ToolEpisodeGenerator(split="dev", journal_path=tmp_path / "dev.sqlite").episode(
        2, family="update", language="zh", select_one=False)
    evaluator = ToolEvaluator(tiny(), EpisodeCompiler(CompactTokenizer()))
    def policy(messages):
        key, updated = re.fullmatch(r"把明文记忆中的(.+)改成(.+)。", messages[0]["content"]).groups()
        if len(messages) == 1:
            return native("memory.read", {"key": key}), True, 1
        if messages[-1]["name"] == "memory.read":
            record = messages[-1]["content"]["record"]
            return native("memory.update", {"record_id": record["record_id"], "text": updated}), True, 1
        return f"已更新：{updated}。", True, 1
    evaluator.generate = policy
    report = evaluator.episode(row)
    assert report["end_to_end_success"] and len(report["observations"]) == 2


def test_tool_evaluation_rejects_training_worlds_and_malformed_generated_actions(tmp_path):
    generator = ToolEpisodeGenerator(split="dev", journal_path=tmp_path / "dev.sqlite")
    row = generator.episode(1, family="write", select_one=False)
    evaluator = ToolEvaluator(tiny(), EpisodeCompiler(CompactTokenizer()))
    evaluator.generate = lambda messages: ("<tool_call><function=memory.write>", True, 1)
    report = evaluator.episode(row)
    assert report["error"] and not report["predicted_calls"] and not report["end_to_end_success"]
    row["split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        evaluator.episode(row)


def test_generation_budget_does_not_execute_a_tool_before_assistant_end(tmp_path):
    row = ToolEpisodeGenerator(split="dev", journal_path=tmp_path / "dev.sqlite").episode(
        1, family="write", select_one=False)
    evaluator = ToolEvaluator(tiny(), EpisodeCompiler(CompactTokenizer()))
    evaluator.generate = lambda messages: (native("memory.write", {"key": "unexpected", "text": "value"}), False, 1)
    report = evaluator.episode(row)
    assert report["stop_reason"] == "generation_budget" and report["actual_explicit_store"] == {}
    assert not report["predicted_calls"] and not report["requested_operation_success"]


def test_c_tools_model_export_does_not_require_optimizer_restore_and_restores_frozen_auxiliary(tmp_path):
    from test_tool_training import build
    trainer = build(tmp_path)
    trainer.run(max_micro_steps=2, install_signals=False)
    torch.manual_seed(991)
    exported = tiny()
    rng = torch.get_rng_state().clone()
    # Adapter construction consumes RNG, while checkpoint loading does not
    # restore the saved training RNG. The exported values match regardless of
    # initialization seed, including the frozen Consolidator.
    _, manifest = load_c_tool_weights(exported, trainer.store, base_identity="tiny-tools")
    assert manifest["progress"]["stage"] == "C_tools"
    for name, value in trainer.model.state_dict().items():
        if not name.startswith(("base.", "head.")) or ".lora_" in name:
            assert torch.equal(value, exported.state_dict()[name]), name
    assert not torch.equal(torch.get_rng_state(), rng)
