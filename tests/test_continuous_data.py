from pathlib import Path

import pytest

from test_tool_training import CompactTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.continuous import FrameCompiler, ContinuousGenerator
from dsrsi.harness.memory_tools import HarnessStore


def test_append_frames_keep_old_assistant_tokens_and_supervise_only_body_and_end():
    c = EpisodeCompiler(CompactTokenizer())
    frames = FrameCompiler(c)
    first = frames.frame({"role": "user", "content": "Remember Q7-492."}, wall_time=1)
    answer = frames.frame({"role": "assistant", "content": "Q7-492"}, wall_time=2)
    old = first.input_ids + answer.input_ids
    next_message = frames.frame({"role": "user", "content": "Now continue."}, wall_time=3)
    combined = old + next_message.input_ids
    assert combined[:len(old)] == old
    assert all(label == -100 for label in answer.labels[:answer.read_at + 1])
    assert answer.labels[answer.read_at + 1] != -100
    assert answer.labels.count(c.tokenizer.eos_token_id) == 1
    assert answer.labels[-1] == -100  # template newline
    assert all(label == -100 for label in first.labels + next_message.labels)


def test_native_qwen_history_rewrite_is_not_applied_to_an_executed_tape():
    root = Path("artifacts/tokenizer/Qwen3.5-0.8B")
    if not (root / "tokenizer.json").exists():
        pytest.skip("native tokenizer artifact is checked in integration environment")
    from transformers import AutoTokenizer
    c = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    messages = [{"role": "user", "content": "保存偏好"}, {"role": "assistant", "content": "已保存。"}]
    old = c.encode(messages)
    new = c.encode(messages + [{"role": "user", "content": "现在继续"}])
    assert new[:len(old)] != old  # native rendering strips old empty <think>
    compiler = FrameCompiler(c)
    tape = [compiler.frame(message, wall_time=number).input_ids for number, message in enumerate(messages)]
    assert tape[0] + tape[1] == old
    assert compiler.frame({"role": "tool", "name": "memory.write", "content": {"ok": True}}, wall_time=3).input_ids


def test_teacher_plan_future_actions_are_invisible_until_consumed_reference(tmp_path):
    c = EpisodeCompiler(CompactTokenizer())
    generator = ContinuousGenerator(c, interference_tokens=0)
    store = HarnessStore(tmp_path / "journal.sqlite", "life", database_id=generator.database_id)
    initial = store.state_dict()
    store.close()
    a = generator.plan(0, life_id="life", journal=tmp_path / "journal.sqlite", harness_reference=initial,
                       clock=0, startup=True, family="save")
    b = generator.plan(0, life_id="life", journal=tmp_path / "journal.sqlite", harness_reference=initial,
                       clock=0, startup=True, family="save")
    assert a == b
    hidden = HarnessStore.from_state_dict(initial)
    assert hidden.memory == {}
    action = next(frame for frame in a["frames"] if frame["supervision_kind"] == "tool")
    observed = HarnessStore.from_state_dict(action["harness_after"])
    assert len(observed.memory) == 1
    assert action["role"] == "assistant"
    assert all(label == -100 for f in a["frames"] if f["role"] == "tool" for label in f["labels"])
    hidden.close()
    observed.close()
