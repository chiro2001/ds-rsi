import json

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_episode_objective import SmallVocabularyTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.training.runner import TrainConfig, Trainer
from dsrsi.training.control import ControlChannel


def build(path, *, resume=False, budget=100000):
    torch.manual_seed(87)
    model = tiny()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    cfg = TrainConfig(student_token_budget=budget, distractors=2, seed=11,
                      gradient_accumulation=4, save_every_micro=100)
    return Trainer(model, compiler, path, cfg, base_identity="fixed-native-tiny", resume=resume)


def test_actual_episode_training_resume_mid_accumulation_matches_full_run(tmp_path):
    a = build(tmp_path / "full")
    assert a.run(max_micro_steps=9, install_signals=False) == "stopped"
    b = build(tmp_path / "resumed")
    assert b.run(max_micro_steps=6, install_signals=False) == "stopped"
    assert b.progress.accumulation_index == 2
    c = build(tmp_path / "resumed", resume=True)
    assert c.run(max_micro_steps=9, install_signals=False) == "stopped"
    assert a.progress == c.progress and a.cursor == c.cursor
    for name, value in a.model.state_dict().items():
        assert torch.equal(value, c.model.state_dict()[name]), name
    for p, q in zip(a.model.parameters(), c.model.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    full = [json.loads(line) for line in (a.run_dir / "events.jsonl").read_text().splitlines()]
    resumed = [json.loads(line) for line in (c.run_dir / "events.jsonl").read_text().splitlines()]
    full = [(e["episode_id"], e["loss"], e["student_input_tokens"]) for e in full if e["event"] == "micro"]
    resumed = [(e["episode_id"], e["loss"], e["student_input_tokens"]) for e in resumed if e["event"] == "micro"]
    assert full == resumed


def test_pause_request_at_start_saves_and_old_request_does_not_repause(tmp_path):
    trainer = build(tmp_path)
    ControlChannel(tmp_path).request("pause")
    assert trainer.run(install_signals=False) == "paused"
    assert trainer.progress.micro_step == 0
    resumed = build(tmp_path, resume=True)
    assert resumed.run(max_micro_steps=1, install_signals=False) == "stopped"
    assert resumed.progress.micro_step == 1


def test_budget_completion_flushes_partial_gradients_and_marks_completed(tmp_path):
    trainer = build(tmp_path, budget=1)
    assert trainer.run(install_signals=False) == "completed"
    assert trainer.progress.student_tokens >= 1
    assert trainer.progress.optimizer_step == 1 and trainer.progress.accumulation_index == 0
    _, manifest = trainer.store.load_latest(identity=trainer.identity)
    assert manifest["status"] == "completed"
