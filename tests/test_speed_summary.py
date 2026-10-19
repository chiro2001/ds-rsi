import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "summarize_ab_speed", Path(__file__).resolve().parents[1] / "scripts/summarize_ab_speed.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def logs(root):
    paths = []
    for rank, seconds in enumerate(([1.0, 2.0], [1.2, 1.8])):
        path = root / f"rank{rank}.jsonl"
        rows = [{"event": "micro", "episode_id": f"{step}:{rank}",
                 "micro_seconds": seconds[step - 1], "student_input_tokens": 100,
                 "teacher_input_tokens": 10,
                 "progress": {"micro_step": step, "student_tokens": step * 200,
                              "teacher_tokens": step * 20}}
                for step in (1, 2)]
        rows.append({"event": "checkpoint", "save_seconds": 0.8,
                     "progress": {"micro_step": 2}})
        path.write_text("\n".join(json.dumps(row) for row in rows))
        paths.append(path)
    return paths


def test_parallel_rank_tokens_add_but_elapsed_times_do_not(tmp_path):
    result = module.summarize(logs(tmp_path))
    assert result["student_tokens"] == 400
    assert result["episodes"] == 4
    assert result["compute_seconds"] == pytest.approx(3.2)
    assert result["student_tokens_per_second"] == pytest.approx(125)
    assert result["seconds_per_12_episodes"] == pytest.approx(9.6)
    assert result["checkpoint_save_seconds_in_window"] == pytest.approx(0.8)
    assert result["student_tokens_per_second_including_recorded_saves"] == pytest.approx(100)


def test_rejects_missing_or_replayed_rank_micro(tmp_path):
    paths = logs(tmp_path)
    text = paths[1].read_text().splitlines()
    paths[1].write_text("\n".join(text[1:]))
    with pytest.raises(ValueError, match="same committed"):
        module.summarize(paths)
    paths = logs(tmp_path)
    paths[0].write_text(paths[0].read_text() + "\n" + paths[0].read_text().splitlines()[0])
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize(paths)


def test_checks_global_progress_and_token_ledger(tmp_path):
    paths = logs(tmp_path)
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["progress"]["student_tokens"] = 900
        path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="token ledger"):
        module.summarize(paths)
