import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


spec = importlib.util.spec_from_file_location("observe_cuda_process",
    Path(__file__).resolve().parents[1] / "scripts/observe_cuda_process.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def test_proc_stat_parser_handles_parentheses_and_spaces():
    values = ["S"] + ["0"] * 18 + ["12345", "999"]
    assert observer.process_start_ticks("123 (a (test) child) " + " ".join(values)) == 12345


@pytest.fixture
def child():
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", "timing-fixture"],
                               stdin=subprocess.PIPE)
    yield process
    if not process.stdin.closed:
        process.stdin.close()
    process.wait(timeout=10)


@pytest.mark.skipif(not hasattr(os, "pidfd_open") or not hasattr(time, "CLOCK_BOOTTIME"),
                    reason="Linux process timing interface")
def test_real_child_exit_and_backward_utc_do_not_corrupt_elapsed(monkeypatch, child):
    ticks = observer.process_start_ticks(Path(f"/proc/{child.pid}/stat").read_text())
    before = time.clock_gettime(time.CLOCK_BOOTTIME)
    assert 0 <= before - ticks / os.sysconf("SC_CLK_TCK") < 10
    readings = iter([1000.0, 997.0])
    monkeypatch.setattr(observer, "realtime_now", lambda: next(readings))
    result = observer.observe_process(child.pid, required_arguments=["timing-fixture"],
        on_attached=lambda _: child.stdin.close())
    assert result["exit_observed"] and result["process_duration_seconds"] > 0
    assert result["observer_duration_realtime_seconds"] == -3
    assert result["observer_duration_monotonic_seconds"] >= 0
    assert result["observer_realtime_minus_monotonic_seconds"] <= -3
    assert not result["signals_sent"] and not result["training_state_modified"]


@pytest.mark.skipif(not hasattr(os, "pidfd_open") or not hasattr(time, "CLOCK_BOOTTIME"),
                    reason="Linux process timing interface")
def test_wrong_command_refused_without_signaling_child(child):
    with pytest.raises(ValueError, match="command differs"):
        observer.observe_process(child.pid, required_arguments=["not-this-process"])
    assert child.poll() is None


def test_observation_result_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "result.json"
    observer.write_new_json(path, {"elapsed": 1})
    with pytest.raises(FileExistsError):
        observer.write_new_json(path, {"elapsed": 2})
    assert '"elapsed": 1' in path.read_text()
