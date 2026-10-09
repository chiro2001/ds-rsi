import json
import signal
import subprocess
import sys

import pytest
import torch

from dsrsi.training.checkpoint import CheckpointStore, Progress, capture_training_state
from dsrsi.training.control import ControlChannel


def test_control_import_does_not_initialize_torch_or_accelerator_runtime():
    code = "import sys; from dsrsi.training.control import ControlChannel; assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def committed(tmp_path, status):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    progress = Progress("A")
    store = CheckpointStore(tmp_path / "checkpoints")
    pending = store.begin(progress)
    store.save_rank(pending, capture_training_state(
        model, opt, progress=progress, data_cursor={}, life_state={}))
    return store.commit(pending, world_size=1, identity={}, progress=progress, status=status)


def test_acknowledged_pause_does_not_pause_resumed_run_again(tmp_path):
    channel = ControlChannel(tmp_path)
    request = channel.request("pause")
    assert channel.poll() == request
    channel.acknowledge(request, committed(tmp_path, "paused"))
    assert ControlChannel(tmp_path).poll() is None


def test_new_request_is_not_erased_by_acknowledging_previous_request(tmp_path):
    channel = ControlChannel(tmp_path)
    old = channel.request("checkpoint")
    assert channel.poll() == old
    new = channel.request("pause")
    channel.acknowledge(old, committed(tmp_path, "checkpoint"))
    assert channel.poll() == new


def test_signal_handler_only_records_request(tmp_path):
    channel = ControlChannel(tmp_path)
    signals = [getattr(signal, name) for name in ("SIGUSR1", "SIGUSR2", "SIGTERM", "SIGINT")]
    previous = {s: signal.getsignal(s) for s in signals}
    try:
        channel.install_signal_handlers()
        signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
        signal.getsignal(signal.SIGUSR2)(signal.SIGUSR2, None)
        assert not (tmp_path / "control").exists()
        request = channel.poll()
        assert request.action == "pause"
        channel.acknowledge(request, committed(tmp_path, "paused"))
        assert channel.poll() is None
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_uncommitted_or_wrong_status_does_not_acknowledge_pause(tmp_path):
    channel = ControlChannel(tmp_path)
    request = channel.request("pause")
    with pytest.raises(ValueError, match="committed"):
        channel.acknowledge(request, tmp_path / "missing")
    with pytest.raises(ValueError, match="status"):
        channel.acknowledge(request, committed(tmp_path, "checkpoint"))
    assert channel.poll() == request
