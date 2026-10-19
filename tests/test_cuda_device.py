import pytest

from dsrsi.training import device


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.cuda, "device_count", lambda: 1)
    calls = []
    monkeypatch.setattr(device.torch.cuda, "set_device", calls.append)
    monkeypatch.setattr(device.torch, "use_deterministic_algorithms", lambda enabled: None)
    return calls


@pytest.mark.parametrize("requested", ["cuda", "cuda:0"])
def test_single_cuda_device_is_selected(cuda, requested, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert device.resolve_device(requested) == ("cuda:0", 1)
    assert cuda == [0]
    assert device.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("requested", ["cuda:-1", "cuda:x", "cuda:"])
def test_malformed_cuda_selection_is_rejected_before_device_initialization(cuda, requested):
    with pytest.raises(ValueError, match="CUDA device must"):
        device.resolve_device(requested)
    assert not cuda


@pytest.mark.parametrize("available,index", [(False, 0), (True, 1)])
def test_unavailable_cuda_is_rejected(cuda, monkeypatch, available, index):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="unavailable"):
        device.resolve_device(f"cuda:{index}")
    assert not cuda


@pytest.mark.parametrize("world,rank", [(2, 0), (1, 1), (0, 0)])
def test_cuda_does_not_silently_change_the_checkpoint_topology(cuda, monkeypatch, world, rank):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    with pytest.raises(ValueError, match="topology migration"):
        device.resolve_device("cuda")
    assert not cuda
