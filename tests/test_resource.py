import subprocess
from pathlib import Path

import pytest

from dsrsi.training import resource
from dsrsi.training import device


@pytest.fixture
def host22(monkeypatch):
    monkeypatch.setattr(resource.socket, "gethostname", lambda: "host22")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")


@pytest.mark.parametrize("mapping", ["", "8,9,10,11,12,13,14,15", "12", "0", "8,8", "9,8", "8,x"])
def test_rejects_old_busy_and_ambiguous_mappings(host22, monkeypatch, mapping):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", mapping)
    with pytest.raises(RuntimeError):
        resource.authorized_mapping()


def test_refuses_previous_host_even_with_new_mapping(monkeypatch):
    monkeypatch.setattr(resource.socket, "gethostname", lambda: "host21")
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "8,9,10,11")
    with pytest.raises(RuntimeError, match="host22"):
        resource.authorized_mapping()


def test_physical_health_and_occupancy_coordinates(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "8,9,10,11")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "Health Status : OK" if args[3] == "health" else "No process in device.")

    monkeypatch.setattr(resource.subprocess, "run", run)
    resource.verify_health(resource.authorized_mapping(), require_idle=True)
    assert [(args[3], args[5], args[7]) for args in calls] == [
        (kind, str(card), str(chip)) for card in (4, 5) for chip in (0, 1) for kind in ("health", "proc-mem")]


@pytest.mark.parametrize("health,process", [("Health Status : Alarm", "No process in device."),
                                           ("Health Status : OK", "Process ID : 123"),
                                           ("Health Status : OK", "")])
def test_refuses_unhealthy_busy_or_unknown_occupancy(monkeypatch, health, process):
    monkeypatch.setattr(resource.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, health if args[3] == "health" else process))
    with pytest.raises(RuntimeError):
        resource.verify_health((8,), require_idle=True)


def test_refuses_physical_indices_in_compact_runtime(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "8,9,10,11")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11")
    with pytest.raises(RuntimeError, match="logical indices"):
        resource.authorized_mapping()


def test_runtime_rejects_extra_device_nodes_before_npu_initialization(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "8,9,10,11")
    monkeypatch.setattr(device.Path, "glob", lambda *args: [Path(f"/dev/davinci{i}") for i in (0, 8, 9, 10, 11)])
    with pytest.raises(RuntimeError, match="isolated device nodes"):
        device.resolve_device("npu")


@pytest.mark.parametrize("world,rank,requested", [(5, 0, "npu"), (4, 4, "npu"),
                                                (1, 0, "npu:4"), (1, 0, "npu:-1")])
def test_runtime_rejects_outside_logical_topology(host22, monkeypatch, world, rank, requested):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "8,9,10,11")
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(device.Path, "glob", lambda *args: [Path(f"/dev/davinci{i}") for i in (8, 9, 10, 11)])
    with pytest.raises((RuntimeError, ValueError)):
        device.resolve_device(requested)


def test_container_reservations_respect_physical_renaming_and_explicit_visibility():
    def container(mapping, visible=None):
        return {"HostConfig": {"Devices": [{"PathOnHost": f"/dev/davinci{host}",
                 "PathInContainer": f"/dev/davinci{local}"} for host, local in mapping]},
                "Config": {"Env": [] if visible is None else [f"ASCEND_RT_VISIBLE_DEVICES={visible}"]}}

    assert resource.reserved_physical_chips(container([(14, 0)], "0")) == {14}
    assert resource.reserved_physical_chips(container([(i, i) for i in range(8, 12)], "0,1,2,3")) == {8, 9, 10, 11}
    broad = [(i, i) for i in range(16)]
    assert resource.reserved_physical_chips(container(broad)) == set()
    assert resource.reserved_physical_chips(container(broad, "8")) == {8}
    assert resource.reserved_physical_chips(container(broad, "0,1,2,3")) == {0, 1, 2, 3}
    assert resource.reserved_physical_chips(container(broad, "bad")) == set(range(16))
