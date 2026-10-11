import subprocess
import json
import time
from pathlib import Path

import pytest

from dsrsi.training import resource
from dsrsi.training import device


@pytest.fixture
def host22(monkeypatch):
    monkeypatch.setattr(resource.socket, "gethostname", lambda: "host22")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2")


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


def test_three_newly_authorized_idle_chips_have_compact_logical_mapping(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "12,14,15")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2")
    assert resource.authorized_mapping() == (12, 14, 15)


@pytest.mark.parametrize("physical,logical,expected", [("12", "0", (12,)), ("14,15", "0,1", (14, 15))])
def test_dedicated_performance_and_training_subsets(host22, monkeypatch, physical, logical, expected):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", physical)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", logical)
    assert resource.authorized_mapping() == expected


@pytest.mark.parametrize("physical,logical", [("13", "0"), ("8,9,10,11", "0,1,2,3")])
def test_previous_or_unreleased_chips_are_no_longer_authorized(host22, monkeypatch, physical, logical):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", physical)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", logical)
    with pytest.raises(RuntimeError, match="currently authorized"):
        resource.authorized_mapping()


def test_physical_health_and_occupancy_coordinates(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "12,14,15")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "Health Status : OK" if args[3] == "health" else "No process in device.")

    monkeypatch.setattr(resource.subprocess, "run", run)
    resource.verify_health(resource.authorized_mapping(), require_idle=True)
    assert [(args[3], args[5], args[7]) for args in calls] == [
        (kind, str(card), str(chip)) for card, chip in ((6, 0), (7, 0), (7, 1)) for kind in ("health", "proc-mem")]


@pytest.mark.parametrize("health,process", [("Health Status : Alarm", "No process in device."),
                                           ("Health Status : OK", "Process ID : 123"),
                                           ("Health Status : OK", "")])
def test_refuses_unhealthy_busy_or_unknown_occupancy(monkeypatch, health, process):
    monkeypatch.setattr(resource.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, health if args[3] == "health" else process))
    with pytest.raises(RuntimeError):
        resource.verify_health((8,), require_idle=True)


def test_refuses_physical_indices_in_compact_runtime(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "12,14,15")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "12,14,15")
    with pytest.raises(RuntimeError, match="logical indices"):
        resource.authorized_mapping()


def test_runtime_rejects_extra_device_nodes_before_npu_initialization(host22, monkeypatch):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "12,14,15")
    monkeypatch.setattr(device.Path, "glob", lambda *args: [Path(f"/dev/davinci{i}") for i in (0, 12, 14, 15)])
    with pytest.raises(RuntimeError, match="isolated device nodes"):
        device.resolve_device("npu")


@pytest.mark.parametrize("world,rank,requested", [(4, 0, "npu"), (3, 3, "npu"),
                                                (1, 0, "npu:3"), (1, 0, "npu:-1")])
def test_runtime_rejects_outside_logical_topology(host22, monkeypatch, world, rank, requested):
    monkeypatch.setenv("DSRSI_PHYSICAL_CHIPS", "12,14,15")
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(device.Path, "glob", lambda *args: [Path(f"/dev/davinci{i}") for i in (12, 14, 15)])
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


@pytest.mark.parametrize("age,reported_chips,success", [(0, [12, 14, 15], True),
    (301, [12, 14, 15], False), (0, [8, 9, 10], False)])
def test_unpaired_die_health_requires_fresh_matching_host_report(tmp_path, monkeypatch, age, reported_chips, success):
    path = tmp_path / "health.json"
    path.write_text(json.dumps({"host": "host22", "physical_chips": reported_chips,
        "health": "OK", "verified_unix": time.time() - age}))
    monkeypatch.setenv("DSRSI_HOST_HEALTH_FILE", str(path))
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if str(p) == "/dev/davinci13" else original(p))
    def run(args, **kwargs):
        if args[5] == "6":
            raise subprocess.CalledProcessError(215, args, output="Invalid card id.")
        return subprocess.CompletedProcess(args, 0, "Health Status : OK")
    monkeypatch.setattr(resource.subprocess, "run", run)
    if success:
        resource.verify_health((12, 14, 15))
    else:
        with pytest.raises(RuntimeError, match="attestation"):
            resource.verify_health((12, 14, 15))
    # Host idle checks can never fall back to a report.
    with pytest.raises(subprocess.CalledProcessError):
        resource.verify_health((12, 14, 15), require_idle=True)


def test_single_die_inventory_failure_is_avoided_with_fresh_readonly_attestation(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    path.write_text(json.dumps({"host": "host22", "physical_chips": [12],
        "health": "OK", "verified_unix": time.time()}))
    monkeypatch.setenv("DSRSI_HOST_HEALTH_FILE", str(path))
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if str(p) == "/dev/davinci13" else original(p))
    def unavailable(*args, **kwargs):
        raise AssertionError("single-die inventory must use its verified host report")
    monkeypatch.setattr(resource.subprocess, "run", unavailable)
    resource.verify_health((12,))
    report = json.loads(path.read_text())
    report["verified_unix"] -= 301
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="attestation"):
        resource.verify_health((12,))
