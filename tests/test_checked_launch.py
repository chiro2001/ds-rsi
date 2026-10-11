import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location("checked_launch", Path("scripts/run_a322_checked.py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def container(chips=(14, 15)):
    return {"State": {"Status": "running"}, "HostConfig": {"Privileged": False, "Devices": [
        {"PathOnHost": f"/dev/davinci{i}", "PathInContainer": f"/dev/davinci{i}"} for i in chips]},
        "Config": {"Labels": {"dsrsi.project": "ds-rsi"}, "Env": [
            "DSRSI_PHYSICAL_CHIPS=" + ",".join(map(str, chips)),
            "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, range(len(chips))))]}}


def prepare(monkeypatch, info=None):
    m = module()
    monkeypatch.setattr(m.socket, "gethostname", lambda: "host22")
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: json.dumps([info or container()]))
    return m


def test_failed_health_or_busy_devices_never_launch_the_command(monkeypatch):
    m = prepare(monkeypatch)
    calls = []
    def refuse(chips, *, require_idle):
        assert chips == (14, 15) and require_idle
        raise RuntimeError("device is busy")
    monkeypatch.setattr(m, "verify_health", refuse)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="busy"):
        m.run_checked("ds-rsi-a322-train-chip14-15", ["torchrun", "train.py"])
    assert calls == []


def test_single_device_preflight_checks_the_exact_physical_chip_before_launch(monkeypatch):
    m = prepare(monkeypatch)
    order = []
    monkeypatch.setattr(m, "verify_health", lambda chips, **k: order.append(("health", chips, k)))
    monkeypatch.setattr(m.subprocess, "run", lambda args, **k: order.append(("launch", args, k)))
    m.run_checked("ds-rsi-a322-train-chip14-15", ["python", "evaluate.py"], logical_devices=[1],
                  environment=["PYTHONPATH=/workspace/src"])
    assert order[0] == ("health", (15,), {"require_idle": True})
    assert order[1][1] == ["docker", "exec", "--env", "PYTHONPATH=/workspace/src",
                           "ds-rsi-a322-train-chip14-15", "python", "evaluate.py"]


@pytest.mark.parametrize("mutation", [
    lambda c: c["HostConfig"].update(Privileged=True),
    lambda c: c["HostConfig"]["Devices"][0].update(PathInContainer="/dev/davinci0"),
    lambda c: c["Config"]["Env"].append("ASCEND_RT_VISIBLE_DEVICES=1,0"),
])
def test_privilege_renaming_and_visibility_mismatch_are_rejected_before_health(monkeypatch, mutation):
    info = container()
    mutation(info)
    m = prepare(monkeypatch, info)
    def unexpected(*a, **k):
        raise AssertionError("preflight should not reach a health query or launch")
    monkeypatch.setattr(m, "verify_health", unexpected)
    monkeypatch.setattr(m.subprocess, "run", unexpected)
    with pytest.raises(RuntimeError):
        m.run_checked("ds-rsi-a322-train-chip14-15", ["python", "train.py"])


def test_single_die_refreshes_only_the_read_only_attestation_without_mapping_its_sibling(monkeypatch):
    info = container((12,))
    info["Config"]["Env"].append("DSRSI_HOST_HEALTH_FILE=/run/health/report.json")
    info["Mounts"] = [{"Type": "bind", "RW": False, "Source": "/host/health", "Destination": "/run/health"}]
    m = prepare(monkeypatch, info)
    order = []
    monkeypatch.setattr(m, "verify_health", lambda chips, **k: order.append(("idle", chips)))
    monkeypatch.setattr(m, "write_host_health", lambda chips, path, **k: order.append(("report", chips, path)))
    monkeypatch.setattr(m.subprocess, "run", lambda args, **k: order.append(("exec", args)))
    m.run_checked("ds-rsi-a322-perf-chip12", ["python", "profile.py"])
    assert order[0] == ("idle", (12,))
    assert order[1] == ("report", (12,), Path("/host/health/report.json"))
    assert order[2][0] == "exec"


def test_device_authorization_cannot_be_overridden_by_command_environment(monkeypatch):
    m = prepare(monkeypatch)
    with pytest.raises(ValueError, match="authorization"):
        m.run_checked("ds-rsi-a322-train-chip14-15", ["python", "train.py"],
                      environment=["ASCEND_RT_VISIBLE_DEVICES=2"])
