"""Current hardware authorization; usable by the host without PyTorch."""

import os
import json
from pathlib import Path
import re
import socket
import subprocess
import time

from dsrsi.utils.atomic import atomic_json


AUTHORIZED_HOST = "host22"  # SSH alias: a3-22; use --uts=host in Docker.
AUTHORIZED_CHIPS = (12, 14, 15)


def reserved_physical_chips(container):
    """Interpret an explicit container reservation, including renamed nodes.

    Broad all-device development containers without a visibility mask do not
    reserve every chip; their actual running processes are checked separately.
    """
    ordered = []
    for node in container["HostConfig"].get("Devices") or []:
        host = re.fullmatch(r"/dev/davinci(\d+)", node["PathOnHost"])
        local = re.fullmatch(r"/dev/davinci(\d+)", node["PathInContainer"])
        if host and local:
            ordered.append((int(local.group(1)), int(host.group(1))))
    physical = {host for _, host in ordered}
    if len(physical) <= 4:
        return physical
    visible = next((entry.split("=", 1)[1] for entry in container["Config"].get("Env") or []
                    if entry.startswith("ASCEND_RT_VISIBLE_DEVICES=")), None)
    if visible is None:
        return set()
    if not re.fullmatch(r"\d+(,\d+)*", visible):
        return physical  # An explicit but unparseable reservation is ambiguous.
    indices = [int(value) for value in visible.split(",")]
    ordered.sort()
    if any(index >= len(ordered) for index in indices):
        return physical
    return {ordered[index][1] for index in indices}


def authorized_mapping():
    if socket.gethostname() != AUTHORIZED_HOST:
        raise RuntimeError(f"NPU execution is currently authorized on {AUTHORIZED_HOST} (a3-22) only")
    physical = os.environ.get("DSRSI_PHYSICAL_CHIPS", "")
    if not re.fullmatch(r"\d+(,\d+)*", physical):
        raise RuntimeError("DSRSI_PHYSICAL_CHIPS must explicitly list physical chips")
    chips = tuple(int(value) for value in physical.split(","))
    if (chips != tuple(sorted(set(chips))) or not set(chips) <= set(AUTHORIZED_CHIPS)
            or len(chips) > len(AUTHORIZED_CHIPS)):
        raise RuntimeError("only ordered, distinct subsets of a3-22 physical chip12,14,15 are currently authorized")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "") != ",".join(str(i) for i in range(len(chips))):
        raise RuntimeError("isolated container visibility must use compact logical indices matching physical chips")
    return chips


def _recent_host_health(chips, physical):
    report_path = os.environ.get("DSRSI_HOST_HEALTH_FILE")
    # npu-smi hides an A3 card when its sibling die is absent from /dev.
    # Only this case may use a fresh, read-only host preflight report.
    if not report_path or Path(f"/dev/davinci{physical ^ 1}").exists():
        raise RuntimeError(f"physical chip{physical} has no usable host health attestation")
    report = json.loads(Path(report_path).read_text())
    age = time.time() - report["verified_unix"]
    if (report["host"] != AUTHORIZED_HOST or report["physical_chips"] != list(chips)
            or report["health"] != "OK" or not -5 <= age <= 300):
        raise RuntimeError("host health attestation is stale or differs from the chip mapping")


def verify_health(chips, *, require_idle=False):
    """Check physical card/chip coordinates, before loading the NPU runtime.

    Idle checks belong to the host launcher, before any ranks start. Runtime
    health checks must allow peers from the same job already using the devices.
    """
    for physical in chips:
        # A3 inventory may fail entirely (-9005), rather than merely hide one
        # card, when a container has only one die. Validate its fresh read-only
        # host attestation directly; never map the unallocated sibling die.
        if (not require_idle and os.environ.get("DSRSI_HOST_HEALTH_FILE")
                and not Path(f"/dev/davinci{physical ^ 1}").exists()):
            _recent_host_health(chips, physical)
            continue
        card, chip = divmod(physical, 2)
        try:
            result = subprocess.run(["npu-smi", "info", "-t", "health", "-i", str(card), "-c", str(chip)],
                                    capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            if require_idle or exc.returncode != 215 or "Invalid card id" not in (exc.stdout or ""):
                raise
            _recent_host_health(chips, physical)
            continue
        if re.search(r"Health Status\s*:\s*OK\b", result.stdout) is None:
            raise RuntimeError(f"physical chip{physical} is not verified healthy")
        if require_idle:
            result = subprocess.run(["npu-smi", "info", "-t", "proc-mem", "-i", str(card), "-c", str(chip)],
                                    capture_output=True, text=True, check=True)
            if "No process in device." not in result.stdout:
                raise RuntimeError(f"physical chip{physical} has a process or unverifiable occupancy; refuse to launch")


def write_host_health(chips, path, *, require_idle):
    if socket.gethostname() != AUTHORIZED_HOST or any(
            not Path(f"/dev/davinci{physical}").exists() for chip in chips for physical in (chip, chip ^ 1)):
        raise RuntimeError("host health reports must be generated on the host with both die nodes visible")
    verify_health(chips, require_idle=require_idle)
    atomic_json(Path(path), {"host": AUTHORIZED_HOST, "physical_chips": list(chips),
                            "health": "OK", "verified_unix": time.time(), "idle_checked": require_idle})
