"""Current hardware authorization; usable by the host without PyTorch."""

import os
import re
import socket
import subprocess


AUTHORIZED_HOST = "host22"  # SSH alias: a3-22; use --uts=host in Docker.
AUTHORIZED_CHIPS = (8, 9, 10, 11)


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
    if len(physical) <= len(AUTHORIZED_CHIPS):
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
    if chips != tuple(sorted(set(chips))) or not set(chips) <= set(AUTHORIZED_CHIPS):
        raise RuntimeError("only ordered, distinct a3-22 physical chip8-11 mappings are authorized")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "") != ",".join(str(i) for i in range(len(chips))):
        raise RuntimeError("isolated container visibility must use compact logical indices matching physical chips")
    return chips


def verify_health(chips, *, require_idle=False):
    """Check physical card/chip coordinates, before loading the NPU runtime.

    Idle checks belong to the host launcher, before any ranks start. Runtime
    health checks must allow peers from the same job already using the devices.
    """
    for physical in chips:
        card, chip = divmod(physical, 2)
        result = subprocess.run(["npu-smi", "info", "-t", "health", "-i", str(card), "-c", str(chip)],
                                capture_output=True, text=True, check=True)
        if re.search(r"Health Status\s*:\s*OK\b", result.stdout) is None:
            raise RuntimeError(f"physical chip{physical} is not verified healthy")
        if require_idle:
            result = subprocess.run(["npu-smi", "info", "-t", "proc-mem", "-i", str(card), "-c", str(chip)],
                                    capture_output=True, text=True, check=True)
            if "No process in device." not in result.stdout:
                raise RuntimeError(f"physical chip{physical} has a process or unverifiable occupancy; refuse to launch")
