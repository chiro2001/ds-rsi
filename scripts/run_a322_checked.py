#!/usr/bin/env python3
"""Host-only, fail-closed device preflight followed by one project command."""

import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsrsi.training.resource import AUTHORIZED_HOST, verify_health, write_host_health

CONTAINERS = {"ds-rsi-a322-perf-chip12": (12,), "ds-rsi-a322-train-chip14-15": (14, 15)}
ENV_KEYS = {"PYTHONPATH", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"}


def run_checked(container, command, *, logical_devices=None, environment=()):
    if socket.gethostname() != AUTHORIZED_HOST or container not in CONTAINERS:
        raise RuntimeError("only the authorized host22 project containers may launch")
    if not command:
        raise ValueError("a command is required")
    expected = CONTAINERS[container]
    selected = tuple(range(len(expected))) if logical_devices is None else tuple(logical_devices)
    if not selected or selected != tuple(sorted(set(selected))) or any(i < 0 or i >= len(expected) for i in selected):
        raise ValueError("logical devices must be an ordered distinct subset of the container mapping")
    for entry in environment:
        if entry.split("=", 1)[0] not in ENV_KEYS or "=" not in entry:
            raise ValueError("environment overrides may not alter authorization or device visibility")
    info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
    if (info["State"]["Status"] != "running" or info["HostConfig"]["Privileged"]
            or info["Config"].get("Labels", {}).get("dsrsi.project") != "ds-rsi"):
        raise RuntimeError("container must be running, non-privileged and owned by ds-rsi")
    nodes = []
    for device in info["HostConfig"].get("Devices") or []:
        match = re.fullmatch(r"/dev/davinci(\d+)", device["PathOnHost"])
        if match:
            if device["PathOnHost"] != device["PathInContainer"]:
                raise RuntimeError("physical device nodes must retain their original names")
            nodes.append(int(match[1]))
    env = dict(entry.split("=", 1) for entry in info["Config"].get("Env") or [] if "=" in entry)
    if (tuple(sorted(nodes)) != expected or env.get("DSRSI_PHYSICAL_CHIPS") != ",".join(map(str, expected))
            or env.get("ASCEND_RT_VISIBLE_DEVICES") != ",".join(map(str, range(len(expected))))):
        raise RuntimeError("container devices and visibility differ from the authorized mapping")
    active = tuple(expected[i] for i in selected)
    # A raised health/occupancy error always precedes docker exec in this same
    # process; an asynchronous caller cannot accidentally ignore it and launch.
    verify_health(active, require_idle=True)
    if any((chip ^ 1) not in expected for chip in expected):
        report = Path(env.get("DSRSI_HOST_HEALTH_FILE", ""))
        mounts = [m for m in info.get("Mounts", []) if m.get("Type") == "bind" and not m.get("RW")
                  and report.is_relative_to(Path(m["Destination"]))]
        if not report.is_absolute() or len(mounts) != 1:
            raise RuntimeError("single-die execution requires a read-only host health report mount")
        mount = mounts[0]
        host_report = Path(mount["Source"]) / report.relative_to(Path(mount["Destination"]))
        write_host_health(expected, host_report, require_idle=False)
    arguments = ["docker", "exec"] + [item for entry in environment for item in ("--env", entry)] + [container, *command]
    return subprocess.run(arguments, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True, choices=tuple(CONTAINERS))
    parser.add_argument("--logical-devices", help="single-device evaluations may choose 0 or 1; distributed jobs use all")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    devices = None if args.logical_devices is None else tuple(int(i) for i in args.logical_devices.split(","))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    run_checked(args.container, command, logical_devices=devices, environment=args.env)


if __name__ == "__main__":
    main()
