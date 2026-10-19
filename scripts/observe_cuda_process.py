#!/usr/bin/env python3
"""Read-only elapsed-time observation, independent of CLOCK_REALTIME updates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import time
from time import time as realtime_now


def process_start_ticks(stat_text):
    # comm (field 2) may contain spaces or parentheses. starttime is field 22.
    return int(stat_text[stat_text.rindex(")") + 2:].split()[19])


def observe_process(pid, *, required_arguments=(), required_pairs=None, on_attached=None):
    if not hasattr(os, "pidfd_open") or not hasattr(time, "CLOCK_BOOTTIME"):
        raise RuntimeError("Linux pidfd and CLOCK_BOOTTIME are required; no UTC fallback")
    proc = Path("/proc") / str(pid)
    descriptor = os.pidfd_open(pid)
    try:
        ticks = process_start_ticks((proc / "stat").read_text())
        arguments = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        if proc.stat().st_uid != os.geteuid() or any(x not in arguments for x in required_arguments):
            raise ValueError("process owner or command differs from the submitted job")
        for flag, expected in (required_pairs or {}).items():
            if arguments.count(flag) != 1 or arguments[arguments.index(flag) + 1:][:1] != [expected]:
                raise ValueError("process arguments differ from the submitted job")
        if process_start_ticks((proc / "stat").read_text()) != ticks:
            raise RuntimeError("process identity changed during attachment")
        hz = os.sysconf("SC_CLK_TCK")
        start = ticks / hz
        boot = time.clock_gettime(time.CLOCK_BOOTTIME)
        mono, utc = time.monotonic(), realtime_now()
        if start > boot:
            raise RuntimeError("process start and CLOCK_BOOTTIME origins differ")
        info = {"pid": pid, "method": "proc-starttime-pidfd-CLOCK_BOOTTIME-v1",
                "start_clock_ticks": ticks, "clock_ticks_per_second": hz,
                "process_start_boottime_seconds": start,
                "observer_start_boottime_seconds": boot,
                "observer_start_monotonic_seconds": mono, "observer_start_realtime_seconds": utc,
                "start_quantization_seconds": 1 / hz,
                "signals_sent": False, "training_state_modified": False}
        if on_attached is not None:
            on_attached(info)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            events = poller.poll(50_000)
            if not events:
                continue
            if any(mask & select.POLLNVAL for _, mask in events):
                raise RuntimeError("invalid pidfd")
            if any(mask & (select.POLLIN | select.POLLHUP) for _, mask in events):
                break
            raise RuntimeError("unexpected pidfd events")
        end = time.clock_gettime(time.CLOCK_BOOTTIME)
        end_mono, end_utc = time.monotonic(), realtime_now()
        return {**info, "process_exit_boottime_seconds": end,
                "process_duration_seconds": end - start,
                "observer_duration_monotonic_seconds": end_mono - mono,
                "observer_duration_realtime_seconds": end_utc - utc,
                "observer_realtime_minus_monotonic_seconds": (end_utc - utc) - (end_mono - mono),
                "exit_observed": True, "returncode_observed": False,
                "includes_machine_suspend_time": True}
    finally:
        os.close(descriptor)


def write_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(str(path) + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("/mnt/e/ds-rsi").resolve()
    submission, output = Path(args.submission).resolve(), Path(args.output).resolve()
    if not submission.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("observation inputs and outputs must stay on Windows E:\\ds-rsi")
    record = json.loads(submission.read_text())
    command = record["command"]
    script = next(x for x in command if Path(x).name == "train_ab.py")
    pairs = {flag: command[command.index(flag) + 1] for flag in ("--run-dir", "--max-micro-steps")}
    if not Path(pairs["--run-dir"]).resolve().is_relative_to(root):
        raise ValueError("submitted training run must stay on E:")
    if output.exists():
        raise FileExistsError(output)
    provenance = {"submission": str(submission),
                  "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
                  "observer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def attached(info):
        write_new_json(str(output) + ".observing.json", {**info, **provenance})
        print(json.dumps({"observer_attached": info}), flush=True)

    result = observe_process(record["pid"], required_arguments=[script],
                             required_pairs=pairs, on_attached=attached)
    write_new_json(output, {**result, **provenance})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
