#!/usr/bin/env python3
"""Launch one project CUDA job with external storage and an inherited GPU lock."""

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def check_storage_paths(command, code, root):
    flags = {"--base", "--run-dir", "--source-run", "--target-run", "--run-root",
             "--output", "--output-dir", "--journal", "--corpus", "--manifest"}
    for index, value in enumerate(command[:-1]):
        if value in flags:
            path = Path(command[index + 1])
            path = (path if path.is_absolute() else code / path).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f"{value} must remain under Windows E:\\ds-rsi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--python", default="/mnt/e/ds-rsi/envs/train-cu130-torch210/bin/python")
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path("/mnt/e/ds-rsi").resolve()
    code = Path(args.code_root).resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (socket.gethostname().lower() != "gamepc" or not command
            or not code.is_relative_to(root) or not (code / "src/dsrsi").is_dir()):
        raise RuntimeError("expected a complete project snapshot on the authorized 5080 host")
    check_storage_paths(command, code, root)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock = os.open(control / "gpu.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another ds-rsi GPU job still holds the project lock") from None
    if shutil.disk_usage(root).free < 20 * (1 << 30):
        raise RuntimeError("Windows E: has less than the reserved 20 GiB free")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    devices = list(csv.reader(io.StringIO(output)))
    if (len(devices) != 1 or "RTX 5080" not in devices[0][0]
            or float(devices[0][1]) - float(devices[0][2]) < 6144 or float(devices[0][3]) >= 85):
        raise RuntimeError("5080 identity, available memory or temperature check failed")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(code / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               WORLD_SIZE="1", RANK="0", LOCAL_RANK="0",
               CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for variable, folder in (("HF_HOME", "huggingface"), ("PIP_CACHE_DIR", "pip"),
                             ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                             ("TORCH_HOME", "torch"), ("CUDA_CACHE_PATH", "cuda"), ("XDG_CACHE_HOME", "xdg")):
        path = root / "cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    (root / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(root / "tmp")
    os.set_inheritable(lock, True)
    job = {"pid": os.getpid(), "label": args.label, "code_root": str(code),
           "python": args.python, "command": command, "started_unix": time.time(),
           "windows_free_bytes": shutil.disk_usage(root).free}
    (control / "gpu-job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "checked_launch", **job}), flush=True)
    os.chdir(code)
    os.execvpe(args.python, [args.python, *command], env)


if __name__ == "__main__":
    main()
