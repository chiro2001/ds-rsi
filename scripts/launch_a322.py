#!/usr/bin/env python3
"""Host-side idle check and isolated project container on a3-22."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsrsi.training.resource import authorized_mapping, reserved_physical_chips, write_host_health


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base", default="/home/l00886679/models/Qwen3.5-0.8B")
    parser.add_argument("--name", default="ds-rsi-a322-npu")
    parser.add_argument("--image", default="local/dsv41-lab-env:v1")
    parser.add_argument("--chips", default="12,14,15", help="ordered physical chips; verify current resource authorization")
    parser.add_argument("--released-reservation", action="append", default=[],
                        help="container whose idle device reservation the user explicitly released; does not modify it")
    parser.add_argument("--refresh-health-only", action="store_true",
                        help="refresh an existing container's host health report; no container mutation")
    args = parser.parse_args()
    os.environ["DSRSI_PHYSICAL_CHIPS"] = args.chips
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(args.chips.split(","))))
    chips = authorized_mapping()
    if Path(args.name).name != args.name:
        raise ValueError("container name must not contain path components")
    health_directory = Path.home() / ".cache/ds-rsi/a322-health" / args.name
    health_report = health_directory / "report.json"
    if args.refresh_health_only:
        write_host_health(chips, health_report, require_idle=False)
        print(json.dumps({"refreshed_host_health": str(health_report), "physical_chips": chips}))
        return
    project, base = Path(args.project).resolve(), Path(args.base).resolve()
    if not (project / "pyproject.toml").is_file() or not (base / "model.safetensors.index.json").is_file():
        raise ValueError("project and complete local base checkpoint must exist")
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True)) if ids else []
    released = set(args.released_reservation)
    if released - {container["Name"].lstrip("/") for container in containers}:
        raise ValueError("released reservation must identify an existing running container")
    for container in containers:
        if (reserved_physical_chips(container) & set(chips)
                and container["Name"].lstrip("/") not in released):
            raise RuntimeError(f"selected chips are reserved by running container {container['Name']}")
    write_host_health(chips, health_report, require_idle=True)
    command = ["docker", "run", "--detach", "--name", args.name, "--uts=host", "--network=host",
               "--add-host", "host22:127.0.0.1",
               "--shm-size=8g", "--ulimit", "memlock=-1", "--label", "dsrsi.project=ds-rsi",
               "--label", f"dsrsi.physical-chips={args.chips}", "--workdir", "/workspace",
               "--env", f"DSRSI_PHYSICAL_CHIPS={args.chips}",
               "--env", f"ASCEND_RT_VISIBLE_DEVICES={os.environ['ASCEND_RT_VISIBLE_DEVICES']}",
               "--env", "DSRSI_HOST_HEALTH_FILE=/run/dsrsi-host-health/report.json",
               "--env", "OMP_NUM_THREADS=4",
               "--env", "PYTHONPATH=/workspace/src", "--env", "TOKENIZERS_PARALLELISM=false"]
    for chip in chips:
        command.extend(["--device", f"/dev/davinci{chip}:/dev/davinci{chip}"])
    for node in ("davinci_manager", "devmm_svm", "hisi_hdc"):
        command.extend(["--device", f"/dev/{node}:/dev/{node}"])
    for root in ("driver", "firmware"):
        command.extend(["--volume", f"/usr/local/Ascend/{root}:/usr/local/Ascend/{root}:ro"])
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        raise RuntimeError("npu-smi executable is required")
    command.extend(["--mount", f"type=bind,src={Path(npu_smi).resolve()},dst=/usr/local/bin/npu-smi,readonly",
                    "--mount", f"type=bind,src={project},dst=/workspace",
                    "--mount", f"type=bind,src={health_directory},dst=/run/dsrsi-host-health,readonly",
                    "--mount", f"type=bind,src={base},dst=/models/Qwen3.5-0.8B,readonly",
                    args.image, "sleep", "infinity"])
    subprocess.run(command, check=True)
    print(json.dumps({"host": "a3-22", "physical_chips": chips, "container": args.name,
                      "project": str(project), "privileged": False,
                      "user_released_reservations": sorted(released)}))


if __name__ == "__main__":
    main()
