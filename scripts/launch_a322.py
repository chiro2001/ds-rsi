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
from dsrsi.training.resource import authorized_mapping, reserved_physical_chips, verify_health


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base", default="/home/l00886679/models/Qwen3.5-0.8B")
    parser.add_argument("--name", default="ds-rsi-a322-npu")
    parser.add_argument("--image", default="local/dsv41-lab-env:v1")
    args = parser.parse_args()
    os.environ["DSRSI_PHYSICAL_CHIPS"] = "8,9,10,11"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
    chips = authorized_mapping()
    project, base = Path(args.project).resolve(), Path(args.base).resolve()
    if not (project / "pyproject.toml").is_file() or not (base / "model.safetensors.index.json").is_file():
        raise ValueError("project and complete local base checkpoint must exist")
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True)) if ids else []
    for container in containers:
        if reserved_physical_chips(container) & set(chips):
            raise RuntimeError(f"selected chips are reserved by running container {container['Name']}")
    verify_health(chips, require_idle=True)
    command = ["docker", "run", "--detach", "--name", args.name, "--uts=host", "--network=host",
               "--add-host", "host22:127.0.0.1",
               "--shm-size=8g", "--ulimit", "memlock=-1", "--label", "dsrsi.project=ds-rsi",
               "--label", "dsrsi.physical-chips=8,9,10,11", "--workdir", "/workspace",
               "--env", "DSRSI_PHYSICAL_CHIPS=8,9,10,11", "--env", "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3",
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
                    "--mount", f"type=bind,src={base},dst=/models/Qwen3.5-0.8B,readonly",
                    args.image, "sleep", "infinity"])
    subprocess.run(command, check=True)
    print(json.dumps({"host": "a3-22", "physical_chips": chips, "container": args.name,
                      "project": str(project), "privileged": False}))


if __name__ == "__main__":
    main()
