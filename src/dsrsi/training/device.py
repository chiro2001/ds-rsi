"""Select a local device while enforcing the currently authorized chip mapping."""

import os
import re
import subprocess

import torch


def resolve_device(requested):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not requested.startswith("npu"):
        if requested != "cpu":
            raise ValueError("only CPU and authorized NPU execution are implemented")
        return requested, world_size
    device = f"npu:{local_rank}" if world_size > 1 or requested == "npu" else requested
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "") != "8,9,10,11,12,13,14,15":
        raise RuntimeError("NPU execution requires the explicitly authorized chip8-15 mapping")
    for card in range(4, 8):
        for chip in (0, 1):
            result = subprocess.run(["npu-smi", "info", "-t", "health", "-i", str(card), "-c", str(chip)],
                                    capture_output=True, text=True, check=True)
            if re.search(r"Health Status\s*:\s*OK\b", result.stdout) is None:
                raise RuntimeError("authorized devices are not verified healthy; refuse to train on known RAS errors")
    import torch_npu  # noqa: F401
    if world_size > 8 or not 0 <= int(device.split(":")[1]) < 8:
        raise RuntimeError("requested rank is outside the authorized device mapping")
    torch.npu.set_device(torch.device(device))
    return device, world_size
