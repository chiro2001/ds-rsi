"""Select a local device while enforcing the currently authorized chip mapping."""

import os
import re
from pathlib import Path

import torch

from .resource import authorized_mapping, verify_health


def resolve_device(requested):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "cuda" or requested.startswith("cuda:"):
        if world_size != 1 or local_rank != 0:
            raise ValueError("CUDA currently supports one rank; topology migration must be explicit")
        match = re.fullmatch(r"cuda:(\d+)", requested)
        if requested != "cuda" and match is None:
            raise ValueError("CUDA device must be cuda or cuda:<logical index>")
        index = int(match.group(1)) if match else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA device is unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        return f"cuda:{index}", world_size
    if not requested.startswith("npu"):
        if requested != "cpu":
            raise ValueError("supported devices are CPU, single-rank CUDA and authorized NPU")
        return requested, world_size
    device = f"npu:{local_rank}" if world_size > 1 or requested == "npu" else requested
    chips = authorized_mapping()
    nodes = {int(path.name.removeprefix("davinci")) for path in Path("/dev").glob("davinci*")
             if path.name.removeprefix("davinci").isdigit()}
    if nodes != set(chips):
        raise RuntimeError("isolated device nodes must retain exactly the authorized physical chip names")
    if not 1 <= world_size <= len(chips) or not 0 <= local_rank < world_size:
        raise RuntimeError("requested parallel topology is outside the authorized mapping")
    if re_match := re.fullmatch(r"npu:(\d+)", device):
        index = int(re_match.group(1))
    else:
        raise ValueError("NPU device must be npu or npu:<logical index>")
    if index >= len(chips):
        raise RuntimeError("requested device is outside the authorized mapping")
    verify_health(chips)
    import torch_npu  # noqa: F401
    if torch.npu.device_count() != len(chips):
        raise RuntimeError("runtime device count differs from the authorized physical mapping")
    torch.npu.set_device(torch.device(device))
    torch.use_deterministic_algorithms(True)
    return device, world_size
