"""Select a local device while enforcing the currently authorized chip mapping."""

import os
import re
from pathlib import Path

import torch

from .resource import authorized_mapping, verify_health


def resolve_device(requested):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not requested.startswith("npu"):
        if requested != "cpu":
            raise ValueError("only CPU and authorized NPU execution are implemented")
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
