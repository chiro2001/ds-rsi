"""FP32 LoRA on the language layers after Writer; no extra inference tokens."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    start_layer: int | None = None
    end_layer: int | None = None

    def __post_init__(self):
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")


class LoRALinear(nn.Module):
    def __init__(self, original, *, rank, alpha):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("LoRA requires an existing Linear module")
        self.original = original.requires_grad_(False)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.scale = alpha / rank
        self.enabled = True
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, device=original.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, device=original.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, hidden):
        output = self.original(hidden)
        if not self.enabled:
            return output
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = F.linear(F.linear(hidden.float(), self.lora_a), self.lora_b) * self.scale
        return output + delta.to(output.dtype)


def install_lora(model, config=LoRAConfig()):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed; never overwrite trained adapters")
    start = model.writer_after + 1 if config.start_layer is None else config.start_layer
    end = len(model.base.layers) if config.end_layer is None else config.end_layer
    if not model.writer_after < start < end <= len(model.base.layers):
        raise ValueError("LoRA layers must follow Writer's frozen source feature site")
    plan = []
    for index in range(start, end):
        layer = model.base.layers[index]
        targets = [("mlp.up_proj", layer.mlp, "up_proj"), ("mlp.down_proj", layer.mlp, "down_proj")]
        if layer.block_type == "full_attention":
            targets += [("self_attn.q_proj", layer.self_attn, "q_proj"),
                        ("self_attn.o_proj", layer.self_attn, "o_proj")]
        else:
            targets += [("linear_attn.out_proj", layer.linear_attn, "out_proj")]
        for name, parent, attribute in targets:
            original = getattr(parent, attribute)
            if not isinstance(original, nn.Linear):
                raise ValueError(f"unsupported LoRA target: layer {index} {name}")
            plan.append((f"base.layers.{index}.{name}", parent, attribute, original))
    for _, parent, attribute, original in plan:
        setattr(parent, attribute, LoRALinear(original, rank=config.rank, alpha=config.alpha))
    layout = {**asdict(config), "start_layer": start, "end_layer": end,
              "targets": [name for name, *_ in plan]}
    model.lora_layout = layout
    return layout


@contextmanager
def lora_disabled(model):
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    previous = [m.enabled for m in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled
