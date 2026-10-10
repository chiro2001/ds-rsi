"""Qwen-independent trainable modules with the proposed production dimensions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryConfig:
    hidden_size: int = 1024
    memory_dim: int = 256
    slots: int = 8
    heads: int = 4
    intermediate: int = 512
    key_dim: int = 128
    keys: int = 2
    summary_slots: int = 16
    reader_alpha: float = 0.05
    writer_anchor_weight: float = 0.0
    reader_gate_mode: str = "token"

    def __post_init__(self):
        for name in ("hidden_size", "memory_dim", "slots", "heads", "intermediate", "key_dim", "keys", "summary_slots"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.memory_dim % self.heads:
            raise ValueError("memory_dim must be divisible by heads")
        if not math.isfinite(self.writer_anchor_weight) or not 0 <= self.writer_anchor_weight <= 1:
            raise ValueError("writer anchor weight must be finite and in [0,1]")
        if self.reader_gate_mode not in {"token", "lookup"}:
            raise ValueError("reader gate mode must be token or lookup")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _attention(q, k, v, valid, heads):
    """Small cross-attention, FP32 scores, safe all-masked queries."""
    batch, queries, width = q.shape
    source = k.shape[1]
    if valid.ndim == 2:
        valid = valid[:, None, :].expand(-1, queries, -1)
    if valid.shape != (batch, queries, source) or valid.dtype != torch.bool:
        raise ValueError("attention validity must be [B,S] or [B,Q,S] boolean")
    dim = width // heads
    qh = q.reshape(batch, queries, heads, dim).transpose(1, 2)
    kh = k.reshape(batch, source, heads, dim).transpose(1, 2)
    vh = v.reshape(batch, source, heads, dim).transpose(1, 2)
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) / math.sqrt(dim)
        mask = valid[:, None]
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = scores.softmax(-1).masked_fill(~mask, 0)
        out = torch.matmul(probabilities, vh.float())
    return out.transpose(1, 2).reshape(batch, queries, width).to(q.dtype)


class SlotCompressor(nn.Module):
    def __init__(self, input_width: int, config: MemoryConfig, slots: int):
        super().__init__()
        dim = config.memory_dim
        self.heads = config.heads
        self.input_width = input_width
        self.input_norm = RMSNorm(input_width)
        self.queries = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.query_norm = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(input_width, dim, bias=False)
        self.v = nn.Linear(input_width, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, config.intermediate)
        self.final_norm = RMSNorm(dim)

    def forward(self, source: torch.Tensor, valid: torch.Tensor | None = None):
        if source.ndim != 3 or source.shape[-1] != self.input_width:
            raise ValueError("compressor input must be [B,S,input_width]")
        if valid is None:
            valid = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)
        if valid.shape != source.shape[:2] or valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError("every source must have a nonempty boolean validity mask")
        # A masked NaN in padding must not enter either the scores or A @ V.
        h = self.input_norm(source.masked_fill(~valid[..., None], 0))
        z = self.queries.to(source.dtype)[None].expand(source.shape[0], -1, -1)
        q = self.q(self.query_norm(z))
        z = z + self.o(_attention(q, self.k(h), self.v(h), valid, self.heads))
        return self.final_norm(z + self.ffn(self.ffn_norm(z)))


class MemoryWriter(SlotCompressor):
    def __init__(self, config: MemoryConfig):
        super().__init__(config.hidden_size, config, config.slots)
        self.anchor_weight = config.writer_anchor_weight

    def forward(self, source, valid=None):
        if not self.anchor_weight:
            return super().forward(source, valid)
        if source.ndim != 3 or source.shape[-1] != self.input_width:
            raise ValueError("writer input must be [B,S,input_width]")
        if valid is None:
            valid = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)
        if valid.shape != source.shape[:2] or valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError("every writer source must have nonempty boolean validity")
        h = self.input_norm(source.masked_fill(~valid[..., None], 0))
        z = self.queries.to(source.dtype)[None].expand(source.shape[0], -1, -1)
        values = self.v(h)
        attended = _attention(self.q(self.query_norm(z)), self.k(h), values, valid, self.heads)
        # Adaptive contiguous bins over *valid* token order. Every observed
        # position has a direct gradient path, even if learned attention has
        # saturated elsewhere. Short sources use overlapping nonempty bins.
        slots = z.shape[1]
        lengths = valid.sum(-1)[:, None, None]
        index = torch.arange(slots, device=source.device)[None, :, None]
        begin = index * lengths // slots
        end = ((index + 1) * lengths + slots - 1) // slots
        ranks = valid.long().cumsum(-1)[:, None] - 1
        bins = (ranks >= begin) & (ranks < end) & valid[:, None]
        with torch.autocast(device_type=source.device.type, enabled=False):
            weights = bins.float() / bins.sum(-1, keepdim=True).float()
            anchored = torch.matmul(weights, values.float()).to(values.dtype)
        z = z + self.o(attended + self.anchor_weight * anchored)
        return self.final_norm(z + self.ffn(self.ffn_norm(z)))


class Consolidator(SlotCompressor):
    def __init__(self, config: MemoryConfig):
        super().__init__(config.memory_dim, config, config.summary_slots)


class MemoryReader(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.memory_dim = config.memory_dim
        self.heads = config.heads
        self.h_norm = RMSNorm(config.hidden_size)
        self.z_norm = RMSNorm(config.memory_dim)
        self.q = nn.Linear(config.hidden_size, config.memory_dim, bias=False)
        self.k = nn.Linear(config.memory_dim, config.memory_dim, bias=False)
        self.v = nn.Linear(config.memory_dim, config.memory_dim, bias=False)
        self.o = nn.Linear(config.memory_dim, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.alpha = nn.Parameter(torch.tensor(config.reader_alpha))

    def lookup_gate(self, hidden):
        return self.gate(self.h_norm(hidden)).sigmoid()

    def forward(self, hidden, slots, valid=None, *, slot_features=None, enabled=True, read_gate=None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("reader hidden must be [B,T,hidden_size]")
        if not enabled or slots is None or slots.shape[1] == 0:
            return hidden
        if slots.ndim != 3 or slots.shape[0] != hidden.shape[0] or slots.shape[-1] != self.memory_dim:
            raise ValueError("reader slots must be [B,M,memory_dim]")
        if valid is None:
            valid = torch.ones(slots.shape[:2], dtype=torch.bool, device=slots.device)
        if valid.dtype != torch.bool or valid.ndim not in (2, 3):
            raise ValueError("reader validity must be boolean")
        any_valid = valid if valid.ndim == 2 else valid.any(1)
        if any_valid.shape != slots.shape[:2]:
            raise ValueError("reader slot mask dimensions differ")
        z = slots if slot_features is None else slots + slot_features
        z = self.z_norm(z.masked_fill(~any_valid[..., None], 0))
        h = self.h_norm(hidden)
        delta = self.o(_attention(self.q(h), self.k(z), self.v(z), valid, self.heads))
        if read_gate is not None and read_gate.shape != (hidden.shape[0], 1, 1):
            raise ValueError("lookup gate must be one scalar per life, shared across the span")
        gate = self.gate(h).sigmoid() if read_gate is None else read_gate
        return hidden + (self.alpha.to(delta.dtype) * gate.to(delta.dtype) * delta).to(hidden.dtype)


class AddressEncoder(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.h_norm = RMSNorm(config.hidden_size)
        self.z_norm = RMSNorm(config.memory_dim)
        self.query = nn.Linear(config.hidden_size, config.key_dim, bias=False)
        self.key_queries = nn.Parameter(torch.randn(config.keys, config.memory_dim) * 0.02)
        self.key = nn.Linear(config.memory_dim, config.key_dim, bias=False)

    def encode_query(self, hidden: torch.Tensor, *, detach_source=True):
        h = hidden.detach() if detach_source else hidden
        return F.normalize(self.query(self.h_norm(h)).float(), dim=-1, eps=1e-6)

    def encode_keys(self, slots: torch.Tensor, valid=None, *, detach_source=True):
        if slots.ndim != 3:
            raise ValueError("address slots must be [B,S,D]")
        if valid is None:
            valid = torch.ones(slots.shape[:2], device=slots.device, dtype=torch.bool)
        if valid.shape != slots.shape[:2] or valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError("address source must have valid slots")
        z = slots.detach() if detach_source else slots
        z = self.z_norm(z.masked_fill(~valid[..., None], 0))
        with torch.autocast(device_type=z.device.type, enabled=False):
            score = torch.einsum("jd,bsd->bjs", self.key_queries.float(), z.float()) / math.sqrt(z.shape[-1])
            probability = score.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
            pooled = torch.matmul(probability, z.float()).to(z.dtype)
        return F.normalize(self.key(pooled).float(), dim=-1, eps=1e-6)


class TimeSourceEncoder(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.down = nn.Linear(8, 128)
        self.up = nn.Linear(128, config.memory_dim)

    def forward(self, features: torch.Tensor):
        if features.shape[-1] != 8:
            raise ValueError("time/source features must have width 8")
        return self.up(F.silu(self.down(features)))
