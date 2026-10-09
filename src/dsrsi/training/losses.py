"""Event-normalized objectives for latent memory training.

Routing input stop-gradients belong in the address encoder: detach source
features before their trainable projections, not the projected keys themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _connected_zero(tensor: torch.Tensor) -> torch.Tensor:
    # Empty slice keeps an autograd path without summing unused fp16 logits or
    # -inf routing scores. Such sums can overflow or turn the zero into NaN.
    return tensor.reshape(-1)[:0].float().sum()


@dataclass
class LossSum:
    total: torch.Tensor
    count: torch.Tensor

    def mean(self) -> torch.Tensor:
        return self.total / self.count.clamp_min(1)


def _event_reduce(values: torch.Tensor, event_ids: torch.Tensor, batch: int, zero: torch.Tensor) -> LossSum:
    sums = zero.expand(batch).clone().index_add(0, event_ids, values)
    lengths = torch.bincount(event_ids, minlength=batch).to(sums.dtype)
    means = sums / lengths.clamp_min(1)
    return LossSum(means.sum(), (lengths > 0).sum().to(sums.dtype))


def response_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> LossSum:
    """labels target token positions; predict labels[t] with logits[t-1]."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    targets = labels[:, 1:]
    positions = (targets != IGNORE_INDEX).nonzero(as_tuple=False)
    zero = _connected_zero(logits)
    if positions.numel() == 0:
        return LossSum(zero, zero.detach())
    batch_ids, token_ids = positions.unbind(1)
    selected = logits[batch_ids, token_ids].float()
    losses = F.cross_entropy(selected, targets[batch_ids, token_ids], reduction="none")
    return _event_reduce(losses, batch_ids, logits.shape[0], zero)


def response_cross_entropy_from_hidden(
    hidden: torch.Tensor, head_weight: torch.Tensor, labels: torch.Tensor, *, chunk_positions: int = 128,
) -> LossSum:
    """Apply a large vocabulary head only at supervised prediction positions."""
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or head_weight.ndim != 2:
        raise ValueError("expected hidden [B,T,H], weight [V,H], labels [B,T]")
    if hidden.shape[-1] != head_weight.shape[-1] or chunk_positions < 1:
        raise ValueError("invalid head dimension/chunk size")
    targets = labels[:, 1:]
    positions = (targets != IGNORE_INDEX).nonzero(as_tuple=False)
    zero = _connected_zero(hidden)
    if positions.numel() == 0:
        return LossSum(zero, zero.detach())
    values, batches = [], []
    for part in positions.split(chunk_positions):
        batch_ids, token_ids = part.unbind(1)
        selected_hidden = hidden[batch_ids, token_ids]
        logits = F.linear(selected_hidden, head_weight)
        values.append(F.cross_entropy(logits.float(), targets[batch_ids, token_ids], reduction="none"))
        batches.append(batch_ids)
    return _event_reduce(torch.cat(values), torch.cat(batches), hidden.shape[0], zero)


def routing_scores(query: torch.Tensor, keys: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    """query [B,D], keys [B,R,J,D] or [R,J,D]; aggregate independent addresses."""
    if query.ndim != 2 or keys.ndim not in (3, 4) or temperature <= 0:
        raise ValueError("invalid query/key dimensions or temperature")
    if keys.ndim == 3:
        keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
    if keys.shape[0] != query.shape[0] or keys.shape[-1] != query.shape[-1] or keys.shape[2] < 1:
        raise ValueError("incompatible query/key dimensions")
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    per_key = torch.einsum("bd,brjd->brj", q, k) / temperature
    return torch.logsumexp(per_key, dim=-1) - math.log(keys.shape[2])


def required_support_loss(
    scores: torch.Tensor, positive_groups: torch.Tensor, eligible: torch.Tensor,
    negatives: torch.Tensor | None = None,
) -> LossSum:
    """Each nonempty group must be covered; alternative supports are positives.

    Other valid support groups are excluded from negatives. Empty groups are
    unused padding, not missing positive evidence.
    """
    if scores.ndim != 2 or positive_groups.ndim != 3:
        raise ValueError("expected scores [B,R] and positive_groups [B,G,R]")
    if (positive_groups.shape[0] != scores.shape[0] or positive_groups.shape[2] != scores.shape[1]
            or eligible.shape != scores.shape):
        raise ValueError("support/eligibility dimensions differ")
    if positive_groups.dtype != torch.bool or eligible.dtype != torch.bool:
        raise ValueError("support/eligibility masks must be boolean")
    if torch.any(positive_groups & ~eligible.unsqueeze(1)):
        raise ValueError("positive evidence is ineligible/future/cross-stream")
    all_positive = positive_groups.any(dim=1)
    if negatives is None:
        negatives = eligible & ~all_positive
    elif negatives.shape != scores.shape or negatives.dtype != torch.bool:
        raise ValueError("invalid negative mask")
    if torch.any(negatives & (all_positive | ~eligible)):
        raise ValueError("negative mask contains valid positive or ineligible evidence")

    active = positive_groups.any(dim=-1)
    indices = active.nonzero(as_tuple=False)
    zero = _connected_zero(scores)
    if indices.numel() == 0:
        return LossSum(zero, zero.detach())
    batch_ids, group_ids = indices.unbind(1)
    group_positive = positive_groups[batch_ids, group_ids]
    denominator = group_positive | negatives[batch_ids]
    values = scores[batch_ids].float()
    losses = (torch.logsumexp(values.masked_fill(~denominator, -torch.inf), dim=-1)
              - torch.logsumexp(values.masked_fill(~group_positive, -torch.inf), dim=-1))
    return _event_reduce(losses, batch_ids, scores.shape[0], zero)


def topk_tail_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, topk: int = 32) -> torch.Tensor:
    """KL on teacher top-k categories plus one tail category, retaining tail mass."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student/teacher logits must have matching [...,V] shape")
    vocabulary = student_logits.shape[-1]
    if topk < 1 or vocabulary < 2:
        raise ValueError("invalid top-k/vocabulary")
    topk = min(topk, vocabulary - 1)
    student = student_logits.float().reshape(-1, vocabulary)
    teacher = teacher_logits.detach().float().reshape(-1, vocabulary)
    if student.shape[0] == 0:
        return _connected_zero(student_logits)
    indices = teacher.topk(topk, dim=-1).indices
    mask = torch.zeros_like(student, dtype=torch.bool).scatter(1, indices, True)
    q_norm = torch.logsumexp(student, dim=-1, keepdim=True)
    p_norm = torch.logsumexp(teacher, dim=-1, keepdim=True)
    q_top = student.gather(1, indices) - q_norm
    p_top = teacher.gather(1, indices) - p_norm
    q_tail = torch.logsumexp(student.masked_fill(mask, -torch.inf), dim=-1, keepdim=True) - q_norm
    p_tail = torch.logsumexp(teacher.masked_fill(mask, -torch.inf), dim=-1, keepdim=True) - p_norm
    values = (p_top.exp() * (p_top - q_top)).sum(dim=-1)
    values = values + (p_tail.exp() * (p_tail - q_tail)).squeeze(-1)
    return values.mean()
