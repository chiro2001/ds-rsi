"""Scoped support-record attention supervision for the A fact curriculum."""

from contextlib import contextmanager
from dataclasses import replace
import math

import torch
import torch.nn.functional as F

from .episode_objective import EpisodeObjective


READER_SUPPORT_VERSION = "reader-support-mass-v1"


def support_attention_loss(reader, hidden, slots, valid, slot_features, positive_slots, prediction_mask):
    """Mean negative log of head-mean support mass at causal target positions.

    Recompute deterministic Reader q/k projections inside its autocast scope;
    scores match the actual FP32 attention. No V/output or student logits change.
    Only supervised rows enter logsumexp, avoiding undefined all-masked rows.
    """
    batch, length, _ = hidden.shape
    if prediction_mask.shape != (batch, length) or prediction_mask.dtype != torch.bool:
        raise ValueError("Reader supervision prediction mask differs from hidden positions")
    if positive_slots.shape != slots.shape[:2] or positive_slots.dtype != torch.bool:
        raise ValueError("Reader supervision support mask differs from packed records")
    if valid.ndim == 2:
        valid = valid[:, None].expand(-1, length, -1)
    if valid.shape != (batch, length, slots.shape[1]) or valid.dtype != torch.bool:
        raise ValueError("Reader supervision validity layout differs")
    if not prediction_mask.any():
        raise ValueError("Reader supervision requires causal target positions")
    any_valid = valid.any(1)
    z = slots if slot_features is None else slots + slot_features
    z = reader.z_norm(z.masked_fill(~any_valid[..., None], 0))
    q = reader.q(reader.h_norm(hidden))
    k = reader.k(z)
    width = q.shape[-1] // reader.heads
    qh = q.reshape(batch, length, reader.heads, width).transpose(1, 2)
    kh = k.reshape(batch, slots.shape[1], reader.heads, width).transpose(1, 2)
    selected_valid = valid[prediction_mask]
    selected_positive = positive_slots[:, None].expand_as(valid)[prediction_mask]
    if not (selected_valid & selected_positive).any(-1).all():
        raise ValueError("a supervised Reader position has no valid supporting slot")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) / math.sqrt(width)
        scores = scores.transpose(1, 2)[prediction_mask]
        logp = F.log_softmax(scores.masked_fill(~selected_valid[:, None], -torch.inf), dim=-1)
        support_logp = torch.logsumexp(logp.masked_fill(~selected_positive[:, None], -torch.inf), dim=-1)
        log_mass = torch.logsumexp(support_logp, dim=-1) - math.log(reader.heads)
        loss = -log_mass.mean()
        mass = log_mass.exp().mean()
    return loss, mass


@contextmanager
def capture_reader_support(model, support_ids, *, end_token_id=None):
    if getattr(model, "_reader_support_capture_active", False):
        raise ValueError("nested Reader support supervision is unsupported")
    handles, captures, current = [], [], {}

    def begin(_module, args, kwargs):
        current.clear()
        labels, batches = kwargs.get("labels"), kwargs.get("records")
        if labels is None or not batches or not kwargs.get("memory_enabled", True):
            return
        if any(len(records) < 2 for records in batches):
            return
        masks = [[record.record_id in support_ids for record in records for _ in range(record.slots.shape[0])]
                 for records in batches]
        if not all(any(row) for row in masks):
            return  # Missing-evidence controls retain their original objective.
        maximum = max(map(len, masks))
        positive = torch.tensor([row + [False] * (maximum - len(row)) for row in masks],
                                device=labels.device, dtype=torch.bool)
        prediction = torch.zeros_like(labels, dtype=torch.bool)
        prediction[:, :-1] = labels[:, 1:] != -100
        if end_token_id is not None:
            prediction[:, :-1] &= labels[:, 1:] != end_token_id
        current.update(positive=positive, prediction=prediction)

    def end(_module, _args, _kwargs, _output):
        current.clear()

    def observe(name):
        def hook(reader, args, kwargs, _output):
            if not current or not kwargs.get("enabled", True):
                return
            loss, mass = support_attention_loss(reader, args[0], args[1], args[2],
                kwargs.get("slot_features"), current["positive"], current["prediction"])
            captures.append((name, loss, mass))
        return hook

    model._reader_support_capture_active = True
    try:
        handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(model.register_forward_hook(end, with_kwargs=True, always_call=True))
        handles.extend(reader.register_forward_hook(observe(name), with_kwargs=True)
                       for name, reader in model.readers.items())
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        current.clear()
        del model._reader_support_capture_active


class ReaderSupportObjective(EpisodeObjective):
    def __init__(self, *args, reader_support_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(reader_support_weight) or reader_support_weight < 0:
            raise ValueError("Reader support weight must be finite and nonnegative")
        if reader_support_weight and (self.stage != "A" or self.memory_candidates not in (1, 2)):
            raise ValueError("Reader support v1 requires A with one or two candidates")
        self.reader_support_weight = reader_support_weight

    def __call__(self, row, **kwargs):
        captures = []
        if self.reader_support_weight and self.memory_candidates > 1 and row["family"] == "single_fact":
            if len(row["probes"]) != 1 or len(row["probes"][0]["support_groups"]) != 1:
                raise ValueError("Reader support v1 requires one fact response and support group")
            support = set(row["probes"][0]["support_groups"][0])
            with capture_reader_support(self.model, support,
                                        end_token_id=self.compiler.tokenizer.eos_token_id) as captures:
                result = super().__call__(row, **kwargs)
            if result.covered_queries == result.queries and len(captures) != len(self.model.readers):
                raise ValueError("Reader support did not observe exactly one answer forward per Reader")
        else:
            result = super().__call__(row, **kwargs)
        if captures:
            loss = torch.stack([item[1] for item in captures]).mean()
            result = replace(result, loss=result.loss + self.reader_support_weight * loss)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("nonfinite Reader support objective")
            result.reader_support_loss = float(loss.detach())
            result.reader_support_mass = float(torch.stack([item[2].detach() for item in captures]).mean())
        else:
            result.reader_support_loss, result.reader_support_mass = 0.0, None
        result.reader_support_readers = len(captures)
        return result
