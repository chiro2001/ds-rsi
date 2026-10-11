"""Scoped reuse of identical native inputs before memory injection.

This is an experimental timing primitive, not enabled by any training runner.
Only deterministic frozen prefix layers are reused. The scope ends before a
microbatch checkpoint; no reusable state is persisted or shared across lives.
"""

from contextlib import contextmanager

import torch


@contextmanager
def frozen_prefix_reuse(model):
    if getattr(model, "_prefix_reuse_active", False):
        raise ValueError("nested prefix reuse is unsupported")
    last = model.writer_after

    def parameter_signature():
        params = list(model.base.embed_tokens.parameters())
        for layer in model.base.layers[:last + 1]:
            params.extend(layer.parameters())
        if model.base.training or any(p.requires_grad for p in params):
            raise ValueError("prefix reuse requires an eval-mode frozen backbone prefix")
        return tuple((id(p), p._version) for p in params)

    parameter_signature()
    original_prepare, original_layer, original_forward = model._prepare, model._layer, model.forward
    cache, current = {}, {}
    default_mask = {}
    stats = {"hits": 0, "skipped_layers": 0}

    def same_mask(mask):
        old = cache.get("mask")
        if old is None or mask is None:
            return old is None and mask is None
        if old._version != cache["mask_version"]:
            return False
        # Avoid synchronizing the NPU merely to compare masks. Equivalent but
        # distinct caller masks conservatively miss; implicit all-valid masks
        # are shared by forward() below for the identical input object.
        return old is mask

    def forward(*args, **kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        if kwargs.get("attention_mask") is None:
            if not (default_mask.get("ids") is ids and default_mask.get("ids_version") == ids._version
                    and default_mask["mask"]._version == default_mask["mask_version"]):
                mask = torch.ones_like(ids, dtype=torch.bool)
                default_mask.clear()
                default_mask.update(ids=ids, ids_version=ids._version, mask=mask, mask_version=mask._version)
            kwargs["attention_mask"] = default_mask["mask"]
        return original_forward(*args, **kwargs)

    def prepare(ids, attention_mask=None):
        result = original_prepare(ids, attention_mask)
        signature = parameter_signature()
        frequency = model.base.rotary_emb.inv_freq
        rotary = (id(frequency), frequency._version)
        hit = (cache.get("ids") is ids and cache.get("ids_version") == ids._version
               and cache.get("parameters") == signature and cache.get("rotary") == rotary
               and cache.get("hidden_version") == cache["hidden"]._version and same_mask(attention_mask)) if cache else False
        current.clear()
        current.update(ids=ids, ids_version=ids._version, mask=attention_mask,
                       mask_version=None if attention_mask is None else attention_mask._version,
                       parameters=signature, rotary=rotary, expected=0, hit=hit)
        if hit:
            stats["hits"] += 1
        return result

    def layer(index, hidden, *args, **kwargs):
        if index <= last and current.get("expected") == index:
            current["expected"] += 1
            if current["hit"]:
                stats["skipped_layers"] += 1
                return cache["hidden"] if index == 0 else hidden
            output = original_layer(index, hidden, *args, **kwargs)
            if index == last:
                if output.requires_grad:
                    raise ValueError("prefix output unexpectedly retains a trainable graph")
                cache.clear()
                cache.update({k: v for k, v in current.items() if k not in {"expected", "hit"}})
                cache.update(hidden=output.detach(), hidden_version=output._version)
            return output
        return original_layer(index, hidden, *args, **kwargs)

    model._prefix_reuse_active = True
    model._prepare, model._layer = prepare, layer
    model.forward = forward
    try:
        yield stats
    finally:
        model._prepare, model._layer = original_prepare, original_layer
        model.forward = original_forward
        del model._prefix_reuse_active
        cache.clear()
        current.clear()
        default_mask.clear()
