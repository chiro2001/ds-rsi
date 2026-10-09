"""Read-only A/B weights loading; never restore optimizer, cursors or RNG."""

import torch

from dsrsi.training.checkpoint import CheckpointError


def load_ab_weights(model, store, *, base_identity, committed=None):
    directory, manifest = store.load_committed() if committed is None else committed
    identity = manifest["identity"]
    if identity["base"] != base_identity:
        raise CheckpointError("evaluation base differs from checkpoint's frozen base")
    config = identity["train_config"]
    if config["stage"] not in {"A", "B"} or config["stage"] != manifest["progress"]["stage"]:
        raise CheckpointError("evaluation loader supports explicit A/B checkpoints")
    model.address.requires_grad_(config["stage"] == "B")
    model.consolidator.requires_grad_(False)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    reference = store.load_rank(directory, manifest, rank=0)
    for label, current in (("weights", parameters), ("buffers", buffers)):
        saved = reference[label]
        if current.keys() != saved.keys():
            raise CheckpointError(f"evaluation {label} names differ")
        for name, value in current.items():
            if value.shape != saved[name].shape or value.dtype != saved[name].dtype:
                raise CheckpointError(f"evaluation {label} shape/dtype differs: {name}")
    # Pending gradients/RNG differ legitimately. Model weights and buffers must
    # be identical at every committed coordinated optimizer boundary.
    for rank in range(1, manifest["world_size"]):
        payload = store.load_rank(directory, manifest, rank=rank)
        for label in ("weights", "buffers"):
            if payload[label].keys() != reference[label].keys() or any(
                    not torch.equal(value, payload[label][name]) for name, value in reference[label].items()):
                raise CheckpointError(f"rank {rank} {label} differ; cannot export one rank as the model")
        del payload
    with torch.no_grad():
        for label, current in (("weights", parameters), ("buffers", buffers)):
            for name, value in current.items():
                value.copy_(reference[label][name].to(value.device))
    return directory, manifest
