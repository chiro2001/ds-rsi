"""Read-only A/B weights loading; never restore optimizer, cursors or RNG."""

import torch

from dsrsi.training.checkpoint import CheckpointError, frozen_auxiliary_parameters


def load_ab_weights(model, store, *, base_identity, committed=None):
    directory, manifest = store.load_committed() if committed is None else committed
    identity = manifest["identity"]
    if identity["base"] != base_identity:
        raise CheckpointError("evaluation base differs from checkpoint's frozen base")
    if identity.get("memory_architecture", model.memory_architecture_identity()) != model.memory_architecture_identity():
        raise CheckpointError("memory architecture differs from checkpoint")
    config = identity["train_config"]
    if config["stage"] not in {"A", "B"} or config["stage"] != manifest["progress"]["stage"]:
        raise CheckpointError("evaluation loader supports explicit A/B checkpoints")
    model.address.requires_grad_(config["stage"] == "B")
    model.consolidator.requires_grad_(False)
    return _load_groups(model, store, directory, manifest)


def load_c_tool_weights(model, store, *, base_identity, committed=None):
    from dsrsi.model.lora import LoRAConfig, install_lora
    directory, manifest = store.load_committed() if committed is None else committed
    identity = manifest["identity"]
    if identity["base"] != base_identity or manifest["progress"]["stage"] != "C_tools":
        raise CheckpointError("C-tools evaluation base/stage mismatch")
    if identity.get("memory_architecture", model.memory_architecture_identity()) != model.memory_architecture_identity():
        raise CheckpointError("C-tools memory architecture differs from checkpoint")
    if identity["train_config"]["stage"] != "C_tools":
        raise CheckpointError("C-tools config/stage mismatch")
    model.address.requires_grad_(True)
    model.consolidator.requires_grad_(False)
    layout = identity["lora"]
    actual = install_lora(model, LoRAConfig(**{key: layout[key] for key in ("rank", "alpha", "start_layer", "end_layer")}))
    if actual != layout:
        raise CheckpointError("C-tools LoRA layout differs from checkpoint")
    return _load_groups(model, store, directory, manifest, require_frozen=True)


def _load_groups(model, store, directory, manifest, *, require_frozen=False):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    reference = store.load_rank(directory, manifest, rank=0)
    if require_frozen and "frozen_auxiliary" not in reference:
        raise CheckpointError("C-tools evaluation requires complete frozen auxiliary state")
    groups = [("weights", parameters), ("buffers", buffers)]
    if "frozen_auxiliary" in reference:
        groups.append(("frozen_auxiliary", frozen_auxiliary_parameters(model)))
    for label, current in groups:
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
        for label, _ in groups:
            if payload[label].keys() != reference[label].keys() or any(
                    not torch.equal(value, payload[label][name]) for name, value in reference[label].items()):
                raise CheckpointError(f"rank {rank} {label} differ; cannot export one rank as the model")
        del payload
    with torch.no_grad():
        for label, current in groups:
            for name, value in current.items():
                value.copy_(reference[label][name].to(value.device))
    return directory, manifest


def load_context_weights(model, store, *, base_identity, committed=None):
    from dsrsi.model.lora import LoRAConfig, install_lora
    from dsrsi.training.context_runner import configure_context_trainable
    directory, manifest = store.load_committed() if committed is None else committed
    identity = manifest["identity"]
    if identity["base"] != base_identity or manifest["progress"]["stage"] != "C_context":
        raise CheckpointError("continuous evaluation base/stage mismatch")
    if identity["memory_architecture"] != model.memory_architecture_identity():
        raise CheckpointError("continuous memory architecture differs")
    layout = identity["lora"]
    actual = install_lora(model, LoRAConfig(**{key: layout[key] for key in ("rank", "alpha", "start_layer", "end_layer")}))
    if actual != layout:
        raise CheckpointError("continuous LoRA layout differs")
    configure_context_trainable(model)
    return _load_groups(model, store, directory, manifest, require_frozen=True)
