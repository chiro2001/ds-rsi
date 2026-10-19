"""Lower a stopped CUDA A/B run's AdamW rate without resetting training state."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from dsrsi.data.revisions import data_identity
from dsrsi.model.cuda_execution import cuda_execution_identity
from dsrsi.utils.atomic import atomic_json
from .checkpoint import CheckpointError, CheckpointStore, Progress, file_sha256
from .topology import require_equal


def lower_learning_rate(source_run, target_run, *, learning_rate, device="cuda:0", dry_run=False):
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if target_run.exists() or source_run == target_run:
        raise CheckpointError("learning-rate migration requires a new target run")
    if not (source_run / "checkpoints/latest.json").is_file():
        raise CheckpointError("learning-rate migration requires a committed source")
    source = CheckpointStore(source_run / "checkpoints")
    directory, manifest = source.load_committed()
    pointer = (source.root / "latest.json").read_bytes()
    identity = copy.deepcopy(manifest["identity"])
    config = identity["train_config"]
    old_rate = config["learning_rate"]
    if (type(learning_rate) not in {int, float} or not math.isfinite(learning_rate)
            or not 0 < learning_rate < old_rate):
        raise CheckpointError("new learning rate must be finite, positive and strictly lower")
    progress = Progress(**manifest["progress"])
    if (manifest["world_size"] != 1 or progress.stage not in {"A", "B"}
            or identity.get("execution_semantics") != "ab-episode-v2-explicit-optimizer-backend"):
        raise CheckpointError("learning-rate migration requires single-rank stateless A/B")
    if manifest["status"] not in {"stopped", "paused"} or progress.accumulation_index:
        raise CheckpointError("learning-rate migration requires a stopped empty accumulation boundary")
    if identity.get("data") != data_identity(config["data_revision"]):
        raise CheckpointError("learning-rate migration data/compiler identity differs")
    if (config.get("gdn_inverse") is not None
            or identity.get("cuda_execution", {}).get("version") != "cuda-reference-gdn-eager-v1"):
        raise CheckpointError("learning-rate migration requires reference CUDA execution")
    if identity["cuda_execution"] != cuda_execution_identity(device):
        raise CheckpointError("learning-rate migration cannot change the CUDA runtime")
    payload = source.load_rank(directory, manifest, rank=0)
    if payload["life_state"] != {"mode": "m_only", "streams": {}, "harness_refs": {}}:
        raise CheckpointError("learning-rate migration cannot change persistent life state")
    if any(value is not None for value in payload["gradients"].values()):
        raise CheckpointError("learning-rate migration cannot discard pending gradients")
    if payload["rng"]["accelerator"] is None or payload["rng"]["accelerator"]["kind"] != "cuda":
        raise CheckpointError("learning-rate migration requires the existing CUDA RNG")
    if any(payload[name] != {"class": None, "state": None} for name in ("scheduler", "scaler")):
        raise CheckpointError("scheduled or scaled optimization requires its own explicit migration")
    if payload["optimizer"]["class"] != "torch.optim.adamw.AdamW":
        raise CheckpointError("learning-rate migration supports the existing AdamW optimizer")
    groups = payload["optimizer"]["state"]["param_groups"]
    if not groups or any(group["lr"] != old_rate or "initial_lr" in group for group in groups):
        raise CheckpointError("source AdamW learning rates do not match the configuration")
    migration = {
        "kind": "lower-adamw-learning-rate-preserve-full-state-v1",
        "source_run": str(source_run), "source_checkpoint": directory.name,
        "source_manifest_sha256": file_sha256(directory / "manifest.json"),
        "source_learning_rate": old_rate, "target_learning_rate": float(learning_rate),
        "progress": manifest["progress"], "data_cursor": payload["data_cursor"],
        "optimizer_moments_and_counters_reset": False, "rng_reset": False,
        "budget_changed": False, "data_cursor_reset": False,
        "source_control_requests_copied": False, "same_trajectory_as_source_claimed": False,
        "changed_payload_fields": [f"optimizer.state.param_groups[{i}].lr" for i in range(len(groups))],
    }
    if dry_run:
        return {**migration, "dry_run": True}
    converted = copy.deepcopy(payload)
    for group in converted["optimizer"]["state"]["param_groups"]:
        group["lr"] = float(learning_rate)
    config["learning_rate"] = float(learning_rate)
    # Trainer already preserves this provenance collection during full resume.
    # The kind explicitly identifies a numerical optimizer policy change.
    identity.setdefault("execution_migrations", []).append(migration)
    if (source.root / "latest.json").read_bytes() != pointer:
        raise CheckpointError("source advanced during learning-rate migration")
    target_run.mkdir(parents=True, exist_ok=False)
    target = CheckpointStore(target_run / "checkpoints")
    pending = target.begin(progress)
    target.save_rank(pending, converted)
    result = target.commit(pending, world_size=1, identity=identity, progress=progress, status="stopped")
    checked, checked_manifest = target.load_committed()
    actual = target.load_rank(checked, checked_manifest, rank=0)
    require_equal(converted, actual, "migrated learning-rate state")
    for group in actual["optimizer"]["state"]["param_groups"]:
        group["lr"] = old_rate
    require_equal(payload, actual, "all unchanged training state")
    source.load_committed()
    if (source.root / "latest.json").read_bytes() != pointer:
        raise CheckpointError("source advanced during learning-rate migration")
    report = {**migration, "target_run": str(target_run), "target_checkpoint": result.name,
              "target_manifest_sha256": file_sha256(result / "manifest.json"),
              "all_payload_fields_preserved_except_optimizer_group_lr": True}
    atomic_json(target_run / "learning-rate-migration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from .device import resolve_device
    torch.set_num_threads(2)
    device, _ = resolve_device(args.device)
    print(json.dumps(lower_learning_rate(args.source_run, args.target_run, learning_rate=args.learning_rate,
        device=device, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
