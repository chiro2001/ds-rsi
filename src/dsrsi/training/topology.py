"""Explicit A/B replica reduction at a clean optimizer boundary.

This preserves training state, but intentionally changes future global batches.
Stateful C streams and partially accumulated gradients cannot use this path.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from dsrsi.data.revisions import LEGACY_AB_REVISION, data_identity
from dsrsi.utils.atomic import atomic_json
from .checkpoint import CheckpointError, CheckpointStore, Progress, file_sha256


def require_equal(left, right, path):
    if type(left) is not type(right):
        raise CheckpointError(f"rank state differs: {path}")
    if isinstance(left, torch.Tensor):
        equal = left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        equal = left.keys() == right.keys()
        if equal:
            for key in left:
                require_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        equal = len(left) == len(right)
        if equal:
            for index, (a, b) in enumerate(zip(left, right)):
                require_equal(a, b, f"{path}[{index}]")
    else:
        equal = left == right
    if not equal:
        raise CheckpointError(f"rank state differs: {path}")


def reduce_ab_replicas(source_run, target_run, *, world_size, dry_run=False):
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if target_run.exists() or source_run == target_run:
        raise CheckpointError("topology conversion requires a new target run directory")
    source = CheckpointStore(source_run / "checkpoints")
    directory, manifest = source.load_committed()
    old_size = manifest["world_size"]
    if not 2 <= world_size < old_size:
        raise CheckpointError("only replica reduction to at least two ranks is supported")
    progress = Progress(**manifest["progress"])
    identity = copy.deepcopy(manifest["identity"])
    if (progress.stage not in {"A", "B"}
            or identity.get("execution_semantics") != "ab-episode-v4-query-weighted-coordinated-gradients"):
        raise CheckpointError("conversion requires the stateless query-weighted A/B executor")
    if manifest["status"] not in {"paused", "stopped"} or progress.accumulation_index:
        raise CheckpointError("conversion requires a stopped/paused clean optimizer boundary")
    revision = identity["train_config"].get("data_revision", LEGACY_AB_REVISION)
    if identity.get("data") != data_identity(revision):
        raise CheckpointError("conversion data/compiler identity differs from source")
    payloads = [source.load_rank(directory, manifest, rank=rank) for rank in range(old_size)]
    reference = payloads[0]
    next_ordinal = reference["data_cursor"].get("next_ordinal")
    if not isinstance(next_ordinal, int) or next_ordinal < 0 or next_ordinal % old_size:
        raise CheckpointError("source cursor is not a contiguous replica boundary")
    shared = ("weights", "frozen_auxiliary", "buffers", "optimizer", "scheduler", "scaler")
    for rank, payload in enumerate(payloads):
        if payload["data_cursor"] != {"schema_version": 1, "next_ordinal": next_ordinal + rank}:
            raise CheckpointError("source rank cursors are not contiguous")
        if payload["life_state"] != {"mode": "m_only", "streams": {}, "harness_refs": {}}:
            raise CheckpointError("conversion cannot discard persistent life or harness state")
        if any(grad is not None for grad in payload["gradients"].values()):
            raise CheckpointError("conversion cannot discard accumulated gradients")
        for key in shared:
            require_equal(reference[key], payload[key], key)
    migration = {
        "kind": "ab-replica-reduction-at-optimizer-boundary-v1",
        "source_run": str(source_run), "source_checkpoint": directory.name,
        "source_manifest_sha256": file_sha256(directory / "manifest.json"),
        "source_world_size": old_size, "target_world_size": world_size,
        "next_global_ordinal": next_ordinal, "progress": manifest["progress"],
        "rank_rng_mapping": {str(rank): rank for rank in range(world_size)},
        "inactive_source_rng_ranks": list(range(world_size, old_size)),
        "inactive_rng_preserved_in_source_checkpoint": True,
        "optimizer_and_counters_reset": False,
        "same_trajectory_as_original_topology": False,
        "future_global_episodes_per_optimizer_step": world_size * identity["train_config"]["gradient_accumulation"],
        "source_control_requests_copied": False,
    }
    identity.setdefault("topology_migrations", []).append(migration)
    if dry_run:
        return {**migration, "dry_run": True}
    target_run.mkdir(parents=True, exist_ok=False)
    target = CheckpointStore(target_run / "checkpoints")
    pending = target.begin(progress)
    for rank in range(world_size):
        payload = copy.deepcopy(payloads[rank])
        payload["world_size"] = world_size
        payload["data_cursor"]["next_ordinal"] = next_ordinal + rank
        target.save_rank(pending, payload)
    converted = target.commit(pending, world_size=world_size, identity=identity,
                              progress=progress, status="stopped")
    # Validate the complete committed generation before advertising success.
    verified, converted_manifest = target.load_committed()
    for rank in range(world_size):
        got = target.load_rank(verified, converted_manifest, rank=rank)
        expected = copy.deepcopy(payloads[rank])
        expected["world_size"] = world_size
        require_equal(expected, got, f"converted.rank{rank}")
    report = {**migration, "target_run": str(target_run), "target_checkpoint": converted.name,
              "target_manifest_sha256": file_sha256(converted / "manifest.json"),
              "retained_rank_payloads_exact_match_except_world_size": True}
    atomic_json(target_run / "topology-conversion.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reduce_ab_replicas(args.source_run, args.target_run,
                                     world_size=args.world_size, dry_run=args.dry_run),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
