"""Explicit, audited NPU A/B replicas to one CUDA device at an empty boundary."""

import argparse
import copy
import json
from pathlib import Path

import torch

from dsrsi.data.revisions import data_identity
from dsrsi.model.cuda_execution import cuda_execution_identity
from dsrsi.utils.atomic import atomic_json
from .checkpoint import CheckpointError, CheckpointStore, Progress, file_sha256
from .topology import require_equal


def new_cuda_rng(device, seed):
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    with torch.cuda.device(index), torch.random.fork_rng(devices=[index]):
        torch.cuda.manual_seed(seed)
        state = torch.cuda.get_rng_state(index).cpu().clone()
    return {"kind": "cuda", "state": state}


def migrate_ab_to_cuda(source_run, target_run, *, device="cuda:0", cuda_seed=20261010,
                       gradient_accumulation=None, dry_run=False):
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if target_run.exists() or source_run == target_run:
        raise CheckpointError("hardware migration requires a new target run")
    if not (source_run / "checkpoints/latest.json").is_file():
        raise CheckpointError("hardware migration requires a committed source")
    if type(cuda_seed) is not int or not 0 <= cuda_seed < 2**63:
        raise CheckpointError("CUDA seed must be a nonnegative 63-bit integer")
    source = CheckpointStore(source_run / "checkpoints")
    directory, manifest = source.load_committed()
    source_pointer = (source.root / "latest.json").read_bytes()
    identity = copy.deepcopy(manifest["identity"])
    progress = Progress(**manifest["progress"])
    old_size = manifest["world_size"]
    config = identity["train_config"]
    if (old_size < 2 or progress.stage not in {"A", "B"}
            or config.get("stage", progress.stage) != progress.stage
            or identity.get("execution_semantics") != "ab-episode-v4-query-weighted-coordinated-gradients"):
        raise CheckpointError("hardware migration requires stateless distributed A/B execution")
    if manifest["status"] not in {"paused", "stopped"} or progress.accumulation_index:
        raise CheckpointError("hardware migration requires a stopped empty accumulation boundary")
    if config.get("gdn_inverse") is not None or "gdn_execution" in identity or "cuda_execution" in identity:
        raise CheckpointError("hardware migration requires the original NPU GDN execution")
    if identity.get("data") != data_identity(config["data_revision"]):
        raise CheckpointError("hardware migration data/compiler identity differs")
    accumulation = old_size * config["gradient_accumulation"]
    if gradient_accumulation is not None and (
            type(gradient_accumulation) is not int or gradient_accumulation != accumulation):
        raise CheckpointError("CUDA accumulation must preserve the source global episode batch")
    payloads = [source.load_rank(directory, manifest, rank=r) for r in range(old_size)]
    reference = payloads[0]
    frontier = reference["data_cursor"].get("next_ordinal")
    if type(frontier) is not int or frontier < 0:
        raise CheckpointError("hardware migration requires a contiguous source cursor")
    for rank, payload in enumerate(payloads):
        if payload["data_cursor"] != {"schema_version": 1, "next_ordinal": frontier + rank}:
            raise CheckpointError("source replica cursors are not contiguous")
        if payload["life_state"] != {"mode": "m_only", "streams": {}, "harness_refs": {}}:
            raise CheckpointError("hardware migration cannot discard persistent life state")
        if any(value is not None for value in payload["gradients"].values()):
            raise CheckpointError("hardware migration cannot discard pending gradients")
        if payload["rng"]["accelerator"] is None or payload["rng"]["accelerator"]["kind"] != "npu":
            raise CheckpointError("hardware migration requires explicit NPU RNG provenance")
        for label in ("weights", "frozen_auxiliary", "buffers", "optimizer", "scheduler", "scaler"):
            require_equal(reference[label], payload[label], label)
    runtime = cuda_execution_identity(device)
    migration = {
        "kind": "npu-replicas-to-single-cuda-preserve-optimizer-v1",
        "source_run": str(source_run), "source_checkpoint": directory.name,
        "source_manifest_sha256": file_sha256(directory / "manifest.json"),
        "source_world_size": old_size, "target_world_size": 1,
        "source_accumulation": config["gradient_accumulation"], "target_accumulation": accumulation,
        "global_episode_batch_preserved": True, "episodes_per_optimizer_step": accumulation,
        "next_global_ordinal": frontier, "progress": manifest["progress"],
        "optimizer_and_counters_reset": False, "source_control_requests_copied": False,
        "portable_rng_source_rank": 0, "inactive_source_rng_ranks": list(range(1, old_size)),
        "all_original_npu_rng_preserved_in_source": True,
        "cuda_rng_explicitly_initialized": True, "cuda_seed": cuda_seed,
        "same_trajectory_as_source_claimed": False, "cuda_execution": runtime,
    }
    if dry_run:
        return {**migration, "dry_run": True}
    identity["execution_semantics"] = "ab-episode-v2-explicit-optimizer-backend"
    config["gradient_accumulation"] = accumulation
    identity["cuda_execution"] = runtime
    identity.setdefault("hardware_migrations", []).append(migration)
    converted_payload = copy.deepcopy(reference)
    converted_payload["world_size"] = 1
    converted_payload["rng"]["accelerator"] = new_cuda_rng(device, cuda_seed)
    target_run.mkdir(parents=True, exist_ok=False)
    target = CheckpointStore(target_run / "checkpoints")
    pending = target.begin(progress)
    target.save_rank(pending, converted_payload)
    converted = target.commit(pending, world_size=1, identity=identity, progress=progress, status="stopped")
    checked, checked_manifest = target.load_committed()
    require_equal(converted_payload, target.load_rank(checked, checked_manifest, rank=0), "converted CUDA state")
    # Compare all unchanged fields, including Python/NumPy/CPU RNG and optimizer.
    portable = copy.deepcopy(converted_payload)
    portable["world_size"] = old_size
    portable["rng"]["accelerator"] = reference["rng"]["accelerator"]
    require_equal(reference, portable, "preserved source state")
    source.load_committed()
    if (source.root / "latest.json").read_bytes() != source_pointer:
        raise CheckpointError("source checkpoint advanced during hardware migration")
    report = {**migration, "target_run": str(target_run), "target_checkpoint": converted.name,
              "target_manifest_sha256": file_sha256(converted / "manifest.json"),
              "all_payload_fields_preserved_except_world_size_and_accelerator_rng": True}
    atomic_json(target_run / "hardware-migration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--cuda-seed", type=int, default=20261010)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from .device import resolve_device
    torch.set_num_threads(2)
    device, _ = resolve_device(args.device)
    print(json.dumps(migrate_ab_to_cuda(args.source_run, args.target_run, device=device,
        cuda_seed=args.cuda_seed, gradient_accumulation=args.grad_accum, dry_run=args.dry_run),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
