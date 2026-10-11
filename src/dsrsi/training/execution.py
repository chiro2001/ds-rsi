"""Explicit execution optimization migration, preserving all training payloads."""

import argparse
import copy
import json
from pathlib import Path

from dsrsi.data.revisions import data_identity
from dsrsi.training.checkpoint import CheckpointError, CheckpointStore, file_sha256
from dsrsi.utils.atomic import atomic_json


def enable_prefix_reuse(source_run, target_run, *, dry_run=False):
    return _enable_execution(source_run, target_run, optimization="prefix_reuse", dry_run=dry_run)


def enable_scripted_gdn(source_run, target_run, *, dry_run=False):
    return _enable_execution(source_run, target_run, optimization="scripted_gdn", dry_run=dry_run)


def _enable_execution(source_run, target_run, *, optimization, dry_run):
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if target_run.exists() or source_run == target_run:
        raise CheckpointError("execution migration requires a new target run")
    if not (source_run / "checkpoints/latest.json").is_file():
        raise CheckpointError("execution migration requires an existing committed source")
    source = CheckpointStore(source_run / "checkpoints")
    directory, manifest = source.load_committed()
    identity = copy.deepcopy(manifest["identity"])
    config = identity["train_config"]
    if (manifest["progress"]["stage"] not in {"A", "B"}
            or identity["execution_semantics"] != "ab-episode-v4-query-weighted-coordinated-gradients"):
        raise CheckpointError("execution migration requires the stateless distributed A/B executor")
    if manifest["status"] not in {"stopped", "paused"} or manifest["progress"]["accumulation_index"]:
        raise CheckpointError("execution migration requires a stopped optimizer boundary")
    if optimization == "prefix_reuse" and config.get("reuse_frozen_prefix", False):
        raise CheckpointError("source already uses prefix reuse")
    if optimization == "scripted_gdn" and config.get("gdn_inverse") is not None:
        raise CheckpointError("source already uses a GDN execution override")
    if config.get("gdn_inverse") is not None:
        from dsrsi.model.gdn_execution import gdn_execution_identity
        if config["gdn_inverse"] != "scripted_rows" or identity.get("gdn_execution") != gdn_execution_identity():
            raise CheckpointError("source GDN execution identity differs")
    if data_identity(config["data_revision"]) != identity["data"]:
        raise CheckpointError("execution migration data identity differs")
    payloads = [source.load_rank(directory, manifest, rank=r) for r in range(manifest["world_size"])]
    from .topology import require_equal
    for payload in payloads:
        if payload["life_state"] != {"mode": "m_only", "streams": {}, "harness_refs": {}}:
            raise CheckpointError("execution migration cannot discard persistent life state")
        if any(g is not None for g in payload["gradients"].values()):
            raise CheckpointError("execution migration cannot discard pending gradients")
        for label in ("weights", "frozen_auxiliary", "buffers", "optimizer", "scheduler", "scaler"):
            require_equal(payloads[0][label], payload[label], label)
    if optimization == "prefix_reuse":
        implementation = Path(__file__).resolve().parents[1] / "model/frozen_prefix_reuse.py"
        config["reuse_frozen_prefix"] = True
        identity["prefix_reuse"] = {"version": "native-frozen-prefix-reuse-v1", "sha256": file_sha256(implementation)}
        kind, detail = "enable-frozen-prefix-reuse-at-optimizer-boundary-v1", {"prefix_reuse": identity["prefix_reuse"]}
    else:
        from dsrsi.model.gdn_execution import gdn_execution_identity
        config["gdn_inverse"] = "scripted_rows"
        identity["gdn_execution"] = gdn_execution_identity()
        kind, detail = "enable-scripted-gdn-at-optimizer-boundary-v1", {"gdn_execution": identity["gdn_execution"]}
    report = {"kind": kind,
              "source_run": str(source_run), "source_checkpoint": directory.name,
              "source_manifest_sha256": file_sha256(directory / "manifest.json"),
              "world_size": manifest["world_size"], "progress": manifest["progress"],
              "optimizer_and_counters_reset": False, "training_payloads_modified": False,
              "source_control_requests_copied": False, **detail}
    if dry_run:
        return {**report, "dry_run": True}
    identity.setdefault("execution_migrations", []).append(report)
    from .checkpoint import Progress
    target_run.mkdir(parents=True, exist_ok=False)
    target = CheckpointStore(target_run / "checkpoints")
    pending = target.begin(Progress(**manifest["progress"]))
    for payload in payloads:
        target.save_rank(pending, payload)
    result = target.commit(pending, world_size=manifest["world_size"], identity=identity,
                           progress=Progress(**manifest["progress"]), status="stopped")
    directory2, manifest2 = target.load_committed()
    for rank, payload in enumerate(payloads):
        require_equal(payload, target.load_rank(directory2, manifest2, rank=rank), f"rank{rank}")
    result_report = {**report, "target_run": str(target_run), "target_checkpoint": result.name,
                     "target_manifest_sha256": file_sha256(result / "manifest.json"), "exact_payload_match": True}
    atomic_json(target_run / "execution-migration.json", result_report)
    return result_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--optimization", choices=["prefix_reuse", "scripted_gdn"], default="prefix_reuse")
    args = parser.parse_args()
    print(json.dumps(_enable_execution(args.source_run, args.target_run,
                                      optimization=args.optimization, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
