#!/usr/bin/env python3
"""Verify full-state recovery from a converted real A/B checkpoint."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp


def worker(rank, base, root, size, requested_device):
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, capture_training_state, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.training.distributed import CollectiveContext
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.topology import require_equal
    os.environ.update(LOCAL_RANK=str(rank), WORLD_SIZE=str(size))
    torch.set_num_threads(2)
    device, _ = resolve_device(requested_device)
    context = CollectiveContext.initialize(device=device, backend="hccl" if requested_device == "npu" else "gloo",
        rank=rank, world_size=size, init_method="file://" + str(Path(root) / "rendezvous"), timeout_seconds=180)
    started = time.monotonic()
    try:
        def load():
            _, manifest = CheckpointStore(Path(root) / "full/checkpoints").load_committed()
            config = TrainConfig(**manifest["identity"]["train_config"])
            torch.manual_seed(config.seed)
            checkpoint_root = Path(base)
            index = json.loads((checkpoint_root / "model.safetensors.index.json").read_text())["weight_map"]
            identity = {"tokenizer": tokenizer_fingerprint(checkpoint_root),
                "weights": {name: file_sha256(checkpoint_root / name) for name in sorted(set(index.values()))}}
            model = MemoryQwen.from_local_checkpoint(base, device=device)
            compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
            return model, compiler, identity, config
        model, compiler, identity, config = context.run_local("load migration proof artifacts", load)
        def trainer(name):
            return Trainer(model, compiler, Path(root) / name, config, base_identity=identity,
                           collectives=context, resume=True)
        def snapshot(instance):
            return capture_training_state(model, instance.optimizer, progress=instance.progress,
                data_cursor=instance.cursor, life_state=instance.life, rank=rank, world_size=size, device=device)
        full = trainer("full")
        initial_micro = full.progress.micro_step
        initial_ordinal = full.cursor["next_ordinal"]
        assert full.progress.accumulation_index == 0
        assert full.run(max_micro_steps=initial_micro + 5, install_signals=False) == "stopped"
        reference = snapshot(full)
        del full
        resumed = trainer("resumed")
        objective = resumed.objective
        calls = 0
        def pause(row, **kwargs):
            nonlocal calls
            result = objective(row, **kwargs)
            calls += 1
            if rank == size - 1 and calls == 3:
                resumed.control._signal_action = "pause"
            return result
        resumed.objective = pause
        assert resumed.run(max_micro_steps=initial_micro + 5, install_signals=False) == "paused"
        assert resumed.progress.micro_step == initial_micro + 3
        assert resumed.progress.accumulation_index == 3
        resumed = trainer("resumed")
        assert resumed.run(max_micro_steps=initial_micro + 5, install_signals=False) == "stopped"
        context.run_local("compare migrated training and recovery", lambda: require_equal(reference, snapshot(resumed), "state"))
        assert resumed.cursor["next_ordinal"] == initial_ordinal + 5 * size
        report = {"rank": rank, "device": device, "initial_ordinal": initial_ordinal,
            "cursor": resumed.cursor, "progress": asdict(resumed.progress),
            "exact_complete_state_match": True, "paused_after_new_microbatches": 3,
            "paused_pending_gradient_microbatches": 3, "elapsed_seconds": time.monotonic() - started}
        (Path(root) / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from dsrsi.training.topology import reduce_ab_replicas
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    conversions = {name: reduce_ab_replicas(args.source_run, root / name, world_size=args.world_size)
                   for name in ("full", "resumed")}
    mp.spawn(worker, args=(args.base, str(root), args.world_size, args.device), nprocs=args.world_size, join=True)
    report = {"status": "actual_checkpoint_replica_reduction_and_full_state_recovery_verified",
        "device": args.device, "world_size": args.world_size, "conversions": conversions,
        "ranks": [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(args.world_size)],
        "same_trajectory_as_four_rank_training_claimed": False, "trained_recall_claimed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
