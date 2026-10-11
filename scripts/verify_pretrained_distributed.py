#!/usr/bin/env python3
"""CPU/Gloo or authorized NPU/HCCL full-state recovery; no capability claim."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp


def assert_equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"{path}: tensors differ")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def worker(rank, base, run_root, threads, identity, requested_device, world_size, controls, writer_anchor_weight,
           reader_gate_mode, micro_steps, source_run, source_checkpoint, compare_prefix_reuse):
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.memory.modules import MemoryConfig
    from dsrsi.training.checkpoint import (CheckpointStore, capture_rng, restore_rng,
                                         capture_training_state, file_sha256)
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.pinned import load_pinned
    from dsrsi.training.distributed import CollectiveContext
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.device import resolve_device

    torch.set_num_threads(threads)
    os.environ.update(LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    device, _ = resolve_device(requested_device)
    context = CollectiveContext.initialize(device=device, backend="hccl" if device.startswith("npu") else "gloo",
                                           rank=rank, world_size=world_size,
                                           init_method="file://" + str(Path(run_root) / "rendezvous"),
                                           timeout_seconds=180)
    started = time.monotonic()
    try:
        source_metadata = {}
        def load():
            random.seed(20261010)
            np.random.seed(20261010)
            torch.manual_seed(20261010)
            if source_run:
                store = CheckpointStore(Path(source_run) / "checkpoints")
                committed = load_pinned(store, source_checkpoint)
                model = MemoryQwen.from_local_checkpoint(base, device=device,
                    memory_architecture=committed[1]["identity"]["memory_architecture"])
                checkpoint, _ = load_ab_weights(model, store, base_identity=identity, committed=committed)
                source_metadata.update(checkpoint=checkpoint.name,
                    manifest_sha256=file_sha256(checkpoint / "manifest.json"),
                    source_weights_used_but_test_optimizer_and_cursor_started_fresh=True)
            else:
                model = MemoryQwen.from_local_checkpoint(base, device=device,
                    memory_config=MemoryConfig(writer_anchor_weight=writer_anchor_weight, reader_gate_mode=reader_gate_mode))
            return model, EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
        model, compiler = context.run_local("load actual base", load)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()
                   if not name.startswith(("base.", "head."))}
        initial_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        initial_rng = capture_rng(device)
        config = TrainConfig(student_token_budget=100000,
                             seed=20261010, save_every_micro=100, **controls)
        def trainer(directory, resume=False, optimized=None):
            active_config = config if optimized is None else replace(config, reuse_frozen_prefix=optimized)
            return Trainer(model, compiler, Path(run_root) / directory, active_config,
                           base_identity=identity, collectives=context, resume=resume)
        def reset():
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in initial:
                        p.copy_(initial[name])
                    p.grad = None
                for name, buffer in model.named_buffers():
                    buffer.copy_(initial_buffers[name])
            restore_rng(initial_rng, device)
        def snapshot(instance):
            return capture_training_state(model, instance.optimizer, progress=instance.progress,
                                          data_cursor=instance.cursor, life_state=instance.life,
                                          device=device, rank=rank, world_size=world_size)
        full = trainer("full", optimized=False if compare_prefix_reuse else None)
        auxiliary, natural_keep, unknown_time_keep = 0, 0, 0
        full_objective = full.objective
        def observe(row, **kwargs):
            nonlocal auxiliary, natural_keep, unknown_time_keep
            auxiliary += int(row.get("annotations", {}).get("codec_task") == "indexed_character")
            is_natural = bool(row.get("annotations", {}).get("natural_keep"))
            natural_keep += int(is_natural)
            unknown_time_keep += int(is_natural and not row["annotations"]["keep_wall_time_known"])
            return full_objective(row, **kwargs)
        full.objective = observe
        assert full.run(max_micro_steps=micro_steps, install_signals=False) == "stopped"
        global_auxiliary = context.totals(auxiliary, 0, 0)[0]
        global_natural, global_unknown, _ = context.totals(natural_keep, unknown_time_keep, 0)
        if config.data_revision in {"ab-v5-indexed-digits", "ab-v6-natural-keep"} and not global_auxiliary:
            raise AssertionError("recovery proof did not exercise any indexed-character episodes")
        if config.data_revision == "ab-v6-natural-keep" and (not global_natural or not global_unknown):
            raise AssertionError("recovery proof did not exercise natural continuation and unknown-time memory")
        reference = snapshot(full)
        del full
        reset()
        if compare_prefix_reuse:
            optimized = trainer("optimized", optimized=True)
            assert optimized.run(max_micro_steps=micro_steps, install_signals=False) == "stopped"
            context.run_local("compare original and prefix reuse states",
                              lambda: assert_equal(reference, snapshot(optimized)))
            del optimized
            reset()
        resumed = trainer("resumed")
        original = resumed.objective
        calls = 0
        def pause(row, **kwargs):
            nonlocal calls
            result = original(row, **kwargs)
            calls += 1
            if rank == world_size - 1 and calls == 3:
                resumed.control._signal_action = "pause"
            return result
        resumed.objective = pause
        assert resumed.run(max_micro_steps=micro_steps, install_signals=False) == "paused"
        assert resumed.progress.micro_step == 3 and resumed.progress.accumulation_index == 3
        resumed = trainer("resumed", resume=True)
        assert resumed.run(max_micro_steps=micro_steps, install_signals=False) == "stopped"
        context.run_local("compare complete training states", lambda: assert_equal(reference, snapshot(resumed)))
        report = {"rank": rank, "device": device, "progress": asdict(resumed.progress), "cursor": resumed.cursor,
                  "exact_state_match": True, "worker_pause_micro_step": 3,
                  "pending_gradient_microbatches": resumed.progress.accumulation_index,
                  "observed_auxiliary_episodes": auxiliary, "global_observed_auxiliary_episodes": global_auxiliary,
                  "observed_natural_keep_episodes": natural_keep,
                  "global_natural_keep_episodes": global_natural,
                  "global_unknown_time_keep_episodes": global_unknown,
                  "source": source_metadata,
                  "original_vs_prefix_reuse_exact_state_match": True if compare_prefix_reuse else None,
                  "elapsed_seconds": time.monotonic() - started}
        (Path(run_root) / f"result_rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads-per-rank", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--memory-candidates", type=int, default=8)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--curriculum", choices=["mixed", "facts", "locations"], default="mixed")
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument("--keep-fraction", type=float)
    parser.add_argument("--keep-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--writer-anchor-weight", type=float, default=0)
    parser.add_argument("--reader-gate-mode", choices=["token", "lookup"], default="token")
    parser.add_argument("--data-revision", default="ab-v4-fields-and-language")
    parser.add_argument("--micro-steps", type=int, default=5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--reuse-frozen-prefix", action="store_true")
    parser.add_argument("--compare-prefix-reuse", action="store_true",
                        help="also compare unoptimized and optimized complete training states")
    parser.add_argument("--source-run", help="load trained memory weights, without touching the source run")
    parser.add_argument("--source-checkpoint", help="explicit committed source generation for the read-only proof")
    args = parser.parse_args()
    if args.source_checkpoint and not args.source_run:
        parser.error("--source-checkpoint requires --source-run")
    if args.threads_per_rank < 1 or args.world_size < 2 or args.micro_steps < 4 or args.grad_accum <= 3:
        raise ValueError("threads must be positive, world size at least two and accumulation greater than the pause step3")
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from dsrsi.data.compiler import tokenizer_fingerprint
    from dsrsi.training.checkpoint import file_sha256
    base = Path(args.base).resolve()
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(set(index.values()))}}
    controls = {"memory_candidates": args.memory_candidates, "counterfactual_weight": args.counterfactual_weight,
                "curriculum": args.curriculum, "distractors": args.distractors,
                "keep_fraction": args.keep_fraction, "keep_weight": args.keep_weight,
                "learning_rate": args.lr}
    controls["data_revision"] = args.data_revision
    controls["gradient_accumulation"] = args.grad_accum
    controls["reuse_frozen_prefix"] = args.reuse_frozen_prefix or args.compare_prefix_reuse
    mp.spawn(worker, args=(str(base), str(root), args.threads_per_rank, identity, args.device, args.world_size, controls,
                           args.writer_anchor_weight, args.reader_gate_mode, args.micro_steps,
                           args.source_run, args.source_checkpoint, args.compare_prefix_reuse),
             nprocs=args.world_size, join=True)
    report = {"status": f"pretrained_{args.device}_{'hccl' if args.device == 'npu' else 'gloo'}_recovery_only",
              "world_size": args.world_size,
              "controls": controls,
              "writer_anchor_weight": args.writer_anchor_weight,
              "reader_gate_mode": args.reader_gate_mode,
              "source_run": args.source_run, "source_checkpoint": args.source_checkpoint,
              "compare_prefix_reuse": args.compare_prefix_reuse,
              "threads_per_rank": args.threads_per_rank, "base_identity": identity,
              "ranks": [json.loads((root / f"result_rank_{rank}.json").read_text()) for rank in range(args.world_size)],
              "trained_recall_claimed": False, "hccl_verified": args.device == "npu"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
