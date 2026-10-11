#!/usr/bin/env python3
"""Compare original/optimized training and optimized mid-gradient recovery."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import random

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-steps", type=int, default=50)
    parser.add_argument("--data-revision", choices=["ab-v5-indexed-digits", "ab-v6-natural-keep"],
                        default="ab-v5-indexed-digits")
    parser.add_argument("--gdn-inverse", choices=["scripted_rows"],
                        help="profiling-only GDN test; both sides retain frozen prefix reuse")
    args = parser.parse_args()
    if args.micro_steps < 4:
        parser.error("at least four microsteps required")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.training.checkpoint import (CheckpointStore, file_sha256, capture_rng, restore_rng,
                                         capture_training_state)
    from dsrsi.training.device import resolve_device
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.topology import require_equal
    torch.set_num_threads(2)
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("this proof uses one isolated device; no HCCL group")
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=False)
    base = Path(args.base)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(set(index.values()))}}
    source = CheckpointStore(Path(args.source_run) / "checkpoints")
    committed = source.load_committed()
    random.seed(20261010)
    np.random.seed(20261010)
    torch.manual_seed(20261010)
    model = MemoryQwen.from_local_checkpoint(base, device=device,
        memory_architecture=committed[1]["identity"]["memory_architecture"])
    checkpoint, manifest = load_ab_weights(model, source, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    initial = {name: p.detach().clone() for name, p in model.named_parameters()
               if not name.startswith(("base.", "head."))}
    initial_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    initial_rng = capture_rng(device)
    config = TrainConfig(student_token_budget=100000, gradient_accumulation=6,
                         distractors=7, seed=20261010, curriculum="facts", memory_candidates=1,
                         data_revision=args.data_revision, counterfactual_weight=1,
                         keep_fraction=0.4, keep_weight=0.5, reuse_frozen_prefix=bool(args.gdn_inverse))
    def reset():
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])
                parameter.grad = None
            for name, buffer in model.named_buffers():
                buffer.copy_(initial_buffers[name])
        restore_rng(initial_rng, device)
    def trainer(name, optimized, resume=False):
        return Trainer(model, compiler, root / name, replace(config, reuse_frozen_prefix=optimized or bool(args.gdn_inverse)),
                       base_identity=identity, resume=resume)
    def scope():
        if not args.gdn_inverse:
            return nullcontext()
        from dsrsi.model.experimental_gdn import experimental_gdn_inverse
        return experimental_gdn_inverse(model, args.gdn_inverse)
    def snapshot(t):
        return capture_training_state(model, t.optimizer, progress=t.progress, data_cursor=t.cursor,
                                      life_state=t.life, device=device)
    base_trainer = trainer("baseline", False)
    coverage = {"auxiliary": 0, "natural_keep": 0, "unknown_wall_keep": 0, "legacy_keep": 0, "full_code": 0}
    original = base_trainer.objective
    def observe(row, **kwargs):
        a = row.get("annotations", {})
        if a.get("natural_keep"):
            coverage["natural_keep"] += 1
            coverage["unknown_wall_keep"] += int(not a.get("keep_wall_time_known"))
        elif row["family"] == "language_preservation":
            coverage["legacy_keep"] += 1
        elif a.get("codec_task") == "indexed_character":
            coverage["auxiliary"] += 1
        else:
            coverage["full_code"] += 1
        return original(row, **kwargs)
    base_trainer.objective = observe
    assert base_trainer.run(max_micro_steps=args.micro_steps, install_signals=False) == "stopped"
    if not coverage["auxiliary"]:
        raise AssertionError("proof did not exercise indexed-character episodes")
    if args.data_revision == "ab-v6-natural-keep" and not all(coverage[k] for k in coverage):
        raise AssertionError(f"proof did not cover the v6 episode kinds: {coverage}")
    reference = snapshot(base_trainer)
    del base_trainer
    reset()
    full = trainer("optimized", True)
    with scope():
        assert full.run(max_micro_steps=args.micro_steps, install_signals=False) == "stopped"
    require_equal(reference, snapshot(full), "original-vs-optimized")
    del full
    reset()
    part = trainer("resumed", True)
    with scope():
        assert part.run(max_micro_steps=3, install_signals=False) == "stopped"
    assert part.progress.accumulation_index == 3
    del part
    resumed = trainer("resumed", True, resume=True)
    with scope():
        assert resumed.run(max_micro_steps=args.micro_steps, install_signals=False) == "stopped"
    require_equal(reference, snapshot(resumed), "optimized-recovery")
    report = {"status": ("pretrained_gdn_training_equivalence_and_recovery_only" if args.gdn_inverse else
                         "pretrained_prefix_reuse_training_equivalence_and_recovery_only"),
              "source_checkpoint": checkpoint.name, "source_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
              "source_weights_used_but_test_optimizer_and_cursor_started_fresh": True,
              "base_identity": identity, "device": device, "world_size": 1, "config": asdict(config),
              "micro_steps": args.micro_steps, "pause_micro_step": 3, "pause_accumulation_index": 3,
              "observed_auxiliary_episodes": coverage["auxiliary"], "episode_coverage": coverage,
              "gdn_inverse": args.gdn_inverse, "common_prefix_reuse": bool(args.gdn_inverse),
              "profiling_scope_not_persisted_in_test_checkpoints": bool(args.gdn_inverse),
              "final_progress": asdict(resumed.progress),
              "original_vs_optimized_exact_state_match": True, "optimized_recovery_exact_state_match": True,
              "formal_training_ledger_modified": False, "trained_recall_claimed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
