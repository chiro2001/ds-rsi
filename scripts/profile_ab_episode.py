#!/usr/bin/env python3
"""Read-only A/B checkpoint timing, without optimizer steps or run mutations."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--frozen-prefix-reuse", action="store_true",
                        help="experimental scoped reuse; verifies exact losses and every parameter gradient first")
    parser.add_argument("--compare-prefix-reuse", action="store_true",
                        help="alternate baseline/reuse trials on the same copy to control timing drift")
    args = parser.parse_args()
    if args.compare_prefix_reuse:
        args.frozen_prefix_reuse = True
    if args.repeats < 1:
        parser.error("positive repeat count required")
    torch.set_num_threads(2)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.training.episode_objective import EpisodeObjective
    from dsrsi.model.frozen_prefix_reuse import frozen_prefix_reuse
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("isolated profiling uses one logical device and no HCCL group")
    synchronize = torch.npu.synchronize if device.startswith("npu") else lambda: None
    root = Path(args.base)
    names = set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values())
    identity = {"tokenizer": tokenizer_fingerprint(root),
                "weights": {name: file_sha256(root / name) for name in sorted(names)}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    if data_identity(config["data_revision"]) != committed[1]["identity"]["data"]:
        raise ValueError("profiling data identity differs from checkpoint")
    model = MemoryQwen.from_local_checkpoint(root, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture"))
    checkpoint, manifest = load_ab_weights(model, store, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    objective = EpisodeObjective(model, compiler, stage=config["stage"],
        memory_candidates=config.get("memory_candidates", 8), keep_weight=config.get("keep_weight", 0.2),
        counterfactual_weight=config.get("counterfactual_weight", 0),
        counterfactual_margin=config.get("counterfactual_margin", 2))
    generator = make_generator(revision=config["data_revision"], seed=config["seed"], distractors=config["distractors"])
    selected = {}
    for world in range(1000):
        row = generator.episode(world, variant=0, family="single_fact", language=args.language)
        kind = row["annotations"].get("codec_task", "full_code")
        if kind not in selected:
            selected[kind] = (world, row)
        if "full_code" in selected and ("indexed_character" in selected or config["data_revision"] != "ab-v5-indexed-digits"):
            break
    selected["keep"] = (0, generator.episode(0, variant=0, family="language_preservation", language=args.language))
    model.train()
    results = {}
    for kind, (world, row) in selected.items():
        counter = (generator.episode(world, variant=1, family="single_fact", language=args.language)
                   if kind != "keep" and config.get("counterfactual_weight", 0) else None)
        kwargs = {"counterfactual_row": counter} if counter is not None else {}
        equivalence = None
        if args.frozen_prefix_reuse:
            model.zero_grad(set_to_none=True)
            reference = objective(row, **kwargs)
            reference.loss.backward()
            expected_loss = reference.loss.detach().clone()
            expected_grads = {name: None if p.grad is None else p.grad.detach().clone()
                              for name, p in model.named_parameters() if p.requires_grad}
            model.zero_grad(set_to_none=True)
            with frozen_prefix_reuse(model) as reuse:
                actual = objective(row, **kwargs)
                actual.loss.backward()
            checks = [torch.equal(actual.loss, expected_loss)]
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                old = expected_grads[name]
                checks.append((old is None and p.grad is None) or
                              (old is not None and p.grad is not None and torch.equal(old, p.grad)))
            if not all(checks):
                raise AssertionError("prefix reuse changed the loss or a parameter gradient")
            equivalence = {"exact_loss_and_all_gradients": True, "gradient_tensors_checked": len(expected_grads), **reuse}
            del reference, actual, expected_grads, expected_loss
        # Warm-up then uninstrumented timing, without optimizer updates.
        totals, baseline_totals = [], []
        for repeat in range(args.repeats + 1):
            modes = [args.frozen_prefix_reuse]
            if args.compare_prefix_reuse:
                modes = [False, True] if repeat % 2 == 0 else [True, False]
            for use_reuse in modes:
                model.zero_grad(set_to_none=True)
                synchronize()
                started = time.perf_counter()
                with frozen_prefix_reuse(model) if use_reuse else nullcontext():
                    output = objective(row, **kwargs)
                    output.loss.backward()
                synchronize()
                elapsed = time.perf_counter() - started
                if repeat:
                    (baseline_totals if args.compare_prefix_reuse and not use_reuse else totals).append(elapsed)
                del output
        components = defaultdict(list)

        def measured(name, function, *values, **options):
            synchronize()
            started = time.perf_counter()
            try:
                return function(*values, **options)
            finally:
                synchronize()
                components[name].append(time.perf_counter() - started)

        original_sources, original_forward, original_head = objective._sources, model.forward, model.head.forward
        objective._sources = lambda episode: measured("source_features_writer_keys", original_sources, episode)
        model.forward = lambda *values, **options: measured(
            "answer_forward_including_ce" if options.get("labels") is not None else
            "keep_teacher_forward" if options.get("memory_enabled") is False else "contrast_prefix_forward",
            original_forward, *values, **options)
        model.head.forward = lambda *values, **options: measured("explicit_head_projection", original_head, *values, **options)
        try:
            model.zero_grad(set_to_none=True)
            with frozen_prefix_reuse(model) if args.frozen_prefix_reuse else nullcontext():
                output = objective(row, **kwargs)
                measured("backward", output.loss.backward)
            tokens = {"student": output.student_tokens, "teacher": output.teacher_tokens,
                      "supervised": output.supervised_tokens, "padding": output.padding_positions}
            del output
        finally:
            objective._sources, model.forward, model.head.forward = original_sources, original_forward, original_head
            model.zero_grad(set_to_none=True)
        results[kind] = {"tokens": tokens, "uninstrumented_forward_backward_seconds": totals,
                         "equivalence": equivalence,
                         "median_seconds": statistics.median(totals),
                         "instrumented_component_seconds": {name: {"calls": len(values), "total": sum(values)}
                                                            for name, values in components.items()}}
        if baseline_totals:
            results[kind]["paired_comparison"] = {
                "baseline_seconds": baseline_totals, "baseline_median_seconds": statistics.median(baseline_totals),
                "trial_speedups": [a / b for a, b in zip(baseline_totals, totals)],
                "median_speedup": statistics.median(a / b for a, b in zip(baseline_totals, totals)),
                "alternating_execution_order": True}
        print(json.dumps({"case": kind, **results[kind]}), flush=True)
    report = {"status": "isolated_episode_timing_diagnostic", "checkpoint": checkpoint.name,
              "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "checkpoint_progress": manifest["progress"],
              "base_identity": identity, "data_identity": data_identity(config["data_revision"]),
              "device": str(device), "language": args.language, "repeats": args.repeats, "results": results,
              "experimental_frozen_prefix_reuse": args.frozen_prefix_reuse,
              "optimizer_steps_applied": 0, "formal_training_ledger_modified": False,
              "includes_distributed_sync_or_optimizer": False,
              "timing_scope": "Isolated forward/backward; synchronized component timings alter overlap. Explicit head calls cover contrast/keep logits; answer CE's functional projection is inside answer forward.",
              "capability_gate_passed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
