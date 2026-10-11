#!/usr/bin/env python3
"""Read-only 0.8B loss/gradient and alternating timing for experimental GDN."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch


def difference(value, reference):
    return {"max_abs": float((value - reference).abs().max()),
            "relative_l2": float((value - reference).norm() / reference.norm().clamp_min(1e-12)),
            "finite": bool(torch.isfinite(value).all()), "bitwise_equal": bool(torch.equal(value, reference))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--languages", nargs="+", choices=["zh", "en"], default=["zh", "en"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeat count required")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.model.experimental_gdn import experimental_gdn_inverse
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.training.episode_objective import EpisodeObjective
    from dsrsi.utils.atomic import atomic_json
    torch.set_num_threads(2)
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("one isolated device required, without HCCL")
    sync = torch.npu.synchronize if device.startswith("npu") else lambda: None
    base = Path(args.base)
    files = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"].values())
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(files)}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    if data_identity(config["data_revision"]) != committed[1]["identity"]["data"]:
        raise ValueError("profiling data differs from the checkpoint")
    model = MemoryQwen.from_local_checkpoint(base, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture"))
    checkpoint, manifest = load_ab_weights(model, store, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    objective = EpisodeObjective(model, compiler, stage=config["stage"],
        memory_candidates=config.get("memory_candidates", 8), keep_weight=config.get("keep_weight", 0.2),
        counterfactual_weight=config.get("counterfactual_weight", 0),
        counterfactual_margin=config.get("counterfactual_margin", 2),
        reuse_frozen_prefix=config.get("reuse_frozen_prefix", False))
    generator = make_generator(revision=config["data_revision"], seed=config["seed"],
                               distractors=config["distractors"])
    model.train()
    report = {"status": "experimental_full_model_gdn_diagnostic", "checkpoint": checkpoint.name,
              "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "device": device,
              "repeats": args.repeats, "frozen_prefix_reuse": config.get("reuse_frozen_prefix", False),
              "summation_order_changed": True, "formal_training_modified": False,
              "original_trajectory_equivalence_claimed": False, "cases": {}}
    for language in args.languages:
        selected = {}
        for world in range(1000):
            for family in ("single_fact", "language_preservation"):
                row = generator.episode(world, variant=0, family=family, language=language)
                kind = ("natural_keep" if row["annotations"].get("natural_keep") else "legacy_keep") if family == "language_preservation" else row["annotations"]["codec_task"]
                selected.setdefault(kind, (world, row))
            if len(selected) == 4:
                break
        if len(selected) != 4:
            raise ValueError("all four episode kinds are required")
        for kind, (world, row) in selected.items():
            options = {}
            if row["family"] == "single_fact" and config.get("counterfactual_weight", 0):
                options["counterfactual_row"] = generator.episode(world, variant=1, family="single_fact", language=language)
            def execute(candidate):
                model.zero_grad(set_to_none=True)
                with experimental_gdn_inverse(model) if candidate else nullcontext():
                    output = objective(row, **options)
                    output.loss.backward()
                return output
            old = execute(False)
            loss = old.loss.detach().clone()
            gradients = {name: None if p.grad is None else p.grad.detach().clone()
                         for name, p in model.named_parameters() if p.requires_grad}
            del old
            new = execute(True)
            errors = {}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                previous = gradients[name]
                if (previous is None) != (p.grad is None):
                    raise AssertionError(f"gradient participation changed: {name}")
                if previous is not None:
                    errors[name] = difference(p.grad, previous)
            case = {"world": world, "loss_reference": float(loss),
                    "loss_candidate": float(new.loss.detach()), "loss_error": difference(new.loss.detach(), loss),
                    "gradient_errors": errors, "baseline_seconds": [], "candidate_seconds": []}
            del new, gradients, loss
            for repeat in range(args.repeats + 1):
                for candidate in ((False, True) if repeat % 2 == 0 else (True, False)):
                    sync()
                    started = time.perf_counter()
                    output = execute(candidate)
                    sync()
                    elapsed = time.perf_counter() - started
                    del output
                    if repeat:
                        case["candidate_seconds" if candidate else "baseline_seconds"].append(elapsed)
            case["median_speedup"] = statistics.median(a / b for a, b in zip(case["baseline_seconds"], case["candidate_seconds"]))
            case["baseline_median"] = statistics.median(case["baseline_seconds"])
            case["candidate_median"] = statistics.median(case["candidate_seconds"])
            report["cases"][f"{language}/{kind}"] = case
            atomic_json(Path(args.output + ".partial.json"), report)
            print(json.dumps({"case": f"{language}/{kind}", "median_speedup": case["median_speedup"],
                              "loss_error": case["loss_error"],
                              "max_gradient_relative_l2": max(value["relative_l2"] for value in errors.values())}), flush=True)
    model.zero_grad(set_to_none=True)
    atomic_json(Path(args.output), report)


if __name__ == "__main__":
    main()
