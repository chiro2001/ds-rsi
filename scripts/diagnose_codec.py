#!/usr/bin/env python3
"""Pinned-checkpoint diagnostics of code positions, payload sensitivity and readers."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worlds-per-position", type=int, default=2)
    parser.add_argument("--language", choices=["zh", "en", "cross_language"], default="en")
    args = parser.parse_args()
    if args.worlds_per_position < 1:
        parser.error("worlds per position must be positive")
    from dsrsi.training.device import resolve_device
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("diagnostics use one logical device")
    torch.set_num_threads(4)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.codec import (canonical_logits, native_logits, distinguishing_token,
        swapped_payloads, payload_stats, summarize_logits)
    from dsrsi.evaluation.m_only import MOnlyEvaluator
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.episode_objective import EpisodeObjective
    root = Path(args.base)
    weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    base_identity = {"tokenizer": tokenizer_fingerprint(root),
        "weights": {name: file_sha256(root / name) for name in sorted(set(weights.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    revision = config["data_revision"]
    if revision != "ab-v4-fields-and-language" or data_identity(revision) != committed[1]["identity"]["data"]:
        raise ValueError("this stratified diagnostic requires the pinned v4 fields distribution")
    model = MemoryQwen.from_local_checkpoint(root, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture")).eval()
    checkpoint, manifest = load_ab_weights(model, store, base_identity=base_identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    generator = make_generator(revision=revision, seed=config["seed"], split="dev", distractors=config["distractors"])
    objective = EpisodeObjective(model, compiler, stage=config["stage"], memory_candidates=1)
    evaluator = MOnlyEvaluator(model, compiler, memory_candidates=1)
    counts, records = defaultdict(int), []
    with torch.no_grad():
        for world in range(1000 * args.worlds_per_position):
            rows = [generator.episode(world, variant=v, family="single_fact", language=args.language) for v in (0, 1)]
            position = rows[0]["annotations"]["counterfactual_character_position"]
            if counts[position] >= args.worlds_per_position:
                continue
            counts[position] += 1
            episodes = [compiler.compile(row) for row in rows]
            pools = [objective._sources(episode)[0] for episode in episodes]
            for variant in (0, 1):
                episode, opposite = episodes[variant], episodes[1 - variant]
                response = episode.responses[0]
                changed, targets, opposite_targets = distinguishing_token(response, opposite.responses[0])
                support = response.support_groups[0]
                selected = [r for r in pools[variant] if r.record_id in support]
                if len(selected) != 1:
                    raise ValueError("single-record oracle diagnostic needs exactly one support")
                foreign = swapped_payloads(selected, pools[1 - variant])
                trace = {}
                native = native_logits(model, response, selected, life_id=episode.episode_id, reader_trace=trace)
                native_foreign = native_logits(model, response, foreign, life_id=episode.episode_id)
                prefix = evaluator._encode_prefix(response)
                canonical = canonical_logits(model, compiler, response, selected, life_id=episode.episode_id, prefix=prefix)
                canonical_foreign = canonical_logits(model, compiler, response, foreign, life_id=episode.episode_id, prefix=prefix)
                margin = {}
                for name, actual, replaced in (("native", native, native_foreign), ("canonical", canonical, canonical_foreign)):
                    correct_id, opposite_id = targets[changed], opposite_targets[changed]
                    p, q = F.log_softmax(actual[changed], dim=-1), F.log_softmax(replaced[changed], dim=-1)
                    margin[name] = {"correct_payload_logp_advantage": float(p[correct_id] - q[correct_id]),
                        "actual_target_over_opposite_logit": float(p[correct_id] - p[opposite_id]),
                        "foreign_target_over_actual_logit": float(q[opposite_id] - q[correct_id])}
                native_summary = summarize_logits(native, targets, compiler.tokenizer)
                canonical_summary = summarize_logits(canonical, targets, compiler.tokenizer)
                entry = {"world": world, "variant": variant, "episode_id": episode.episode_id,
                    "changed_character_position": position, "changed_token_position": changed,
                    "expected": rows[variant]["probes"][0]["target"],
                    "opposite_expected": rows[1 - variant]["probes"][0]["target"],
                    "native": native_summary, "canonical": canonical_summary,
                    "counterfactual": margin, "reader": trace,
                    "payload": payload_stats(selected[0], foreign[0]),
                    "max_target_nll_path_difference": max(abs(a - b) for a, b in
                        zip(native_summary["target_nll"], canonical_summary["target_nll"]))}
                records.append(entry)
                print(json.dumps({"episode": episode.episode_id, "position": position,
                                  "payload_advantage": margin}, ensure_ascii=False), flush=True)
            if len(counts) == 5 and min(counts.values()) >= args.worlds_per_position:
                break
    if len(counts) != 5 or min(counts.values()) < args.worlds_per_position:
        raise RuntimeError("insufficient independent worlds to cover all character positions")
    def summarize(rows):
        token_count = len(rows[0]["canonical"]["correct"])
        if any(len(r["canonical"]["correct"]) != token_count for r in rows):
            raise ValueError("cannot combine targets with different tokenizer lengths")
        return {"queries": len(rows),
            "teacher_forced_accuracy": {name: [statistics.mean(r[name]["correct"][i] for r in rows)
                for i in range(token_count)] for name in ("native", "canonical")},
            "mean_target_nll": {name: [statistics.mean(r[name]["target_nll"][i] for r in rows)
                for i in range(token_count)] for name in ("native", "canonical")},
            "mean_changed_token_payload_advantage": {name: statistics.mean(
                r["counterfactual"][name]["correct_payload_logp_advantage"] for r in rows) for name in ("native", "canonical")},
            "max_target_nll_path_difference": max(r["max_target_nll_path_difference"] for r in rows)}
    report = {"status": "teacher_forced_codec_diagnostic", "checkpoint": checkpoint.name,
        "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "checkpoint_progress": manifest["progress"],
        "base_identity": base_identity, "data_identity": data_identity(revision), "language": args.language,
        "worlds_per_position": args.worlds_per_position, "device": device, "memory_candidates": 1,
        "source_encoding": "training_batched_independent_events", "canonical_execution_quantum": 1,
        "gold_prefix_used": True, "future_tokens_used_in_canonical_prediction": False,
        "summary": summarize(records), "by_changed_character_position": {str(p): summarize(
            [r for r in records if r["changed_character_position"] == p]) for p in sorted(counts)},
        "records": records, "capability_gate_passed": False}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
