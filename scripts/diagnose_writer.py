#!/usr/bin/env python3
"""Inspect source-only Writer attention and loss of counterfactual digits."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--writer-anchor-weight", type=float, help="explicit untrained architecture challenge")
    args = parser.parse_args()
    from dsrsi.training.device import resolve_device
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("Writer inspection uses one logical device")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.codec import payload_stats
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    torch.set_num_threads(4)
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(root),
        "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    revision = config["data_revision"]
    if data_identity(revision) != committed[1]["identity"]["data"]:
        raise ValueError("source data identity differs")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    compiler = EpisodeCompiler(tokenizer)
    architecture = copy.deepcopy(committed[1]["identity"].get("memory_architecture"))
    if args.writer_anchor_weight is not None:
        if architecture is None:
            raise ValueError("architecture challenge requires recorded source architecture")
        if args.writer_anchor_weight:
            architecture["memory_config"]["writer_anchor_weight"] = args.writer_anchor_weight
            architecture["writer_anchor_version"] = "position_pool_v1"
        else:
            architecture["memory_config"].pop("writer_anchor_weight", None)
            architecture.pop("writer_anchor_version", None)
    model = MemoryQwen.from_local_checkpoint(root, device=device, memory_architecture=architecture).eval()
    checkpoint, manifest = load_ab_weights(model, store, base_identity=identity, committed=committed,
                                           allow_writer_anchor_change=args.writer_anchor_weight is not None)
    generator = make_generator(revision=revision, seed=config["seed"], split="dev", distractors=config["distractors"])
    entries, seen = [], set()
    with torch.no_grad():
        for world in range(100):
            rows = [generator.episode(world, variant=v, family="single_fact", language="en") for v in (0, 1)]
            position = rows[0]["annotations"]["counterfactual_character_position"]
            if position in seen:
                continue
            seen.add(position)
            features, records, profiles = [], [], []
            for row in rows:
                episode = compiler.compile(row)
                span = next(s for s in episode.source if s.event_id in episode.responses[0].support_groups[0])
                ids = torch.tensor([span.input_ids], device=device)
                h = model.source_features(ids)
                features.append(h)
                records.extend(model.write_records(ids, stream_ids=[episode.episode_id], logical_begins=[span.logical_begin],
                    record_ids=[span.event_id], wall_times=[span.wall_time]))
                writer = model.writer
                with model._memory_context(h):
                    normalized = writer.input_norm(h)
                    queries = writer.q(writer.query_norm(writer.queries.to(h.dtype)[None]))
                    keys = writer.k(normalized)
                heads, width = writer.heads, queries.shape[-1] // writer.heads
                q = queries.reshape(1, -1, heads, width).transpose(1, 2).float()
                k = keys.reshape(1, -1, heads, width).transpose(1, 2).float()
                scores = q @ k.transpose(-1, -2) / math.sqrt(width)
                probabilities = scores.softmax(-1)[0]
                code = row["probes"][0]["target"]
                suffix = tokenizer.encode("-" + code[3:], add_special_tokens=False)
                starts = [i for i in range(len(span.input_ids) - len(suffix) + 1)
                          if span.input_ids[i:i + len(suffix)] == suffix]
                if len(starts) != 1 or len(suffix) != 4:
                    raise ValueError("source tokenizer did not identify three individual suffix digits")
                suffix_positions = list(range(starts[0] + 1, starts[0] + 4))
                entropy = -(probabilities * probabilities.clamp_min(1e-38).log()).sum(-1)
                peaks, peak_positions = probabilities.max(-1)
                profiles.append({"source_ids": span.input_ids,
                    "source_tokens": [tokenizer.decode([i]) for i in span.input_ids],
                    "suffix_token_positions": suffix_positions,
                    "suffix_digit_attention_mass_per_head_slot": probabilities[..., suffix_positions].tolist(),
                    "mean_attention_mass_per_suffix_digit": probabilities[..., suffix_positions].mean((0, 1)).tolist(),
                    "mean_entropy": float(entropy.mean()), "minimum_entropy": float(entropy.min()),
                    "maximum_logit_range": float((scores.max(-1).values - scores.min(-1).values).max()),
                    "peak_mass": peaks.tolist(), "peak_positions": peak_positions.tolist()})
            entries.append({"world": world, "changed_character_position": position,
                "answers": [row["probes"][0]["target"] for row in rows], "attention": profiles,
                "source_hidden_relative_change": float((features[0].float() - features[1].float()).square().mean().sqrt()
                    / features[0].float().square().mean().sqrt()), "payload": payload_stats(records[0], records[1])})
            if len(seen) == 5:
                break
    report = {"status": "source_only_writer_attention_diagnostic", "checkpoint": checkpoint.name,
        "checkpoint_progress": manifest["progress"], "manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "base_identity": identity, "data_identity": data_identity(revision), "device": device,
        "memory_architecture": model.memory_architecture_identity(),
        "untrained_architecture_challenge": model.memory_architecture_identity() != manifest["identity"]["memory_architecture"],
        "source_encoding": "single_independent_event", "records": entries, "capability_gate_passed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "attention"} for r in entries], ensure_ascii=False))


if __name__ == "__main__":
    main()
