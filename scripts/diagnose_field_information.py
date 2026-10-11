#!/usr/bin/env python3
"""Held-out linear field decoding from frozen Writer slots and source H11."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-worlds", type=int, default=256)
    parser.add_argument("--dev-worlds", type=int, default=128)
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_worlds < 1 or args.dev_worlds < 1 or not 0 < args.ridge < float("inf"):
        parser.error("world counts and finite regularization must be positive")
    torch.set_num_threads(4)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.field_probe import code_labels, linear_field_probe, CODE_POSITIONS
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("frozen representation probes use one logical device")
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(root),
                "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    revision = config["data_revision"]
    if revision not in {"ab-v4-fields-and-language", "ab-v5-indexed-digits"} or data_identity(revision) != committed[1]["identity"]["data"]:
        raise ValueError("field probes require the pinned v4/v5 full-code distribution")
    model = MemoryQwen.from_local_checkpoint(root, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture")).eval()
    checkpoint, manifest = load_ab_weights(model, store, base_identity=identity, committed=committed)
    model.requires_grad_(False)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    samples, features = {}, {}
    with torch.no_grad():
        for split, worlds in (("train", args.train_worlds), ("dev", args.dev_worlds)):
            view = {"probe_view": "full"} if revision == "ab-v5-indexed-digits" else {}
            generator = make_generator(revision=revision, seed=config["seed"], split=split, distractors=config["distractors"], **view)
            rows, slots, pooled = [], [], []
            for world in range(worlds):
                for variant in (0, 1):
                    row = generator.episode(world, variant=variant, family="single_fact", language=args.language)
                    episode = compiler.compile(row)
                    support = episode.responses[0].support_groups[0]
                    span = next(s for s in episode.source if s.event_id in support)
                    ids = torch.tensor([span.input_ids], device=device)
                    hidden = model.source_features(ids)
                    with model._memory_context(hidden):
                        payload = model.writer(hidden)
                    slots.append(payload[0].float().cpu().flatten())
                    pooled.append(F.adaptive_avg_pool1d(hidden.float().transpose(1, 2), model.memory_config.slots)
                                  .transpose(1, 2)[0].cpu().flatten())
                    rows.append({"world": world, "variant": variant, "episode_id": episode.episode_id,
                                 "source_ids": span.input_ids, "expected": row["probes"][0]["target"],
                                 "changed_character_position": row["annotations"]["counterfactual_character_position"]})
                if (world + 1) % 32 == 0:
                    print(json.dumps({"event": "source_features", "split": split, "worlds": world + 1}), flush=True)
            samples[split] = rows
            features[split] = {"writer_slots": torch.stack(slots), "position_pooled_h11": torch.stack(pooled)}
    if {tuple(r["source_ids"]) for r in samples["train"]} & {tuple(r["source_ids"]) for r in samples["dev"]}:
        raise ValueError("probe train/dev source content overlaps")
    labels = {split: code_labels([r["expected"] for r in rows]) for split, rows in samples.items()}
    results = {}
    for name in features["train"]:
        predictions, metrics = linear_field_probe(features["train"][name], labels["train"],
                                                  features["dev"][name], labels["dev"], ridge=args.ridge)
        by_position = {}
        for position in CODE_POSITIONS:
            pairs = [i for i in range(0, len(samples["dev"]), 2)
                     if samples["dev"][i]["changed_character_position"] == position]
            field = CODE_POSITIONS.index(position)
            by_position[str(position)] = {"pairs": len(pairs), "paired_changed_field_accuracy":
                sum(bool((predictions["dev"][i:i + 2, field] == labels["dev"][i:i + 2, field]).all())
                    for i in pairs) / len(pairs) if pairs else None}
        results[name] = {"feature_dimensions": features["train"][name].shape[1], **metrics,
                         "counterfactual_pairs_by_position": by_position,
                         "dev_predictions": predictions["dev"].tolist()}
    report = {"status": "frozen_representation_linear_probe_only", "checkpoint": checkpoint.name,
              "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "checkpoint_progress": manifest["progress"],
              "base_identity": identity, "data_identity": data_identity(revision), "device": device,
              "language": args.language, "memory_architecture": model.memory_architecture_identity(),
              "ridge": args.ridge, "normalization_and_fitting_split": "train", "hyperparameters_tuned_on_dev": False,
              "source_encoding": "single_independent_support_event", "source_only_no_query_or_completion_input": True,
              "probe_installed_in_model": False, "capability_gate_passed": False,
              "interpretation": "Held-out success lower-bounds linearly accessible information; failure is inconclusive.",
              "train_worlds": args.train_worlds, "dev_worlds": args.dev_worlds, "field_order": list(CODE_POSITIONS),
              "results": results, "dev_samples": [{k: v for k, v in row.items() if k != "source_ids"}
                                                  for row in samples["dev"]]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in result.items() if k != "dev_predictions"}
                      for name, result in results.items()}), flush=True)


if __name__ == "__main__":
    main()
