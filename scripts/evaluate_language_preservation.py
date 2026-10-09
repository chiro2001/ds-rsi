#!/usr/bin/env python3
"""Original-base vs adapted model, with unrelated populated M and no M."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    from dsrsi.training.device import resolve_device
    device, world = resolve_device(args.device)
    if world != 1:
        raise ValueError("language probes run on one logical device")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.language_probes import PROBE_REVISION, probes, matches
    from dsrsi.evaluation.checkpoint import load_ab_weights, load_c_tool_weights, load_context_weights
    from dsrsi.evaluation.m_only import MOnlyEvaluator
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.model.lora import lora_disabled
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(root),
                "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    model = MemoryQwen.from_local_checkpoint(root, device=device).eval()
    loader = {"C_tools": load_c_tool_weights, "C_context": load_context_weights}.get(
        committed[1]["progress"]["stage"], load_ab_weights)
    checkpoint, manifest = loader(model, store, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    evaluator = MOnlyEvaluator(model, compiler, max_new_tokens=args.max_new_tokens, memory_candidates=1)
    records = []
    with torch.no_grad():
        for family, row in probes():
            episode = compiler.compile(row)
            source = evaluator.source_records(episode)
            response = episode.responses[0]
            outputs = {}
            for name, mode in (("original_base", "off"), ("adapted_no_memory", "off"), ("adapted_unrelated_memory", "actual")):
                with lora_disabled(model) if name == "original_base" else nullcontext():
                    answer = evaluator.generate(response, source, life_id=episode.episode_id,
                        expected=row["probes"][0]["target"], mode=mode)
                outputs[name] = {"text": answer.text, "correct": matches(answer.text, answer.expected, family=family),
                                 "stopped_on_end": answer.stopped_on_end}
            records.append({"id": row["episode_id"], "family": family, "language": row["language"],
                            "expected": row["probes"][0]["target"], "outputs": outputs})
    def summarize(rows):
        base_correct = [r for r in rows if r["outputs"]["original_base"]["correct"]]
        return {"queries": len(rows), "base_correct_queries": len(base_correct),
                "accuracy": {name: sum(r["outputs"][name]["correct"] for r in rows) / len(rows)
                             for name in ("original_base", "adapted_no_memory", "adapted_unrelated_memory")},
                "retained_base_correct_with_unrelated_memory": (sum(r["outputs"]["adapted_unrelated_memory"]["correct"]
                    for r in base_correct) / len(base_correct) if base_correct else None)}
    strata = defaultdict(list)
    for row in records:
        strata[f"{row['family']}/{row['language']}"].append(row)
    report = {"status": "elementary_heldout_language_retention_diagnostic", "probe_revision": PROBE_REVISION,
              "probe_source_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/dsrsi/data/language_probes.py"),
              "checkpoint": checkpoint.name, "manifest_sha256": file_sha256(checkpoint / "manifest.json"),
              "checkpoint_progress": manifest["progress"], "base_identity": identity, "device": device,
              "canonical_execution_quantum": 1, "max_new_tokens": args.max_new_tokens,
              "summary": summarize(records), "by_family_language": {k: summarize(v) for k, v in strata.items()},
              "records": records, "broad_language_capability_gate_claimed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
