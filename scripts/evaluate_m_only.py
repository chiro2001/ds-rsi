#!/usr/bin/env python3
"""Held-out free generation and memory ablation on a committed checkpoint."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--families", nargs="+", default=["single_fact"],
                        choices=["single_fact", "paraphrase_coreference", "episode_and_task", "temporal_update",
                                 "missing_evidence", "language_preservation"])
    parser.add_argument("--languages", nargs="+", choices=["zh", "en", "cross_language"], default=["zh"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--include-foreign", action="store_true", help="replace all selected payloads with the other counterfactual world")
    args = parser.parse_args()
    if args.worlds < 1 or args.distractors < 8 or args.threads < 1:
        raise ValueError("positive worlds/threads and at least 8 distractors are required")
    if len(set(args.families)) != len(args.families) or len(set(args.languages)) != len(args.languages):
        raise ValueError("duplicate task families or languages would inflate evaluation counts")
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.generate import EpisodeGenerator
    from dsrsi.evaluation.m_only import MOnlyEvaluator, normalized_answer
    from dsrsi.evaluation.checkpoint import load_ab_weights, load_c_tool_weights
    from dsrsi.evaluation.metrics import summarize_records
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    run = Path(args.run_dir)
    store = CheckpointStore(run / "checkpoints")
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    actual_base = {"tokenizer": tokenizer_fingerprint(root),
                   "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = MemoryQwen.from_local_checkpoint(root)
    compiler = EpisodeCompiler(tokenizer)
    loader = load_c_tool_weights if config["stage"] == "C_tools" else load_ab_weights
    checkpoint_dir, manifest = loader(model, store, base_identity=actual_base, committed=committed)
    evaluator = MOnlyEvaluator(model, compiler, max_new_tokens=args.max_new_tokens)
    generator = EpisodeGenerator(seed=config["seed"], split=args.split, distractors=args.distractors)
    records = []
    model.eval()
    for world in range(args.worlds):
        for family in args.families:
            for language in args.languages:
                for variant in (0, 1):
                    row = generator.episode(world, variant=variant, family=family, language=language)
                    if args.retrieval_only:
                        with torch.no_grad():
                            episode = compiler.compile(row)
                            sources = evaluator.source_records(episode)
                            results = [evaluator.retrieval(response, sources, life_id=episode.episode_id)
                                       for response in episode.responses]
                    else:
                        replacements = None
                        foreign_expected = {}
                        modes = ("actual", "oracle", "off", "shuffled")
                        if args.include_foreign:
                            foreign = generator.episode(world, variant=1 - variant, family=family, language=language)
                            with torch.no_grad():
                                replacements = evaluator.source_records(compiler.compile(foreign))
                            foreign_expected = {p["probe_id"]: p["target"] for p in foreign["probes"]}
                            modes += ("foreign",)
                        results = evaluator.episode(row, modes=modes, replacement_records=replacements)
                        for result in results:
                            if result["mode"] == "foreign":
                                result["counterfactual_payload_answer"] = foreign_expected[result["probe_id"]]
                                result["follows_counterfactual_payload"] = (normalized_answer(result["text"])
                                    == normalized_answer(foreign_expected[result["probe_id"]]))
                    for result in results:
                        result.update(episode_id=row["episode_id"], world_id=row["world_id"], variant=variant,
                                      family=family, language=language)
                        records.append(result)
    summary = summarize_records(records, retrieval_only=args.retrieval_only)
    report = {"status": "heldout_retrieval_only" if args.retrieval_only else "heldout_free_generation", "split": args.split,
              "checkpoint": checkpoint_dir.name, "training_world_size": manifest["world_size"],
              "manifest_sha256": file_sha256(checkpoint_dir / "manifest.json"), "base_identity": actual_base,
              "families": args.families, "languages": args.languages, "distractors": args.distractors,
              "canonical_execution_quantum": 1,
              "foreign_payload_ablation": args.include_foreign and not args.retrieval_only,
              "checkpoint_progress": manifest["progress"], "max_new_tokens": args.max_new_tokens,
              "worlds": args.worlds, "summary": summary, "records": records,
              "capability_gate_passed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checkpoint_progress": manifest["progress"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
