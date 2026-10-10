#!/usr/bin/env python3
"""Held-out tool execution on a committed C-tools model, without teacher rollout."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch


def main():
    from dsrsi.data.tool_generate import TOOL_FAMILIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--journal", required=True, help="separate held-out evaluation journal; never the training journal")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--families", nargs="+", choices=TOOL_FAMILIES, default=list(TOOL_FAMILIES))
    parser.add_argument("--languages", nargs="+", choices=["zh", "en"], default=["zh"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-prefill-diagnostic", action="store_true")
    parser.add_argument("--disable-lora", action="store_true", help="same prompts and harness, original frozen-language baseline")
    args = parser.parse_args()
    if min(args.worlds, args.threads) < 1:
        raise ValueError("positive worlds/threads are required")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.tool_generate import ToolEpisodeGenerator
    from dsrsi.evaluation.checkpoint import load_c_tool_weights
    from dsrsi.evaluation.tools import ToolEvaluator
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.model.lora import lora_disabled
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    torch.set_num_threads(args.threads)
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    base = {"tokenizer": tokenizer_fingerprint(root),
            "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    torch.manual_seed(config["seed"])
    model = MemoryQwen.from_local_checkpoint(root,
        memory_architecture=committed[1]["identity"].get("memory_architecture"))
    directory, manifest = load_c_tool_weights(model, store, base_identity=base, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True),
                               max_query_tokens=config["max_query_tokens"])
    evaluator = ToolEvaluator(model, compiler, max_new_tokens=args.max_new_tokens, max_turns=args.max_turns,
                              parallel_prefill_diagnostic=args.parallel_prefill_diagnostic)
    generator = ToolEpisodeGenerator(seed=config["seed"], split=args.split, journal_path=args.journal)
    with lora_disabled(model) if args.disable_lora else nullcontext():
        records = [evaluator.episode(generator.episode(world, family=family, language=language, select_one=False))
                   for world in range(args.worlds) for family in args.families for language in args.languages]
    summary = {"episodes": len(records), **{metric: sum(row[metric] for row in records) / len(records) for metric in
               ("tool_call_exact_match", "explicit_state_match", "final_answer_exact_match", "end_to_end_success", "requested_operation_success")}}
    report = {"status": "heldout_tool_execution", "split": args.split, "base_identity": base,
              "checkpoint": directory.name, "manifest_sha256": file_sha256(directory / "manifest.json"),
              "training_world_size": manifest["world_size"], "summary": summary, "records": records,
              "prompt_execution_quantum": None if args.parallel_prefill_diagnostic else 1,
              "parallel_prefill_diagnostic": args.parallel_prefill_diagnostic,
              "lora_enabled": not args.disable_lora,
              "journal": str(Path(args.journal).resolve()), "capability_gate_passed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
