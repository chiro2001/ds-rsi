#!/usr/bin/env python3
"""Read-only single-chip Ascend operator trace from a pinned A/B weight copy."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=["full_code", "indexed_character", "natural_keep"], default="full_code")
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.model.frozen_prefix_reuse import frozen_prefix_reuse
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.training.episode_objective import EpisodeObjective
    from dsrsi.utils.atomic import atomic_json
    import torch_npu
    torch.set_num_threads(2)
    device, size = resolve_device(args.device)
    if size != 1 or not device.startswith("npu"):
        raise ValueError("Ascend traces use one isolated NPU; no HCCL group")
    output = Path(args.output_dir).resolve()
    run = Path(args.run_dir).resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError("trace requires a new output directory outside the source run")
    store = CheckpointStore(run / "checkpoints")
    committed = store.load_committed()
    config = committed[1]["identity"]["train_config"]
    if data_identity(config["data_revision"]) != committed[1]["identity"]["data"]:
        raise ValueError("trace data identity differs from the checkpoint")
    base = Path(args.base)
    files = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"].values())
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(files)}}
    model = MemoryQwen.from_local_checkpoint(base, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture"))
    checkpoint, manifest = load_ab_weights(model, store, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    objective = EpisodeObjective(model, compiler, stage=config["stage"],
        memory_candidates=config.get("memory_candidates", 8), keep_weight=config.get("keep_weight", 0.2),
        counterfactual_weight=config.get("counterfactual_weight", 0),
        counterfactual_margin=config.get("counterfactual_margin", 2))
    generator = make_generator(revision=config["data_revision"], seed=config["seed"],
                               distractors=config["distractors"])
    family = "language_preservation" if args.kind == "natural_keep" else "single_fact"
    for world in range(1000):
        row = generator.episode(world, variant=0, family=family, language=args.language)
        if ((args.kind == "natural_keep" and row["annotations"].get("natural_keep"))
                or row["annotations"].get("codec_task") == args.kind):
            break
    else:
        raise ValueError("no episode of the requested kind in this revision")
    options = {}
    if family != "language_preservation" and config.get("counterfactual_weight", 0):
        options["counterfactual_row"] = generator.episode(world, variant=1, family=family, language=args.language)
    reuse = config.get("reuse_frozen_prefix", False)
    model.train()
    def execute():
        model.zero_grad(set_to_none=True)
        with frozen_prefix_reuse(model) if reuse else nullcontext():
            result = objective(row, **options)
            result.loss.backward()
        torch.npu.synchronize()
        return {"loss": float(result.loss.detach()), "answer_ce_diagnostic": result.answer_loss,
                "keep_kl": result.keep_kl, "student_tokens": result.student_tokens,
                "teacher_tokens": result.teacher_tokens}
    execute()  # Warm up; no optimizer, checkpoint or control mutation.
    output.mkdir(parents=True, exist_ok=False)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / "trace")),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1)):
        result = execute()
    model.zero_grad(set_to_none=True)
    report = {"status": "isolated_operator_trace_only", "checkpoint": checkpoint.name,
              "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "device": device,
              "data_identity": data_identity(config["data_revision"]), "base_identity": identity,
              "kind": args.kind, "language": args.language, "world": world,
              "reuse_frozen_prefix": reuse, "optimizer_steps_performed": 0,
              "formal_training_state_modified": False, "profile_overhead_included": True,
              "trained_recall_claimed": False, "result": result,
              "files": [str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file()]}
    atomic_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "result": result, "files": len(report["files"])}), flush=True)


if __name__ == "__main__":
    main()
