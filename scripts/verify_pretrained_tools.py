#!/usr/bin/env python3
"""Actual pretrained C-tools/LoRA two-rank pause/recovery; no ability claim."""

from dataclasses import asdict
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from verify_pretrained_distributed import assert_equal


def canonical_state(payload):
    # Journal location varies across experiment directories; its database ID,
    # visible revision and life ID must match. The journals themselves are
    # restored and verified by ToolTrainer.
    for reference in payload["life_state"]["harness_refs"].values():
        reference["path"] = "<per-rank-journal>"
    return payload


def worker(rank, base, source, root, threads, identity):
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import capture_training_state
    from dsrsi.training.distributed import CollectiveContext
    from dsrsi.training.tool_runner import ToolTrainer, ToolTrainConfig
    torch.set_num_threads(threads)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
                                           init_method="file://" + str(Path(root) / "rendezvous"), timeout_seconds=180)
    started = time.monotonic()
    try:
        config = ToolTrainConfig(student_token_budget=100000, replay_distractors=2, save_every_micro=100)
        def make(directory, resume=False):
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            def load():
                return MemoryQwen.from_local_checkpoint(base), EpisodeCompiler(
                    AutoTokenizer.from_pretrained(base, local_files_only=True), max_query_tokens=config.max_query_tokens)
            model, compiler = context.run_local("load actual C-tools model", load)
            return ToolTrainer(model, compiler, Path(root) / directory, config, base_identity=identity,
                               collectives=context, resume=resume, initialize_from=None if resume else source)
        def snapshot(trainer):
            return canonical_state(capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
                data_cursor=trainer.cursor, life_state=trainer.life, device="cpu", rank=rank, world_size=2))
        full = make("full")
        assert full.run(max_micro_steps=5, install_signals=False) == "stopped"
        reference = snapshot(full)
        lora_parameters = sum(p.numel() for name, p in full.model.named_parameters() if ".lora_" in name)
        del full
        gc.collect()
        part = make("resumed")
        original = part._forward_episode
        calls = 0
        def pause(ordinal):
            nonlocal calls
            result = original(ordinal)
            calls += 1
            if rank == 1 and calls == 3:
                part.control._signal_action = "pause"
            return result
        part._forward_episode = pause
        assert part.run(max_micro_steps=5, install_signals=False) == "paused"
        assert part.progress.accumulation_index == 3
        del part, original, pause
        gc.collect()
        resumed = make("resumed", resume=True)
        assert resumed.run(max_micro_steps=5, install_signals=False) == "stopped"
        context.run_local("compare complete C-tools states", lambda: assert_equal(reference, snapshot(resumed)))
        events = [json.loads(line) for line in (resumed.run_dir / f"events_rank_{rank:05d}.jsonl").read_text().splitlines()]
        micro = [event for event in events if event["event"] == "micro"]
        result = {"rank": rank, "progress": asdict(resumed.progress), "cursor": resumed.cursor,
                  "exact_full_vs_resumed_training_state_match": True, "worker_pause_micro_step": 3,
                  "lora_parameters": lora_parameters,
                  "tool_events": sum(event["tool_events"] for event in micro),
                  "reply_events": sum(event["reply_events"] for event in micro),
                  "families": [event["family"] for event in micro],
                  "elapsed_seconds": time.monotonic() - started}
        (Path(root) / f"result_rank_{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads-per-rank", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from dsrsi.data.compiler import tokenizer_fingerprint
    from dsrsi.training.checkpoint import file_sha256
    base = Path(args.base).resolve()
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(set(index.values()))}}
    mp.spawn(worker, args=(str(base), str(Path(args.init_from).resolve()), str(root), args.threads_per_rank, identity),
             nprocs=2, join=True)
    report = {"status": "pretrained_cpu_c_tools_recovery_only", "world_size": 2, "base_identity": identity,
              "ranks": [json.loads((root / f"result_rank_{rank}.json").read_text()) for rank in range(2)],
              "trained_tool_capability_claimed": False, "trained_continuous_context_claimed": False, "hccl_verified": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
