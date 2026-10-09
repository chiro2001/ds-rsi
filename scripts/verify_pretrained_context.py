#!/usr/bin/env python3
"""Two-rank BF16 continuous-state TBPTT recovery with actual pretrained weights."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from verify_pretrained_distributed import assert_equal


def canonical(payload):
    payload["life_state"]["harness_refs"]["visible"]["path"] = "<per-rank-journal>"
    case = payload["data_cursor"]["case"]
    if case:
        for frame in case["frames"]:
            frame["harness_after"]["path"] = "<per-rank-journal>"
    return payload


def worker(rank, base, source, root, identity, threads):
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import capture_training_state
    from dsrsi.training.context_runner import ContextTrainer, ContextTrainConfig
    from dsrsi.training.distributed import CollectiveContext
    torch.set_num_threads(threads)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
        init_method="file://" + str(Path(root) / "rendezvous"), timeout_seconds=180)
    started = time.monotonic()
    try:
        config = ContextTrainConfig(student_token_budget=100000, gradient_accumulation=2, unroll_tokens=16,
            window=16, rebase_interval=32, retrieval_interval=8, interference_tokens=0, include_tools=False)
        def make(directory, resume=False):
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            def load():
                return MemoryQwen.from_local_checkpoint(base), EpisodeCompiler(
                    AutoTokenizer.from_pretrained(base, local_files_only=True))
            model, compiler = context.run_local("load continuous pretrained model", load)
            return ContextTrainer(model, compiler, Path(root) / directory, config, base_identity=identity,
                collectives=context, resume=resume, initialize_from=None if resume else source)
        def snapshot(trainer):
            return canonical(capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
                data_cursor=trainer.cursor, life_state=trainer.life, device="cpu", rank=rank, world_size=2))
        full = make("full")
        assert full.run(max_micro_steps=16, install_signals=False) == "stopped"
        reference = snapshot(full)
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
        assert part.run(max_micro_steps=16, install_signals=False) == "paused"
        paused = {"clock": part.stream.cache.clock, "frame_offset": part.cursor["frame_offset"],
                  "pending_writer_tokens": 0 if part.stream.pending is None else part.stream.pending.shape[1],
                  "accumulation_index": part.progress.accumulation_index}
        del part, original, pause
        gc.collect()
        resumed = make("resumed", resume=True)
        assert resumed.run(max_micro_steps=16, install_signals=False) == "stopped"
        context.run_local("compare full continuous training states", lambda: assert_equal(reference, snapshot(resumed)))
        if resumed.progress.optimizer_step == 0:
            raise AssertionError("continuous smoke never exercised a supervised optimizer update")
        result = {"rank": rank, "progress": asdict(resumed.progress), "paused": paused,
                  "exact_full_vs_resumed_state_match": True, "logical_clock": resumed.stream.cache.clock,
                  "teacher_clock": resumed.teacher.cache.clock, "position_origin": resumed.stream.cache.position_origin,
                  "window": config.window, "unroll_tokens": config.unroll_tokens,
                  "student_cache_tensor_bytes": resumed.stream.cache.tensor_bytes,
                  "teacher_cache_tensor_bytes": resumed.teacher.cache.tensor_bytes,
                  "memory_records": len(resumed.stream.memory.records), "memory_tensor_bytes": resumed.stream.memory.tensor_bytes,
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
    mp.spawn(worker, args=(str(base), str(Path(args.init_from).resolve()), str(root), identity, args.threads_per_rank),
             nprocs=2, join=True)
    report = {"status": "pretrained_cpu_continuous_training_recovery_only", "world_size": 2, "base_identity": identity,
              "ranks": [json.loads((root / f"result_rank_{rank}.json").read_text()) for rank in range(2)],
              "trained_long_context_capability_claimed": False, "hccl_verified": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
