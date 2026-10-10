#!/usr/bin/env python3
"""Train C-tools adapters, keeping full recoverable state and memory replay."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--init-from")
    parser.add_argument("--allow-untrained-memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--student-tokens", type=int, default=1_000_000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--memory-replay-fraction", type=float, default=0.25)
    parser.add_argument("--replay-distractors", type=int, default=15)
    parser.add_argument("--max-query-tokens", type=int, default=2048)
    parser.add_argument("--max-micro-steps", type=int)
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be combined")
    from dsrsi.training.device import resolve_device
    from dsrsi.training.distributed import CollectiveContext
    device, world_size = resolve_device(args.device)
    if device == "cpu":
        threads = max(1, 8 // world_size) if args.cpu_threads is None else args.cpu_threads
        if threads < 1:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    context = (CollectiveContext.initialize(device=device, backend="hccl" if device.startswith("npu") else "gloo")
               if world_size > 1 else CollectiveContext(device=device, enabled=False))
    try:
        def load():
            from transformers import AutoTokenizer
            from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
            from dsrsi.model.qwen35_memory import MemoryQwen
            from dsrsi.training.checkpoint import file_sha256
            from dsrsi.training.tool_runner import ToolTrainConfig
            from dsrsi.evaluation.checkpoint import memory_architecture_for_run
            root = Path(args.base)
            index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
            identity = {"tokenizer": tokenizer_fingerprint(root),
                        "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
            config = ToolTrainConfig(student_token_budget=args.student_tokens, gradient_accumulation=args.grad_accum,
                                     learning_rate=args.lr, seed=args.seed, lora_rank=args.lora_rank,
                                     lora_alpha=args.lora_alpha, max_query_tokens=args.max_query_tokens,
                                     allow_untrained_memory=args.allow_untrained_memory,
                                     memory_replay_fraction=args.memory_replay_fraction, replay_distractors=args.replay_distractors)
            architecture = memory_architecture_for_run(args.run_dir if args.resume else args.init_from)
            model = MemoryQwen.from_local_checkpoint(root, device=device, memory_architecture=architecture)
            compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True),
                                       max_query_tokens=config.max_query_tokens)
            return identity, model, compiler, config
        identity, model, compiler, config = context.run_local("load C-tools artifacts", load)
        from dsrsi.training.tool_runner import ToolTrainer
        trainer = ToolTrainer(model, compiler, args.run_dir, config, base_identity=identity,
                              initialize_from=args.init_from, resume=args.resume, collectives=context)
        status = trainer.run(max_micro_steps=args.max_micro_steps)
        print(json.dumps({"status": status, "rank": context.rank, "world_size": context.world_size,
                          "progress": trainer.progress.__dict__, "checkpoint": str(trainer.last_checkpoint)}, ensure_ascii=False))
    finally:
        context.close()


if __name__ == "__main__":
    main()
