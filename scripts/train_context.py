#!/usr/bin/env python3
"""Train and resume short unrolls of a continuously evolving Qwen life."""

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
    parser.add_argument("--unroll-tokens", type=int, default=32)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--rebase-interval", type=int, default=4096)
    parser.add_argument("--writer-stride", type=int, default=256)
    parser.add_argument("--retrieval-interval", type=int, default=64)
    parser.add_argument("--memory-tensor-budget", type=int, default=64 << 20)
    parser.add_argument("--interference-tokens", type=int, default=2048)
    parser.add_argument("--no-tools", action="store_true")
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
            from dsrsi.training.context_runner import ContextTrainConfig
            from dsrsi.evaluation.checkpoint import memory_architecture_for_run
            root = Path(args.base)
            index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
            identity = {"tokenizer": tokenizer_fingerprint(root),
                        "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
            config = ContextTrainConfig(student_token_budget=args.student_tokens, gradient_accumulation=args.grad_accum,
                learning_rate=args.lr, seed=args.seed, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                unroll_tokens=args.unroll_tokens, window=args.window, rebase_interval=args.rebase_interval,
                writer_stride=args.writer_stride, retrieval_interval=args.retrieval_interval,
                memory_tensor_budget=args.memory_tensor_budget, interference_tokens=args.interference_tokens,
                include_tools=not args.no_tools, allow_untrained_memory=args.allow_untrained_memory)
            architecture = memory_architecture_for_run(args.run_dir if args.resume else args.init_from)
            return identity, MemoryQwen.from_local_checkpoint(root, device=device, memory_architecture=architecture), EpisodeCompiler(
                AutoTokenizer.from_pretrained(root, local_files_only=True)), config
        identity, model, compiler, config = context.run_local("load continuous artifacts", load)
        from dsrsi.training.context_runner import ContextTrainer
        trainer = ContextTrainer(model, compiler, args.run_dir, config, base_identity=identity, collectives=context,
                                 resume=args.resume, initialize_from=args.init_from)
        status = trainer.run(max_micro_steps=args.max_micro_steps)
        print(json.dumps({"status": status, "rank": context.rank, "world_size": context.world_size,
                          "progress": trainer.progress.__dict__, "clock": trainer.stream.cache.clock,
                          "checkpoint": str(trainer.last_checkpoint)}, ensure_ascii=False))
    finally:
        context.close()


if __name__ == "__main__":
    main()
