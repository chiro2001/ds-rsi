#!/usr/bin/env python3
"""Launch recoverable A/B controls on an explicit local base checkpoint."""

import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    parser.add_argument("--student-tokens", type=int, default=30_000_000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-micro-steps", type=int)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("multi-rank launch needs the coordinated executor; this CLI is explicitly single-rank")
    if args.device.startswith("npu"):
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if visible != "8,9,10,11,12,13,14,15":
            raise RuntimeError("NPU execution requires the explicitly authorized chip8-15 mapping")
        for card in range(4, 8):
            for chip in (0, 1):
                result = subprocess.run(["npu-smi", "info", "-t", "health", "-i", str(card), "-c", str(chip)],
                                        capture_output=True, text=True, check=True)
                if re.search(r"Health Status\s*:\s*OK\b", result.stdout) is None:
                    raise RuntimeError("authorized devices are not verified healthy; refuse to train on known RAS errors")
        import torch_npu  # noqa: F401
        torch.npu.set_device(torch.device(args.device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(8)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import file_sha256
    from dsrsi.training.runner import TrainConfig, Trainer
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(root),
                "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = MemoryQwen.from_local_checkpoint(root, device=args.device)
    compiler = EpisodeCompiler(tokenizer)
    config = TrainConfig(stage=args.stage, student_token_budget=args.student_tokens,
                         gradient_accumulation=args.grad_accum, learning_rate=args.lr,
                         seed=args.seed, distractors=args.distractors)
    trainer = Trainer(model, compiler, args.run_dir, config, base_identity=identity, resume=args.resume)
    status = trainer.run(max_micro_steps=args.max_micro_steps)
    print(json.dumps({"status": status, "progress": trainer.progress.__dict__,
                      "checkpoint": str(trainer.last_checkpoint)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
