#!/usr/bin/env python3
"""Launch recoverable A/B controls on an explicit local base checkpoint."""

import argparse
import copy
import json
from pathlib import Path
import random

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
    parser.add_argument("--data-revision", choices=["ab-v1", "ab-v2-matched-entities", "ab-v3-random-entity-order", "ab-v4-fields-and-language", "ab-v5-indexed-digits", "ab-v6-natural-keep", "ab-v7-fixed-pair-qa"],
                        default="ab-v4-fields-and-language")
    parser.add_argument("--curriculum", choices=["mixed", "facts", "locations"], default="mixed")
    parser.add_argument("--save-every-micro", type=int, default=100)
    parser.add_argument("--memory-candidates", type=int, default=8)
    parser.add_argument("--memory-candidate-curriculum", choices=["single-double-world-v1"],
                        help="explicit new A course: paired-world single/double fact rehearsal; keep remains double")
    parser.add_argument("--memory-trainable-policy", choices=["freeze-reader-gates-v1"],
                        help="explicit new rehearsal course: freeze Reader gate and alpha parameters")
    parser.add_argument("--reader-support-weight", type=float, default=0.0,
                        help="explicit new A objective: support-record attention supervision")
    parser.add_argument("--evidence-distillation-weight", type=float, default=0.0,
                        help="explicit v7 A objective: free-greedy-screened plaintext evidence teacher")
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=2.0)
    parser.add_argument("--keep-fraction", type=float)
    parser.add_argument("--keep-weight", type=float, default=0.2)
    parser.add_argument("--reuse-frozen-prefix", action="store_true")
    parser.add_argument("--gdn-inverse", choices=["scripted_rows"],
                        help="verified original-order rows; existing runs require explicit execution migration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", help="explicit A-to-B or versioned A-to-A weights initialization; resets training state")
    parser.add_argument("--init-checkpoint", help="pin an older committed source generation without changing its latest pointer")
    parser.add_argument("--writer-anchor-weight", type=float, help="explicit new-run position pooling change in [0,1]")
    parser.add_argument("--reader-gate-mode", choices=["token", "lookup"], help="explicit new-run reader decision scope")
    parser.add_argument("--max-micro-steps", type=int)
    parser.add_argument("--cpu-threads", type=int, help="CPU threads per rank; defaults to max(1, 8/world_size)")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be combined")
    if args.init_checkpoint and (args.resume or not args.init_from):
        parser.error("--init-checkpoint requires --init-from and cannot be used with --resume")
    from dsrsi.training.device import resolve_device
    device, world_size = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        threads = max(1, 8 // world_size) if args.cpu_threads is None else args.cpu_threads
        if threads < 1:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(threads)
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import file_sha256
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.distributed import CollectiveContext
    context = (CollectiveContext.initialize(device=device, backend="hccl" if device.startswith("npu") else "gloo")
               if world_size > 1 else CollectiveContext(device=device, enabled=False))
    try:
        def load():
            root = Path(args.base)
            index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
            identity = {"tokenizer": tokenizer_fingerprint(root),
                        "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
            tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
            from dsrsi.evaluation.checkpoint import memory_architecture_for_run
            from dsrsi.memory.modules import MemoryConfig
            architecture = memory_architecture_for_run(args.run_dir if args.resume else args.init_from,
                                                       checkpoint=args.init_checkpoint)
            anchor_change, gate_change = False, False
            if architecture is None:
                model = MemoryQwen.from_local_checkpoint(root, device=device,
                    memory_config=MemoryConfig(writer_anchor_weight=args.writer_anchor_weight or 0,
                                               reader_gate_mode=args.reader_gate_mode or "token"))
            else:
                architecture = copy.deepcopy(architecture)
                if args.writer_anchor_weight is not None:
                    MemoryConfig(writer_anchor_weight=args.writer_anchor_weight)
                    previous = architecture["memory_config"].get("writer_anchor_weight", 0.0)
                    if args.resume and previous != args.writer_anchor_weight:
                        raise ValueError("resume cannot change writer anchors")
                    if args.writer_anchor_weight:
                        architecture["memory_config"]["writer_anchor_weight"] = args.writer_anchor_weight
                        architecture["writer_anchor_version"] = "position_pool_v1"
                    else:
                        architecture["memory_config"].pop("writer_anchor_weight", None)
                        architecture.pop("writer_anchor_version", None)
                    anchor_change = args.init_from is not None and previous != args.writer_anchor_weight
                if args.reader_gate_mode is not None:
                    previous = architecture["memory_config"].get("reader_gate_mode", "token")
                    if args.resume and previous != args.reader_gate_mode:
                        raise ValueError("resume cannot change reader gates")
                    if args.reader_gate_mode == "lookup":
                        architecture["memory_config"]["reader_gate_mode"] = "lookup"
                        architecture["reader_gate_version"] = "lookup_v1"
                    else:
                        architecture["memory_config"].pop("reader_gate_mode", None)
                        architecture.pop("reader_gate_version", None)
                    gate_change = args.init_from is not None and previous != args.reader_gate_mode
                model = MemoryQwen.from_local_checkpoint(root, device=device, memory_architecture=architecture)
            compiler = EpisodeCompiler(tokenizer)
            config = TrainConfig(stage=args.stage, student_token_budget=args.student_tokens,
                                 gradient_accumulation=args.grad_accum, learning_rate=args.lr,
                                 seed=args.seed, distractors=args.distractors, data_revision=args.data_revision,
                                 curriculum=args.curriculum, save_every_micro=args.save_every_micro,
                                 memory_candidates=args.memory_candidates, counterfactual_weight=args.counterfactual_weight,
                                 memory_candidate_curriculum=args.memory_candidate_curriculum,
                                 memory_trainable_policy=args.memory_trainable_policy,
                                 reader_support_weight=args.reader_support_weight,
                                 evidence_distillation_weight=args.evidence_distillation_weight,
                                 reuse_frozen_prefix=args.reuse_frozen_prefix,
                                 gdn_inverse=args.gdn_inverse,
                                 counterfactual_margin=args.counterfactual_margin, keep_fraction=args.keep_fraction,
                                 keep_weight=args.keep_weight)
            return identity, model, compiler, config, anchor_change, gate_change
        identity, model, compiler, config, anchor_change, gate_change = context.run_local("load training artifacts", load)
        trainer = Trainer(model, compiler, args.run_dir, config, base_identity=identity, resume=args.resume,
            collectives=context, initialize_from=args.init_from, initialize_checkpoint=args.init_checkpoint,
            allow_writer_anchor_change=anchor_change,
            allow_reader_gate_change=gate_change)
        status = trainer.run(max_micro_steps=args.max_micro_steps)
        print(json.dumps({"rank": context.rank, "world_size": context.world_size,
                          "status": status, "progress": trainer.progress.__dict__,
                          "checkpoint": str(trainer.last_checkpoint)}, ensure_ascii=False))
    finally:
        context.close()


if __name__ == "__main__":
    main()
