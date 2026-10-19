#!/usr/bin/env python3
"""Profile copied A/B weights on CUDA; never migrate or resume a formal run.

Uses the source optimizer in a disposable, single-rank probe. The data cursor
defaults to the source; an explicit ordinal may select a comparison workload.
CUDA RNG is explicitly initialized, not converted from NPU RNG. Results are
diagnostics, not evidence of equivalence to the source distributed trajectory.
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--start-ordinal", type=int,
                        help="diagnostic-only workload override; never updates the source cursor")
    parser.add_argument("--warmup-episodes", type=int, default=0,
                        help="warm backend caches, then restore the complete initial probe state before timing")
    parser.add_argument("--mode", choices=["fallback", "scripted", "fla"], default="fallback")
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    if args.episodes < 4:
        parser.error("at least four episodes required")
    if args.start_ordinal is not None and args.start_ordinal < 0:
        parser.error("start ordinal must be nonnegative")
    if args.warmup_episodes < 0:
        parser.error("warmup episodes must be nonnegative")
    initialization_started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoTokenizer, __version__ as transformers_version
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import (CheckpointStore, capture_training_state,
                                         file_sha256, restore_training_state)
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.training.topology import require_equal

    store = CheckpointStore(Path(args.source_run) / "checkpoints")
    committed = directory, manifest = store.load_committed()
    source_sha = file_sha256(directory / "manifest.json")
    source_latest = (store.root / "latest.json").read_bytes()
    if manifest["status"] not in {"stopped", "paused"} or manifest["progress"]["accumulation_index"]:
        raise ValueError("probe requires a stopped source at an empty accumulation boundary")
    config = TrainConfig(**manifest["identity"]["train_config"])
    if config.stage not in {"A", "B"} or config.gdn_inverse is not None:
        raise ValueError("probe expects baseline A/B execution")
    base = Path(args.base)
    started = time.perf_counter()
    names = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"].values())
    identity = {"tokenizer": tokenizer_fingerprint(base),
                "weights": {name: file_sha256(base / name) for name in sorted(names)}}
    base_hash_seconds = time.perf_counter() - started
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if args.mode != "fla":
        # Match the NPU PyTorch fallback, including gated normalization/conv.
        hf.FusedRMSNormGated = None
        hf.chunk_gated_delta_rule = None
        hf.fused_recurrent_gated_delta_rule = None
        hf.causal_conv1d_fn = None
        hf.causal_conv1d_update = None
    started = time.perf_counter()
    model = MemoryQwen.from_local_checkpoint(base, device="cuda:0",
        memory_architecture=manifest["identity"]["memory_architecture"])
    torch.cuda.synchronize()
    base_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    load_ab_weights(model, store, base_identity=identity, committed=committed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    probe_config = replace(config, gradient_accumulation=config.gradient_accumulation * manifest["world_size"])
    trainer = Trainer(model, compiler, output / "disposable", probe_config, base_identity=identity)
    source = store.load_rank(directory, manifest, rank=0)
    trainer.optimizer.load_state_dict(copy.deepcopy(source["optimizer"]["state"]))
    trainer.progress = replace(trainer.progress, **manifest["progress"])
    trainer.cursor = copy.deepcopy(source["data_cursor"])
    if args.start_ordinal is not None:
        trainer.cursor["next_ordinal"] = args.start_ordinal
    probe_start_ordinal = trainer.cursor["next_ordinal"]
    source_state_and_trainer_load_seconds = time.perf_counter() - started
    model.train()
    from contextlib import nullcontext
    from dsrsi.model.experimental_gdn import experimental_gdn_inverse

    def snapshot():
        return capture_training_state(model, trainer.optimizer, progress=trainer.progress,
            data_cursor=trainer.cursor, life_state=trainer.life, device="cuda:0")

    def scope():
        return experimental_gdn_inverse(model, "scripted_rows") if args.mode == "scripted" else nullcontext()

    rows = []
    def run(count, record):
        for _ in range(count):
            ordinal = trainer.cursor["next_ordinal"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with scope():
                row, _, result = trainer._forward_episode(ordinal)
                (result.loss / probe_config.gradient_accumulation).backward()
            trainer.progress = replace(trainer.progress,
                micro_step=trainer.progress.micro_step + 1,
                accumulation_index=trainer.progress.accumulation_index + 1,
                student_tokens=trainer.progress.student_tokens + result.student_tokens,
                teacher_tokens=trainer.progress.teacher_tokens + result.teacher_tokens)
            trainer.cursor["next_ordinal"] += 1
            if trainer.progress.accumulation_index == probe_config.gradient_accumulation:
                trainer._optimizer_step()
            torch.cuda.synchronize()
            info = {"ordinal": ordinal, "seconds": time.perf_counter() - started,
                "student_tokens": result.student_tokens, "teacher_tokens": result.teacher_tokens,
                "loss": float(result.loss.detach()), "family": row["family"],
                "annotations": row.get("annotations", {}), "queries": result.queries}
            if not np.isfinite(info["loss"]):
                raise ValueError("non-finite diagnostic loss")
            if record:
                rows.append(info)
                print(json.dumps(info), flush=True)
            del result

    initial = snapshot()
    initialization_seconds = time.perf_counter() - initialization_started
    warmup = {"episodes": args.warmup_episodes, "student_tokens": 0,
              "teacher_tokens": 0, "seconds": 0.0}
    if args.warmup_episodes:
        started = time.perf_counter()
        run(args.warmup_episodes, False)
        warmup.update(seconds=time.perf_counter() - started,
            student_tokens=trainer.progress.student_tokens - initial["progress"]["student_tokens"],
            teacher_tokens=trainer.progress.teacher_tokens - initial["progress"]["teacher_tokens"])
        trainer.progress, trainer.cursor, trainer.life = restore_training_state(
            initial, model, trainer.optimizer, device="cuda:0")
    torch.cuda.reset_peak_memory_stats()
    run(args.episodes, True)
    final = snapshot()
    torch.save(final, output / "final-state.pt")
    recovery = None
    if args.recovery:
        trainer.progress, trainer.cursor, trainer.life = restore_training_state(
            initial, model, trainer.optimizer, device="cuda:0")
        run(3, False)
        partial = snapshot()
        assert partial["progress"]["accumulation_index"] == 3
        proof = CheckpointStore(output / "recovery-checkpoints")
        pending = proof.begin(trainer.progress)
        started = time.perf_counter()
        proof.save_rank(pending, partial)
        proof.commit(pending, world_size=1, identity=trainer.identity, progress=trainer.progress, status="stopped")
        save_seconds = time.perf_counter() - started
        started = time.perf_counter()
        restored, _ = proof.load_latest(identity=trainer.identity, rank=0, world_size=1)
        trainer.progress, trainer.cursor, trainer.life = restore_training_state(
            restored, model, trainer.optimizer, device="cuda:0")
        load_seconds = time.perf_counter() - started
        run(args.episodes - 3, False)
        require_equal(final, snapshot(), "cuda-probe-complete-state-recovery")
        recovery = {"complete_payload_exact_match": True, "pending_gradient_episodes": 3,
                    "save_seconds": save_seconds, "load_seconds": load_seconds}
    # The source is read-only. Check every file hash again, not only latest.
    store.load_committed()
    assert (store.root / "latest.json").read_bytes() == source_latest
    assert file_sha256(directory / "manifest.json") == source_sha
    measured = rows[2:]
    seconds = sum(r["seconds"] for r in measured)
    tokens = sum(r["student_tokens"] for r in measured)
    report = {"status": "cuda_training_copy_diagnostic_only", "mode": args.mode,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": transformers_version, "device": torch.cuda.get_device_name(),
        "source_checkpoint": directory.name, "source_manifest_sha256": source_sha,
        "source_progress": manifest["progress"], "source_world_size": manifest["world_size"],
        "probe_start_ordinal": probe_start_ordinal,
        "source_cursor_overridden_for_workload_comparison": args.start_ordinal is not None,
        "base_path": str(base.resolve()), "output_path": str(output),
        "base_hash_seconds": base_hash_seconds, "base_load_seconds": base_load_seconds,
        "source_state_and_trainer_load_seconds": source_state_and_trainer_load_seconds,
        "initialization_seconds": initialization_seconds,
        "warmup": warmup,
        "probe_config": asdict(probe_config), "episodes": rows,
        "timing_excludes_first_episodes": 2, "measured_student_tokens": tokens,
        "measured_seconds": seconds, "student_tokens_per_second": tokens / seconds,
        "episode_mean_seconds": seconds / len(measured),
        "episode_median_seconds": statistics.median(r["seconds"] for r in measured),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_progress": final["progress"], "recovery": recovery,
        "source_run_modified": False, "formal_training_ledger_modified": False,
        "source_optimizer_state_loaded": True, "accelerator_rng_explicitly_reinitialized": True,
        "source_distributed_trajectory_equivalence_claimed": False,
        "probe_query_normalization_is_single_rank": True,
        "production_migration_verified": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
