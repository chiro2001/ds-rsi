#!/usr/bin/env python3
"""Real CUDA CLI processes: continuous vs external pause and complete resume."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def micro_events(run):
    path = run / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live writer may not have finished its current line.
        if row.get("event") == "micro":
            rows.append(row)
    return rows


def pinned_manifest(run):
    store = run / "checkpoints"
    latest = json.loads((store / "latest.json").read_text())
    directory = store / latest["directory"]
    return directory, json.loads((directory / "manifest.json").read_text())


def train_arguments(base, run, config, maximum):
    options = {"stage": "stage", "student_token_budget": "student-tokens",
               "gradient_accumulation": "grad-accum", "learning_rate": "lr", "seed": "seed",
               "distractors": "distractors", "data_revision": "data-revision", "curriculum": "curriculum",
               "save_every_micro": "save-every-micro", "memory_candidates": "memory-candidates",
               "counterfactual_weight": "counterfactual-weight", "counterfactual_margin": "counterfactual-margin",
               "keep_fraction": "keep-fraction", "keep_weight": "keep-weight"}
    options["memory_candidate_curriculum"] = "memory-candidate-curriculum"
    if config.get("weight_decay", 0.01) != 0.01 or config.get("gradient_clip", 1.0) != 1.0:
        raise ValueError("CLI recovery proof requires the supported optimizer defaults")
    result = ["scripts/train_ab.py", "--base", str(base), "--run-dir", str(run),
              "--resume", "--device", "cuda", "--cpu-threads", "2", "--max-micro-steps", str(maximum)]
    for name, flag in options.items():
        if config.get(name) is not None:
            result += ["--" + flag, str(config[name])]
    if config.get("reuse_frozen_prefix"):
        result += ["--reuse-frozen-prefix"]
    if config.get("gdn_inverse"):
        result += ["--gdn-inverse", config["gdn_inverse"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--new-episodes", type=int, default=188)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_run).resolve()
    _, manifest = pinned_manifest(source)
    initial = manifest["progress"]["micro_step"]
    if manifest["world_size"] != 1 or manifest["progress"]["accumulation_index"]:
        raise ValueError("proof requires an empty single-CUDA source boundary")
    maximum = initial + args.new_episodes
    for name in ("continuous", "interrupted"):
        shutil.copytree(source / "checkpoints", root / name / "checkpoints")
    config = manifest["identity"]["train_config"]
    from dsrsi.training.control import ControlChannel

    def launch(name, *, pause=False, after_micro=None):
        run = root / name
        command = [args.python, str(code / "scripts/run_5080_checked.py"), "--code-root", str(code),
                   "--python", args.python, "--label", f"cuda-recovery-{name}", "--"]
        command += train_arguments(Path(args.base), run, config, maximum)
        log = (root / f"{name}-{'resume' if after_micro else 'initial'}.log").open("w")
        started = time.monotonic()
        child = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=os.environ.copy())
        requested = first_micro = None
        try:
            while child.poll() is None:
                rows = micro_events(run)
                if after_micro is not None and first_micro is None and any(
                        row["progress"]["micro_step"] > after_micro for row in rows):
                    first_micro = time.monotonic() - started
                if pause and requested is None and rows and rows[-1]["progress"]["micro_step"] >= initial + 3:
                    requested = time.monotonic()
                    ControlChannel(run).request("pause")
                if time.monotonic() - started > 1800:
                    # Ask for a recoverable exit, never kill a live backward.
                    ControlChannel(run).request("pause")
                    raise TimeoutError("proof exceeded 30 minutes; pause requested, inspect the existing worker")
                time.sleep(0.05)
            if child.returncode:
                raise RuntimeError(f"CUDA child failed with {child.returncode}; inspect {log.name}")
            return {"elapsed_seconds": time.monotonic() - started,
                    "pause_to_exit_seconds": None if requested is None else time.monotonic() - requested,
                    "resume_to_first_micro_seconds": first_micro}
        finally:
            log.close()

    continuous = launch("continuous")
    interrupted = launch("interrupted", pause=True)
    pause_directory, paused = pinned_manifest(root / "interrupted")
    if paused["status"] != "paused" or not paused["progress"]["accumulation_index"]:
        raise AssertionError("external pause did not preserve pending accumulated gradients")
    resumed = launch("interrupted", after_micro=paused["progress"]["micro_step"])
    # Heavy imports and full-state comparison happen after both GPU jobs exit.
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.topology import require_equal
    states = []
    for name in ("continuous", "interrupted"):
        store = CheckpointStore(root / name / "checkpoints")
        directory, final_manifest = store.load_committed()
        states.append(store.load_rank(directory, final_manifest, rank=0))
        assert final_manifest["status"] == "stopped" and final_manifest["progress"]["micro_step"] == maximum
    require_equal(states[0], states[1], "separate-process-complete-CUDA-recovery")
    a, b = micro_events(root / "continuous"), micro_events(root / "interrupted")
    assert len(a) == len(b) == args.new_episodes
    for x, y in zip(a, b):
        for field in ("episode_id", "student_input_tokens", "teacher_input_tokens", "progress", "loss"):
            assert x[field] == y[field], field
    coverage = {"natural_keep": sum(row.get("natural_keep", False) for row in a),
                "unknown_wall": sum(row.get("natural_keep", False) and not row.get("keep_wall_time_known") for row in a),
                "known_wall": sum(row.get("natural_keep", False) and bool(row.get("keep_wall_time_known")) for row in a),
                "auxiliary": sum(row.get("codec_task") == "indexed_character" for row in a)}
    if coverage["natural_keep"] < 14 or coverage["unknown_wall"] < 6 or coverage["auxiliary"] < 8:
        raise AssertionError(f"insufficient recovery episode coverage: {coverage}")
    source_store = CheckpointStore(source / "checkpoints")
    _, source_manifest = source_store.load_committed()
    assert source_manifest == manifest
    report = {"status": "separate_process_CUDA_full_state_external_pause_recovery_verified",
        "source_run": str(source), "source_progress": manifest["progress"], "new_episodes": args.new_episodes,
        "continuous": continuous, "interrupted": interrupted, "resumed": resumed,
        "paused_progress": paused["progress"], "paused_manifest_sha256": file_sha256(pause_directory / "manifest.json"),
        "final_progress": states[0]["progress"], "final_cursor": states[0]["data_cursor"],
        "complete_payload_exact_match": True, "sample_ids_tokens_losses_exact_match": True,
        "coverage": coverage, "formal_source_modified": False,
        "student_tokens_per_trajectory": states[0]["progress"]["student_tokens"] - manifest["progress"]["student_tokens"],
        "teacher_tokens_per_trajectory": states[0]["progress"]["teacher_tokens"] - manifest["progress"]["teacher_tokens"]}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
