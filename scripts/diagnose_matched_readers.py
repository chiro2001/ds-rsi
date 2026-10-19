#!/usr/bin/env python3
"""Matched-record teacher-forced Reader traces; never a capability evaluation."""

import argparse
import json
from pathlib import Path


def fixed_candidate_ids(row, reference):
    probe = row["probes"][0]
    groups = probe["support_groups"]
    selected = reference["selected_records"]
    if (reference["episode_id"] != row["episode_id"] or reference["expected"] != probe["target"]
            or len(groups) != 1 or len(groups[0]) != 1 or len(selected) != 2
            or len(set(selected)) != 2 or selected[0] not in groups[0]
            or not set(selected) <= {item["event_id"] for item in row["source_events"]}):
        raise ValueError("reference must select the real support first and one distinct in-pool negative")
    return list(selected)


def record_attention_summary(trace, *, slots_per_record, candidate_count):
    result = {}
    for reader, values in trace.items():
        masses = []
        for row in values["attention_mass_per_slot"]:
            if len(row) != slots_per_record * candidate_count:
                raise ValueError("trace slot count differs from the fixed candidate layout")
            grouped = [sum(row[i:i + slots_per_record])
                       for i in range(0, len(row), slots_per_record)]
            total = sum(grouped)
            masses.append({"per_record": grouped,
                           "positive_fraction": grouped[0] / total if total else None})
        result[reader] = {"alpha": values["alpha"], "gate": values["gate"],
                          "relative_residual_rms": values["relative_residual_rms"],
                          "attention_by_prediction_position": masses}
    return result


def ablation_parameter_names(policy, names):
    rules = {
        "old_writer": lambda n: n.startswith("writer."),
        "old_readers": lambda n: n.startswith("readers."),
        "old_gates": lambda n: n.startswith("readers.") and (n.endswith(".alpha") or ".gate." in n),
        "old_reader_values": lambda n: n.startswith("readers.") and n.endswith((".v.weight", ".o.weight")),
        "old_reader_qk": lambda n: n.startswith("readers.") and n.endswith((".q.weight", ".k.weight")),
        "old_time_source": lambda n: n.startswith("time_source."),
    }
    if policy not in rules:
        raise ValueError("unknown diagnostic parameter ablation")
    selected = [name for name in names if rules[policy](name)]
    if not selected:
        raise ValueError("parameter ablation selected no tensors")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal")
    parser.add_argument("--worlds", nargs="+", type=int, default=[0, 8, 11])
    parser.add_argument("--languages", nargs="+", choices=["zh", "en", "cross_language"],
                        default=["zh", "en", "cross_language"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--candidate-counts", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--parameter-ablation", action="store_true",
                        help="restore selected initial parameters on top of trained weights, in RAM only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-items", type=int)
    args = parser.parse_args()
    if len(set(args.worlds)) != len(args.worlds) or any(w < 0 for w in args.worlds):
        parser.error("worlds must be unique and nonnegative")
    if len(set(args.languages)) != len(args.languages):
        parser.error("languages must be unique")
    if len(set(args.candidate_counts)) != len(args.candidate_counts):
        parser.error("candidate counts must be unique")
    import torch
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import data_identity, make_generator
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.codec import native_logits, canonical_logits, summarize_logits
    from dsrsi.evaluation.journal import EvaluationJournal
    from dsrsi.evaluation.m_only import MOnlyEvaluator
    from dsrsi.evaluation.pinned import load_pinned
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.utils.atomic import atomic_json

    device, world_size = resolve_device(args.device)
    if world_size != 1 or args.threads < 1:
        raise ValueError("diagnostic requires one device and positive CPU thread count")
    torch.set_num_threads(args.threads)
    base, run = Path(args.base), Path(args.run_dir)
    output, journal_path = Path(args.output), Path(args.journal or (args.output + ".journal.json"))
    if output.exists() and not args.resume:
        raise FileExistsError("diagnostic output exists; explicit resume is required")
    if device.startswith("cuda"):
        root = Path("/mnt/e/ds-rsi").resolve()
        for path in [base, run, output, journal_path, Path(args.reference_evaluation)]:
            if not path.resolve().is_relative_to(root):
                raise ValueError("CUDA diagnostic files must remain on Windows E:\\ds-rsi")
    store = CheckpointStore(run / "checkpoints")
    snapshots = [("initial", load_pinned(store, args.baseline_checkpoint)),
                 ("trained", load_pinned(store, args.checkpoint))]
    conditions = [(label, committed, None) for label, committed in snapshots]
    baseline_weights = None
    overrides = {}
    if args.parameter_ablation:
        baseline_weights = store.load_rank(*snapshots[0][1], rank=0)["weights"]
        for policy in ("old_writer", "old_readers", "old_gates", "old_reader_values",
                       "old_reader_qk", "old_time_source"):
            overrides["trained_" + policy] = ablation_parameter_names(policy, baseline_weights)
            conditions.append(("trained_" + policy, snapshots[1][1], overrides["trained_" + policy]))
    first = snapshots[0][1][1]
    config, architecture = first["identity"]["train_config"], first["identity"]["memory_architecture"]
    if config["data_revision"] not in {"ab-v5-indexed-digits", "ab-v6-natural-keep"}:
        raise ValueError("matched Reader diagnostic requires the v5/v6 full-code distribution")
    for _, (_, manifest) in snapshots:
        if (manifest["identity"]["memory_architecture"] != architecture
                or manifest["identity"]["data"] != data_identity(config["data_revision"])
                or manifest["identity"]["train_config"] != config):
            raise ValueError("matched snapshots must share architecture, data and training configuration")
    weights = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    base_identity = {"tokenizer": tokenizer_fingerprint(base),
                     "weights": {name: file_sha256(base / name) for name in sorted(set(weights.values()))}}
    reference = json.loads(Path(args.reference_evaluation).read_text())
    if reference["memory_candidates"] != 2 or reference["distractors"] != 8:
        raise ValueError("reference must use oracle2 in the original nine-record pool")
    references = {(row["episode_id"], row["language"]): row for row in reference["records"]
                  if row["mode"] == "oracle" and row["family"] == "single_fact"}
    generator = make_generator(revision=config["data_revision"], seed=config["seed"],
                               split="dev", distractors=8, probe_view="full")
    cases = []
    for world in args.worlds:
        for language in args.languages:
            for variant in (0, 1):
                row = generator.episode(world, variant=variant, family="single_fact", language=language)
                selected = fixed_candidate_ids(row, references[(row["episode_id"], language)])
                cases.append((world, language, variant, row, selected))
    source_root = Path(__file__).resolve().parents[1]
    runtime = {"torch": str(torch.__version__)}
    if device.startswith("cuda"):
        from dsrsi.model.cuda_execution import cuda_execution_identity
        runtime = cuda_execution_identity(device)
        if any(manifest["identity"].get("cuda_execution") != runtime
               for _, (_, manifest) in snapshots):
            raise ValueError("diagnostic CUDA runtime differs from the pinned snapshots")
    identity = {"snapshots": {label: {"checkpoint": directory.name,
                 "manifest_sha256": file_sha256(directory / "manifest.json")}
                 for label, (directory, _) in snapshots}, "base": base_identity,
                "runtime": runtime, "device": device, "data": data_identity(config["data_revision"]),
                "reference_sha256": file_sha256(Path(args.reference_evaluation)),
                "cases": [{"world": w, "language": l, "variant": v, "selected": ids}
                          for w, l, v, _, ids in cases], "candidate_counts": args.candidate_counts,
                "condition_parameter_overrides_from_initial": overrides,
                "script_sha256": file_sha256(Path(__file__)),
                "implementation": {str(p.relative_to(source_root)): file_sha256(p)
                                   for p in sorted((source_root / "src/dsrsi").rglob("*.py"))}}
    jobs = [f"{label}/{w}/{l}/{v}/k{k}" for label, _, _ in conditions
            for w, l, v, _, _ in cases for k in args.candidate_counts]
    journal = EvaluationJournal(journal_path, identity, jobs, resume=args.resume,
                                stop_after_items=args.stop_after_items)
    journal.install_signal_handlers()
    torch.manual_seed(config["seed"])
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    model = MemoryQwen.from_local_checkpoint(base, device=device, memory_architecture=architecture).eval()
    evaluator = MOnlyEvaluator(model, compiler, memory_candidates=2)
    for label, committed, names in conditions:
        load_ab_weights(model, store, base_identity=base_identity, committed=committed)
        if names is not None:
            parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name in names:
                    if (parameters[name].shape != baseline_weights[name].shape
                            or parameters[name].dtype != baseline_weights[name].dtype):
                        raise ValueError("ablation parameter shape/dtype mismatch")
                    parameters[name].copy_(baseline_weights[name].to(parameters[name].device))
        model.eval()
        for world, language, variant, row, ids in cases:
            episode = compiler.compile(row)
            response = episode.responses[0]
            cached = {}

            def compute(candidate_count):
                if "pool" not in cached:
                    with torch.no_grad():
                        pool = evaluator.source_records(episode)
                        cached["pool"] = {r.record_id: r for r in pool}
                        cached["prefix"] = evaluator._encode_prefix(response)
                records = [cached["pool"][key] for key in ids[:candidate_count]]
                if any(r.slots.shape[0] != model.memory_config.slots for r in records):
                    raise ValueError("diagnostic requires uncompressed per-record slot layout")
                targets = response.input_ids[response.prompt_length:]
                with torch.no_grad():
                    plain = native_logits(model, response, records, life_id=episode.episode_id)
                    trace = {}
                    native = native_logits(model, response, records, life_id=episode.episode_id,
                                           reader_trace=trace)
                    if not torch.equal(plain, native):
                        raise RuntimeError("Reader instrumentation changed native forward logits")
                    canonical = canonical_logits(model, compiler, response, records,
                        life_id=episode.episode_id, prefix=cached["prefix"])
                a = summarize_logits(native, targets, compiler.tokenizer)
                b = summarize_logits(canonical, targets, compiler.tokenizer)
                return {"snapshot": label, "world": world, "language": language, "variant": variant,
                        "episode_id": episode.episode_id, "candidate_count": candidate_count,
                        "selected_ids": ids[:candidate_count], "expected": row["probes"][0]["target"],
                        "parameter_overrides_from_initial": names or [],
                        "native": a, "canonical": b,
                        "max_target_nll_path_difference": max(abs(x - y) for x, y in
                            zip(a["target_nll"], b["target_nll"])),
                        "native_instrumentation_logits_exact_match": True,
                        "reader": record_attention_summary(trace, slots_per_record=model.memory_config.slots,
                                                           candidate_count=candidate_count), "raw_reader_trace": trace}

            for count in args.candidate_counts:
                key = f"{label}/{world}/{language}/{variant}/k{count}"
                journal.evaluate(key, lambda count=count: compute(count))
                print(json.dumps({"item": key, "committed": len(journal.results), "total": len(jobs)}), flush=True)
    journal.finish()
    report = {"status": "matched_candidate_teacher_forced_reader_diagnostic",
              "identity": identity, "sampling": "handpicked mechanism cases; not representative capability scoring",
              "gold_prefix_used": True, "future_targets_used_in_prediction": False,
              "formal_training_modified": False, "free_generation_capability_claimed": False,
              "M5_passed": False, "records": [item["value"] for item in journal.results]}
    atomic_json(output, report)
    print(json.dumps({"status": report["status"], "items": len(report["records"])}), flush=True)


if __name__ == "__main__":
    from dsrsi.evaluation.journal import EvaluationPaused
    try:
        main()
    except EvaluationPaused as error:
        print(json.dumps({"status": "paused", "message": str(error)}), flush=True)
