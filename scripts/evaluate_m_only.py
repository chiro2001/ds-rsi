#!/usr/bin/env python3
"""Held-out free generation and memory ablation on a committed checkpoint."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", help="pin a committed generation instead of following latest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", help="defaults to OUTPUT.journal.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-items", type=int, help="safe evaluation pause for recovery verification")
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--worlds-per-character-position", type=int,
                        help="optional diagnostic strata; requires only single_fact and pinned v4/v5 code data")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--families", nargs="+", default=["single_fact"],
                        choices=["single_fact", "paraphrase_coreference", "episode_and_task", "temporal_update",
                                 "missing_evidence", "language_preservation"])
    parser.add_argument("--languages", nargs="+", choices=["zh", "en", "cross_language"], default=["zh"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-data-revision", choices=["ab-v1", "ab-v2-matched-entities", "ab-v3-random-entity-order", "ab-v4-fields-and-language", "ab-v5-indexed-digits"],
                        help="explicit distribution challenge; default matches the pinned training revision")
    parser.add_argument("--include-foreign", action="store_true", help="replace all selected payloads with the other counterfactual world")
    parser.add_argument("--modes", nargs="+", choices=["actual", "oracle", "oracle_foreign", "off", "shuffled", "foreign"],
                        default=["actual", "oracle", "off", "shuffled"])
    parser.add_argument("--memory-candidates", type=int, default=8)
    parser.add_argument("--probe-view", choices=["full", "indexed"], default="full",
                        help="v5 auxiliary view is diagnostic only; formal recall uses full codes")
    args = parser.parse_args()
    if args.worlds < 1 or args.distractors < 8 or args.threads < 1:
        raise ValueError("positive worlds/threads and at least 8 distractors are required")
    if (len(set(args.families)) != len(args.families) or len(set(args.languages)) != len(args.languages)
            or len(set(args.modes)) != len(args.modes)):
        raise ValueError("duplicate task families or languages would inflate evaluation counts")
    torch.set_num_threads(args.threads)
    from dsrsi.training.device import resolve_device
    device, world_size = resolve_device(args.device)
    if world_size != 1:
        raise ValueError("held-out evaluation currently runs on one logical device")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import LEGACY_AB_REVISION, data_identity, make_generator
    from dsrsi.evaluation.m_only import MOnlyEvaluator, normalized_answer
    from dsrsi.evaluation.checkpoint import load_ab_weights, load_c_tool_weights, load_context_weights
    from dsrsi.evaluation.metrics import summarize_records
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    run = Path(args.run_dir)
    store = CheckpointStore(run / "checkpoints")
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    actual_base = {"tokenizer": tokenizer_fingerprint(root),
                   "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    from dsrsi.evaluation.journal import EvaluationJournal
    from dsrsi.evaluation.pinned import load_pinned
    journal_path = Path(args.journal or (args.output + ".journal.json"))
    if args.resume and args.checkpoint is None:
        args.checkpoint = EvaluationJournal.read(journal_path)["identity"]["checkpoint"]
    committed = load_pinned(store, args.checkpoint)
    config = committed[1]["identity"]["train_config"]
    data_revision = config.get("data_revision", LEGACY_AB_REVISION)
    if "data" in committed[1]["identity"] and data_identity(data_revision) != committed[1]["identity"]["data"]:
        raise RuntimeError("evaluation data/compiler identity differs from the pinned training revision")
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = MemoryQwen.from_local_checkpoint(root, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture"))
    compiler = EpisodeCompiler(tokenizer)
    loader = {"C_tools": load_c_tool_weights, "C_context": load_context_weights}.get(config["stage"], load_ab_weights)
    checkpoint_dir, manifest = loader(model, store, base_identity=actual_base, committed=committed)
    evaluator = MOnlyEvaluator(model, compiler, max_new_tokens=args.max_new_tokens, memory_candidates=args.memory_candidates)
    eval_revision = args.eval_data_revision or data_revision
    if args.probe_view != "full" and eval_revision != "ab-v5-indexed-digits":
        raise ValueError("indexed probe view requires the explicit v5 data revision")
    view = {"probe_view": args.probe_view} if eval_revision == "ab-v5-indexed-digits" else {}
    generator = make_generator(revision=eval_revision, seed=config["seed"], split=args.split,
                               distractors=args.distractors, **view)
    if args.worlds_per_character_position is not None:
        if (args.worlds_per_character_position < 1 or args.families != ["single_fact"]
                or eval_revision not in {"ab-v4-fields-and-language", "ab-v5-indexed-digits"}):
            raise ValueError("character-stratified diagnostics require positive counts and only v4/v5 single_fact")
        from dsrsi.evaluation.sampling import stratified_code_worlds
        worlds = stratified_code_worlds(generator, worlds_per_position=args.worlds_per_character_position,
                                       language=args.languages[0])
        sampling = {"kind": "counterfactual_character_strata", "world_indices": worlds,
                    "worlds_per_character_position": args.worlds_per_character_position}
    else:
        worlds = list(range(args.worlds))
        sampling = {"kind": "consecutive_worlds", "world_indices": worlds}
    records = []
    modes = tuple(args.modes)
    if args.include_foreign and "foreign" not in modes:
        modes += ("foreign",)
    jobs = [f"{world}/{family}/{language}/{variant}" for world in worlds for family in args.families
            for language in args.languages for variant in (0, 1)]
    source_root = Path(__file__).resolve().parents[1]
    runtime = {"torch": str(torch.__version__)}
    if device.startswith("cuda"):
        from dsrsi.model.cuda_execution import cuda_execution_identity
        runtime = cuda_execution_identity(device)
    journal_identity = {
        "checkpoint": checkpoint_dir.name, "manifest_sha256": file_sha256(checkpoint_dir / "manifest.json"),
        "base": actual_base, "device": device,
        "runtime": runtime,
        "config": {key: value for key, value in vars(args).items()
                   if key not in {"resume", "stop_after_items", "output", "journal", "checkpoint"}},
        "sampling": sampling, "evaluation_data_identity": data_identity(eval_revision),
        "implementation": {str(path.relative_to(source_root)): file_sha256(path)
                           for path in sorted((source_root / "src/dsrsi").rglob("*.py"))},
        "script_sha256": file_sha256(Path(__file__)),
    }
    journal = EvaluationJournal(journal_path, journal_identity, jobs, resume=args.resume,
                                 stop_after_items=args.stop_after_items)
    journal.install_signal_handlers()
    model.eval()
    for world in worlds:
        for family in args.families:
            for language in args.languages:
                for variant in (0, 1):
                    row = generator.episode(world, variant=variant, family=family, language=language)
                    def compute():
                        if args.retrieval_only:
                            with torch.no_grad():
                                episode = compiler.compile(row)
                                sources = evaluator.source_records(episode)
                                return [evaluator.retrieval(response, sources, life_id=episode.episode_id)
                                        for response in episode.responses]
                        replacements = None
                        foreign_expected = {}
                        if args.include_foreign or any(m in modes for m in ("foreign", "oracle_foreign")):
                            foreign = generator.episode(world, variant=1 - variant, family=family, language=language)
                            with torch.no_grad():
                                replacements = evaluator.source_records(compiler.compile(foreign))
                            foreign_expected = {p["probe_id"]: p["target"] for p in foreign["probes"]}
                        results = evaluator.episode(row, modes=modes, replacement_records=replacements)
                        for result in results:
                            if result["mode"] in ("foreign", "oracle_foreign"):
                                result["counterfactual_payload_answer"] = foreign_expected[result["probe_id"]]
                                result["follows_counterfactual_payload"] = (normalized_answer(result["text"])
                                    == normalized_answer(foreign_expected[result["probe_id"]]))
                        return results
                    results = copy.deepcopy(journal.evaluate(f"{world}/{family}/{language}/{variant}", compute))
                    for result in results:
                        result.update(episode_id=row["episode_id"], world_id=row["world_id"], variant=variant,
                                      family=family, language=language)
                        result["codec_task"] = row["annotations"].get("codec_task", "full_code" if family == "single_fact" else None)
                        result["answer_character_position"] = row["annotations"].get("answer_character_position")
                        records.append(result)
    summary = summarize_records(records, retrieval_only=args.retrieval_only)
    by_task, by_position = defaultdict(list), defaultdict(list)
    for record in records:
        if record.get("codec_task") is not None:
            by_task[record["codec_task"]].append(record)
        if record.get("answer_character_position") is not None:
            by_position[str(record["answer_character_position"])].append(record)
    report = {"status": "heldout_retrieval_only" if args.retrieval_only else "heldout_free_generation", "split": args.split,
              "checkpoint": checkpoint_dir.name, "training_world_size": manifest["world_size"],
              "manifest_sha256": file_sha256(checkpoint_dir / "manifest.json"), "base_identity": actual_base,
              "families": args.families, "languages": args.languages, "distractors": args.distractors,
              "canonical_execution_quantum": 1,
              "training_data_revision": data_revision, "data_revision": eval_revision,
              "evaluation_data_identity": data_identity(eval_revision), "distribution_challenge": eval_revision != data_revision,
              "device": device,
              "memory_candidates": args.memory_candidates,
              "probe_view": args.probe_view, "auxiliary_probe_diagnostic": args.probe_view == "indexed",
              "modes": modes if not args.retrieval_only else [],
              "foreign_payload_ablation": (args.include_foreign or any(m in args.modes for m in ("foreign", "oracle_foreign"))) and not args.retrieval_only,
              "checkpoint_progress": manifest["progress"], "max_new_tokens": args.max_new_tokens,
              "worlds": len(worlds), "sampling": sampling, "summary": summary, "records": records,
              "by_codec_task": {key: summarize_records(rows, retrieval_only=args.retrieval_only) for key, rows in by_task.items()},
              "by_answer_character_position": {key: summarize_records(rows, retrieval_only=args.retrieval_only)
                                               for key, rows in by_position.items()},
              "capability_gate_passed": False}
    journal.finish()
    from dsrsi.utils.atomic import atomic_json
    atomic_json(Path(args.output), report)
    print(json.dumps({"checkpoint_progress": manifest["progress"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    from dsrsi.evaluation.journal import EvaluationPaused
    try:
        main()
    except EvaluationPaused as exc:
        print(json.dumps({"status": "paused", "message": str(exc)}, ensure_ascii=False), flush=True)
