#!/usr/bin/env python3
"""Broad held-out language retention: natural-text NLL plus generated probes."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch

PROBE_MODES = (("original_base", "off"), ("adapted_no_memory", "off"),
               ("adapted_unrelated_memory", "actual"))


def parse_modes(value):
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"base", "off", "unrelated"}:
        raise argparse.ArgumentTypeError("use unique modes from base,off,unrelated")
    return modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", help="pin an explicit committed checkpoint directory name")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--windows-per-language", type=int, default=80)
    parser.add_argument("--protocols", default="raw,native")
    parser.add_argument("--modes", type=parse_modes, default=("base", "off", "unrelated"))
    parser.add_argument("--probe-set", choices=("both", "v1", "extended", "none"), default="both")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--journal", help="atomic progress path; defaults to OUTPUT.journal.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-items", type=int, help="safe evaluation interruption for recovery verification")
    args = parser.parse_args()
    if not args.skip_nll and not (args.corpus and args.manifest):
        parser.error("--corpus and --manifest are required unless --skip-nll is used")
    protocols = tuple(part.strip() for part in args.protocols.split(","))
    if set(protocols) - {"raw", "native"} or len(set(protocols)) != len(protocols):
        parser.error("use unique protocols from raw,native")
    if min(args.windows_per_language, args.bootstrap, args.max_new_tokens) < 1 or args.window < 16:
        parser.error("budgets must be positive; NLL windows require at least sixteen tokens")
    if args.skip_nll and args.probe_set == "none":
        parser.error("at least one NLL or generation task is required")
    from dsrsi.data.broad_language import (CORPUS_REVISION, PROBE_REVISION, documents_for_language,
                                           extended_probes, file_sha256, load_corpus, select_windows)
    from dsrsi.data.language_probes import PROBE_REVISION as V1_REVISION
    from dsrsi.data.language_probes import matches, probes as v1_probes
    from dsrsi.evaluation.language_nll import (aggregate, build_scoring_sequence, document_totals,
                                               paired_bootstrap, retention_gate, score_sequence)
    from dsrsi.evaluation.checkpoint import load_ab_weights, load_c_tool_weights, load_context_weights
    from dsrsi.evaluation.m_only import MOnlyEvaluator
    from dsrsi.evaluation.pinned import load_pinned
    from dsrsi.evaluation.journal import EvaluationJournal
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.model.lora import lora_disabled
    from dsrsi.training.checkpoint import CheckpointStore
    from dsrsi.training.device import resolve_device
    device, world = resolve_device(args.device)
    if world != 1:
        raise ValueError("language retention probes run on one logical device")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    root = Path(args.base)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"tokenizer": tokenizer_fingerprint(root),
                "weights": {name: file_sha256(root / name) for name in sorted(set(index.values()))}}
    store = CheckpointStore(Path(args.run_dir) / "checkpoints")
    journal_path = Path(args.journal or (args.output + ".journal.json"))
    if args.resume and args.checkpoint is None:
        args.checkpoint = EvaluationJournal.read(journal_path)["identity"]["checkpoint"]
    committed = load_pinned(store, args.checkpoint)
    model = MemoryQwen.from_local_checkpoint(root, device=device,
        memory_architecture=committed[1]["identity"].get("memory_architecture")).eval()
    loader = {"C_tools": load_c_tool_weights, "C_context": load_context_weights}.get(
        committed[1]["progress"]["stage"], load_ab_weights)
    checkpoint, manifest = loader(model, store, base_identity=identity, committed=committed)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    documents, window_plan, selected = [], {}, []
    if not args.skip_nll:
        documents = load_corpus(args.corpus, args.manifest)
        for language in ("zh", "en"):
            window_plan[language] = select_windows(tokenizer, documents_for_language(documents, language),
                                                   window=args.window, budget=args.windows_per_language)
            if not window_plan[language]:
                raise ValueError(f"no usable NLL windows for {language}")
    if args.probe_set in ("both", "v1"):
        selected += [("v1", family, row) for family, row in v1_probes()]
    if args.probe_set in ("both", "extended"):
        selected += [("extended", family, row) for family, row in extended_probes()]
    jobs = [f"nll/{document.document_id}/{number}/{protocol}/{mode}"
            for language in ("zh", "en") for document, number, _ in window_plan.get(language, [])
            for protocol in protocols for mode in args.modes]
    jobs += [f"probe/{probe_set}/{row['episode_id']}/{name}"
             for probe_set, _, row in selected for name, _ in PROBE_MODES]
    source_root = Path(__file__).resolve().parents[1]
    journal_identity = {"checkpoint": checkpoint.name, "run_dir": str(Path(args.run_dir).resolve()),
        "manifest_sha256": file_sha256(checkpoint / "manifest.json"), "base": identity,
        "device": device, "corpus_sha256": file_sha256(args.corpus) if not args.skip_nll else None,
        "corpus_manifest_sha256": file_sha256(args.manifest) if not args.skip_nll else None,
        "config": {"window": args.window, "windows_per_language": args.windows_per_language,
                   "protocols": list(protocols), "modes": list(args.modes), "probe_set": args.probe_set,
                   "max_new_tokens": args.max_new_tokens, "bootstrap": args.bootstrap,
                   "skip_nll": args.skip_nll},
        "implementation": {str(path.relative_to(source_root)): file_sha256(path)
                           for path in sorted((source_root / "src/dsrsi").rglob("*.py"))},
        "script_sha256": file_sha256(__file__)}
    journal = EvaluationJournal(journal_path, journal_identity, jobs, resume=args.resume,
                                 stop_after_items=args.stop_after_items)
    journal.install_signal_handlers()
    report = {"status": "broad_heldout_language_retention_evaluation",
              "scope_note": ("held-out natural-text NLL with an unrelated memory record plus fixed "
                             "generated probes; retention regression evidence, not a capability gate"),
              "corpus_revision": CORPUS_REVISION, "probe_revision": PROBE_REVISION,
              "v1_probe_revision": V1_REVISION,
              "checkpoint": checkpoint.name, "manifest_sha256": file_sha256(checkpoint / "manifest.json"),
              "checkpoint_progress": manifest["progress"], "base_identity": identity, "device": device,
              "nll_execution": "parallel_native_teacher_forcing",
              "generation_execution": "canonical_single_token_streaming",
              "requested_modes": list(args.modes),
              "broad_language_capability_gate_claimed": False}

    unrelated = {}
    if "unrelated" in args.modes:
        source_text = {"zh": "白杨-941208的箱封码是J8-672。",
                       "en": "The seal code for Poplar-941208 is J8-672."}
        for language, text in source_text.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            unrelated[language] = model.write_records(
                torch.tensor([ids], device=device), stream_ids=[f"language-retention-{language}"],
                logical_begins=[0], record_ids=[f"language-unrelated-{language}"])[0]

    if not args.skip_nll:
        manifest_document = json.loads(Path(args.manifest).read_text())
        nll = {"window": args.window, "windows_per_language": args.windows_per_language,
               "corpus_manifest": str(args.manifest),
               "corpus_manifest_sha256": file_sha256(args.manifest),
               "corpus_file_sha256": manifest_document["corpus_sha256"],
               "documents": len(documents),
               "by_language_documents": {language: len(documents_for_language(documents, language))
                                         for language in ("zh", "en")},
               "document_sha256": {document.document_id: document.sha256 for document in documents},
               "parity": {}}
        records = []
        for language in ("zh", "en"):
            for document, number, window in window_plan[language]:
                for protocol in protocols:
                    sequence = build_scoring_sequence(tokenizer, window, protocol=protocol, language=language)
                    for mode in args.modes:
                        def compute_score():
                            with lora_disabled(model) if mode == "base" else nullcontext():
                                return score_sequence(model, sequence, mode=mode,
                                    record=unrelated.get(language), life_id=f"language-retention-{language}")
                        score = journal.evaluate(f"nll/{document.document_id}/{number}/{protocol}/{mode}",
                                                 compute_score)
                        records.append({"document_id": document.document_id, "language": language,
                                        "protocol": protocol, "mode": mode, "window": number, **score})
        nll["selected_windows"] = {"by_language": {
            language: len([record for record in records if record["language"] == language
                           and record["protocol"] == protocols[0]
                           and record["mode"] == args.modes[0]])
            for language in ("zh", "en")}}
        nll["selected_windows"]["by_document"] = {
            document.document_id: sum(row.document_id == document.document_id
                                     for row, _, _ in window_plan[document.language])
            for document in documents}
        strata = defaultdict(list)
        for record in records:
            strata[(record["language"], record["protocol"])].append(record)
        for (language, protocol), rows in sorted(strata.items()):
            split = {"windows": {"base": [r for r in rows if r["mode"] == "base"],
                                 "off": [r for r in rows if r["mode"] == "off"],
                                 "unrelated": [r for r in rows if r["mode"] == "unrelated"]}}
            stratum = {"aggregate": {}, "paired": {}, "gate": {}}
            for mode in args.modes:
                stratum["aggregate"][mode] = aggregate(split["windows"][mode]) if split["windows"][mode] else None
            if "base" in args.modes:
                base_totals = document_totals(split["windows"]["base"])
                for mode in ("off", "unrelated"):
                    if mode not in args.modes or not split["windows"][mode]:
                        continue
                    variant_totals = document_totals(split["windows"][mode])
                    paired = paired_bootstrap(base_totals, variant_totals, iterations=args.bootstrap)
                    stratum["paired"][f"{mode}_vs_base"] = paired
                    stratum["gate"][f"{mode}_vs_base"] = retention_gate(
                        stratum["aggregate"]["base"], stratum["aggregate"][mode], paired)
            nll["parity"][f"{language}/{protocol}"] = stratum
        report["nll"] = nll
        report["nll_records"] = records

    if args.probe_set != "none":
        compiler = EpisodeCompiler(tokenizer)
        evaluator = MOnlyEvaluator(model, compiler, max_new_tokens=args.max_new_tokens, memory_candidates=1)
        records = []
        with torch.no_grad():
            for probe_set, family, row in selected:
                episode = compiler.compile(row)
                source = evaluator.source_records(episode)
                response = episode.responses[0]
                outputs = {}
                for name, mode in PROBE_MODES:
                    def compute_probe():
                        with lora_disabled(model) if name == "original_base" else nullcontext():
                            answer = evaluator.generate(response, source, life_id=episode.episode_id,
                                expected=row["probes"][0]["target"], mode=mode)
                        return {"text": answer.text,
                                "correct": matches(answer.text, answer.expected, family=family),
                                "stopped_on_end": answer.stopped_on_end}
                    outputs[name] = journal.evaluate(f"probe/{probe_set}/{row['episode_id']}/{name}", compute_probe)
                records.append({"probe_set": probe_set, "id": row["episode_id"], "family": family,
                                "language": row["language"], "expected": row["probes"][0]["target"],
                                "outputs": outputs})

        def summarize(rows):
            if not rows:
                return {"queries": 0}
            base_correct = [r for r in rows if r["outputs"]["original_base"]["correct"]]
            return {"queries": len(rows), "base_correct_queries": len(base_correct),
                    "accuracy": {name: sum(r["outputs"][name]["correct"] for r in rows) / len(rows)
                                 for name in ("original_base", "adapted_no_memory", "adapted_unrelated_memory")},
                    "retained_base_correct_with_unrelated_memory": (
                        sum(r["outputs"]["adapted_unrelated_memory"]["correct"] for r in base_correct)
                        / len(base_correct) if base_correct else None)}
        groups = defaultdict(list)
        for row in records:
            groups["all"].append(row)
            groups[row["probe_set"]].append(row)
            groups[f"{row['probe_set']}/{row['family']}/{row['language']}"].append(row)
        report["generation_probes"] = {"summary": {name: summarize(rows) for name, rows in sorted(groups.items())},
                                       "records": records,
                                       "probe_source_sha256": {
                                           "language_probes": file_sha256(
                                               Path(__file__).resolve().parents[1] / "src/dsrsi/data/language_probes.py"),
                                           "broad_language": file_sha256(
                                               Path(__file__).resolve().parents[1] / "src/dsrsi/data/broad_language.py")},
                                       "max_new_tokens": args.max_new_tokens}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal.finish()
    report["evaluation_identity"] = journal_identity
    report["journal_items"] = len(journal.results)
    from dsrsi.utils.atomic import atomic_json
    atomic_json(output, report)
    summary = {"status": report["status"], "checkpoint": checkpoint.name}
    if "nll" in report:
        summary["nll"] = {key: {mode: (value["aggregate"][mode]["mean_nll"]
                                       if value["aggregate"].get(mode) else None)
                                for mode in report["requested_modes"]}
                          for key, value in report["nll"]["parity"].items()}
        summary["gate"] = {key: {name: gate["passed"]
                                 for name, gate in value["gate"].items()}
                           for key, value in report["nll"]["parity"].items()}
    if "generation_probes" in report:
        summary["probes"] = report["generation_probes"]["summary"].get("all")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    from dsrsi.evaluation.journal import EvaluationPaused
    try:
        main()
    except EvaluationPaused as exc:
        print(json.dumps({"status": "paused", "message": str(exc)}, ensure_ascii=False), flush=True)
