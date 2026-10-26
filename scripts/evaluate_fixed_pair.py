#!/usr/bin/env python3
"""Fixed-pair M-only recall and plaintext-teacher upper bounds; resumable."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "run-dir", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-items", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p.add_argument("--languages", nargs="+", default=["zh", "en", "cross_language"])
    p.add_argument("--views", nargs="+", default=["memory", "support_text", "pair_text"],
                   choices=["memory", "support_text", "pair_text", "off", "foreign"])
    p.add_argument("--source-layout", choices=["native", "varied"], default="native")
    p.add_argument("--probe-view", choices=["full", "mixed", "indexed"], default="full")
    p.add_argument("--distractors", type=int, default=7)
    p.add_argument("--memory-candidates", type=int, choices=[1, 2], default=2)
    p.add_argument("--max-new-tokens", type=int, default=16)
    args = p.parse_args()
    if (args.worlds < 1 or args.threads < 1 or len(set(args.languages)) != len(args.languages)
            or len(set(args.views)) != len(args.views)
            or not set(args.languages) <= {"zh", "en", "cross_language"}):
        raise ValueError("invalid fixed-pair evaluation plan")
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.fixed_pair import FixedPairEpisodeGenerator
    from dsrsi.evaluation.fixed_records import generate_fixed_records, evidence_teacher_response
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.evaluation.journal import EvaluationJournal
    from dsrsi.evaluation.pinned import load_pinned
    from dsrsi.evaluation.m_only import MOnlyEvaluator
    from dsrsi.model.qwen35_memory import MemoryQwen
    from dsrsi.training.checkpoint import CheckpointStore, file_sha256
    from dsrsi.training.device import resolve_device
    from dsrsi.utils.atomic import atomic_json
    device, world = resolve_device(args.device)
    if world != 1:
        raise ValueError("one device is required")
    torch.set_num_threads(args.threads)
    base, run, output = Path(args.base), Path(args.run_dir), Path(args.output)
    journal_path = Path(str(output) + ".journal.json")
    if device.startswith("cuda"):
        root = Path("/mnt/e/ds-rsi").resolve()
        if any(not value.resolve().is_relative_to(root) for value in (base, run, output)):
            raise ValueError("CUDA paths must remain on E:")
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    store = CheckpointStore(run / "checkpoints")
    if args.resume:
        args.checkpoint = EvaluationJournal.read(journal_path)["identity"]["checkpoint"]
    committed = load_pinned(store, args.checkpoint)
    m = committed[1]
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    base_id = dict(tokenizer=tokenizer_fingerprint(base),
                   weights={n: file_sha256(base / n) for n in sorted(set(index.values()))})
    torch.manual_seed(m["identity"]["train_config"]["seed"])
    model = MemoryQwen.from_local_checkpoint(base, device=device,
                                            memory_architecture=m["identity"].get("memory_architecture"))
    directory, m = load_ab_weights(model, store, base_identity=base_id, committed=committed)
    model.eval()
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(base, local_files_only=True))
    generator = FixedPairEpisodeGenerator(seed=m["identity"]["train_config"]["seed"], split=args.split,
                                          distractors=args.distractors, source_layout=args.source_layout,
                                          probe_view=args.probe_view)
    jobs = [f"{w}/{lang}/{v}/{probe}/{view}" for w in range(args.worlds) for lang in args.languages
            for v in (0, 1) for probe in ("p1", "p2") for view in args.views]
    code = Path(__file__).resolve().parents[1]
    runtime = dict(torch=str(torch.__version__))
    if device.startswith("cuda"):
        from dsrsi.model.cuda_execution import cuda_execution_identity
        runtime = cuda_execution_identity(device)
    identity = dict(checkpoint=directory.name, manifest_sha256=file_sha256(directory / "manifest.json"),
                    base=base_id, runtime=runtime,
                    config={k: v for k, v in vars(args).items() if k not in
                            {"resume", "checkpoint", "stop_after_items", "output"}},
                    implementation={str(x.relative_to(code)): file_sha256(x) for x in
                                    sorted((code / "src/dsrsi").rglob("*.py"))},
                    script_sha256=file_sha256(Path(__file__)))
    journal = EvaluationJournal(journal_path, identity, jobs, resume=args.resume,
                                 stop_after_items=args.stop_after_items)
    journal.install_signal_handlers()
    records = []
    evaluator = MOnlyEvaluator(model, compiler)
    for w in range(args.worlds):
        for lang in args.languages:
            for variant in (0, 1):
                row = generator.episode(w, variant=variant, family="single_fact", language=lang)
                episode = compiler.compile(row)
                written = None
                foreign = None
                for response, probe in zip(episode.responses, row["probes"]):
                    for view in args.views:
                        key = f"{w}/{lang}/{variant}/{probe['probe_id']}/{view}"
                        def compute():
                            nonlocal written, foreign
                            selected = []
                            query = response
                            source_tokens = 0
                            if view in {"memory", "foreign"}:
                                if written is None:
                                    with torch.no_grad():
                                        written = evaluator.source_records(episode)
                                    source_tokens += sum(len(span.input_ids) for span in episode.source)
                                ids = row["annotations"]["fixed_memory_candidate_ids"]
                                selected = [r for r in written if r.record_id in ids]
                                if args.memory_candidates == 1:
                                    selected = [r for r in selected if r.record_id in probe["support_groups"][0]]
                                if view == "foreign":
                                    if foreign is None:
                                        opposite = generator.episode(w, variant=1 - variant, family="single_fact", language=lang)
                                        with torch.no_grad():
                                            opposite_episode = compiler.compile(opposite)
                                            foreign = evaluator.source_records(opposite_episode)
                                        source_tokens += sum(len(span.input_ids) for span in opposite_episode.source)
                                    by_id = {r.record_id: r for r in foreign}
                                    from dataclasses import replace
                                    selected = [replace(r, slots=by_id[r.record_id].slots) for r in selected]
                            elif view in {"support_text", "pair_text"}:
                                query = evidence_teacher_response(compiler, row, probe,
                                                                  scope="support" if view == "support_text" else "pair")
                            result = generate_fixed_records(model, compiler, query, selected,
                                                            life_id=episode.episode_id, expected=probe["target"],
                                                            max_new_tokens=args.max_new_tokens)
                            value = dict(result["answer"], view=view, query_input_tokens=result["input_tokens"],
                                         source_input_tokens=source_tokens,
                                         episode_id=row["episode_id"], language=lang, world=w, variant=variant,
                                         pair_kind=row["annotations"]["pair_kind"],
                                         codec_task=row["annotations"]["codec_task"],
                                         fixed_pair_ids=row["annotations"]["fixed_memory_candidate_ids"])
                            value["exact_token_match"] = (value["stopped_on_end"] and
                                                          value["token_ids"] + [compiler.tokenizer.eos_token_id]
                                                          == query.input_ids[query.prompt_length:])
                            return value
                        records.append(copy.deepcopy(journal.evaluate(key, compute)))
    summary = {}
    for view in args.views:
        part = [r for r in records if r["view"] == view]
        groups = defaultdict(list)
        for r in part:
            groups[(r["world"], r["language"], r["variant"])].append(r)
        summary[view] = dict(queries=len(part), correct=sum(r["exact_match"] for r in part),
                             exact_token_matches=sum(r["exact_token_match"] for r in part),
                             both_questions_correct=sum(all(r["exact_match"] for r in rows) for rows in groups.values()),
                             paired_worlds=len(groups), query_input_tokens=sum(r["query_input_tokens"] for r in part),
                             source_input_tokens=sum(r["source_input_tokens"] for r in part),
                             actual_input_tokens=sum(r["query_input_tokens"] + r["source_input_tokens"] for r in part))
    journal.finish()
    report = dict(status="fixed_pair_free_generation_diagnostic", checkpoint=directory.name,
                   manifest_sha256=identity["manifest_sha256"], sampling=identity["config"],
                   summary=summary, records=records, M5_passed=False, source_training_state_modified=False)
    if device.startswith("cuda"):
        report["cuda_memory"] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    atomic_json(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
