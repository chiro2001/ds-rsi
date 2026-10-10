#!/usr/bin/env python3
"""Audit real-tokenizer alignment and answer-free indexed counterfactuals."""

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
    from dsrsi.data.revisions import make_generator, data_identity
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    compiler = EpisodeCompiler(tokenizer)
    counts, positions = Counter(), Counter()
    maxima = {"source": 0, "response": 0}
    examples = []
    for split in ("train", "dev"):
        generator = make_generator(revision="ab-v5-indexed-digits", seed=20261010, split=split, distractors=7)
        for world in range(80):
            for language in ("zh", "en", "cross_language"):
                rows = [generator.episode(world, variant=v, family="single_fact", language=language) for v in (0, 1)]
                episodes = [compiler.compile(row) for row in rows]
                responses = [e.responses[0] for e in episodes]
                assert responses[0].prompt_ids == responses[1].prompt_ids
                suffixes = [r.input_ids[r.prompt_length:] for r in responses]
                changed = next(i for i, (a, b) in enumerate(zip(*suffixes)) if a != b)
                task = rows[0]["annotations"]["codec_task"]
                for response in responses:
                    assert response.retrieval_query_index == response.prompt_length - 1
                    assert all(x == -100 for x in response.labels[:response.prompt_length])
                    assert response.write_enabled is False
                if task == "indexed_character":
                    assert changed == 0 and all(len(s) == 2 for s in suffixes)
                    positions[rows[0]["annotations"]["answer_character_position"]] += 1
                if split == "dev":
                    assert task == "full_code"
                counts[f"{split}/{task}"] += 2
                maxima["source"] = max(maxima["source"], *(len(s.input_ids) for e in episodes for s in e.source))
                maxima["response"] = max(maxima["response"], *(len(r.input_ids) for r in responses))
                if task == "indexed_character" and len(examples) < 6:
                    examples.append({"language": language, "query": rows[0]["probes"][0]["query"],
                        "targets": [r["probes"][0]["target"] for r in rows], "first_difference": changed,
                        "supervised_token_count": len(suffixes[0])})
    report = {"status": "real_tokenizer_indexed_counterfactual_audit_passed", "counts": dict(counts),
        "auxiliary_suffix_positions": dict(positions), "maximum_tokens": maxima,
        "tokenizer_fingerprint": tokenizer_fingerprint(args.tokenizer),
        "data_identity": data_identity("ab-v5-indexed-digits"), "examples": examples,
        "actual_training_consumption": False, "capability_gate_passed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
