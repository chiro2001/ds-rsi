import json
from pathlib import Path

import pytest
import torch

from dsrsi.data.broad_language import (CORPUS_REVISION, CorpusDocument, extended_probes,
                                       generated_cases, load_corpus, select_windows, text_sha256,
                                       token_windows)
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.language_probes import matches
from dsrsi.evaluation.language_nll import (aggregate, build_scoring_sequence, document_totals,
                                           paired_bootstrap, retention_gate, score_sequence)
from test_compiler import BoundaryTokenizer


class NativeTokenizer(BoundaryTokenizer):
    def decode(self, tokens, **kwargs):
        return "".join(chr(max(token - 3, 0)) for token in tokens)


def test_generated_cases_are_deterministic_complete_and_frozen():
    cases = generated_cases()
    families = {}
    for family, zh, en, answers in cases:
        families[family] = families.get(family, 0) + 1
        assert zh and en and len(answers) == 2 and all(answers)
        assert matches(answers[0], answers[0], family=family)
        assert not matches("definitely-not-the-answer", answers[0], family=family)
    assert families == {"arithmetic": 8, "reading": 6, "logic": 4, "code": 3, "format": 3}
    assert text_sha256(json.dumps(cases, ensure_ascii=False)) == (
        "30c1288d5397e2b77cd4958b5693cee6ffe5c8078c90dc50bf20cb54426a10ca")
    assert generated_cases() == cases


def test_extended_probes_compile_answer_free_and_unique():
    rows = extended_probes()
    assert len(rows) == 48 and len({row["episode_id"] for _, row in rows}) == 48
    assert {family for family, _ in rows} == {"arithmetic", "reading", "logic", "code", "format"}
    compiler = EpisodeCompiler(NativeTokenizer(), max_query_tokens=10000)
    for family, row in rows:
        assert row["split"] == "dev" and not row["probes"][0]["support_groups"]
        response = compiler.compile(row).responses[0]
        assert response.retrieval_query_index == response.prompt_length - 1
        assert not response.write_enabled
        assert matches(row["probes"][0]["target"], row["probes"][0]["target"], family=family)


def test_token_windows_drop_partial_tail_and_respect_limit():
    assert token_windows(list(range(10)), window=4, limit=3) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert token_windows(list(range(10)), window=4, limit=1) == [[0, 1, 2, 3]]
    assert token_windows([1, 2, 3], window=4, limit=2) == []
    with pytest.raises(ValueError):
        token_windows([1, 2, 3], window=1, limit=2)
    with pytest.raises(ValueError):
        token_windows([1, 2, 3], window=2, limit=0)


def test_select_windows_round_robins_across_documents():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [int(part) for part in text.split() if part.isdigit()]

    documents = [CorpusDocument("en-a", "en", "A", 1, " ".join(str(i) for i in range(10))),
                 CorpusDocument("en-b", "en", "B", 2, " ".join(str(i) for i in range(100, 110))),
                 CorpusDocument("en-c", "en", "C", 3, "x")]
    selected = select_windows(Tokenizer(), documents, window=4, budget=4)
    assert [(document.document_id, number) for document, number, _ in selected] == [
        ("en-a", 0), ("en-b", 0), ("en-a", 1), ("en-b", 1)]
    assert selected[0][2] == [0, 1, 2, 3] and selected[1][2] == [100, 101, 102, 103]
    assert len(select_windows(Tokenizer(), documents, window=4, budget=50)) == 4
    with pytest.raises(ValueError):
        select_windows(Tokenizer(), documents, window=4, budget=0)


def test_corpus_manifest_roundtrip_and_tamper_rejection(tmp_path):
    documents = [{"document_id": "en-a", "language": "en", "title": "A", "revision_id": 11,
                  "url": "https://example.invalid/11", "license": "CC BY-SA 4.0", "text": "Alpha text."},
                 {"document_id": "zh-b", "language": "zh", "title": "B", "revision_id": 22,
                  "url": "https://example.invalid/22", "license": "CC BY-SA 4.0", "text": "中文文本。"}]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in documents))
    manifest = {"schema": CORPUS_REVISION, "corpus_sha256": text_sha256(corpus.read_text()),
                "documents": [{"document_id": row["document_id"], "language": row["language"],
                               "title": row["title"], "revision_id": row["revision_id"],
                               "sha256": text_sha256(row["text"])} for row in documents]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    loaded = load_corpus(corpus, manifest_path)
    assert [(document.document_id, document.language, document.title) for document in loaded] == [
        (row["document_id"], row["language"], row["title"]) for row in documents]
    corpus.write_text(corpus.read_text().replace("Alpha", "Beta"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_corpus(corpus, manifest_path)
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in documents))
    manifest["documents"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    with pytest.raises(ValueError, match="differs from its manifest"):
        load_corpus(corpus, manifest_path)


def test_scoring_sequences_mask_retrieval_prefix():
    tokenizer = NativeTokenizer()
    raw = build_scoring_sequence(tokenizer, list(range(20, 40)), protocol="raw", language="en")
    assert raw.query_index == 0 and raw.labels[0] == -100
    assert raw.supervised_tokens == 19 and raw.labels[1:] == list(range(21, 40))
    native = build_scoring_sequence(tokenizer, list(range(20, 60)), protocol="native", language="zh")
    assert native.query_index == len(native.input_ids) - native.supervised_tokens - 1
    assert all(label == -100 for label in native.labels[:native.query_index + 1])
    assert native.labels[native.query_index + 1:] == native.input_ids[native.query_index + 1:]
    with pytest.raises(ValueError):
        build_scoring_sequence(tokenizer, list(range(20, 30)), protocol="raw", language="en")
    with pytest.raises(ValueError):
        build_scoring_sequence(tokenizer, list(range(20, 40)), protocol="bogus", language="en")


def test_aggregation_and_paired_bootstrap_are_token_weighted_and_deterministic():
    records = [{"document_id": "a", "tokens": 10, "nll_sum": 20.0},
               {"document_id": "b", "tokens": 30, "nll_sum": 90.0}]
    summary = aggregate(records)
    assert summary == {"windows": 2, "documents": 2, "tokens": 40, "mean_nll": 2.75,
                       "perplexity": pytest.approx(15.6426319, rel=1e-6)}
    base = document_totals(records)
    variant = document_totals([{"document_id": "a", "tokens": 10, "nll_sum": 21.0},
                               {"document_id": "b", "tokens": 30, "nll_sum": 90.0}])
    paired = paired_bootstrap(base, variant, iterations=500, seed=7)
    assert paired == paired_bootstrap(base, variant, iterations=500, seed=7)
    assert paired["documents"] == 2 and paired["mean_delta"] == pytest.approx(0.05)
    gate = retention_gate(summary, summary, paired)
    assert gate["passed"] and set(gate["checks"]) == {"mean_delta", "ci95_high", "max_document_delta"}
    with pytest.raises(ValueError):
        paired_bootstrap(base, document_totals([{"document_id": "a", "tokens": 9, "nll_sum": 1.0}]))
    with pytest.raises(ValueError, match="identical document sets"):
        paired_bootstrap(base, {"a": base["a"]})
    with pytest.raises(ValueError, match="iterations"):
        paired_bootstrap(base, variant, iterations=0)


def test_score_sequence_runs_base_off_and_unrelated_memory_on_tiny_model():
    pytest.importorskip("transformers")
    from test_qwen_bridge import tiny
    torch.manual_seed(3)
    model = tiny().eval()
    tokenizer = NativeTokenizer()
    sequence = build_scoring_sequence(tokenizer, list(range(4, 20)), protocol="raw", language="en")
    record = model.write_records(torch.tensor([[4, 5, 6, 7]]), stream_ids=["language-retention"],
                                 logical_begins=[0], record_ids=["unrelated"])[0]
    off = score_sequence(model, sequence, mode="off")
    base = score_sequence(model, sequence, mode="base")
    unrelated = score_sequence(model, sequence, mode="unrelated", record=record)
    assert off == base  # frozen-lora baseline and memory-off share one numerical path
    assert unrelated["tokens"] == base["tokens"] == sequence.supervised_tokens
    assert all(row["mean_nll"] > 0 for row in (base, unrelated))
    with pytest.raises(ValueError):
        score_sequence(model, sequence, mode="unrelated")
