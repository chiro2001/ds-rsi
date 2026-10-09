from dsrsi.data.language_probes import probes, matches
from dsrsi.data.compiler import EpisodeCompiler
from test_compiler import BoundaryTokenizer


def test_language_probes_are_heldout_with_unrelated_sources_and_answer_free_reads():
    rows = probes()
    assert len(rows) == 36 and len({r["episode_id"] for _, r in rows}) == 36
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=10000)
    for _, row in rows:
        assert row["split"] == "dev" and not row["probes"][0]["support_groups"]
        response = compiler.compile(row).responses[0]
        assert response.retrieval_query_index == response.prompt_length - 1
        assert not response.write_enabled


def test_probe_scoring_accepts_formatting_variants_without_accepting_nonanswers():
    assert matches(' { "x":7.0 } ', '{"x":7}', family="format")
    assert not matches('{"x":"7"}', '{"x":7}', family="format")
    assert not matches('Here is {"x":7}', '{"x":7}', family="format")
    assert matches("PARIS.", "Paris", family="knowledge")
    assert matches("星期五。", "周五", family="reading")
    assert not matches("I do not know", "Paris", family="knowledge")
