import json

from dsrsi.data.generate import EpisodeGenerator
from dsrsi.data.compiler import EpisodeCompiler
from test_compiler import BoundaryTokenizer


def test_generation_resumes_by_ordinal_without_hidden_rng_state():
    generator = EpisodeGenerator(seed=91)
    expected = [generator.episode(index) for index in range(10)]
    resumed = EpisodeGenerator(seed=91)
    assert [resumed.episode(index) for index in range(6, 10)] == expected[6:]
    # Rank striding does not duplicate or alter worlds.
    assert [generator.episode(index) for index in range(0, 10, 2)] == expected[::2]


def test_counterfactuals_change_evidence_and_answer_without_changing_question():
    generator = EpisodeGenerator(seed=27)
    a = generator.episode(3, family="single_fact", variant=0)
    b = generator.episode(3, family="single_fact", variant=1)
    assert a["probes"][0]["query"] == b["probes"][0]["query"]
    assert a["probes"][0]["target"] != b["probes"][0]["target"]
    assert a["counterfactual_group_id"] == b["counterfactual_group_id"]
    assert a["source_events"][:-1] == b["source_events"][:-1]


def test_all_pilot_families_and_languages_compile_with_valid_supports():
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=20000)
    generator = EpisodeGenerator(seed=5)
    for family in ["single_fact", "paraphrase_coreference", "episode_and_task", "temporal_update",
                   "missing_evidence", "language_preservation"]:
        for language in ["zh", "en", "cross_language"]:
            for variant in [0, 1]:
                row = generator.episode(12, family=family, language=language, variant=variant)
                compiled = compiler.compile(row)
                assert compiled.responses
                assert compiled.source_encoding == "independent_events"
                assert all(span.reset_working_before for span in compiled.source)


def test_train_dev_test_worlds_and_fact_question_templates_are_separate():
    rows = [EpisodeGenerator(seed=3, split=split).episode(1, family="single_fact", language="en")
            for split in ["train", "dev", "test"]]
    assert len({row["world_id"] for row in rows}) == 3
    assert len({row["probes"][0]["query"] for row in rows}) == 3
    assert len({row["episode_id"] for row in rows}) == 3
