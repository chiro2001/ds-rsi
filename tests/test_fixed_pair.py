import copy

import pytest

from dsrsi.data.fixed_pair import FIXED_PAIR_REVISION, FixedPairEpisodeGenerator


def test_pair_is_question_independent_and_world_variants_are_matched():
    for kind in ("entity", "attribute"):
        seen = 0
        for world in range(64):
            g = FixedPairEpisodeGenerator(seed=17, distractors=7)
            a, b = [g.episode(world, variant=v, family="single_fact", language="en") for v in (0, 1)]
            if a["annotations"]["pair_kind"] != kind:
                continue
            seen += 1
            ids = a["annotations"]["fixed_memory_candidate_ids"]
            assert len(ids) == len(set(ids)) == 2 and ids == b["annotations"]["fixed_memory_candidate_ids"]
            assert [p["query"] for p in a["probes"]] == [p["query"] for p in b["probes"]]
            assert len({p["query"] for p in a["probes"]}) == 2
            assert {p["support_groups"][0][0] for p in a["probes"]} == set(ids)
            for x, y in zip(a["source_events"], b["source_events"]):
                assert x["event_id"] == y["event_id"] and x["time"] == y["time"]
                assert (x["content"] != y["content"]) == (x["event_id"] in ids)
            for x, y in zip(a["probes"], b["probes"]):
                assert x["target"] != y["target"]
            assert a["data_revision"] == FIXED_PAIR_REVISION and a["episode_id"] != b["episode_id"]
        assert seen > 10


def test_attribute_queries_have_no_conflicting_same_entity_attribute_records():
    g = FixedPairEpisodeGenerator(seed=17, distractors=7, probe_view="full")
    for world in range(64):
        row = g.episode(world, family="single_fact", language="zh")
        if row["annotations"]["pair_kind"] != "attribute":
            continue
        entity = row["annotations"]["target_entity"]
        same = [e for e in row["source_events"] if entity in e["content"]]
        assert len(same) == 2
        assert sum("校验码" in e["content"] for e in same) == 1
        assert sum("箱封码" in e["content"] for e in same) == 1


def test_keep_content_is_unchanged_and_layout_is_not_query_conditioned():
    from dsrsi.data.natural_keep import NaturalKeepEpisodeGenerator
    g = FixedPairEpisodeGenerator(seed=17, distractors=7)
    a = g.episode(1, family="language_preservation", language="en")
    b = NaturalKeepEpisodeGenerator(seed=17, distractors=7).episode(1, family="language_preservation", language="en")
    for key in ("source_events", "probes", "annotations"):
        assert a[key] == b[key]
    g = FixedPairEpisodeGenerator(seed=17, distractors=7, source_layout="varied")
    row = g.episode(1, family="single_fact", language="en")
    assert row == g.episode(1, family="single_fact", language="en")
    assert len(row["probes"]) == 2
    with pytest.raises(ValueError):
        g.episode(0, family="temporal_update")


def test_fixed_decoder_scoring_labels_cannot_change_generated_tokens():
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.evaluation.fixed_records import generate_fixed_records, evidence_teacher_response
    class DecodingTokenizer(SmallVocabularyTokenizer):
        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))
    compiler = EpisodeCompiler(DecodingTokenizer(), max_query_tokens=1200)
    row = FixedPairEpisodeGenerator(seed=17, distractors=2).episode(0, family="single_fact", language="en")
    m = model()
    episode = compiler.compile(row)
    from dsrsi.training.episode_objective import EpisodeObjective
    records, _, _ = EpisodeObjective(m, compiler)._sources(episode)
    selected = [r for r in records if r.record_id in row["annotations"]["fixed_memory_candidate_ids"]]
    response = episode.responses[0]
    a = generate_fixed_records(m, compiler, response, selected, life_id=episode.episode_id,
                               expected="a", max_new_tokens=3)
    altered = copy.deepcopy(response)
    altered.support_groups = [["nonexistent"]]
    altered.input_ids[altered.prompt_length:] = [7] * (len(altered.input_ids) - altered.prompt_length)
    b = generate_fixed_records(m, compiler, altered, selected, life_id=episode.episode_id,
                               expected="b", max_new_tokens=3)
    assert a["answer"]["token_ids"] == b["answer"]["token_ids"]
    assert a["answer"]["selected_records"] == b["answer"]["selected_records"]
    teacher = evidence_teacher_response(compiler, row, row["probes"][0])
    assert teacher.prompt_ids != response.prompt_ids
    other = dict(row["probes"][0], target="unrelated-scoring-label")
    assert evidence_teacher_response(compiler, row, other).prompt_ids == teacher.prompt_ids
