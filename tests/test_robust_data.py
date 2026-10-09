from datetime import datetime
import json

import pytest

from dsrsi.data.robust import AB_DATA_REVISION, RobustEpisodeGenerator, data_identity, make_generator
from dsrsi.data.generate import EpisodeGenerator, FAMILIES
from dsrsi.data.compiler import EpisodeCompiler
from test_compiler import BoundaryTokenizer


def test_counterfactual_pair_keeps_all_negatives_ids_times_and_positions_identical():
    generator = RobustEpisodeGenerator(seed=17, distractors=15)
    for family in FAMILIES:
        a = generator.episode(13, family=family, language="zh", variant=0)
        b = generator.episode(13, family=family, language="zh", variant=1)
        assert {k: v for k, v in a["annotations"].items() if k != "variant"} == {
            k: v for k, v in b["annotations"].items() if k != "variant"}
        assert a["data_revision"] == b["data_revision"] == AB_DATA_REVISION
        assert [p["query"] for p in a["probes"]] == [p["query"] for p in b["probes"]]
        supports = {source for probe in a["probes"] for group in probe["support_groups"] for source in group}
        assert [e for e in a["source_events"] if e["event_id"] not in supports] == [
            e for e in b["source_events"] if e["event_id"] not in supports]
        if family != "missing_evidence":
            assert all(p["target"] != q["target"] for p, q in zip(a["probes"], b["probes"]))


def test_support_position_and_entity_prefix_do_not_identify_positive_records():
    generator = RobustEpisodeGenerator(seed=91, distractors=7)
    positions = set()
    prefixes = set()
    for world in range(160):
        row = generator.episode(world, family="single_fact", language="zh")
        position, = row["annotations"]["support_positions"]
        positions.add(position)
        prefix = row["annotations"]["matched_prefix"]
        prefixes.add(prefix)
        assert all(e["content"].startswith(prefix + "-") for e in row["source_events"])
        entity = row["annotations"]["target_entity"]
        same_entity = [e for e in row["source_events"] if e["content"].startswith(entity) and e["content"].find("箱封码") >= 0]
        assert same_entity and all(e["event_id"] not in row["probes"][0]["support_groups"][0] for e in same_entity)
        assert all(e["event_id"].startswith("r") for e in row["source_events"])
    assert positions == set(range(8))
    assert len(prefixes) == 4


def test_temporal_sources_remain_chronological_after_support_position_randomization():
    generator = RobustEpisodeGenerator(distractors=15)
    for world in range(24):
        row = generator.episode(world, family="temporal_update", language="en")
        times = [datetime.fromisoformat(e["time"]) for e in row["source_events"]]
        assert times == sorted(times)
        old, new = row["annotations"]["support_positions"]
        assert old < new and times[old].hour == 9 and times[new].hour == 16
        assert row["source_events"][old]["event_id"] in row["probes"][1]["support_groups"][0]
        assert row["source_events"][new]["event_id"] in row["probes"][0]["support_groups"][0]


def test_new_revision_compiles_all_families_languages_and_preserves_explicit_missing_controls():
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=20000)
    generator = RobustEpisodeGenerator(seed=5)
    for family in FAMILIES + ["language_preservation"]:
        for language in ("zh", "en", "cross_language"):
            row = generator.episode(12, family=family, language=language)
            compiled = compiler.compile(row)
            assert compiled.responses and len(compiled.source) == len(row["source_events"])
            if family == "missing_evidence":
                assert compiled.responses[0].memory_transform["drop_events"] == compiled.responses[0].support_groups[0]


def test_legacy_factory_preserves_existing_samples_and_unknown_revisions_are_rejected():
    legacy = make_generator(revision="ab-v1", seed=31)
    assert json.dumps(legacy.episode(12), sort_keys=True) == json.dumps(EpisodeGenerator(seed=31).episode(12), sort_keys=True)
    assert data_identity(AB_DATA_REVISION)["files"]["robust.py"]
    with pytest.raises(ValueError):
        make_generator(revision="unknown")
