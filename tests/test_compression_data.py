from collections import Counter

import pytest

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.compression import (COMPRESSION_DEPTHS, COMPRESSION_REVISION, compression_groups,
                                    compression_plan, leaves_for_depth)
from dsrsi.data.revisions import data_identity, make_generator
from test_compiler import BoundaryTokenizer


def generator(split="train", seed=20261010):
    return make_generator(revision=COMPRESSION_REVISION, seed=seed, split=split, distractors=0)


def test_chain_shape_is_three_d_plus_one_with_support_in_the_first_four():
    g = generator()
    depths = Counter()
    support_positions = set()
    for world in range(24):
        row = g.episode(world)
        plan = compression_plan(row)
        assert len(row["source_events"]) == leaves_for_depth(plan["depth"])
        support_positions.add(plan["leaves"].index("support"))
        assert plan["groups"][0] == (0, 4)
        assert plan["support_group"] == 0
        assert plan["support_event"] in plan["leaves"][:4]
        assert row["probes"][0]["support_groups"] == [["support"]]
        for begin, end in plan["groups"][1:]:
            assert end - begin == 3
        depths[plan["depth"]] += 1
    assert set(depths) == set(COMPRESSION_DEPTHS)
    assert support_positions == {0, 1, 2, 3}


def test_groups_tile_the_leaf_sequence_without_overlap_after_the_first_four():
    for depth in COMPRESSION_DEPTHS:
        groups = compression_groups(depth)
        leaves = leaves_for_depth(depth)
        assert len(groups) == depth
        assert groups[0] == (0, 4)
        covered = []
        for begin, end in groups[1:]:
            covered.extend(range(begin, end))
        assert covered == list(range(4, leaves))


def test_counterfactual_variant_changes_only_the_support_content():
    g = generator()
    for world in range(6):
        a, b = g.episode(world, variant=0), g.episode(world, variant=1)
        assert a["probes"][0]["query"] == b["probes"][0]["query"]
        assert a["probes"][0]["target"] != b["probes"][0]["target"]
        actual = [event for event in a["source_events"] if event["event_id"] != "support"]
        other = [event for event in b["source_events"] if event["event_id"] != "support"]
        assert actual == other
        changed = [i for i, (x, y) in enumerate(zip(a["probes"][0]["target"], b["probes"][0]["target"])) if x != y]
        assert changed == [0]


def test_compiled_chain_keeps_the_support_inside_the_first_summary_group():
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=10000)
    g = generator()
    for world in range(4):
        row = g.episode(world)
        episode = compiler.compile(row)
        plan = compression_plan(row)
        support_span = next(span for span in episode.source if span.event_id == "support")
        first = [span.event_id for span in episode.source[:4]]
        assert support_span.event_id in first
        assert len(episode.source) == leaves_for_depth(plan["depth"])
        response = episode.responses[0]
        assert response.write_enabled is False
        assert all(token == -100 for token in response.labels[:response.prompt_length])
        supervised = sum(token != -100 for token in response.labels)
        # Target characters plus the end marker; BoundaryTokenizer keeps each
        # character distinct, so this stays a tight upper bound.
        assert 0 < supervised <= len(row["probes"][0]["target"]) + 1
        assert set(response.memory_transform) == set()


def test_dev_and_test_add_retention_probes_with_true_targets():
    for split in ("dev", "test"):
        row = generator(split=split).episode(3)
        probes = {probe["probe_id"]: probe for probe in row["probes"]}
        assert set(probes) == {"p1", "p_latest", "p_first_negative"}
        latest_id = f"neg-{row['annotations']['leaf_count'] - 2}"
        assert probes["p_latest"]["support_groups"] == [[latest_id]]
        assert probes["p_first_negative"]["support_groups"] == [["neg-0"]]
        for probe in probes.values():
            assert probe["write_enabled"] is False
            assert probe["target"]
    assert [probe["probe_id"] for probe in generator().episode(3)["probes"]] == ["p1"]


def test_revision_identity_carries_the_compression_module_hash():
    identity = data_identity(COMPRESSION_REVISION)
    assert identity["revision"] == COMPRESSION_REVISION
    assert "compression.py" in identity["files"]
    assert identity["files"]["compression.py"]
    assert identity["files"] == data_identity("ab-v4-fields-and-language")["files"] | \
        {"compression.py": identity["files"]["compression.py"]}


def test_unknown_depth_or_extra_distractors_are_rejected():
    with pytest.raises(ValueError, match="depth"):
        leaves_for_depth(3)
    with pytest.raises(ValueError, match="distractors"):
        make_generator(revision=COMPRESSION_REVISION, seed=1, split="train", distractors=7)
    with pytest.raises(ValueError, match="depth"):
        generator().episode(0, depth=3)
    with pytest.raises(ValueError, match="compression_fact"):
        generator().episode(0, family="single_fact")


def test_splits_have_distinct_entities_and_group_annotations_are_validated():
    train, dev = generator().episode(0), generator(split="dev").episode(0)
    assert "-train-" in train["probes"][0]["query"]
    assert "-dev-" in dev["probes"][0]["query"]
    train["annotations"]["compression_groups"] = [(0, 3)]
    with pytest.raises(ValueError, match="groups"):
        compression_plan(train)
