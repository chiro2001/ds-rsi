from collections import Counter

from dsrsi.data.revisions import make_generator, data_identity
from dsrsi.data.robust_order import RandomOrderEpisodeGenerator, AB_ORDER_REVISION
from dsrsi.data.robust_fields import AB_FIELDS_REVISION
from dsrsi.data.compiler import EpisodeCompiler
from test_compiler import BoundaryTokenizer


def test_counterfactual_changes_each_code_character_without_changing_other_evidence():
    g = make_generator(revision=AB_FIELDS_REVISION, seed=11, distractors=7)
    positions = Counter()
    for world in range(200):
        a = g.episode(world, variant=0, family="single_fact", language="en")
        b = g.episode(world, variant=1, family="single_fact", language="en")
        actual, other = a["probes"][0]["target"], b["probes"][0]["target"]
        changed = [i for i, (x, y) in enumerate(zip(actual, other)) if x != y]
        assert changed == [a["annotations"]["counterfactual_character_position"]]
        positions.update(changed)
        assert a["probes"][0]["query"] == b["probes"][0]["query"]
        support = set(a["probes"][0]["support_groups"][0])
        assert [r for r in a["source_events"] if r["event_id"] not in support] == [
            r for r in b["source_events"] if r["event_id"] not in support]
    assert set(positions) == {0, 1, 3, 4, 5}


def test_general_keep_tasks_compile_and_do_not_depend_on_counterfactual_variant():
    g = make_generator(revision=AB_FIELDS_REVISION, seed=21, distractors=3)
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=10000)
    families = set()
    for world in range(100):
        a = g.episode(world, variant=0, family="language_preservation", language="zh")
        b = g.episode(world, variant=1, family="language_preservation", language="zh")
        assert a["probes"] == b["probes"]
        families.add(a["annotations"]["keep_task_family"])
        assert compiler.compile(a).responses and a["probes"][0]["support_groups"] == []
    assert families == {"arithmetic", "reading", "logic", "code", "format", "knowledge"}


def test_existing_v3_source_and_fingerprint_are_preserved():
    assert make_generator(revision=AB_ORDER_REVISION).episode(1) == RandomOrderEpisodeGenerator().episode(1)
    assert "robust_fields.py" not in data_identity(AB_ORDER_REVISION)["files"]
    assert "general_keep.py" in data_identity(AB_FIELDS_REVISION)["files"]
