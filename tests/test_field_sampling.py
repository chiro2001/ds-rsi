from collections import Counter

import pytest

from dsrsi.data.revisions import make_generator
from dsrsi.evaluation.sampling import stratified_code_worlds


@pytest.mark.parametrize("revision", ["ab-v4-fields-and-language", "ab-v5-indexed-digits"])
def test_world_strata_cover_each_character_without_duplicate_or_changed_evidence(revision):
    generator = make_generator(revision=revision, split="dev", seed=20261010)
    selected = stratified_code_worlds(generator, worlds_per_position=4)
    assert len(set(selected)) == len(selected) == 20
    assert selected == sorted(selected)
    assert selected == stratified_code_worlds(generator, worlds_per_position=4, language="zh")
    counts = Counter(generator.episode(i, family="single_fact", language="en")["annotations"]["counterfactual_character_position"]
                     for i in selected)
    assert counts == {0: 4, 1: 4, 3: 4, 4: 4, 5: 4}
    for world in selected:
        a, b = [generator.episode(world, variant=v, family="single_fact", language="en") for v in (0, 1)]
        assert a["probes"][0]["query"] == b["probes"][0]["query"]
        assert a["source_events"] != b["source_events"]


def test_bad_or_unannotated_world_strata_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        stratified_code_worlds(make_generator(), worlds_per_position=0)
    with pytest.raises(ValueError, match="annotations"):
        stratified_code_worlds(make_generator(revision="ab-v1"), worlds_per_position=1)
