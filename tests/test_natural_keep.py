import copy

import pytest
import torch

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.indexed_digits import IndexedDigitEpisodeGenerator
from dsrsi.data.natural_keep import (AB_NATURAL_KEEP_REVISION, NaturalKeepEpisodeGenerator,
                                     continuation_task)
from dsrsi.data.revisions import data_identity
from dsrsi.training.episode_objective import EpisodeObjective
from test_episode_objective import SmallVocabularyTokenizer
from test_lookup_gates import model


def test_v6_fact_and_legacy_keep_rows_match_v5_except_versioned_identity():
    old = IndexedDigitEpisodeGenerator(seed=17, split="train", distractors=2)
    new = NaturalKeepEpisodeGenerator(seed=17, split="train", distractors=2)
    for ordinal in range(16):
        for variant in (0, 1):
            a = old.episode(ordinal, variant=variant, family="single_fact", language="en")
            b = new.episode(ordinal, variant=variant, family="single_fact", language="en")
            for field in ("episode_id", "world_id", "counterfactual_group_id"):
                b[field] = "v5-" + b[field].removeprefix("v6-")
            b["data_revision"] = a["data_revision"]
            assert a == b


def test_natural_keep_rows_are_causal_and_cover_both_wall_time_conditions():
    generator = NaturalKeepEpisodeGenerator(seed=17, split="train", distractors=2)
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=2000)
    counts, time_conditions = {"natural": 0, "legacy": 0}, set()
    for ordinal in range(64):
        row = generator.episode(ordinal, family="language_preservation", language="en")
        if not row["annotations"].get("natural_keep"):
            counts["legacy"] += 1
            continue
        counts["natural"] += 1
        time_conditions.add(row["annotations"]["keep_wall_time_known"])
        assert row["annotations"]["keep_objective"] == "base_kl_only_v1"
        assert row["probes"][0]["support_groups"] == []
        compiled = compiler.compile(row)
        response = compiled.responses[0]
        assert not response.write_enabled
        assert all(label == -100 for label in response.labels[:response.prompt_length])
        assert sum(label != -100 for label in response.labels) > 50
        if not row["annotations"]["keep_wall_time_known"]:
            assert all(span.wall_time is None for span in compiled.source)
            assert response.wall_time is None
        assert row == generator.episode(ordinal, family="language_preservation", language="en")
    assert 16 <= counts["natural"] <= 48 and counts["legacy"] > 0
    assert time_conditions == {False, True}
    assert continuation_task(17, "train", 4, "en") != continuation_task(17, "dev", 4, "en")


def natural_row():
    generator = NaturalKeepEpisodeGenerator(seed=17, split="train", distractors=2)
    return next(row for n in range(64) if (row := generator.episode(
        n, family="language_preservation", language="en"))["annotations"].get("natural_keep"))


def test_continuation_trains_only_base_kl_and_does_not_update_writer():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    module = model()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=2000)
    row = natural_row()
    objective = EpisodeObjective(module, compiler, memory_candidates=1, keep_weight=0.5)
    output = objective(row)
    assert output.loss.item() == pytest.approx(0.5 * output.keep_kl)
    assert output.answer_loss > 0  # CE is diagnostic, absent from the optimized loss
    output.loss.backward()
    assert all(parameter.grad is None for parameter in module.writer.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in module.readers.parameters())
    module.zero_grad(set_to_none=True)
    legacy = copy.deepcopy(row)
    legacy["annotations"].pop("keep_objective")
    previous = objective(legacy)
    assert previous.loss.item() == pytest.approx(0.5 * (previous.answer_loss + previous.keep_kl))
    invalid = copy.deepcopy(row)
    invalid["annotations"]["keep_objective"] = "unknown"
    with pytest.raises(ValueError, match="unknown language"):
        objective(invalid)


def test_natural_keep_identity_pins_generator_and_objective_implementation():
    identity = data_identity(AB_NATURAL_KEEP_REVISION)
    assert identity["revision"] == AB_NATURAL_KEEP_REVISION
    assert identity["keep_objective"] == "half-continuation-base-top32-tail-kl-only-v1"
    assert {"natural_keep.py", "episode_objective.py"} <= identity["files"].keys()
    assert "episode_objective.py" not in data_identity("ab-v5-indexed-digits")["files"]
