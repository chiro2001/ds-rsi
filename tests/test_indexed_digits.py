from collections import Counter

from dsrsi.data.revisions import make_generator, data_identity
from dsrsi.data.compiler import EpisodeCompiler
from test_compiler import BoundaryTokenizer

REV = "ab-v5-indexed-digits"


def test_auxiliary_targets_are_observed_digits_and_pairs_never_put_answer_in_query():
    old = make_generator(revision="ab-v4-fields-and-language", seed=17, distractors=7)
    g = make_generator(revision=REV, seed=17, distractors=7)
    counts = Counter()
    compiler = EpisodeCompiler(BoundaryTokenizer(), max_query_tokens=10000)
    for world in range(200):
        for language in ("zh", "en", "cross_language"):
            a, b = [g.episode(world, variant=v, family="single_fact", language=language) for v in (0, 1)]
            original = old.episode(world, variant=0, family="single_fact", language=language)
            assert a["source_events"] == original["source_events"]
            assert a["probes"][0]["query"] == b["probes"][0]["query"]
            task = a["annotations"]["codec_task"]
            counts[task] += 1
            if task == "indexed_character":
                position = a["annotations"]["answer_character_position"]
                assert position in (3, 4, 5)
                assert a["probes"][0]["target"] == original["probes"][0]["target"][position]
                assert a["probes"][0]["target"] != b["probes"][0]["target"]
                assert len(a["probes"]) == 1
            assert compiler.compile(a).responses[0].write_enabled is False
    assert 0.2 < counts["indexed_character"] / sum(counts.values()) < 0.4


def test_heldout_default_is_complete_code_and_indexed_view_is_explicit():
    for split in ("dev", "test"):
        full = make_generator(revision=REV, split=split)
        indexed = make_generator(revision=REV, split=split, probe_view="indexed")
        for world in range(30):
            a = full.episode(world, family="single_fact", language="en")
            b = indexed.episode(world, family="single_fact", language="en")
            assert a["annotations"]["codec_task"] == "full_code" and len(a["probes"][0]["target"]) == 6
            assert b["annotations"]["codec_task"] == "indexed_character" and len(b["probes"][0]["target"]) == 1
            assert a["source_events"] == b["source_events"]


def test_ordinary_keep_and_old_data_fingerprints_are_unchanged():
    old = make_generator(revision="ab-v4-fields-and-language", seed=13)
    new = make_generator(revision=REV, seed=13)
    for i in range(30):
        a = old.episode(i, family="language_preservation", language="en")
        b = new.episode(i, family="language_preservation", language="en")
        assert a["probes"] == b["probes"] and a["source_events"] == b["source_events"]
    assert "indexed_digits.py" not in data_identity("ab-v4-fields-and-language")["files"]
    assert "indexed_digits.py" in data_identity(REV)["files"]


def test_indexed_curriculum_pending_gradients_restore_without_changing_old_run(tmp_path):
    import torch
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from test_distributed_trainer import equal
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    import pytest
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    config = TrainConfig(data_revision=REV, student_token_budget=100000, distractors=2, seed=17,
                         curriculum="facts", memory_candidates=1, counterfactual_weight=1, keep_fraction=0.4, keep_weight=0.5)
    def make(name, resume=False):
        return Trainer(model(), compiler, tmp_path / name, config, base_identity="indexed-tiny", resume=resume)
    full = make("full")
    full.run(max_micro_steps=5, install_signals=False)
    part = make("part")
    part.run(max_micro_steps=3, install_signals=False)
    assert part.progress.accumulation_index == 3
    resumed = make("part", resume=True)
    resumed.run(max_micro_steps=5, install_signals=False)
    def state(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress, data_cursor=t.cursor, life_state=t.life)
    equal(state(full), state(resumed))
    with pytest.raises(CheckpointError, match="identity"):
        Trainer(model(), compiler, tmp_path / "part",
                TrainConfig(student_token_budget=100000, distractors=2, seed=17, curriculum="facts"),
                base_identity="indexed-tiny", resume=True)
