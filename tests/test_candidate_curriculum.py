from dataclasses import replace
import json

import pytest
import torch

from dsrsi.training.candidate_curriculum import SINGLE_DOUBLE_V1, candidate_count
from dsrsi.training.runner import TrainConfig, Trainer


def test_schedule_preserves_counterfactual_pairs_and_keep_interference():
    counts = []
    for world in range(256):
        a, b = [candidate_count(seed=17, ordinal=world * 2 + variant, family="single_fact",
                                maximum=2, curriculum=SINGLE_DOUBLE_V1) for variant in (0, 1)]
        assert a == b and a in (1, 2)
        counts.append(a)
        assert candidate_count(seed=17, ordinal=world * 2, family="language_preservation",
                               maximum=2, curriculum=SINGLE_DOUBLE_V1) == 2
    assert 80 < counts.count(1) < 176
    assert candidate_count(seed=17, ordinal=4, family="single_fact", maximum=8) == 8


def config():
    return TrainConfig(data_revision="ab-v6-natural-keep", student_token_budget=100000,
                       gradient_accumulation=6, distractors=2, seed=17, curriculum="facts",
                       memory_candidates=2, counterfactual_weight=1, keep_fraction=.4,
                       keep_weight=1, reuse_frozen_prefix=True, memory_candidate_curriculum=SINGLE_DOUBLE_V1)


@pytest.mark.parametrize("changes", [{"stage": "B"}, {"memory_candidates": 1}, {"curriculum": "mixed"},
                                    {"data_revision": "ab-v5-indexed-digits"},
                                    {"memory_candidate_curriculum": "unknown"}])
def test_invalid_course_is_not_silently_reinterpreted(changes):
    with pytest.raises(ValueError, match="single/double"):
        replace(config(), **changes)


def build(path, *, resume=False, cfg=None, initialize_from=None):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    torch.manual_seed(23)
    return Trainer(model(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000), path,
                   cfg or config(), base_identity="candidate-course-tiny", resume=resume,
                   initialize_from=initialize_from)


def test_mixed_course_full_state_resume_preserves_schedule_gradients_and_ledger(tmp_path):
    from dsrsi.training.topology import require_equal
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    a = build(tmp_path / "continuous")
    a.run(max_micro_steps=24, install_signals=False)
    b = build(tmp_path / "interrupted")
    b.run(max_micro_steps=5, install_signals=False)
    assert b.progress.accumulation_index == 5
    c = build(tmp_path / "interrupted", resume=True)
    c.run(max_micro_steps=24, install_signals=False)
    def payload(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress,
                                      data_cursor=t.cursor, life_state=t.life)
    require_equal(payload(a), payload(c), "mixed-course-complete-resume")
    reference = build(tmp_path / "reference", cfg=replace(config(), reuse_frozen_prefix=False))
    reference.run(max_micro_steps=24, install_signals=False)
    require_equal(payload(a), payload(reference), "mixed-course-prefix-reference")
    rows = [json.loads(line) for line in (a.run_dir / "events.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["event"] == "micro"]
    assert {r["memory_candidates_used"] for r in rows if r["family"] == "single_fact"} == {1, 2}
    assert all(r["memory_candidates_used"] == 2 for r in rows if r["family"] == "language_preservation")
    assert a.identity["candidate_curriculum"]["version"] == SINGLE_DOUBLE_V1
    with pytest.raises(CheckpointError, match="identity"):
        build(tmp_path / "interrupted", resume=True,
              cfg=replace(config(), memory_candidate_curriculum=None))


def test_new_course_can_initialize_from_double_course_but_does_not_resume_it(tmp_path):
    source = build(tmp_path / "source", cfg=replace(config(), memory_candidate_curriculum=None))
    source.run(max_micro_steps=6, install_signals=False)
    pointer = (source.store.root / "latest.json").read_bytes()
    assert "candidate_curriculum" not in source.identity
    assert "memory_candidate_curriculum" not in source.identity["train_config"]
    target = build(tmp_path / "target", initialize_from=source.run_dir)
    assert target.progress.micro_step == 0 and not target.optimizer.state
    assert target.cursor["next_ordinal"] == 0
    for name, tensor in source.model.state_dict().items():
        assert torch.equal(tensor, target.model.state_dict()[name]), name
    assert (source.store.root / "latest.json").read_bytes() == pointer


def test_actual_answer_and_counterfactual_forwards_receive_the_scheduled_records(tmp_path):
    trainer = build(tmp_path)
    original = trainer.model.forward
    observed = []

    def capture(*args, **kwargs):
        if kwargs.get("records") is not None:
            observed.extend(len(records) for records in kwargs["records"])
        return original(*args, **kwargs)

    trainer.model.forward = capture
    # Fixed seed17 fixtures cover both counterfactual pairs and a keep query.
    for ordinal, expected_family, count in [(0, "single_fact", 1), (1, "single_fact", 1),
                                            (4, "language_preservation", 2),
                                            (6, "single_fact", 2), (7, "single_fact", 2)]:
        observed.clear()
        row, _, _ = trainer._forward_episode(ordinal)
        assert row["family"] == expected_family
        assert observed == [count] * (1 if expected_family == "language_preservation" else 3)


def test_resume_cli_keeps_the_explicit_candidate_schedule():
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/verify_cuda_resume.py"
    spec = importlib.util.spec_from_file_location("verify_cuda_curriculum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.train_arguments("base", "run", asdict(config()), 240)
    assert args[args.index("--memory-candidate-curriculum") + 1] == SINGLE_DOUBLE_V1
    assert args.count("--resume") == 1
