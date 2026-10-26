from dataclasses import asdict, replace
import copy

import pytest
import torch

from dsrsi.data.fixed_pair import FIXED_PAIR_REVISION
from dsrsi.evaluation.fixed_records import evidence_teacher_response, sparse_distribution
from dsrsi.training.evidence_distillation import (FixedPairDistillationObjective, exact_teacher_match,
                                                  sparse_teacher_kl)
from dsrsi.training.losses import topk_tail_kl


def config():
    from test_memory_trainable import frozen_config
    return replace(frozen_config(), data_revision=FIXED_PAIR_REVISION, evidence_distillation_weight=1)


def oracle_fixture(self, row, probe):
    response = evidence_teacher_response(self.compiler, row, probe)
    suffix = response.input_ids[response.prompt_length:]
    vocabulary = self.model.head.weight.shape[0]
    logits = torch.zeros(len(suffix), vocabulary)
    logits.scatter_(1, torch.tensor(suffix)[:, None], 3.0)
    sparse = sparse_distribution(logits)
    distributions = [tuple(v[i:i + 1] for v in sparse) for i in range(len(suffix))]
    return dict(answer=dict(stopped_on_end=True, token_ids=suffix[:-1]),
                distributions=distributions, input_tokens=len(response.prompt_ids) + len(suffix) - 1), True


def test_sparse_teacher_loss_and_gradients_match_full_topk_tail_reference():
    torch.manual_seed(13)
    a = torch.randn(7, 65, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    teacher = torch.randn_like(a).requires_grad_(True)
    expected = topk_tail_kl(a, teacher)
    actual = sparse_teacher_kl(b, *sparse_distribution(teacher))
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(a.grad, b.grad) and teacher.grad is None
    with pytest.raises(ValueError):
        sparse_teacher_kl(b, torch.zeros(7, 0, dtype=torch.long), torch.zeros(7, 0), torch.zeros(7, 1))


def test_teacher_screen_requires_actual_free_tokens_and_end_not_only_decoded_score():
    from types import SimpleNamespace
    response = SimpleNamespace(input_ids=[8, 9, 3, 4, 1], prompt_length=2)
    result = dict(answer=dict(stopped_on_end=True, token_ids=[3, 4], exact_match=True),
                  distributions=[None] * 3)
    assert exact_teacher_match(result, response, 1)
    for changes in (dict(stopped_on_end=False), dict(token_ids=[3, 5])):
        bad = copy.deepcopy(result)
        bad["answer"].update(changes)
        assert not exact_teacher_match(bad, response, 1)
    result["distributions"].pop()
    assert not exact_teacher_match(result, response, 1)


def test_pair_uses_identical_candidate_order_for_both_queries_and_counts_teacher_cost(monkeypatch, tmp_path):
    from test_candidate_curriculum import build
    monkeypatch.setattr(FixedPairDistillationObjective, "_teacher", oracle_fixture)
    t = build(tmp_path, cfg=config())
    # Select an actual fact row in the deterministic trainer mixture.
    for o in range(100):
        row, _, result = t._forward_episode(o)
        if row["family"] == "single_fact" and t._candidate_count(o, row["family"]) == 2:
            break
    assert result.queries == 2 and result.covered_queries == 2
    assert result.evidence_teacher_attempted == result.evidence_teacher_accepted == 2
    assert result.evidence_loss > 0 and result.teacher_tokens > 0
    ids = row["annotations"]["fixed_memory_candidate_ids"]
    assert result.fixed_candidates_per_query == [ids, ids]
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in t.model.writer.parameters())
    assert all(p.grad is None for p in t.model.base.parameters())


def test_rejected_teachers_do_not_add_loss_but_actual_teacher_compute_is_counted(monkeypatch, tmp_path):
    from test_candidate_curriculum import build
    from dsrsi.data.revisions import make_generator
    t = build(tmp_path, cfg=config())
    g = make_generator(revision=FIXED_PAIR_REVISION, seed=17, distractors=2)
    row, opposite = [g.episode(0, family="single_fact", language="en", variant=v) for v in (0, 1)]
    def rejected(self, row, probe):
        value, _ = oracle_fixture(self, row, probe)
        return value, False
    monkeypatch.setattr(FixedPairDistillationObjective, "_teacher", rejected)
    options = dict(stage="A", memory_candidates=2, counterfactual_weight=1, keep_weight=1,
                   reuse_frozen_prefix=True)
    a = FixedPairDistillationObjective(t.model, t.compiler, evidence_distillation_weight=1, **options)(row, counterfactual_row=opposite)
    b = FixedPairDistillationObjective(t.model, t.compiler, evidence_distillation_weight=0, **options)(row, counterfactual_row=opposite)
    assert torch.equal(a.loss, b.loss) and a.evidence_loss == b.evidence_loss == 0
    assert a.evidence_teacher_attempted == 2 and a.evidence_teacher_accepted == 0
    assert a.teacher_tokens > 0 and b.teacher_tokens == 0 and a.student_tokens == b.student_tokens


def test_preservation_retains_original_loss_and_all_gradients(monkeypatch, tmp_path):
    from test_candidate_curriculum import build
    from dsrsi.training.episode_objective import EpisodeObjective
    t = build(tmp_path, cfg=config())
    row = t.generator.episode(0, family="language_preservation", language="en")
    def unexpected(*args):
        raise AssertionError("preservation must not query evidence teacher")
    monkeypatch.setattr(FixedPairDistillationObjective, "_teacher", unexpected)
    options = dict(stage="A", memory_candidates=2, keep_weight=1, reuse_frozen_prefix=True)
    old = EpisodeObjective(t.model, t.compiler, **options)(row)
    old.loss.backward()
    gradients = {n: None if p.grad is None else p.grad.clone() for n, p in t.model.named_parameters()}
    t.model.zero_grad(set_to_none=True)
    new = FixedPairDistillationObjective(t.model, t.compiler, evidence_distillation_weight=1, **options)(row)
    new.loss.backward()
    assert torch.equal(old.loss, new.loss) and old.teacher_tokens == new.teacher_tokens
    for name, parameter in t.model.named_parameters():
        assert (parameter.grad is None) == (gradients[name] is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, gradients[name]), name


def test_fixed_pair_complete_resume_prefix_reference_and_identity(monkeypatch, tmp_path):
    from test_candidate_curriculum import build
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    from dsrsi.training.topology import require_equal
    monkeypatch.setattr(FixedPairDistillationObjective, "_teacher", oracle_fixture)
    cfg = config()
    a = build(tmp_path / "continuous", cfg=cfg)
    a.run(max_micro_steps=24, install_signals=False)
    b = build(tmp_path / "interrupted", cfg=cfg)
    b.run(max_micro_steps=5, install_signals=False)
    assert b.progress.accumulation_index == 5
    c = build(tmp_path / "interrupted", cfg=cfg, resume=True)
    c.run(max_micro_steps=24, install_signals=False)
    reference = build(tmp_path / "reference", cfg=replace(cfg, reuse_frozen_prefix=False))
    reference.run(max_micro_steps=24, install_signals=False)
    def payload(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress,
                                      data_cursor=t.cursor, life_state=t.life)
    require_equal(payload(a), payload(c), "fixed-pair-full-state-resume")
    require_equal(payload(a), payload(reference), "fixed-pair-prefix-reference")
    assert a.progress.teacher_tokens > 0 and "evidence_distillation" in a.identity
    with pytest.raises(CheckpointError, match="identity"):
        build(tmp_path / "interrupted", cfg=replace(cfg, evidence_distillation_weight=0), resume=True)


def test_versioned_initialization_and_cli_preserve_new_objective(tmp_path):
    from test_candidate_curriculum import build
    from test_memory_trainable import frozen_config
    source = build(tmp_path / "old", cfg=frozen_config())
    source.run(max_micro_steps=6, install_signals=False)
    target = build(tmp_path / "new", cfg=config(), initialize_from=source.run_dir)
    assert target.progress.micro_step == 0 and not target.optimizer.state
    for key, value in source.model.state_dict().items():
        assert torch.equal(value, target.model.state_dict()[key]), key
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/verify_cuda_resume.py"
    spec = importlib.util.spec_from_file_location("fixed_pair_resume_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.train_arguments("base", "run", asdict(config()), 240)
    assert args[args.index("--evidence-distillation-weight") + 1] == "1"
    assert args[args.index("--data-revision") + 1] == FIXED_PAIR_REVISION


@pytest.mark.parametrize("changes", [{"stage": "B"}, {"memory_trainable_policy": None},
                                      {"memory_candidate_curriculum": None}, {"reader_support_weight": 1},
                                      {"evidence_distillation_weight": float("nan")}])
def test_invalid_new_course_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)
