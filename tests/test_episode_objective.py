import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_compiler import BoundaryTokenizer

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.generate import EpisodeGenerator
from dsrsi.training.episode_objective import EpisodeObjective


class SmallVocabularyTokenizer(BoundaryTokenizer):
    def encode(self, text, **kwargs):
        return [token if token == 1 else token % 29 + 2 for token in super().encode(text, **kwargs)]


def test_ab_objective_uses_written_memory_and_reports_all_processed_input():
    model = tiny().train()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    row = EpisodeGenerator(distractors=2).episode(1, family="single_fact", language="en")
    objective = EpisodeObjective(model, compiler, stage="A")
    result = objective(row)
    result.loss.backward()
    assert result.student_tokens > result.supervised_tokens
    assert result.teacher_tokens == 0 and result.covered_queries == 1
    assert sum(float(p.grad.square().sum()) for p in model.writer.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_stage_b_has_real_address_gradient_and_extra_prefix_compute_accounting():
    torch.manual_seed(27)
    model = tiny().train()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    row = EpisodeGenerator(distractors=2).episode(1, family="single_fact", language="en")
    a = EpisodeObjective(model, compiler, stage="A")(row)
    b = EpisodeObjective(model, compiler, stage="B")(row)
    b.loss.backward()
    assert b.student_tokens > a.student_tokens
    assert b.route_loss > 0
    assert sum(float(p.grad.square().sum()) for p in model.address.parameters() if p.grad is not None) > 0


def test_keep_loss_trains_readers_without_erasing_writer_payload():
    model = tiny().train()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    row = EpisodeGenerator(distractors=2).episode(3, family="language_preservation", language="en")
    result = EpisodeObjective(model, compiler, stage="A")(row)
    result.loss.backward()
    assert result.teacher_tokens > 0 and result.keep_kl >= -1e-6
    assert all(p.grad is None for p in model.writer.parameters())
    assert any(p.grad is not None for p in model.readers.parameters())


def test_missing_controls_do_not_force_a_random_value_without_support():
    model = tiny().train()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    row = EpisodeGenerator(distractors=2).episode(5, family="missing_evidence", language="en")
    result = EpisodeObjective(model, compiler, stage="B")(row)
    assert result.covered_queries == 0 and result.route_loss == 0
    assert torch.isfinite(result.loss)
