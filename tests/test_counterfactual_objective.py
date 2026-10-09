import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_episode_objective import SmallVocabularyTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.revisions import make_generator
from dsrsi.training.episode_objective import EpisodeObjective
from dsrsi.training.runner import TrainConfig, Trainer


def test_counterfactual_loss_only_scores_before_the_first_distinguishing_token():
    model = tiny()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    generator = make_generator(distractors=2)
    a = generator.episode(5, variant=0, family="single_fact", language="en")
    b = generator.episode(5, variant=1, family="single_fact", language="en")
    response = compiler.compile(a).responses[0]
    calls = []
    original = model.forward
    def capture(ids, **kwargs):
        if kwargs.get("labels") is None:
            calls.append(ids.detach().clone())
        return original(ids, **kwargs)
    model.forward = capture
    objective = EpisodeObjective(model, compiler, memory_candidates=1, counterfactual_weight=1)
    result = objective(a, counterfactual_row=b)
    result.loss.backward()
    assert result.counterfactual_loss > 0 and result.teacher_tokens == 0
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])
    assert calls[0].tolist() == [response.prompt_ids]  # code variants differ in their first token
    assert sum(float(p.grad.square().sum()) for p in model.writer.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert result.student_tokens > sum(len(s.input_ids) for s in compiler.compile(a).source) + len(response.input_ids)
    with pytest.raises(ValueError, match="opposite variant"):
        objective(a, counterfactual_row=a)


def test_counterfactual_codec_curriculum_restores_pending_gradients_exactly(tmp_path):
    def make(path, resume=False):
        torch.manual_seed(44)
        return Trainer(tiny(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000), path,
            TrainConfig(student_token_budget=100000, distractors=2, curriculum="facts", memory_candidates=1,
                        counterfactual_weight=1, gradient_accumulation=4), base_identity="counter-tiny", resume=resume)
    full = make(tmp_path / "full")
    full.run(max_micro_steps=5, install_signals=False)
    part = make(tmp_path / "resumed")
    part.run(max_micro_steps=3, install_signals=False)
    resumed = make(tmp_path / "resumed", resume=True)
    resumed.run(max_micro_steps=5, install_signals=False)
    from test_distributed_trainer import equal
    from dsrsi.training.checkpoint import capture_training_state
    def state(trainer):
        return capture_training_state(trainer.model,trainer.optimizer,progress=trainer.progress,
                                      data_cursor=trainer.cursor,life_state=trainer.life)
    equal(state(full),state(resumed))  # Recursively asserts equality; it returns None.
