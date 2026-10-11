import pytest
import torch

from dsrsi.model.frozen_prefix_reuse import frozen_prefix_reuse


def test_counterfactual_loss_and_all_gradients_match_with_frozen_prefix_reuse():
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.data.revisions import make_generator
    from dsrsi.training.episode_objective import EpisodeObjective
    m = model().train()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    generator = make_generator(revision="ab-v5-indexed-digits", seed=17, distractors=2)
    objective = EpisodeObjective(m, compiler, memory_candidates=1, counterfactual_weight=1)
    a, b = [generator.episode(0, variant=v, family="single_fact", language="en") for v in (0, 1)]
    original = objective(a, counterfactual_row=b)
    original.loss.backward()
    gradients = {n: None if p.grad is None else p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    with frozen_prefix_reuse(m) as stats:
        reused = objective(a, counterfactual_row=b)
        reused.loss.backward()
    assert stats["hits"] == 1 and stats["skipped_layers"] == m.writer_after + 1
    assert torch.equal(original.loss, reused.loss)
    for name, parameter in m.named_parameters():
        assert (gradients[name] is None) == (parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert original.student_tokens == reused.student_tokens and original.teacher_tokens == reused.teacher_tokens


def test_mutated_input_masks_and_parameters_invalidate_cached_values_and_scope_is_restored():
    from test_qwen_bridge import tiny
    m = tiny().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    original_prepare, original_layer = m._prepare, m._layer
    with frozen_prefix_reuse(m) as stats:
        m.source_features(ids, mask)
        m.source_features(ids, mask)
        assert stats["hits"] == 1
        ids[0, 0] = 5
        m.source_features(ids, mask)
        assert stats["hits"] == 1
        mask[0, -1] = False
        m.source_features(ids, mask)
        assert stats["hits"] == 1
        with torch.no_grad():
            m.base.embed_tokens.weight.add_(0.1)
        m.source_features(ids, mask)
        assert stats["hits"] == 1
        with pytest.raises(ValueError, match="nested"):
            with frozen_prefix_reuse(m):
                pass
    assert m._prepare == original_prepare and m._layer == original_layer
    assert not hasattr(m, "_prefix_reuse_active")


def test_trainable_prefix_is_refused_and_exception_cleans_up_scope():
    from test_qwen_bridge import tiny
    m = tiny().eval()
    m.base.embed_tokens.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        with frozen_prefix_reuse(m):
            pass
    m.base.embed_tokens.weight.requires_grad_(False)
    original = m._prepare
    with pytest.raises(RuntimeError, match="interrupt"):
        with frozen_prefix_reuse(m):
            raise RuntimeError("interrupt")
    assert m._prepare == original and not hasattr(m, "_prefix_reuse_active")
