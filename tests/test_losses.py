import math

import pytest
import torch
import torch.nn.functional as F

from dsrsi.training.losses import (
    response_cross_entropy, response_cross_entropy_from_hidden,
    routing_scores, required_support_loss, topk_tail_kl,
)


def test_response_loss_shifts_targets_and_normalizes_each_event():
    logits = torch.randn(2, 6, 9, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 3, 2, -100], [-100, -100, -100, 7, -100, -100]])
    result = response_cross_entropy(logits, labels)
    want = (F.cross_entropy(logits[0, 1:4], torch.tensor([4, 3, 2]))
            + F.cross_entropy(logits[1, 2:3], torch.tensor([7]))) / 2
    torch.testing.assert_close(result.mean(), want)
    assert result.count == 2
    result.mean().backward()
    assert logits.grad[0, 1].abs().sum() > 0  # answer first token prediction
    assert logits.grad[0, 3].abs().sum() > 0  # end marker prediction
    assert logits.grad[0, 0].abs().sum() == 0  # prompt target
    assert logits.grad[0, 4:].abs().sum() == 0  # after end/padding


def test_unlabeled_rows_have_zero_count_and_connected_zero_gradient():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    result = response_cross_entropy(logits, torch.full((2, 5), -100))
    assert result.count == 0
    result.mean().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_bf16_cross_entropy_reduces_in_fp32():
    logits = torch.randn(2, 6, 9, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 4, 3, -100], [-100, -100, 1, -100, -100, -100]])
    value = response_cross_entropy(logits, labels).mean()
    assert value.dtype == torch.float32
    torch.testing.assert_close(value, response_cross_entropy(logits.float(), labels).mean())
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_unused_fp16_logits_and_masked_routing_scores_do_not_poison_zero():
    logits = torch.full((2, 5, 7), 30000.0, dtype=torch.float16, requires_grad=True)
    value = response_cross_entropy(logits, torch.full((2, 5), -100)).mean()
    assert value == 0 and torch.isfinite(value)
    value.backward()
    scores = torch.tensor([[2.0, 0.0, -torch.inf]], requires_grad=True)
    loss = required_support_loss(scores, torch.tensor([[[True, False, False]]]),
                                 torch.tensor([[True, True, False]])).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad[0, 2] == 0


def test_chunked_head_matches_full_head_loss_and_gradients():
    hidden = torch.randn(3, 8, 5, requires_grad=True)
    weight = torch.randn(11, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4, -100, -100],
                           [-100, -100, -100, 5, 6, -100, -100, -100],
                           [-100] * 8])
    full = response_cross_entropy(F.linear(hidden, weight), labels).mean()
    grads = torch.autograd.grad(full, (hidden, weight))
    h = hidden.detach().clone().requires_grad_()
    w = weight.detach().clone().requires_grad_()
    chunked = response_cross_entropy_from_hidden(h, w, labels, chunk_positions=2)
    chunked_grads = torch.autograd.grad(chunked.mean(), (h, w))
    torch.testing.assert_close(full, chunked.mean())
    for a, b in zip(grads, chunked_grads):
        torch.testing.assert_close(a, b)
    assert chunked.count == 2


def test_multi_address_scores_keep_strong_address_instead_of_vector_mean():
    q = torch.tensor([[1.0, 0.0]], requires_grad=True)
    keys = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)
    scores = routing_scores(q, keys, temperature=0.1)
    assert scores[0, 0] > scores[0, 1]
    scores.sum().backward()
    assert q.grad is not None and keys.grad is not None


def test_multiple_required_groups_cannot_be_satisfied_by_only_one_record():
    positives = torch.tensor([[[True, False, False], [False, True, False]]])
    eligible = torch.ones(1, 3, dtype=torch.bool)
    one_found = torch.tensor([[9.0, -4.0, 3.0]], requires_grad=True)
    all_found = torch.tensor([[9.0, 9.0, 3.0]], requires_grad=True)
    bad = required_support_loss(one_found, positives, eligible)
    good = required_support_loss(all_found, positives, eligible)
    assert bad.mean() > good.mean()
    bad.mean().backward()
    assert one_found.grad[0, 1] < 0  # missing required group is rewarded when raised
    assert one_found.grad[0, 2] > 0


def test_other_required_groups_are_not_false_negatives():
    scores = torch.tensor([[2.0, 3.0, 0.0]])
    groups = torch.tensor([[[True, False, False], [False, True, False]]])
    got = required_support_loss(scores, groups, torch.ones(1, 3, dtype=torch.bool)).mean()
    want = (torch.logsumexp(torch.tensor([2.0, 0.0]), 0) - 2
            + torch.logsumexp(torch.tensor([3.0, 0.0]), 0) - 3) / 2
    torch.testing.assert_close(got, want)
    with pytest.raises(ValueError, match="valid positive"):
        required_support_loss(scores, groups, torch.ones(1, 3, dtype=torch.bool),
                              torch.tensor([[False, True, True]]))


def test_future_or_cross_stream_positives_are_rejected():
    with pytest.raises(ValueError, match="ineligible"):
        required_support_loss(torch.zeros(1, 2), torch.tensor([[[True, False]]]),
                              torch.tensor([[False, True]]))


def test_topk_tail_kl_matches_coarse_probability_distribution_and_detaches_teacher():
    p = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    q = torch.tensor([[0.2, 0.2, 0.3, 0.3]], requires_grad=True)
    loss = topk_tail_kl(q.log(), p.log(), topk=2)
    want = 0.4 * math.log(0.4 / 0.2) + 0.3 * math.log(0.3 / 0.2) + 0.3 * math.log(0.3 / 0.6)
    torch.testing.assert_close(loss, torch.tensor(want))
    loss.backward()
    assert p.grad is None
    assert q.grad is not None


def test_topk_tail_kl_keeps_tail_mass_and_is_stable_for_peaked_student():
    teacher = torch.zeros(1, 5)
    student = torch.tensor([[1000.0, 0, 0, 0, 0]], requires_grad=True)
    value = topk_tail_kl(student, teacher, topk=1)
    assert torch.isfinite(value) and value > 100
    value.backward()
    assert torch.isfinite(student.grad).all()
    torch.testing.assert_close(topk_tail_kl(teacher, teacher, topk=2), torch.tensor(0.0))
