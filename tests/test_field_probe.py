import pytest
import torch

from dsrsi.evaluation.field_probe import code_labels, linear_field_probe, CODE_CLASSES


def test_frozen_linear_information_probe_generalizes_without_mutating_inputs():
    rng = torch.Generator().manual_seed(48)
    labels = torch.stack([torch.randint(n, (512,), generator=rng) for n in CODE_CLASSES], dim=1)
    embedded = torch.cat([torch.nn.functional.one_hot(labels[:, i], n) for i, n in enumerate(CODE_CLASSES)], dim=1).float()
    features = embedded @ torch.randn(embedded.shape[1], 96, generator=rng) + 0.01 * torch.randn(512, 96, generator=rng)
    original = features.clone()
    predictions, metrics = linear_field_probe(features[:384], labels[:384], features[384:], labels[384:])
    assert metrics["dev"]["full_code_accuracy"] == 1
    assert torch.equal(predictions["dev"], labels[384:])
    assert torch.equal(features, original)


def test_development_labels_never_affect_probe_fitting_or_predictions():
    rng = torch.Generator().manual_seed(2)
    x, z = torch.randn(70, 50, generator=rng), torch.randn(30, 50, generator=rng)
    a = torch.stack([torch.randint(n, (70,), generator=rng) for n in CODE_CLASSES], dim=1)
    b = torch.stack([torch.randint(n, (30,), generator=rng) for n in CODE_CLASSES], dim=1)
    p, _ = linear_field_probe(x, a, z, b)
    q, _ = linear_field_probe(x, a, z, torch.zeros_like(b))
    assert torch.equal(p["train"], q["train"]) and torch.equal(p["dev"], q["dev"])
    assert float((p["dev"] == b).float().mean()) < 0.2


def test_code_labels_require_full_ascii_codes_and_bad_features_are_rejected():
    assert code_labels(["A1-203", "Z9-876"]).tolist() == [[0, 1, 2, 0, 3], [25, 9, 8, 7, 6]]
    for bad in ([], ["1"], ["a1-234"], ["A1-２34"], ["A1_234"]):
        with pytest.raises(ValueError):
            code_labels(bad)
    labels = code_labels(["A1-203", "Z9-876"])
    with pytest.raises(ValueError, match="finite"):
        linear_field_probe(torch.full((2, 3), float("nan")), labels, torch.ones(2, 3), labels)
    with pytest.raises(ValueError, match="shapes"):
        linear_field_probe(torch.ones(2, 3), labels, torch.ones(2, 4), labels)
