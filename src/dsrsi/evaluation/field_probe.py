"""Train-only normalized linear probes of frozen source representations.

A successful held-out probe is a lower bound on accessible information. A
failed probe does not prove the information is absent or impossible to decode.
These heads are diagnostic and are never installed in the memory model.
"""

import torch
import torch.nn.functional as F


CODE_POSITIONS = (0, 1, 3, 4, 5)
CODE_CLASSES = (26, 10, 10, 10, 10)


def code_labels(codes):
    rows = []
    for code in codes:
        if (len(code) != 6 or code[2] != "-" or not "A" <= code[0] <= "Z"
                or any(char not in "0123456789" for char in code[1] + code[3:])):
            raise ValueError("probe labels require complete A0-000 style codes")
        rows.append([ord(code[0]) - 65] + [int(code[p]) for p in CODE_POSITIONS[1:]])
    if not rows:
        raise ValueError("empty probe labels")
    return torch.tensor(rows, dtype=torch.long)


def linear_field_probe(train_features, train_labels, dev_features, dev_labels, *, ridge=0.1):
    """Ridge least-squares classifiers; no development labels enter fitting."""
    x, z = [value.detach().to(device="cpu", dtype=torch.float64) for value in (train_features, dev_features)]
    a, b = [value.detach().to(device="cpu") for value in (train_labels, dev_labels)]
    if (x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]
            or x.shape[0] < 2 or z.shape[0] < 1 or x.shape[1] < 1
            or a.shape != (len(x), 5) or b.shape != (len(z), 5)
            or a.dtype != torch.long or b.dtype != torch.long):
        raise ValueError("invalid field-probe feature/label shapes or types")
    if not torch.isfinite(x).all() or not torch.isfinite(z).all() or not 0 < ridge < float("inf"):
        raise ValueError("field-probe inputs and positive regularization must be finite")
    for index, classes in enumerate(CODE_CLASSES):
        if any((labels[:, index] < 0).any() or (labels[:, index] >= classes).any() for labels in (a, b)):
            raise ValueError("field-probe class index out of range")
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x, z = (x - mean) / scale, (z - mean) / scale
    target = torch.cat([F.one_hot(a[:, i], n) for i, n in enumerate(CODE_CLASSES)], dim=1).double()
    prior = target.mean(0)
    kernel = x @ x.T + ridge * len(x) * torch.eye(len(x), dtype=x.dtype)
    solution = torch.linalg.solve(kernel, target - prior)
    scores = {"train": x @ x.T @ solution + prior, "dev": z @ x.T @ solution + prior}
    predictions = {}
    for split, value in scores.items():
        predictions[split] = torch.stack([part.argmax(-1) for part in value.split(CODE_CLASSES, dim=1)], dim=1)
    metrics = {}
    for split, labels in (("train", a), ("dev", b)):
        correct = predictions[split] == labels
        metrics[split] = {"examples": len(labels), "full_code_accuracy": float(correct.all(-1).float().mean()),
                          "field_accuracy": {str(p): float(correct[:, i].float().mean())
                                             for i, p in enumerate(CODE_POSITIONS)}}
    metrics["train_majority_baseline_dev_accuracy"] = {
        str(p): float((b[:, i] == torch.bincount(a[:, i], minlength=CODE_CLASSES[i]).argmax()).float().mean())
        for i, p in enumerate(CODE_POSITIONS)}
    return predictions, metrics
