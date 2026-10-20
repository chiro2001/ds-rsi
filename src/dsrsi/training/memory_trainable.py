"""Explicit memory parameter freezing, with reversible read-only loading."""

from contextlib import contextmanager


FREEZE_READER_GATES_V1 = "freeze-reader-gates-v1"


def reader_gate_parameters(model):
    return {name: p for name, p in model.named_parameters()
            if name.startswith("readers.") and (name.endswith(".alpha") or ".gate." in name)}


def apply_memory_trainable_policy(model, policy):
    if policy is None:
        return []
    if policy != FREEZE_READER_GATES_V1:
        raise ValueError("unknown explicit memory trainable policy")
    parameters = reader_gate_parameters(model)
    if not parameters or len(parameters) != len(model.readers) * 3:
        raise ValueError("reader gate policy requires alpha, linear weight and bias per Reader")
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    return sorted(parameters)


@contextmanager
def source_memory_trainable_policy(model, policy):
    # A saved frozen parameter lives in frozen_auxiliary, rather than weights.
    # Loading must honor that source partition without changing the caller's
    # target training policy, including when validation raises an exception.
    original = {name: (p, p.requires_grad) for name, p in reader_gate_parameters(model).items()}
    try:
        apply_memory_trainable_policy(model, policy)
        yield
    finally:
        for parameter, requires_grad in original.values():
            parameter.requires_grad_(requires_grad)
