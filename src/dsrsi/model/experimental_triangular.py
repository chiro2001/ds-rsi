"""Profiling-only alternatives to the HF GDN strictly-lower row transform.

These algorithms change FP32 summation order. They are not wired into any
training or inference runner. Original/alternative gradients and complete
model behavior must be measured before an execution change is considered.
"""

import torch


def inverse_row_reference(lower):
    result = lower.clone()
    for index in range(1, lower.shape[-1]):
        row = result[..., index, :index].clone()
        sub = result[..., :index, :index].clone()
        result[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return result + torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype)


def inverse_solve(lower):
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype)
    return torch.linalg.solve_triangular(identity - lower, identity.expand_as(lower),
                                         upper=False, unitriangular=True)


def inverse_doubling(lower):
    """Finite Neumann product: (I+L)(I+L²)(I+L⁴)... for nilpotent L."""
    n = lower.shape[-1]
    identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
    result, power = identity + lower, lower
    exponent = 2
    while exponent < n:
        power = power @ power
        result = result @ (identity + power)
        exponent *= 2
    return result


def inverse_blocks(lower):
    """Batched recursive block inversion with no per-row tensor writes."""
    n = lower.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("block inversion requires a power-of-two width")
    if n == 1:
        return torch.ones_like(lower)
    middle = n // 2
    diagonal = torch.stack((lower[..., :middle, :middle], lower[..., middle:, middle:]), dim=-3)
    inverses = inverse_blocks(diagonal)
    first, last = inverses[..., 0, :, :], inverses[..., 1, :, :]
    bottom_left = last @ lower[..., middle:, :middle] @ first
    top = torch.cat((first, torch.zeros_like(first)), dim=-1)
    bottom = torch.cat((bottom_left, last), dim=-1)
    return torch.cat((top, bottom), dim=-2)


METHODS = {"reference": inverse_row_reference, "solve": inverse_solve,
           "doubling": inverse_doubling, "blocks": inverse_blocks}
