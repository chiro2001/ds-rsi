import pytest
import torch

from dsrsi.model.experimental_triangular import METHODS


@pytest.mark.parametrize("method", ["solve", "doubling", "blocks"])
@pytest.mark.parametrize("width", [4, 16, 64])
def test_inverse_and_derivatives_match_double_precision_row_reference(method, width):
    torch.manual_seed(31)
    values = (torch.randn(2, 3, width, width, dtype=torch.float64) * 0.04).requires_grad_()
    lower = values.tril(-1)
    reference = METHODS["reference"](lower)
    candidate = METHODS[method](lower)
    identity = torch.eye(width, dtype=torch.float64).expand_as(lower)
    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((identity - lower) @ candidate, identity, rtol=1e-12, atol=1e-12)
    direction = torch.randn_like(reference)
    old = torch.autograd.grad((reference * direction).sum(), values, retain_graph=True)[0]
    new = torch.autograd.grad((candidate * direction).sum(), values)[0]
    torch.testing.assert_close(new, old, rtol=1e-11, atol=1e-11)


def test_block_inverse_rejects_unsupported_width():
    with pytest.raises(ValueError, match="power-of-two"):
        METHODS["blocks"](torch.zeros(2, 5, 5))
