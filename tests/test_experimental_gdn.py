import pytest
import torch

from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from dsrsi.model.experimental_gdn import experimental_gdn_inverse, make_chunk_rule


@pytest.mark.parametrize("length", [13, 64, 97])
def test_full_chunk_output_state_and_all_input_gradients_match_with_bounded_fp32_error(length):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    inputs = [torch.randn(1, length, 2, 8) * 0.2 for _ in range(3)]
    inputs += [-torch.rand(1, length, 2) * 0.1, torch.rand(1, length, 2)]
    initial = torch.randn(1, 2, 8, 8) * 0.1
    def execute(function):
        values = [value.clone().requires_grad_() for value in inputs]
        state = initial.clone().requires_grad_()
        output, final = function(*values, initial_state=state, output_final_state=True,
                                 use_qk_l2norm_in_kernel=True)
        gradients = torch.autograd.grad(output.square().sum() + final.square().sum(), [*values, state])
        return output, final, gradients
    old = execute(torch_chunk_gated_delta_rule)
    new = execute(make_chunk_rule())
    torch.testing.assert_close(old[0], new[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(old[1], new[1], rtol=2e-5, atol=2e-6)
    for a, b in zip(old[2], new[2]):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=1e-5)


def test_scoped_gdn_patch_is_restored_on_error_and_refuses_nested_or_modified_backends():
    from test_qwen_bridge import tiny
    model = tiny()
    layers = [layer.linear_attn for layer in model.base.layers if hasattr(layer, "linear_attn")]
    with pytest.raises(RuntimeError, match="interrupted"):
        with experimental_gdn_inverse(model) as info:
            assert info["summation_order_changed"] and info["layers"] == len(layers)
            with pytest.raises(ValueError, match="nested"):
                with experimental_gdn_inverse(model):
                    pass
            raise RuntimeError("interrupted")
    assert not hasattr(model, "_experimental_gdn_active")
    assert all(layer.chunk_gated_delta_rule is torch_chunk_gated_delta_rule for layer in layers)
    layers[0].chunk_gated_delta_rule = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="original pinned"):
        with experimental_gdn_inverse(model):
            pass
