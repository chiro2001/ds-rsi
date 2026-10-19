import pytest

pytest.importorskip("transformers")
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from dsrsi.model.cuda_execution import reference_gdn_loading, require_reference_kernels
from test_qwen_bridge import tiny


def test_reference_loading_restores_all_global_hooks_even_after_exception(monkeypatch):
    marker = object()
    monkeypatch.setattr(hf, "chunk_gated_delta_rule", marker)
    with pytest.raises(RuntimeError, match="construction failed"):
        with reference_gdn_loading():
            assert hf.chunk_gated_delta_rule is None and hf.FusedRMSNormGated is None
            raise RuntimeError("construction failed")
    assert hf.chunk_gated_delta_rule is marker


def test_bound_reference_kernels_survive_global_fast_path_restoration(monkeypatch):
    marker = object()
    monkeypatch.setattr(hf, "chunk_gated_delta_rule", marker)
    with reference_gdn_loading():
        model = tiny()
    assert hf.chunk_gated_delta_rule is marker
    require_reference_kernels(model)
    model.base.layers[0].linear_attn.chunk_gated_delta_rule = marker
    with pytest.raises(ValueError, match="selection differs"):
        require_reference_kernels(model)
