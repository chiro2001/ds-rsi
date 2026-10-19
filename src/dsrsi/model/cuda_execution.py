"""Explicit CUDA reference kernels and runtime identity for recoverable runs."""

from contextlib import contextmanager
import hashlib
import inspect
import os
import platform
from importlib.metadata import version
from pathlib import Path

import torch


@contextmanager
def reference_gdn_loading():
    """Bind reference functions to new layers; restore HF globals afterwards."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    names = ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")
    originals = {name: getattr(hf, name) for name in names}
    try:
        for name in names:
            setattr(hf, name, None)
        yield
    finally:
        for name, value in originals.items():
            setattr(hf, name, value)


def require_reference_kernels(model):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    layers = [layer.linear_attn for layer in model.base.layers if hasattr(layer, "linear_attn")]
    if not layers or model.base.config._attn_implementation != "eager":
        raise ValueError("CUDA reference execution requires eager attention and GDN layers")
    for layer in layers:
        if (layer.chunk_gated_delta_rule is not hf.torch_chunk_gated_delta_rule
                or layer.recurrent_gated_delta_rule is not hf.torch_recurrent_gated_delta_rule
                or layer.causal_conv1d_fn is not None
                or layer.causal_conv1d_update is not hf.torch_causal_conv1d_update
                or type(layer.norm) is not hf.Qwen3_5RMSNormGated):
            raise ValueError("CUDA reference kernel selection differs")


def cuda_execution_identity(device="cuda:0", *, experimental=False):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    root = Path(__file__).resolve().parents[1]
    files = ("model/cuda_execution.py", "model/qwen35_memory.py", "training/device.py",
             "training/runner.py", "training/checkpoint.py", "training/control.py")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("CUDA execution identity requires an available CUDA device")
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("CUDA execution requires deterministic algorithms with TF32 disabled")
    return {
        "version": "cuda-reference-gdn-eager-v1" if not experimental else "cuda-auto-experimental-v1",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "python": platform.python_version(), "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: version(name) for name in ("transformers", "numpy", "safetensors", "tokenizers")},
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cpu_threads": torch.get_num_threads(),
        "deterministic": True, "tf32": False,
        "implementation": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        "hf_reference": {name: hashlib.sha256(inspect.getsource(getattr(hf, name)).encode()).hexdigest()
                         for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                                      "torch_causal_conv1d_update", "Qwen3_5RMSNormGated")},
    }
