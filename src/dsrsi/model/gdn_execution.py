"""Identity and scoped execution for the verified original-order GDN rows."""

from contextlib import contextmanager
import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from .experimental_gdn import experimental_gdn_inverse


def gdn_execution_identity():
    root = Path(__file__).resolve().parent
    try:
        npu_version = version("torch-npu")
    except PackageNotFoundError:
        npu_version = None
    return {
        "version": "gdn-scripted-original-row-order-v1",
        "method": "scripted_rows",
        "summation_order_changed": False,
        "reference_source_sha256": hashlib.sha256(inspect.getsource(torch_chunk_gated_delta_rule).encode()).hexdigest(),
        "implementation": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                           for name in ("gdn_execution.py", "experimental_gdn.py", "experimental_triangular.py")},
        "torch_version": str(torch.__version__),
        "torch_npu_version": npu_version,
        "transformers_version": transformers.__version__,
    }


@contextmanager
def scripted_gdn_scope(model):
    # The faster algorithms that change arithmetic order remain profiling-only.
    with experimental_gdn_inverse(model, "scripted_rows") as information:
        yield information
