"""Scoped profiling-only replacement of the pinned HF GDN inverse block."""

import ast
from contextlib import contextmanager
import hashlib
import inspect

from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from .experimental_triangular import METHODS

_INVERSE_BLOCK = """for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
"""


def make_chunk_rule(method="doubling"):
    if method not in METHODS or method == "reference":
        raise ValueError("choose an experimental inverse method")
    source = inspect.getsource(torch_chunk_gated_delta_rule)
    tree = ast.parse(source)
    function = tree.body[0]
    expected = [ast.dump(node, include_attributes=False) for node in ast.parse(_INVERSE_BLOCK).body]
    matches = [index for index in range(len(function.body) - 1)
               if [ast.dump(node, include_attributes=False) for node in function.body[index:index + 2]] == expected]
    if len(matches) != 1:
        raise ValueError("HF GDN inverse block differs; refuse an unreviewed transformation")
    index = matches[0]
    function.body[index:index + 2] = ast.parse("attn = _dsrsi_inverse(attn)").body
    function.name = "experimental_chunk_gated_delta_rule"
    ast.fix_missing_locations(tree)
    namespace = dict(torch_chunk_gated_delta_rule.__globals__)
    namespace["_dsrsi_inverse"] = METHODS[method]
    # Input is the local pinned dependency source, never user-provided code.
    exec(compile(tree, "<dsrsi-experimental-gdn>", "exec"), namespace)
    rule = namespace[function.name]
    rule._reference_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return rule


@contextmanager
def experimental_gdn_inverse(model, method="doubling"):
    if getattr(model, "_experimental_gdn_active", False):
        raise ValueError("nested experimental GDN scope is unsupported")
    layers = [layer.linear_attn for layer in model.base.layers if hasattr(layer, "linear_attn")]
    if not layers or any(layer.chunk_gated_delta_rule is not torch_chunk_gated_delta_rule for layer in layers):
        raise ValueError("scope requires the original pinned PyTorch GDN fallback")
    candidate = make_chunk_rule(method)
    originals = [layer.chunk_gated_delta_rule for layer in layers]
    model._experimental_gdn_active = True
    try:
        for layer in layers:
            layer.chunk_gated_delta_rule = candidate
        yield {"method": method, "layers": len(layers),
               "reference_source_sha256": candidate._reference_source_sha256,
               "summation_order_changed": True}
    finally:
        for layer, original in zip(layers, originals):
            layer.chunk_gated_delta_rule = original
        del model._experimental_gdn_active
