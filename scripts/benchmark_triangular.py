#!/usr/bin/env python3
"""Isolated FP32 GDN inverse/gradient comparison; no model or optimizer writes."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def errors(actual, reference):
    return {"max_abs": float((actual - reference).abs().max()),
            "relative_l2": float((actual - reference).norm() / reference.norm().clamp_min(1e-12)),
            "finite": bool(torch.isfinite(actual).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeat count required")
    from dsrsi.model.experimental_triangular import METHODS
    from dsrsi.training.device import resolve_device
    from dsrsi.utils.atomic import atomic_json
    torch.set_num_threads(2)
    device, size = resolve_device(args.device)
    if size != 1:
        raise ValueError("inverse microbenchmark uses one isolated device")
    sync = torch.npu.synchronize if device.startswith("npu") else lambda: None
    dispatch = {}
    if device.startswith("npu"):
        for operation in ("aten::linalg_solve_triangular", "aten::linalg_solve_triangular.out"):
            dispatch[operation] = torch._C._dispatch_has_kernel_for_dispatch_key(operation, "PrivateUse1")
    cases = {}
    for batch, heads, chunks in ((1, 16, 3), (8, 16, 1)):
        generator = torch.Generator().manual_seed(371)
        keys = torch.randn(batch, heads, chunks, 64, 128, generator=generator)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        beta = torch.randn(batch, heads, chunks, 64, generator=generator).sigmoid()
        g = -torch.rand(batch, heads, chunks, 64, generator=generator) * 0.2
        g = g.cumsum(-1)
        decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
        values = (-((keys * beta[..., None]) @ keys.transpose(-1, -2)) * decay).tril(-1).to(device)
        direction = torch.randn(values.shape, generator=generator).to(device)
        def execute(method):
            inputs = values.clone().requires_grad_()
            output = METHODS[method](inputs.tril(-1))
            gradient = torch.autograd.grad((output * direction).sum(), inputs)[0]
            return output.detach(), gradient.detach()
        reference = execute("reference")
        results = {}
        for method in METHODS:
            try:
                candidate = execute(method)
                results[method] = {"inverse": errors(candidate[0], reference[0]),
                                   "gradient": errors(candidate[1], reference[1]), "seconds": []}
            except RuntimeError as exc:
                results[method] = {"error": str(exc)}
        methods = [name for name in METHODS if "error" not in results[name]]
        for repeat in range(args.repeats):
            order = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
            for method in order:
                sync()
                started = time.perf_counter()
                execute(method)
                sync()
                results[method]["seconds"].append(time.perf_counter() - started)
        for method in methods:
            results[method]["median_seconds"] = statistics.median(results[method]["seconds"])
            results[method]["speedup_vs_reference"] = (
                statistics.median(results["reference"]["seconds"]) / results[method]["median_seconds"])
        cases[f"{batch}x{heads}x{chunks}x64x64"] = results
        print(json.dumps({"shape": [batch, heads, chunks, 64, 64], "results": results}), flush=True)
    report = {"status": "isolated_triangular_inverse_diagnostic", "device": device,
              "dtype": "float32", "lower_matrix_source": "normalized random K, sigmoid beta, cumulative negative decay",
              "repeats": args.repeats, "rotating_execution_order": True, "cases": cases,
              "solve_privateuse1_dispatch": dispatch, "formal_training_modified": False,
              "full_model_equivalence_or_speedup_claimed": False}
    atomic_json(Path(args.output), report)


if __name__ == "__main__":
    main()
