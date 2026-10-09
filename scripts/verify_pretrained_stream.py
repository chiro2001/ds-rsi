#!/usr/bin/env python3
"""Measure BF16 continuous-state fidelity on actual pretrained language weights."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from dsrsi.model.qwen35_memory import MemoryQwen
from dsrsi.model.streaming import StreamExecutor
from dsrsi.state.life_state import LifeState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(93)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = MemoryQwen.from_local_checkpoint(args.base).eval()
    executor = StreamExecutor(model)
    tokens = torch.tensor([tokenizer.encode("青杉仓的校验码是Q7-492。现在继续讨论采样设备的位置，设备在东侧柜。", add_special_tokens=False)])
    def new_life(quantum=1):
        return executor.new_life("probe", window=16, rebase_interval=16, writer_stride=8,
                                 retrieval_interval=4, execution_quantum=quantum)
    with torch.no_grad():
        baseline = model.base(tokens, use_cache=False).last_hidden_state
        unbounded = executor.new_life("reference", window=None, rebase_interval=None, execution_quantum=None)
        cold = executor.forward(tokens, unbounded, write=False, memory_enabled=False)
        cold_diff = float((cold.hidden.float() - baseline.float()).abs().max())
        whole = executor.forward(tokens, new_life())
        state = new_life()
        outputs = []
        for index in range(tokens.shape[1]):
            step = executor.forward(tokens[:, index:index + 1], state)
            state = step.state
            outputs.append(step.hidden)
        stepped = torch.cat(outputs, 1)
        difference = (whole.hidden.float() - stepped.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(whole.hidden.float(), stepped.float(), dim=-1)
        diagnostic_parallel = executor.forward(tokens, new_life(None))
        parallel_difference = (diagnostic_parallel.hidden.float() - stepped.float()).abs()
        prefix = executor.forward(tokens[:, :7], new_life())
        original = executor.forward(tokens[:, 7:], prefix.state)
        restored = executor.forward(tokens[:, 7:], LifeState.from_state_dict(prefix.state.state_dict()))
        restored_diff = float((original.hidden.float() - restored.hidden.float()).abs().max())
    if cold_diff != 0 or restored_diff != 0:
        raise AssertionError(f"native/snapshot identity violated: {cold_diff}, {restored_diff}")
    if not torch.isfinite(difference).all() or float(difference.max()) != 0:
        raise AssertionError("canonical stream execution differs by external segmentation")
    report = {"status": "pretrained_bf16_stream_integration_only", "tokens": tokens.shape[1],
              "native_cold_max_abs_diff": cold_diff, "chunk_vs_step_max_abs_diff": float(difference.max()),
              "chunk_vs_step_mean_abs_diff": float(difference.mean()), "chunk_vs_step_min_cosine": float(cosine.min()),
              "canonical_execution_quantum": 1,
              "parallel_diagnostic_max_abs_diff": float(parallel_difference.max()),
              "snapshot_restore_max_abs_diff": restored_diff, "logical_clock": state.cache.clock,
              "position_origin": state.cache.position_origin, "cache_tensor_bytes": state.cache.tensor_bytes,
              "memory_records": len(state.memory.records), "memory_tensor_bytes": state.memory.tensor_bytes,
              "trained_long_context_claimed": False, "elapsed_seconds": time.monotonic() - start}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
