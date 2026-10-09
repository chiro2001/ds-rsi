#!/usr/bin/env python3
"""CPU integration on actual pretrained weights; does not claim trained recall."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.generate import EpisodeGenerator
from dsrsi.model.qwen35_memory import MemoryQwen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    episode = EpisodeCompiler(tokenizer).compile(EpisodeGenerator(distractors=0).episode(
        1, family="single_fact", language="zh"))
    model = MemoryQwen.from_local_checkpoint(args.base).eval()
    source = episode.source[0]
    response = episode.responses[0]
    inputs = torch.tensor([response.input_ids])
    labels = torch.tensor([response.labels])
    with torch.no_grad():
        native = model.base(inputs, use_cache=False).last_hidden_state
        disabled = model(inputs, memory_enabled=False).hidden
    disabled_diff = float((native.float() - disabled.float()).abs().max())
    if disabled_diff != 0:
        raise AssertionError(f"disabled memory differs from native backbone: {disabled_diff}")
    records = model.write_records(torch.tensor([source.input_ids]), stream_ids=[episode.episode_id],
                                  logical_begins=[source.logical_begin], record_ids=[source.event_id])
    model.train()
    out = model(inputs, records=[records], query_indices=[response.retrieval_query_index],
                logical_begins=[response.logical_begin], stream_ids=[episode.episode_id], labels=labels)
    early_diff = float((out.hidden[:, :response.memory_read_from].float()
                        - native[:, :response.memory_read_from].float()).detach().abs().max())
    if early_diff != 0:
        raise AssertionError(f"future lookup changes earlier prompt: {early_diff}")
    loss = out.loss.mean()
    loss.backward()
    writer_grad = sum(float(p.grad.float().square().sum()) for p in model.writer.parameters() if p.grad is not None)
    if writer_grad <= 0 or not torch.isfinite(loss):
        raise AssertionError("actual future answer does not train Writer")
    if any(p.grad is not None for p in model.base.parameters()):
        raise AssertionError("frozen base acquired gradients")
    report = {"status": "pretrained_cpu_integration_only", "base": args.base,
              "base_parameters": sum(p.numel() for p in model.base.parameters()),
              "memory_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "base_dtype": str(model.base.embed_tokens.weight.dtype),
              "source_tokens": len(source.input_ids), "response_tokens": len(response.input_ids),
              "disabled_max_abs_diff": disabled_diff, "pre_lookup_prefix_max_abs_diff": early_diff,
              "answer_loss": float(loss.detach()), "writer_grad_squared_norm": writer_grad,
              "base_has_gradients": False, "elapsed_seconds": time.monotonic() - started,
              "trained_recall_claimed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
