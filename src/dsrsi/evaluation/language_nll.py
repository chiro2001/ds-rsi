"""Held-out natural-text NLL retention scoring on the native training path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from dsrsi.training.losses import IGNORE_INDEX

PROTOCOLS = ("raw", "native")
MODES = ("base", "off", "unrelated")
INSTRUCTIONS = {"zh": "请自然地接着写下面的文本，不要添加解释或标题：",
                "en": "Continue the text below naturally. Do not add explanations or headings:"}
UNRELATED_TEXT = {"zh": "白杨-941208的箱封码是J8-672。",
                  "en": "The seal code for Poplar-941208 is J8-672."}


def unrelated_source_ids(tokenizer, language, *, source_format="native_event"):
    """Use the same native user-event encoding as A/B Writer inputs by default."""
    text = UNRELATED_TEXT[language]
    if source_format == "raw_text":
        return tokenizer.encode(text, add_special_tokens=False)
    if source_format != "native_event":
        raise ValueError("unknown unrelated-memory source format")
    from dsrsi.data.compiler import EpisodeCompiler
    return EpisodeCompiler(tokenizer).encode([{"role": "user", "content": text}])


@dataclass(frozen=True)
class ScoringSequence:
    input_ids: list[int]
    labels: list[int]
    query_index: int

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for value in self.labels[1:] if value != IGNORE_INDEX)

    def __post_init__(self):
        if len(self.input_ids) != len(self.labels) or len(self.input_ids) < 3:
            raise ValueError("scoring sequences require aligned nonempty ids and labels")
        if self.labels[0] != IGNORE_INDEX:
            raise ValueError("the first position is never a prediction target")
        if any(value != IGNORE_INDEX for value in self.labels[:self.query_index + 1]):
            raise ValueError("the retrieval index must not see supervised tokens")
        if self.supervised_tokens < 1:
            raise ValueError("a scoring sequence requires at least one target")


def build_scoring_sequence(tokenizer, tokens, *, protocol: str, language: str = "en") -> ScoringSequence:
    """Turn one token window into raw continuation or native chat continuation."""
    tokens = [int(token) for token in tokens]
    if protocol not in PROTOCOLS:
        raise ValueError("unknown scoring protocol")
    if len(tokens) < 16:
        raise ValueError("windows require at least sixteen tokens")
    if protocol == "raw":
        labels = [IGNORE_INDEX] + tokens[1:]
        return ScoringSequence(tokens, labels, 0)
    if language not in INSTRUCTIONS:
        raise ValueError("native scoring requires a supported language")
    cut = max(2, int(len(tokens) * 0.55))
    prefix, suffix = tokens[:cut], tokens[cut:]
    if len(suffix) < 4:
        raise ValueError("native scoring requires a nonempty continuation")
    content = INSTRUCTIONS[language] + "\n\n" + tokenizer.decode(prefix, skip_special_tokens=True)
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer.encode(text, add_special_tokens=False)
    if not prompt:
        raise ValueError("chat template produced an empty prompt")
    return ScoringSequence(prompt + suffix, [IGNORE_INDEX] * len(prompt) + suffix, len(prompt) - 1)


def score_sequence(model, sequence: ScoringSequence, *, mode: str, record=None, life_id: str = "language-retention",
                   logical_begin: int = 4096, device=None) -> dict:
    """One teacher-forced pass; returns summed and mean token NLL."""
    if mode not in MODES:
        raise ValueError("unknown memory mode")
    if mode == "unrelated" and record is None:
        raise ValueError("unrelated-memory scoring requires a record")
    device = model.base.embed_tokens.weight.device if device is None else device
    input_ids = torch.tensor([sequence.input_ids], device=device)
    labels = torch.tensor([sequence.labels], device=device)
    with torch.no_grad():
        output = model.forward(
            input_ids, records=[[record]] if mode == "unrelated" else None,
            query_indices=[sequence.query_index], logical_begins=[logical_begin],
            stream_ids=[life_id], labels=labels, memory_enabled=mode == "unrelated")
    mean_nll = float(output.loss.total)
    if not math.isfinite(mean_nll):
        raise ValueError("non-finite language NLL")
    return {"tokens": sequence.supervised_tokens, "mean_nll": mean_nll,
            "nll_sum": mean_nll * sequence.supervised_tokens}


def aggregate(records) -> dict:
    """Token-weighted mean NLL and perplexity for one mode/protocol stratum."""
    records = list(records)
    tokens = sum(record["tokens"] for record in records)
    if tokens < 1:
        raise ValueError("aggregation requires scored tokens")
    nll_sum = sum(record["nll_sum"] for record in records)
    mean = nll_sum / tokens
    return {"windows": len(records), "documents": len({record["document_id"] for record in records}),
            "tokens": tokens, "mean_nll": mean,
            "perplexity": math.exp(min(mean, 60.0))}


def document_totals(records) -> dict:
    """(document_id -> [nll_sum, tokens]) for paired comparisons."""
    totals = {}
    for record in records:
        row = totals.setdefault(record["document_id"], [0.0, 0])
        row[0] += record["nll_sum"]
        row[1] += record["tokens"]
    return totals


def paired_bootstrap(base_totals, variant_totals, *, iterations: int = 2000, seed: int = 20261011) -> dict:
    """Document-level paired bootstrap of the mean NLL difference."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if base_totals.keys() != variant_totals.keys():
        raise ValueError("paired comparison requires identical document sets")
    shared = sorted(base_totals)
    if not shared:
        raise ValueError("paired comparison requires shared documents")
    deltas = []
    for document_id in shared:
        base_sum, base_tokens = base_totals[document_id]
        variant_sum, variant_tokens = variant_totals[document_id]
        if base_tokens != variant_tokens or base_tokens < 1:
            raise ValueError("paired documents require identical token counts")
        deltas.append((variant_sum - base_sum) / base_tokens)
    generator = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [deltas[generator.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low = samples[int(0.025 * (iterations - 1))]
    high = samples[int(0.975 * (iterations - 1))]
    return {"documents": len(deltas), "mean_delta": sum(deltas) / len(deltas),
            "ci95_low": low, "ci95_high": high, "max_document_delta": max(deltas)}


def retention_gate(base, variant, paired, *, max_mean_delta: float = 0.05, max_ci_high: float = 0.10,
                   max_document_delta: float = 0.25) -> dict:
    """Conservative regression thresholds for one language/protocol stratum."""
    checks = {"mean_delta": paired["mean_delta"] <= max_mean_delta,
              "ci95_high": paired["ci95_high"] <= max_ci_high,
              "max_document_delta": paired["max_document_delta"] <= max_document_delta}
    return {"passed": all(checks.values()), "checks": checks,
            "thresholds": {"max_mean_delta": max_mean_delta, "max_ci95_high": max_ci_high,
                           "max_document_delta": max_document_delta},
            "variant_mean_nll": variant["mean_nll"], "base_mean_nll": base["mean_nll"]}
