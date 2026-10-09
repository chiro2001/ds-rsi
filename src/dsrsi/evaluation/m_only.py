"""Answer-free cached generation, explicit oracle/actual/off/content-shuffle modes."""

from dataclasses import asdict, dataclass, replace
import unicodedata

import torch

from dsrsi.state.streaming_cache import StreamingCache
from dsrsi.training.losses import routing_scores


@dataclass
class GeneratedAnswer:
    probe_id: str
    mode: str
    text: str
    token_ids: list[int]
    stopped_on_end: bool
    expected: str
    exact_match: bool
    actual_support_coverage: bool
    selected_records: list[str]


def normalized_answer(text):
    return unicodedata.normalize("NFKC", text).strip().rstrip("。.!！ ")


class MOnlyEvaluator:
    def __init__(self, model, compiler, *, max_new_tokens=32):
        if max_new_tokens < 1:
            raise ValueError("generation budget must be positive")
        self.model, self.compiler, self.max_new_tokens = model, compiler, max_new_tokens

    def source_records(self, episode):
        if episode.source_encoding != "independent_events":
            raise ValueError("M-only controls require explicit independent source encoding")
        if any(len(span.input_ids) > 256 for span in episode.source):
            raise ValueError("control evidence longer than Writer span requires explicit support mapping")
        records = []
        device = self.model.base.embed_tokens.weight.device
        for span in episode.source:
            records.extend(self.model.write_records(torch.tensor([span.input_ids], device=device),
                                                     stream_ids=[episode.episode_id],
                                                     logical_begins=[span.logical_begin], record_ids=[span.event_id],
                                                     wall_times=[span.wall_time]))
        return records

    def _encode_prefix(self, response):
        device = self.model.base.embed_tokens.weight.device
        prompt = torch.tensor([response.prompt_ids], device=device)
        if not prompt.shape[1]:
            raise ValueError("empty answer-free prefix")
        cache = StreamingCache(self.model.base.config.layer_types, window=2048, rebase_interval=None)
        cache.clock = response.logical_begin
        cache.position_origin = response.logical_begin
        for index in range(prompt.shape[1] - 1):
            cache = self.model.forward_cached(prompt[:, index:index + 1], cache, memory_enabled=False).cache
        return prompt, cache

    def _prompt_forward(self, response, records, *, life_id, mode, replacement_records=None, prefix=None):
        if mode not in {"actual", "oracle", "off", "shuffled", "foreign"}:
            raise ValueError("invalid ablation mode")
        if response.mode != "m_only" or response.write_enabled:
            raise ValueError("evaluation must not write supervised completions")
        if response.memory_transform.get("merge_events"):
            raise ValueError("compression evaluation requires its explicit teacher/student path")
        dropped = set(response.memory_transform.get("drop_events", []))
        pool = [r for r in records if r.record_id not in dropped]
        device = self.model.base.embed_tokens.weight.device
        # Reset all working state; absolute memory time remains independent of
        # the fresh prefix's local rotary coordinate system.
        # Use the same canonical single-token numerical path as StreamExecutor.
        # Parallel BF16 GDN prefill is a diagnostic path, not interchangeable.
        prompt, cache = self._encode_prefix(response) if prefix is None else prefix
        if cache.clock != response.logical_begin + prompt.shape[1] - 1:
            raise ValueError("shared answer-free prefix has a different logical range")
        selected = []
        actual_coverage = False
        groups = [set(group) for group in response.support_groups]
        lookup_position = cache.clock

        def selector(features, position):
            nonlocal selected, actual_coverage
            available = [r for r in pool if r.stream_id == life_id and r.visible_after <= position]
            if not available or mode == "off":
                actual_coverage = not groups
                selected = []
                return []
            with self.model._memory_context(features):
                q = self.model.address.encode_query(features)
            keys = torch.stack([r.keys.to(device) for r in available])
            scores = routing_scores(q, keys)[0]
            ordering = torch.argsort(scores, descending=True, stable=True).tolist()
            actual = {available[index].record_id for index in ordering[:8]}
            actual_coverage = all(actual & group for group in groups)
            if mode == "oracle":
                positives = [i for i, record in enumerate(available) if any(record.record_id in group for group in groups)]
                ordering = positives + [i for i in ordering if i not in positives]
            selected = [available[index] for index in ordering[:8]]
            if mode == "shuffled":
                if len(available) < 2:
                    raise ValueError("content-shuffle ablation needs at least two independent records")
                by_id = {record.record_id: available[(index + 1) % len(available)].slots for index, record in enumerate(available)}
                selected = [replace(record, slots=by_id[record.record_id]) for record in selected]
            elif mode == "foreign":
                if not replacement_records or any(r.stream_id == life_id for r in replacement_records):
                    raise ValueError("foreign-content ablation requires independent counterfactual-life payloads")
                replacements = {r.record_id: r.slots for r in replacement_records}
                if any(r.record_id not in replacements for r in selected):
                    raise ValueError("foreign-content ablation lacks a matching payload")
                selected = [replace(r, slots=replacements[r.record_id]) for r in selected]
            return selected

        output = self.model.forward_cached(prompt[:, -1:], cache, selector=selector,
                                            stream_id=life_id, memory_enabled=True,
                                            lookup_position=lookup_position, lookup_wall_time=response.wall_time)
        available_ids = {r.record_id for r in pool if r.stream_id == life_id and r.visible_after <= lookup_position}
        return output, selected, actual_coverage, all(available_ids & group for group in groups), lookup_position

    @torch.no_grad()
    def retrieval(self, response, records, *, life_id):
        _, selected, coverage, evidence_available, _ = self._prompt_forward(response, records, life_id=life_id, mode="actual")
        return {"probe_id": response.event_id, "required_groups": len(response.support_groups),
                "evidence_available": evidence_available, "support_coverage": coverage,
                "selected_records": [r.record_id for r in selected]}

    @torch.no_grad()
    def generate(self, response, records, *, life_id, expected, mode="actual", replacement_records=None, prefix=None):
        output, selected, actual_coverage, _, lookup_position = self._prompt_forward(
            response, records, life_id=life_id, mode=mode, replacement_records=replacement_records, prefix=prefix)
        device = self.model.base.embed_tokens.weight.device
        generated = []
        stopped = False
        for _ in range(self.max_new_tokens):
            token = int(self.model.head(output.hidden[:, -1]).argmax(-1).item())
            if token == self.compiler.tokenizer.eos_token_id:
                stopped = True
                break
            generated.append(token)
            output = self.model.forward_cached(torch.tensor([[token]], device=device), output.cache,
                                                records=selected, stream_id=life_id,
                                                memory_enabled=mode != "off",
                                                lookup_position=lookup_position, lookup_wall_time=response.wall_time)
        text = self.compiler.tokenizer.decode(generated, skip_special_tokens=True)
        return GeneratedAnswer(response.event_id, mode, text, generated, stopped, expected,
                               normalized_answer(text) == normalized_answer(expected), actual_coverage,
                               [r.record_id for r in selected])

    @torch.no_grad()
    def episode(self, row, *, modes=("actual", "oracle", "off", "shuffled"), replacement_records=None):
        if row.get("split") == "train":
            raise ValueError("held-out evaluation cannot use training worlds")
        self.model.eval()
        episode = self.compiler.compile(row)
        records = self.source_records(episode)
        expected = {probe["probe_id"]: probe["target"] for probe in row["probes"]}
        results = []
        for response in episode.responses:
            # Only this immutable, answer-free prefix is shared across modes.
            # forward_cached forks its cache; no generated state is reused.
            prefix = self._encode_prefix(response)
            for mode in modes:
                results.append(asdict(self.generate(response, records, life_id=episode.episode_id,
                    expected=expected[response.event_id], mode=mode, replacement_records=replacement_records, prefix=prefix)))
        return results
