import copy

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_episode_objective import SmallVocabularyTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.revisions import make_generator
from dsrsi.evaluation.codec import canonical_logits, native_logits, distinguishing_token, swapped_payloads
from dsrsi.training.episode_objective import EpisodeObjective


def test_diagnostic_scores_each_target_before_consuming_it_and_agrees_with_native_path():
    torch.manual_seed(27)
    model = tiny().eval()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    generator = make_generator(distractors=2)
    a = generator.episode(0, variant=0, family="single_fact", language="en")
    b = generator.episode(0, variant=1, family="single_fact", language="en")
    episode, opposite = compiler.compile(a), compiler.compile(b)
    response = episode.responses[0]
    difference, targets, other_targets = distinguishing_token(response, opposite.responses[0])
    assert targets[:difference] == other_targets[:difference] and targets[difference] != other_targets[difference]
    with torch.no_grad():
        pool = EpisodeObjective(model, compiler)._sources(episode)[0]
    selected = [r for r in pool if r.record_id in response.support_groups[0]]
    consumed = []
    original = model.forward_cached
    def capture(ids, cache, **kwargs):
        consumed.extend(ids[0].tolist())
        return original(ids, cache, **kwargs)
    model.forward_cached = capture
    canonical = canonical_logits(model, compiler, response, selected, life_id=episode.episode_id)
    assert consumed == response.prompt_ids + targets[:-1]
    assert canonical.shape[0] == len(targets)
    trace = {}
    native = native_logits(model, response, selected, life_id=episode.episode_id, reader_trace=trace)
    assert torch.allclose(canonical, native, atol=5e-5, rtol=5e-5)
    assert set(trace) == set(model.readers)
    assert all(len(x["gate"]) == len(targets) for x in trace.values())
    assert all(len(x["attention_mass_per_slot"]) == len(targets) for x in trace.values())
    assert all(abs(sum(mass) - 1) < 1e-6 for x in trace.values() for mass in x["attention_mass_per_slot"])
    # Alter a later teacher-forced token; an earlier predicted position cannot change.
    changed = copy.deepcopy(response)
    changed.input_ids[-2] = (changed.input_ids[-2] + 1) % model.base.config.vocab_size
    other = canonical_logits(model, compiler, changed, selected, life_id=episode.episode_id)
    assert torch.equal(other[:-1], canonical[:-1])


def test_counterfactual_swap_preserves_address_and_time_metadata():
    from dsrsi.memory.store import MemoryRecord
    actual = MemoryRecord("source", "life", torch.ones(2, 4), torch.ones(2, 3), 1, 3, 4)
    foreign = MemoryRecord("source", "other-life", torch.full((2, 4), 8.0), torch.zeros(2, 3), 20, 30, 31)
    swapped = swapped_payloads([actual], [foreign])[0]
    assert torch.equal(swapped.slots, foreign.slots)
    assert torch.equal(swapped.keys, actual.keys)
    assert (swapped.stream_id, swapped.source_begin, swapped.source_end, swapped.visible_after) == ("life", 1, 3, 4)


def test_reader_trace_matches_actual_fp32_attention_inside_bfloat16_autocast(monkeypatch):
    import math
    import dsrsi.memory.modules as modules

    torch.manual_seed(31)
    model = tiny().eval()
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    row = make_generator(distractors=2).episode(0, family="single_fact", language="en")
    episode = compiler.compile(row)
    with torch.no_grad():
        records = EpisodeObjective(model, compiler)._sources(episode)[0][:2]
    response = episode.responses[0]
    expected = []
    original = modules._attention

    def capture(q, k, v, valid, heads):
        batch, queries, width = q.shape
        if valid.ndim == 2:
            valid = valid[:, None].expand(-1, queries, -1)
        dim = width // heads
        qh = q.reshape(batch, queries, heads, dim).transpose(1, 2)
        kh = k.reshape(batch, k.shape[1], heads, dim).transpose(1, 2)
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = qh.float() @ kh.float().transpose(-1, -2) / math.sqrt(dim)
            mask = valid[:, None]
            scores = scores.masked_fill(~mask, -torch.inf)
            scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            expected.append(scores.softmax(-1).masked_fill(~mask, 0)[0].mean(0))
        return original(q, k, v, valid, heads)

    monkeypatch.setattr(modules, "_attention", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plain = native_logits(model, response, records, life_id=episode.episode_id)
        expected.clear()
        trace = {}
        traced = native_logits(model, response, records, life_id=episode.episode_id, reader_trace=trace)
    assert torch.equal(plain, traced)
    positions = list(range(response.prompt_length - 1, len(response.input_ids) - 1))
    assert len(expected) == len(model.readers)
    for item, actual in zip(trace.values(), expected):
        observed = torch.tensor(item["attention_mass_per_slot"])
        assert torch.allclose(observed, actual[positions], atol=1e-7, rtol=0)
