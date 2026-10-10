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
