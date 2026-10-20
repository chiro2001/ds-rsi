import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("matched_readers",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_matched_readers.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def fixture_pair():
    row = {"episode_id": "case", "probes": [{"target": "A1-234", "support_groups": [["positive"]]}],
           "source_events": [{"event_id": "positive"}, {"event_id": "negative"}]}
    reference = {"episode_id": "case", "expected": "A1-234", "selected_records": ["positive", "negative"]}
    return row, reference


def test_fixed_candidates_never_silently_follow_new_address_ranking():
    row, reference = fixture_pair()
    assert diagnostic.fixed_candidate_ids(row, reference) == ["positive", "negative"]
    for ids in [["negative", "positive"], ["positive", "positive"], ["positive", "outside"]]:
        changed = copy.deepcopy(reference)
        changed["selected_records"] = ids
        with pytest.raises(ValueError):
            diagnostic.fixed_candidate_ids(row, changed)


def test_attention_is_grouped_per_record_and_all_masked_mass_is_not_divided():
    trace = {"12": {"alpha": .1, "gate": [.2, .2], "relative_residual_rms": [.01, 0],
                    "attention_mass_per_slot": [[.1, .2, .3, .4], [0, 0, 0, 0]]}}
    result = diagnostic.record_attention_summary(trace, slots_per_record=2, candidate_count=2)["12"]
    assert result["attention_by_prediction_position"][0]["per_record"] == pytest.approx([.3, .7])
    assert result["attention_by_prediction_position"][0]["positive_fraction"] == pytest.approx(.3)
    assert result["attention_by_prediction_position"][1]["positive_fraction"] is None
    with pytest.raises(ValueError, match="slot count"):
        diagnostic.record_attention_summary(trace, slots_per_record=3, candidate_count=2)


def test_parameter_ablations_are_explicit_memory_subsets_and_never_base_parameters():
    names = ["base.layer.weight", "writer.q.weight", "readers.12.q.weight", "readers.12.k.weight",
             "readers.12.v.weight", "readers.12.o.weight", "readers.12.alpha",
             "readers.12.gate.weight", "readers.12.gate.bias", "time_source.up.weight"]
    assert diagnostic.ablation_parameter_names("old_writer", names) == ["writer.q.weight"]
    assert diagnostic.ablation_parameter_names("old_reader_qk", names) == names[2:4]
    assert diagnostic.ablation_parameter_names("old_reader_values", names) == names[4:6]
    assert diagnostic.ablation_parameter_names("old_gates", names) == names[6:9]
    assert diagnostic.ablation_parameter_names("old_time_source", names) == names[9:]
    assert "base.layer.weight" not in diagnostic.ablation_parameter_names("old_readers", names)
    with pytest.raises(ValueError):
        diagnostic.ablation_parameter_names("unknown", names)
    with pytest.raises(ValueError, match="no tensors"):
        diagnostic.ablation_parameter_names("old_writer", ["base.layer.weight"])


def test_free_generation_uses_existing_answer_free_evaluator_and_rejects_candidate_drift():
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Answer:
        selected_records: list
        text: str = "generated without gold prefix"

    class Evaluator:
        def generate(self, response, records, **kwargs):
            assert response is prompt and records is fixed
            assert kwargs == dict(life_id="life", expected="gold for scoring", mode="oracle", prefix=prefix)
            return Answer([r.record_id for r in records])

    prompt, prefix = object(), object()
    fixed = [SimpleNamespace(record_id="support"), SimpleNamespace(record_id="distractor")]
    result = diagnostic.generate_fixed_candidates(Evaluator(), prompt, fixed,
        life_id="life", expected="gold for scoring", prefix=prefix)
    assert result["selected_records"] == ["support", "distractor"]

    class Drifted(Evaluator):
        def generate(self, *args, **kwargs):
            return Answer(["distractor", "support"])

    with pytest.raises(ValueError, match="fixed candidate"):
        diagnostic.generate_fixed_candidates(Drifted(), prompt, fixed,
            life_id="life", expected="gold for scoring", prefix=prefix)


def test_real_free_generation_is_independent_of_the_gold_suffix_and_expected_answer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_qwen_bridge import tiny
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.data.revisions import make_generator
    from dsrsi.evaluation.m_only import MOnlyEvaluator

    class DecodingTokenizer(SmallVocabularyTokenizer):
        def decode(self, tokens, **kwargs):
            return " ".join(str(token) for token in tokens)

    torch.manual_seed(7)
    compiler = EpisodeCompiler(DecodingTokenizer(), max_query_tokens=1000)
    evaluator = MOnlyEvaluator(tiny().eval(), compiler, max_new_tokens=2, memory_candidates=1)
    row = make_generator(revision="ab-v6-natural-keep", seed=17, split="dev", distractors=2).episode(
        0, family="single_fact", language="en")
    episode = compiler.compile(row)
    response = episode.responses[0]
    records = [r for r in evaluator.source_records(episode) if r.record_id in response.support_groups[0]]
    prefix = evaluator._encode_prefix(response)
    a = diagnostic.generate_fixed_candidates(evaluator, response, records,
        life_id=episode.episode_id, expected=row["probes"][0]["target"], prefix=prefix)
    changed = copy.deepcopy(response)
    changed.input_ids[changed.prompt_length:] = [3] * (len(changed.input_ids) - changed.prompt_length)
    b = diagnostic.generate_fixed_candidates(evaluator, changed, records,
        life_id=episode.episode_id, expected="a different scoring target", prefix=prefix)
    assert a["token_ids"] == b["token_ids"] and a["text"] == b["text"]
    assert a["expected"] != b["expected"]
