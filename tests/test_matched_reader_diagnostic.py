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
