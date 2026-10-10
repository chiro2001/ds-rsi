from dsrsi.evaluation.metrics import code_field_summary


def test_code_fields_do_not_confuse_one_letter_with_whole_memory_content():
    rows = [{"expected": "Q7-492", "text": "Q1-999"}, {"expected": "R8-315", "text": "R8-111"},
            {"expected": "T2-804", "text": "unknown"}, {"expected": "a city", "text": "Paris"}]
    summary = code_field_summary(rows)
    assert summary["queries"] == 3 and summary["valid_format"] == 2 / 3
    assert summary["letter_accuracy"] == 2 / 3 and summary["digit_accuracy"] == 1 / 3
    assert summary["suffix_accuracy"] == 0
    assert summary["suffix_digit_accuracy"] == [0, 2 / 3, 0]


def test_counterfactual_field_scoring_uses_the_explicit_replaced_payload_target():
    rows = [{"expected": "Q7-492", "counterfactual_payload_answer": "R7-492", "text": "R7-492。"}]
    assert code_field_summary(rows)["letter_accuracy"] == 0
    assert code_field_summary(rows, target_key="counterfactual_payload_answer")["letter_accuracy"] == 1
