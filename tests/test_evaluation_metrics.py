from dsrsi.evaluation.metrics import summarize_records


def test_retrieval_summary_excludes_unavailable_and_no_support_queries():
    def row(groups, available, covered, family="single_fact"):
        return {"required_groups": groups, "evidence_available": available, "support_coverage": covered,
                "family": family, "language": "zh"}
    report = summarize_records([row(1, True, True), row(2, True, False), row(1, False, False),
                                row(0, True, True, "language_preservation")], retrieval_only=True)
    assert report["queries"] == 4 and report["eligible_queries"] == 2
    assert report["required_group_coverage_at_8"] == 0.5
    assert report["excluded_missing_or_no_support_queries"] == 2
    assert report["by_family_language"]["language_preservation/zh"]["required_group_coverage_at_8"] is None


def test_paired_accuracy_requires_both_variants_of_the_same_query_to_succeed():
    def row(world, variant, correct):
        return {"mode": "actual", "family": "single_fact", "language": "zh", "world_id": world,
                "probe_id": "p1", "variant": variant, "exact_match": correct,
                "actual_support_coverage": True, "stopped_on_end": True}
    report = summarize_records([row("w0", 0, True), row("w0", 1, False), row("w1", 0, True)])
    assert report["actual"]["exact_match"] == 2 / 3
    assert report["actual"]["complete_counterfactual_pairs"] == 1
    assert report["actual"]["paired_exact_match"] == 0
