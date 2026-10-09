"""Separate memory coverage, free answer accuracy and paired-world accuracy."""

from collections import defaultdict


def retrieval_summary(records):
    eligible = [r for r in records if r["required_groups"] and r["evidence_available"]]
    limits = {r.get("retrieval_k", 8) for r in records}
    if len(limits) > 1:
        raise ValueError("cannot aggregate mixed retrieval limits as one coverage metric")
    limit = next(iter(limits), 8)
    return {"queries": len(records), "eligible_queries": len(eligible),
            f"required_group_coverage_at_{limit}": (sum(r["support_coverage"] for r in eligible) / len(eligible)
                                              if eligible else None),
            "excluded_missing_or_no_support_queries": len(records) - len(eligible)}


def summarize_records(records, *, retrieval_only=False):
    if retrieval_only:
        summary = retrieval_summary(records)
        strata = defaultdict(list)
        for row in records:
            strata[f"{row['family']}/{row['language']}"].append(row)
        summary["by_family_language"] = {key: retrieval_summary(values) for key, values in strata.items()}
        return summary
    modes = defaultdict(list)
    for row in records:
        modes[row["mode"]].append(row)
    summary = {}
    for mode, values in modes.items():
        pairs = defaultdict(list)
        for row in values:
            pairs[(row["family"], row["language"], row["world_id"], row["probe_id"])].append(row)
        complete = [pair for pair in pairs.values() if len(pair) == 2 and {r["variant"] for r in pair} == {0, 1}]
        summary[mode] = {"queries": len(values),
                         "exact_match": sum(r["exact_match"] for r in values) / len(values),
                         "end_rate": sum(r["stopped_on_end"] for r in values) / len(values),
                         "actual_support_coverage": sum(r["actual_support_coverage"] for r in values) / len(values),
                         "complete_counterfactual_pairs": len(complete),
                         "paired_exact_match": (sum(all(r["exact_match"] for r in pair) for pair in complete) / len(complete)
                                                 if complete else None)}
        if mode in ("foreign", "oracle_foreign"):
            summary[mode]["follows_counterfactual_payload"] = (sum(r["follows_counterfactual_payload"] for r in values) / len(values)
                if all("follows_counterfactual_payload" in r for r in values) else None)
    return summary
