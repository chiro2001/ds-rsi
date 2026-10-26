"""Source-only U/P/C payload packaging for generated-weight memory pilots."""

import copy

from .fixed_pair import FixedPairEpisodeGenerator


SHINE_DATA_VERSION = "shine-source-payload-upc-v1"


def shine_episode(world, *, variant=0, task="U", split="train", language="zh", seed=20261026):
    if task not in {"U", "P", "C"}:
        raise ValueError("SHINE task must be U, P or C")
    g = FixedPairEpisodeGenerator(seed=seed, split=split, distractors=7, probe_view="full")
    row = copy.deepcopy(g.episode(world, variant=variant, family="single_fact", language=language))
    ids = row["annotations"]["fixed_memory_candidate_ids"]
    sources = [event for event in row["source_events"] if event["event_id"] in ids]
    if task == "U":
        binding = row["annotations"]["pair_bindings"][0]
        row["source_events"] = [event for event in sources if event["event_id"] == binding["event_id"]]
        first = row["probes"][0]
        second = copy.deepcopy(first)
        second["probe_id"] = "p2"
        name = ("箱封码" if binding["attribute"] == "seal" else "校验码") if row["query_language"] == "zh" else (
            "seal code" if binding["attribute"] == "seal" else "verification code")
        second["query"] = (f"请给出{binding['entity']}所登记的{name}，仅输出完整编码。" if row["query_language"] == "zh" else
                           f"Give the complete {name} recorded for {binding['entity']}. Output the code only.")
        row["probes"] = [first, second]
    elif task == "P":
        merged = dict(sources[-1], event_id="joint-source", content="\n".join(s["content"] for s in sources))
        row["source_events"] = [merged]
        for probe in row["probes"]:
            probe["support_groups"] = [["joint-source"]]
    else:
        row["source_events"] = sources
    row["annotations"]["fixed_memory_candidate_ids"] = [s["event_id"] for s in row["source_events"]]
    row["annotations"]["payload_task"] = task
    row["data_revision"] = SHINE_DATA_VERSION
    for name in ("episode_id", "world_id", "counterfactual_group_id"):
        row[name] = f"shine-{task}-" + row[name]
    return row
