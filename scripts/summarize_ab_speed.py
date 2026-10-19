#!/usr/bin/env python3
"""Normalize historical A/B rank logs by global tokens and episode count."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(paths, first_micro=None, last_micro=None):
    ranks, checkpoints = [], []
    for path in map(Path, paths):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        micro = {}
        for row in rows:
            if row.get("event") != "micro":
                continue
            step = row["progress"]["micro_step"]
            if step in micro:
                raise ValueError(f"duplicate micro {step} in {path}; inspect rollback before timing")
            if row["micro_seconds"] <= 0:
                raise ValueError("micro timing must be positive")
            micro[step] = row
        ranks.append(micro)
        checkpoints.append([row for row in rows if row.get("event") == "checkpoint"])
    if not ranks or not ranks[0]:
        raise ValueError("nonempty rank logs are required")
    if any(rank.keys() != ranks[0].keys() for rank in ranks[1:]):
        raise ValueError("rank logs do not cover the same committed micro steps")
    steps = sorted(step for step in ranks[0]
                   if (first_micro is None or step >= first_micro)
                   and (last_micro is None or step <= last_micro))
    if not steps or steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("requested timing window must be nonempty and contiguous")
    times, tokens, teachers, samples = [], [], [], []
    for step in steps:
        rows = [rank[step] for rank in ranks]
        if any(row["progress"] != rows[0]["progress"] for row in rows[1:]):
            raise ValueError(f"rank global progress differs at micro {step}")
        total = sum(row["student_input_tokens"] for row in rows)
        teacher = sum(row["teacher_input_tokens"] for row in rows)
        if step - 1 in ranks[0]:
            prior, current = ranks[0][step - 1]["progress"], rows[0]["progress"]
            if (current["student_tokens"] - prior["student_tokens"] != total
                    or current["teacher_tokens"] - prior["teacher_tokens"] != teacher):
                raise ValueError(f"token ledger differs from summed ranks at micro {step}")
        elapsed = max(row["micro_seconds"] for row in rows)
        times.append(elapsed)
        tokens.append(total)
        teachers.append(teacher)
        samples.append({"micro": step, "seconds": elapsed, "student_tokens": total,
                        "teacher_tokens": teacher,
                        "episode_ids": [row["episode_id"] for row in rows]})
    seconds, episodes = sum(times), len(steps) * len(ranks)
    save_seconds = sum(row["save_seconds"] for row in checkpoints[0]
                       if steps[0] <= row["progress"]["micro_step"] <= steps[-1])
    return {
        "rank_count": len(ranks), "first_micro": steps[0], "last_micro": steps[-1],
        "global_micro_steps": len(steps), "episodes": episodes,
        "student_tokens": sum(tokens), "teacher_tokens": sum(teachers),
        "compute_seconds": seconds, "checkpoint_save_seconds_in_window": save_seconds,
        "student_tokens_per_second": sum(tokens) / seconds,
        "student_tokens_per_second_including_recorded_saves": sum(tokens) / (seconds + save_seconds),
        "global_micro_mean_seconds": seconds / len(steps),
        "global_micro_median_seconds": statistics.median(times),
        "episode_mean_seconds": seconds / episodes,
        "seconds_per_12_episodes": seconds / episodes * 12,
        "micro_steps_over_5_seconds": sum(t > 5 for t in times),
        "micro_max_seconds": max(times),
        "timing_scope": "sum of per-micro slowest rank; excludes startup, log writes and unrecorded gaps",
        "rank_token_counts_summed_once": True, "rank_elapsed_times_not_added": True,
        "input_logs": [{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                       for path in paths],
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-log", nargs="+", required=True)
    parser.add_argument("--first-micro", type=int)
    parser.add_argument("--last-micro", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = summarize(args.event_log, args.first_micro, args.last_micro)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
