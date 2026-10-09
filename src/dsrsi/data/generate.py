"""Stateless episode generation: seed/split/ordinal determine the complete world."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

RESET = ["kv", "gdn", "conv", "pending_writer"]
FAMILIES = ["single_fact", "paraphrase_coreference", "episode_and_task", "temporal_update", "missing_evidence"]
WEIGHTS = [0.40, 0.25, 0.20, 0.10, 0.05]
BASE_TIME = datetime(2026, 1, 12, 9, tzinfo=timezone(timedelta(hours=8)))
LOCATIONS = [("东侧柜第二层", "the second shelf of the east cabinet"),
             ("西侧架第三层", "the third shelf of the west rack"),
             ("北侧柜第一层", "the first shelf of the north cabinet"),
             ("南侧架第四层", "the fourth shelf of the south rack")]


class EpisodeGenerator:
    def __init__(self, *, seed=20261010, split="train", distractors=7):
        if split not in {"train", "dev", "test"} or distractors < 0:
            raise ValueError("invalid split/distractor count")
        self.seed = seed
        self.split = split
        self.distractors = distractors

    def _rng(self, ordinal):
        value = hashlib.sha256(f"{self.seed}:{self.split}:{ordinal}".encode()).digest()
        return random.Random(int.from_bytes(value, "big"))

    def episode(self, ordinal: int, *, variant=0, family=None, language=None):
        if ordinal < 0 or variant not in (0, 1):
            raise ValueError("invalid ordinal/variant")
        rng = self._rng(ordinal)
        family = family or rng.choices(FAMILIES, WEIGHTS)[0]
        language = language or rng.choices(["zh", "en", "cross_language"], [0.6, 0.3, 0.1])[0]
        if family not in FAMILIES + ["language_preservation"] or language not in {"zh", "en", "cross_language"}:
            raise ValueError("unknown family/language")
        src_lang = rng.choice(["zh", "en"]) if language == "cross_language" else language
        query_lang = ("en" if src_lang == "zh" else "zh") if language == "cross_language" else language
        entity = f"{'青杉' if src_lang == 'zh' else 'Cedar'}-{rng.randrange(10000):04d}-{ordinal}"
        code = f"{chr(65 + rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000):03d}"
        value = chr(65 + (ord(code[0]) - 65 + variant) % 26) + code[1:]
        events = []

        def event(content, event_id=None, hour=None):
            number = len(events)
            when = BASE_TIME + timedelta(minutes=number) if hour is None else BASE_TIME.replace(hour=hour)
            result = {"event_id": event_id or f"n{number}", "time": when.isoformat(),
                      "role": "user", "content": content, "commit_boundary": True}
            events.append(result)
            return result["event_id"]

        def probe(text, target, groups, probe_id="p1", transform=None):
            return {"probe_id": probe_id, "mode": "m_only", "query": text, "target": target,
                    "query_time": BASE_TIME.replace(hour=18).isoformat(), "support_groups": groups,
                    "allow_tools": [], "write_enabled": False, "memory_transform": transform or {}}

        # Separate independently encoded records make negative/support labels exact.
        for number in range(self.distractors):
            other = f"{'海松' if src_lang == 'zh' else 'Pine'}-{rng.randrange(10000):04d}-{ordinal}-{number}"
            other_code = f"{chr(65 + rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000):03d}"
            event(f"{other}本次校验码为 {other_code}。" if src_lang == "zh" else f"The code for {other} is {other_code}.")
        heldout = 0 if self.split == "train" else (1 if self.split == "dev" else 2)
        if family in {"single_fact", "missing_evidence"}:
            support = event(f"{entity}本次校验码为 {value}。" if src_lang == "zh" else f"The code for {entity} is {value}.", "fact")
            queries = ([f"{entity}的校验码是什么？请只回答校验码。", f"上回登记{entity}时使用了哪个码？", f"请找回给{entity}分配的校验字符串。"]
                       if query_lang == "zh" else [f"What is the code for {entity}? Give only the code.", f"Which code was recorded for {entity}?", f"Recover the verification string assigned to {entity}."])
            target = value
            transform = {}
            if family == "missing_evidence":
                transform = {"drop_events": [support]}
                target = "当前可用记忆没有足够信息确定这个校验码。" if query_lang == "zh" else "The available memory does not contain enough evidence to determine that code."
            probes = [probe(queries[heldout], target, [[support]], transform=transform)]
        elif family == "paraphrase_coreference":
            location = LOCATIONS[(rng.randrange(len(LOCATIONS)) + variant) % len(LOCATIONS)]
            support = event(f"{entity}项目的采样设备放在{location[0]}。" if src_lang == "zh" else f"The sampling device for {entity} is stored on {location[1]}.", "location")
            text = ([f"{entity}项目那台设备收在哪里？", f"要继续{entity}的采样了，上次设备放到了什么地方？", f"请回忆{entity}的仪器保存位置。"]
                    if query_lang == "zh" else [f"Where did we store the device for {entity}?", f"We are resuming {entity}; where was its equipment left?", f"Recall the storage location of the instrument used by {entity}."])
            probes = [probe(text[heldout], location[0] + "。" if query_lang == "zh" else location[1] + ".", [[support]])]
        elif family == "temporal_update":
            a, b = rng.sample(LOCATIONS, 2)
            if variant:
                a, b = b, a
            old = event(f"{entity}设备放在{a[0]}。" if src_lang == "zh" else f"The device for {entity} is on {a[1]}.", "old", hour=9)
            new = event(f"刚把{entity}设备搬到了{b[0]}。" if src_lang == "zh" else f"We moved the device for {entity} to {b[1]}.", "new", hour=16)
            now = ([f"{entity}设备现在在哪里？", f"请给出{entity}仪器最新的存放位置。", f"此刻要取回{entity}的设备，应该去哪里？"]
                   if query_lang == "zh" else [f"Where is the {entity} device now?", f"Give the latest storage location for the {entity} instrument.", f"Where should we go to collect the {entity} equipment at present?"])[heldout]
            past = ([f"今天上午10点时，{entity}设备在哪里？", f"回到今天10点，{entity}仪器当时收在哪里？", f"请查明上午10点那个时刻{entity}设备的位置。"]
                    if query_lang == "zh" else [f"Where was the {entity} device at 10 AM today?", f"At 10 AM today, where had the {entity} instrument been stored?", f"Recover the location of the {entity} equipment as of 10 AM."])[heldout]
            probes = [probe(now, b[0] + "。" if query_lang == "zh" else b[1] + ".", [[new]], "now"),
                      probe(past, a[0] + "。" if query_lang == "zh" else a[1] + ".", [[old]], "past")]
        elif family == "episode_and_task":
            choose_queue = (rng.randrange(2) + variant) % 2 == 0
            if choose_queue:
                source_zh = f"{entity}项目试了实时同步和本地队列。实时同步在断网后丢失了未确认写入，本地队列能保留并补传，因此最后选了本地队列。"
                source_en = f"For {entity}, we tried live synchronization and a local queue. Live synchronization lost unacknowledged writes during outages; the queue retained and retried them. We selected the local queue."
                target_zh = "本地队列，因为它能保留断网时未确认的写入并在恢复后补传。"
                target_en = "The local queue, because it retains unacknowledged writes during outages and retries them afterward."
            else:
                source_zh = f"{entity}项目试了实时同步和本地队列。本地队列出现队列溢出导致写入丢失，实时同步在实际测试里完成了所有确认，因此最后选了实时同步。"
                source_en = f"For {entity}, we tried live synchronization and a local queue. The local queue overflowed and lost writes; live synchronization completed all acknowledgements in our tests. We selected live synchronization."
                target_zh = "实时同步，因为本地队列溢出丢失了写入，而实时同步完成了所有确认。"
                target_en = "Live synchronization, because the local queue overflowed and lost writes while live synchronization completed all acknowledgements."
            support = event(source_zh if src_lang == "zh" else source_en, "decision")
            text = ([f"{entity}项目上次最后选了什么同步方案？为什么？", f"回顾{entity}的实验结论，我们用了哪种同步方式，依据是什么？", f"请说明{entity}同步方案最终决策及其实际测试原因。"]
                    if query_lang == "zh" else [f"Which synchronization approach did {entity} finally choose, and why?", f"Review the {entity} experiment: what synchronization method was adopted, and on what evidence?", f"Explain the final synchronization decision for {entity} and the observed reason."])[heldout]
            probes = [probe(text, target_zh if query_lang == "zh" else target_en, [[support]])]
        else:
            a, b = rng.randrange(10), rng.randrange(10)
            text = f"{a}加{b}是多少？请只回答结果。" if query_lang == "zh" else f"What is {a} plus {b}? Give only the result."
            probes = [probe(text, str(a + b), [])]
        return {"schema_version": "0.1", "status": "generated_training_episode",
                "episode_id": f"{self.split}-e{ordinal:08d}-v{variant}", "world_id": f"{self.split}-w{ordinal:08d}",
                "counterfactual_group_id": f"{self.split}-cf{ordinal:08d}", "split": self.split,
                "family": family, "language": language, "source_language": src_lang,
                "query_language": query_lang, "source_encoding": "independent_events",
                "source_events": events, "probes": probes,
                "annotations": {"query_resets": RESET, "evidence_available": family != "missing_evidence",
                                "stateless_seed": self.seed, "ordinal": ordinal, "variant": variant}}


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible delayed-memory episodes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000, help="number of independent worlds")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--counterfactual-pairs", action="store_true")
    parser.add_argument("--keep-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError("choose a new output or explicit --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = EpisodeGenerator(seed=args.seed, split=args.split, distractors=args.distractors)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    counts = {}
    try:
        with os.fdopen(descriptor, "w") as stream:
            for ordinal in range(args.count):
                for variant in range(2 if args.counterfactual_pairs else 1):
                    episode = generator.episode(ordinal, variant=variant, family="language_preservation" if args.keep_only else None)
                    stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
                    counts[episode["family"]] = counts.get(episode["family"], 0) + 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"worlds": args.count, "episodes": sum(counts.values()), "families": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
