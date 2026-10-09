"""Versioned A/B controls with matched entities, attributes and source positions."""

from datetime import timedelta
import hashlib
from pathlib import Path
import random
import re

from .generate import BASE_TIME, EpisodeGenerator, LOCATIONS

LEGACY_AB_REVISION = "ab-v1"
AB_DATA_REVISION = "ab-v2-matched-entities"


def data_identity(revision):
    if revision not in (LEGACY_AB_REVISION, AB_DATA_REVISION):
        raise ValueError("unknown A/B data revision")
    root = Path(__file__).resolve().parent
    names = ("generate.py", "compiler.py", "tools.py")
    if revision == AB_DATA_REVISION:
        names += ("robust.py",)
    return {"revision": revision, "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                           for name in names}}


def make_generator(*, revision=AB_DATA_REVISION, **kwargs):
    if revision == LEGACY_AB_REVISION:
        return EpisodeGenerator(**kwargs)
    if revision == AB_DATA_REVISION:
        return RobustEpisodeGenerator(**kwargs)
    raise ValueError("unknown A/B data revision")


class RobustEpisodeGenerator(EpisodeGenerator):
    def episode(self, ordinal, *, variant=0, family=None, language=None):
        # Positive/program truth templates remain the established controls.
        # A separate keyed RNG fixes all negative payloads, IDs and positions
        # across the two counterfactual variants.
        row = EpisodeGenerator(seed=self.seed, split=self.split, distractors=0).episode(
            ordinal, variant=variant, family=family, language=language)
        digest = hashlib.sha256(f"{AB_DATA_REVISION}:{self.seed}:{self.split}:{ordinal}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        lang, family = row["source_language"], row["family"]
        prefix = rng.choice(("青杉", "海松", "红枫", "白杨") if lang == "zh" else ("Cedar", "Pine", "Maple", "Poplar"))
        serial = rng.randrange(1_000_000)
        entity = f"{prefix}-{serial:06d}"
        old_pattern = r"(?:青杉|Cedar)-\d{4}-\d+"
        for source in row["source_events"]:
            source["content"] = re.sub(old_pattern, entity, source["content"])
        for probe in row["probes"]:
            probe["query"] = re.sub(old_pattern, entity, probe["query"])

        positives = row["source_events"]
        count = self.distractors + len(positives)
        positions = sorted(rng.sample(range(count), len(positives)))
        ids = [f"r{number:08x}" for number in rng.sample(range(1 << 30), count)]
        remap = {}
        positive_at = {}
        for position, source in zip(positions, positives):
            remap[source["event_id"]] = ids[position]
            positive_at[position] = source
        events = []
        negatives = []

        def code():
            return f"{chr(65 + rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000):03d}"

        def negative(number):
            # Shared prefixes and nearby serials have the same length. Every
            # third distractor uses the exact entity but a different attribute.
            same_entity = number % 3 == 0
            other = entity if same_entity else f"{prefix}-{(serial + number + 1) % 1_000_000:06d}"
            if family in ("single_fact", "missing_evidence", "language_preservation"):
                attribute = "箱封码" if same_entity else "校验码"
                attribute_en = "seal code" if same_entity else "code"
                value = code()
                content = (f"{other}本次{attribute}为 {value}。" if lang == "zh"
                           else f"The {attribute_en} for {other} is {value}.")
            elif family in ("paraphrase_coreference", "temporal_update"):
                value = rng.choice(LOCATIONS)
                item = "维修工具" if same_entity else "采样设备"
                item_en = "repair tools" if same_entity else "sampling device"
                content = (f"{other}项目的{item}放在{value[0]}。" if lang == "zh"
                           else f"The {item_en} for {other} is stored on {value[1]}.")
            else:
                queue = rng.choice((True, False))
                if same_entity:
                    content = (f"{other}项目的打包材料比较了纸盒和木盒。纸盒在跌落试验中破裂，木盒完好，因此选了木盒。"
                               if lang == "zh" else f"For {other}, cardboard and wooden boxes were compared. Cardboard broke in drop tests; wood remained intact. We selected wooden boxes.")
                elif queue:
                    content = (f"{other}项目试了实时同步和本地队列。实时同步在断网后丢失未确认写入，本地队列能保留并补传，因此选了本地队列。"
                               if lang == "zh" else f"For {other}, live synchronization lost unacknowledged writes during outages; a local queue retained and retried them. We selected the local queue.")
                else:
                    content = (f"{other}项目试了实时同步和本地队列。本地队列溢出丢失写入，实时同步完成所有确认，因此选了实时同步。"
                               if lang == "zh" else f"For {other}, a local queue overflowed and lost writes; live synchronization completed all acknowledgements. We selected live synchronization.")
            return {"role": "user", "content": content, "commit_boundary": True}, same_entity

        def wall_time(position):
            # Preserve chronological wall time and the actual 9 AM/4 PM update
            # anchors without giving other tasks a fixed support timestamp.
            anchors = [(-1, -60.0), (count, 480.0)]
            if family == "temporal_update":
                anchors[1:1] = [(positions[0], 0.0), (positions[1], 420.0)]
            for (left, begin), (right, end) in zip(anchors, anchors[1:]):
                if left <= position <= right:
                    minutes = begin + (end - begin) * (position - left) / (right - left)
                    return (BASE_TIME + timedelta(minutes=minutes)).isoformat()
            raise AssertionError("source position outside chronological anchors")

        negative_number = 0
        for position in range(count):
            if position in positive_at:
                source = dict(positive_at[position])
            else:
                source, same_entity = negative(negative_number)
                negatives.append({"event_id": ids[position], "same_entity_different_attribute": same_entity})
                negative_number += 1
            source.update(event_id=ids[position], time=wall_time(position))
            events.append(source)
        for probe in row["probes"]:
            probe["support_groups"] = [[remap[event] for event in group] for group in probe["support_groups"]]
            transform = probe["memory_transform"]
            if "drop_events" in transform:
                transform["drop_events"] = [remap[event] for event in transform["drop_events"]]
        row.update(source_events=events, data_revision=AB_DATA_REVISION)
        row["annotations"].update(target_entity=entity, matched_prefix=prefix, support_positions=positions,
                                  negative_kinds=negatives)
        return row
