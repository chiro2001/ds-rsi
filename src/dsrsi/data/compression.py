"""D-stage compression curriculum: 3d+1 leaves with the support in the first four.

The protocol (docs/semantic-compression.md) fixes the shape of a compression
episode: the answer-bearing support must live in the *first* group of four
leaves, and every later compression step merges the previous summary with
exactly three fresh leaves, so a depth-d episode really experiences a chain of
length d over 3d+1 leaves. Support is in the oldest group; later leaves are
independent facts about unasked entities, so retaining them is never required
by the supervised target.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import random

COMPRESSION_REVISION = "ab-d-compression-v1"
COMPRESSION_DEPTHS = (1, 2, 4, 8)
COMPRESSION_RESET = ["kv", "gdn", "conv", "pending_writer"]
BASE_TIME = datetime(2026, 1, 12, 9, tzinfo=timezone(timedelta(hours=8)))


def leaves_for_depth(depth: int) -> int:
    """Leaf count of a depth-d chain: 4 leaves then three per extra level."""
    if type(depth) is not int or depth not in COMPRESSION_DEPTHS:
        raise ValueError(f"compression depth must be one of {COMPRESSION_DEPTHS}")
    return 3 * depth + 1


def compression_groups(depth: int) -> list[tuple[int, int]]:
    """Half-open leaf ranges merged at each step: level 1 takes leaves 0..4,
    level k takes three fresh leaves after the previous summary."""
    leaves_for_depth(depth)
    groups = [(0, 4)]
    for level in range(2, depth + 1):
        start = 3 * level - 2
        groups.append((start, start + 3))
    return groups


def compression_plan(row) -> dict:
    """Validated chain layout for one generated row."""
    events = [event["event_id"] for event in row["source_events"]]
    annotations = row.get("annotations", {})
    depth = annotations.get("compression_depth")
    if depth not in COMPRESSION_DEPTHS or annotations.get("leaf_count") != leaves_for_depth(depth):
        raise ValueError("row does not carry a valid compression depth/leaf count")
    if len(events) != leaves_for_depth(depth) or len(set(events)) != len(events):
        raise ValueError("compression rows require exactly 3d+1 distinct leaf events")
    support = annotations.get("support_event")
    groups = compression_groups(depth)
    if [tuple(group) for group in annotations.get("compression_groups", [])] != groups:
        raise ValueError("declared compression groups differ from the depth chain")
    support_group = next((index for index, (begin, end) in enumerate(groups) if support in events[begin:end]), None)
    if support_group is None:
        raise ValueError("support event is not part of the compression chain")
    return {"depth": depth, "leaves": events, "groups": groups, "support_event": support,
            "support_group": support_group, "support_probe_id": annotations.get("support_probe_id")}


class CompressionEpisodeGenerator:
    """Deterministic 3d+1 leaf episodes for the D content stage.

    Extra A/B distractors are not meaningful here: the chain length *is* the
    distractor structure, so a nonzero `distractors` value is rejected instead
    of being silently ignored.
    """

    def __init__(self, *, seed=20261010, split="train", distractors=0):
        if split not in {"train", "dev", "test"}:
            raise ValueError("invalid split")
        if distractors:
            raise ValueError("compression episodes derive their length from depth; distractors must be 0")
        self.seed, self.split = seed, split

    def _rng(self, ordinal):
        value = hashlib.sha256(f"{self.seed}:{self.split}:{ordinal}".encode()).digest()
        return random.Random(int.from_bytes(value, "big"))

    def episode(self, ordinal: int, *, variant=0, family=None, language=None, depth=None):
        if ordinal < 0 or variant not in (0, 1):
            raise ValueError("invalid ordinal/variant")
        if family is not None and family != "compression_fact":
            raise ValueError("compression generator only produces compression_fact")
        rng = self._rng(ordinal)
        depth = COMPRESSION_DEPTHS[ordinal % len(COMPRESSION_DEPTHS)] if depth is None else depth
        total = leaves_for_depth(depth)
        language = language or rng.choices(["zh", "en", "cross_language"], [0.6, 0.3, 0.1])[0]
        if language not in {"zh", "en", "cross_language"}:
            raise ValueError("unknown language")
        src_lang = rng.choice(["zh", "en"]) if language == "cross_language" else language
        query_lang = ("en" if src_lang == "zh" else "zh") if language == "cross_language" else language
        prefix = "青杉" if src_lang == "zh" else "Cedar"
        other_prefix = "海松" if src_lang == "zh" else "Pine"
        entity = f"{prefix}-{self.split}-{rng.randrange(10000):04d}-{ordinal}"
        code = f"{chr(65 + rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000):03d}"
        value = chr(65 + (ord(code[0]) - 65 + variant) % 26) + code[1:]
        events = []

        def event(content, event_id):
            number = len(events)
            events.append({"event_id": event_id, "time": (BASE_TIME + timedelta(minutes=number)).isoformat(),
                           "role": "user", "content": content, "commit_boundary": True})
            return event_id

        support = event(f"{entity}本次校验码为 {value}。" if src_lang == "zh"
                        else f"The code for {entity} is {value}.", "support")
        negatives = []
        for number in range(total - 1):
            other = f"{other_prefix}-{self.split}-{rng.randrange(10000):04d}-{ordinal}-{number}"
            other_code = f"{chr(65 + rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000):03d}"
            negatives.append((other, other_code))
            event(f"{other}本次校验码为 {other_code}。" if src_lang == "zh"
                  else f"The code for {other} is {other_code}.", f"neg-{number}")
        first = events[:4]
        rng.shuffle(first)
        events[:4] = first
        for index, source in enumerate(events):
            source["time"] = (BASE_TIME + timedelta(minutes=index)).isoformat()
        query = (f"{entity}的校验码是什么？请只回答校验码。" if query_lang == "zh"
                 else f"What is the code for {entity}? Give only the code.")
        probes = [{"probe_id": "p1", "mode": "m_only", "query": query, "target": value,
                   "query_time": BASE_TIME.replace(hour=18).isoformat(), "support_groups": [[support]],
                   "allow_tools": [], "write_enabled": False, "memory_transform": {}}]
        if self.split != "train":
            # Retention diagnostics only: the newest leaf sits inside the final
            # summary group, the first negative is compressed at level 1. Both keep
            # their true program targets so the report shows what survived
            # instead of labelling compressed detail as unknowable.
            latest = events[-1]["event_id"]
            latest_entity, latest_code = ((entity, value) if latest == support
                                           else negatives[int(latest.removeprefix("neg-"))])
            probes.append({"probe_id": "p_latest", "mode": "m_only",
                           "query": (f"{latest_entity}的校验码是什么？请只回答校验码。" if query_lang == "zh"
                                     else f"What is the code for {latest_entity}? Give only the code."),
                           "target": latest_code, "query_time": BASE_TIME.replace(hour=18).isoformat(),
                           "support_groups": [[latest]], "allow_tools": [], "write_enabled": False,
                           "memory_transform": {}})
            first_negative = next(source["event_id"] for source in events if source["event_id"] != support)
            first_entity, first_code = negatives[int(first_negative.removeprefix("neg-"))]
            probes.append({"probe_id": "p_first_negative", "mode": "m_only",
                           "query": (f"{first_entity}的校验码是什么？请只回答校验码。" if query_lang == "zh"
                                     else f"What is the code for {first_entity}? Give only the code."),
                           "target": first_code, "query_time": BASE_TIME.replace(hour=18).isoformat(),
                           "support_groups": [[first_negative]], "allow_tools": [], "write_enabled": False,
                           "memory_transform": {}})
        return {"schema_version": "0.1", "status": "generated_training_episode",
                "episode_id": f"{self.split}-d{depth}-e{ordinal:08d}-v{variant}",
                "world_id": f"{self.split}-d{depth}-w{ordinal:08d}",
                "counterfactual_group_id": f"{self.split}-d{depth}-cf{ordinal:08d}", "split": self.split,
                "family": "compression_fact", "language": language, "source_language": src_lang,
                "query_language": query_lang, "source_encoding": "independent_events",
                "source_events": events, "probes": probes,
                "annotations": {"query_resets": COMPRESSION_RESET, "evidence_available": True,
                                "stateless_seed": self.seed, "ordinal": ordinal, "variant": variant,
                                "compression_depth": depth, "leaf_count": total,
                                "compression_groups": compression_groups(depth),
                                "support_event": support, "support_probe_id": "p1",
                                },
                "data_revision": COMPRESSION_REVISION}
