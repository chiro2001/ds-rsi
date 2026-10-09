"""Eliminate numeric-order hints while preserving counterfactual evidence pairs."""

import hashlib
import random
import re

from .robust import RobustEpisodeGenerator

AB_ORDER_REVISION = "ab-v3-random-entity-order"


class RandomOrderEpisodeGenerator(RobustEpisodeGenerator):
    def episode(self, ordinal, *, variant=0, family=None, language=None):
        row = super().episode(ordinal, variant=variant, family=family, language=language)
        prefix = row["annotations"]["matched_prefix"]
        pattern = re.compile(re.escape(prefix) + r"-\d{6}")
        entities = sorted({match.group() for event in row["source_events"] for match in pattern.finditer(event["content"])}
                          | {row["annotations"]["target_entity"]})
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"{AB_ORDER_REVISION}:{self.seed}:{self.split}:{ordinal}".encode()).digest(), "big"))
        center = int(row["annotations"]["target_entity"].rsplit("-", 1)[1])
        radius = max(2, len(entities) * 2)
        offsets = rng.sample(range(-radius, radius + 1), len(entities))
        replacements = {name: f"{prefix}-{(center + offset) % 1_000_000:06d}" for name, offset in zip(entities, offsets)}
        for event in row["source_events"]:
            event["content"] = pattern.sub(lambda match: replacements[match.group()], event["content"])
        for probe in row["probes"]:
            probe["query"] = pattern.sub(lambda match: replacements[match.group()], probe["query"])
        row["annotations"]["target_entity"] = replacements[row["annotations"]["target_entity"]]
        ordered = sorted(replacements.values())
        row["annotations"]["target_numeric_rank"] = ordered.index(row["annotations"]["target_entity"])
        row["annotations"]["distinct_entities"] = len(ordered)
        row["data_revision"] = AB_ORDER_REVISION
        for name in ("episode_id", "world_id", "counterfactual_group_id"):
            row[name] = "v3-" + row[name]
        return row
