"""Counterfactuals cover every code character, with diverse ordinary-task keep."""

import hashlib
import random
import re

from .robust_order import RandomOrderEpisodeGenerator
from .general_keep import keep_task

AB_FIELDS_REVISION = "ab-v4-fields-and-language"


class AllFieldsEpisodeGenerator(RandomOrderEpisodeGenerator):
    def episode(self, ordinal, *, variant=0, family=None, language=None):
        row = super().episode(ordinal, variant=variant, family=family, language=language)
        if row["family"] in ("single_fact", "missing_evidence"):
            support = row["probes"][0]["support_groups"][0][0]
            source = next(e for e in row["source_events"] if e["event_id"] == support)
            match = re.search(r"[A-Z][0-9]-[0-9]{3}", source["content"])
            if match is None:
                raise ValueError("fact source lacks its programmatic code")
            old = match.group()
            base = chr(65 + (ord(old[0]) - 65 - variant) % 26) + old[1:]
            rng = random.Random(int.from_bytes(hashlib.sha256(
                f"{AB_FIELDS_REVISION}:{self.seed}:{self.split}:{ordinal}".encode()).digest(), "big"))
            position = rng.choice((0, 1, 3, 4, 5))
            offset = rng.randrange(1, 26 if position == 0 else 10)
            value = list(base)
            if variant:
                start = 65 if position == 0 else 48
                size = 26 if position == 0 else 10
                value[position] = chr(start + (ord(value[position]) - start + offset) % size)
            value = "".join(value)
            source["content"] = source["content"][:match.start()] + value + source["content"][match.end():]
            if row["family"] == "single_fact":
                row["probes"][0]["target"] = value
            row["annotations"]["counterfactual_character_position"] = position
        if row["family"] == "language_preservation":
            task = keep_task(self.seed, self.split, ordinal, row["query_language"])
            row["probes"][0].update(query=task["query"], target=task["target"])
            row["annotations"]["keep_task_family"] = task["family"]
        row["data_revision"] = AB_FIELDS_REVISION
        for name in ("episode_id", "world_id", "counterfactual_group_id"):
            row[name] = "v4-" + row[name].removeprefix("v3-")
        return row
