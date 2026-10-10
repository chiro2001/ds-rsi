"""Auxiliary first-output digit retrieval; held-out defaults stay full-code."""

import hashlib

from .robust_fields import AllFieldsEpisodeGenerator

AB_INDEXED_REVISION = "ab-v5-indexed-digits"


class IndexedDigitEpisodeGenerator(AllFieldsEpisodeGenerator):
    def __init__(self, *args, probe_view=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_view = probe_view or ("mixed" if self.split == "train" else "full")
        if self.probe_view not in {"mixed", "full", "indexed"}:
            raise ValueError("unknown codec probe view")

    def episode(self, ordinal, *, variant=0, family=None, language=None):
        row = super().episode(ordinal, variant=variant, family=family, language=language)
        if row["family"] == "single_fact":
            position = row["annotations"]["counterfactual_character_position"]
            draw = int.from_bytes(hashlib.sha256(
                f"{AB_INDEXED_REVISION}:{self.seed}:{self.split}:{ordinal}:view".encode()).digest()[:8], "big") % 2
            indexed = self.probe_view == "indexed" or (
                self.probe_view == "mixed" and position >= 3 and draw == 0)
            probe = row["probes"][0]
            row["annotations"]["codec_task"] = "full_code"
            if indexed:
                entity = row["annotations"]["target_entity"]
                if row["query_language"] == "en":
                    if position == 0:
                        query = f"Return only the uppercase letter in {entity}'s identifier code."
                    elif position == 1:
                        query = f"Return only the digit before the hyphen in {entity}'s identifier code."
                    else:
                        name = ("first", "second", "third")[position - 3]
                        query = (f"In {entity}'s identifier code, what is the {name} digit of the three-digit suffix "
                                 "after the hyphen? Return one digit only.")
                else:
                    if position == 0:
                        query = f"请只回答{entity}标识编码中的大写字母。"
                    elif position == 1:
                        query = f"请只回答{entity}标识编码连字符之前的数字。"
                    else:
                        query = f"请只回答{entity}标识编码连字符之后三位数字中，从左数第{position - 2}位。只写一个数字。"
                probe["query"], probe["target"] = query, probe["target"][position]
                row["annotations"].update(codec_task="indexed_character", answer_character_position=position)
        row["data_revision"] = AB_INDEXED_REVISION
        for name in ("episode_id", "world_id", "counterfactual_group_id"):
            row[name] = "v5-" + row[name].removeprefix("v4-")
        return row
