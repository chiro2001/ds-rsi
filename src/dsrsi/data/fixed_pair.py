"""Versioned two-question worlds with a question-independent candidate pair."""

import copy
import hashlib
import random
import re

from .natural_keep import NaturalKeepEpisodeGenerator


FIXED_PAIR_REVISION = "ab-v7-fixed-pair-qa"
_CODE = re.compile(r"[A-Z][0-9]-[0-9]{3}")
_ENTITY = re.compile(r"(?:青杉|海松|红枫|白杨|Cedar|Pine|Maple|Poplar)-[0-9]{6}")


def _rng(seed, split, ordinal, purpose):
    digest = hashlib.sha256(f"{FIXED_PAIR_REVISION}:{seed}:{split}:{ordinal}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _question(entity, attribute, position, language, split):
    zh = language == "zh"
    name = ("箱封码" if attribute == "seal" else "校验码") if zh else (
        "seal code" if attribute == "seal" else "verification code")
    if position is None:
        if zh:
            return (f"{entity}的{name}是什么？请只回答编码。" if split == "train" else
                    f"请找回给{entity}登记的{name}，只输出编码。")
        return (f"What is {entity}'s {name}? Give only the code." if split == "train" else
                f"Recover the {name} registered for {entity}. Output only the code.")
    if position == 0:
        return (f"请只回答{entity}的{name}中的大写字母。" if zh else
                f"Return only the uppercase letter in {entity}'s {name}.")
    if position == 1:
        return (f"请只回答{entity}的{name}连字符之前的数字。" if zh else
                f"Return only the digit before the hyphen in {entity}'s {name}.")
    ordinal = ("first", "second", "third")[position - 3]
    return (f"请只回答{entity}的{name}连字符之后三位数字中，从左数第{position - 2}位。只写一个数字。" if zh else
            f"In {entity}'s {name}, what is the {ordinal} digit of the three-digit suffix after the hyphen? Return one digit only.")


def _content(entity, code, attribute, language, layout, rng):
    if language == "zh":
        name = "箱封码" if attribute == "seal" else "校验码"
        choices = (f"{entity}本次{name}为 {code}。", f"登记的编码是 {code}，对应{entity}的{name}。",
                   f"{entity}的{name}：{code}。这条登记已确认。")
    else:
        name = "seal code" if attribute == "seal" else "code"
        choices = (f"The {name} for {entity} is {code}.",
                   f"Recorded value: {code}. This is the {name} assigned to {entity}.",
                   f"For {entity}, the {name} is {code}. This entry is confirmed.")
    return choices[0] if layout == "native" else rng.choice(choices)


class FixedPairEpisodeGenerator(NaturalKeepEpisodeGenerator):
    def __init__(self, *args, source_layout="native", **kwargs):
        super().__init__(*args, **kwargs)
        if source_layout not in {"native", "varied"} or self.distractors < 2:
            raise ValueError("fixed pairs require at least two distractors and a known source layout")
        self.source_layout = source_layout

    def episode(self, ordinal, *, variant=0, family=None, language=None):
        if family not in {None, "single_fact", "language_preservation"}:
            raise ValueError("fixed-pair revision supports explicit facts and preservation only")
        requested = family or "single_fact"
        if requested == "language_preservation":
            row = super().episode(ordinal, variant=variant, family=requested, language=language)
        else:
            # Start from one immutable base world; both evidence variants share
            # entities, questions, ordering, timestamps and all negative values.
            row = super().episode(ordinal, variant=0, family=requested, language=language)
            if variant not in (0, 1):
                raise ValueError("invalid evidence variant")
            rng = _rng(self.seed, self.split, ordinal, "pair")
            first_id = row["probes"][0]["support_groups"][0][0]
            first_entity = row["annotations"]["target_entity"]
            sources = row["source_events"]
            candidates = [i for i, event in enumerate(sources)
                          if _ENTITY.search(event["content"]).group() != first_entity
                          and ("本次校验码" in event["content"] or event["content"].startswith("The code for "))]
            partner = rng.choice(candidates)
            second_id = sources[partner]["event_id"]
            attribute_pair = bool(rng.randrange(2))
            position = row["annotations"]["counterfactual_character_position"]
            indexed = row["annotations"].get("codec_task") == "indexed_character"
            bindings = []
            for index, event in enumerate(sources):
                old = event["content"]
                entity, code = _ENTITY.search(old).group(), _CODE.search(old).group()
                attribute = "seal" if ("箱封码" in old or "seal code" in old) else "verification"
                if index == partner and attribute_pair:
                    entity, attribute = first_entity, "seal"
                elif attribute_pair and event["event_id"] != first_id and entity == first_entity:
                    # Other old seal distractors must not become conflicting
                    # answers to the newly introduced seal-code query.
                    prefix, serial = first_entity.rsplit("-", 1)
                    entity = f"{prefix}-{(int(serial) + 100000 + index) % 1000000:06d}"
                if event["event_id"] in {first_id, second_id}:
                    if variant:
                        chars = list(code)
                        base, size = (65, 26) if position == 0 else (48, 10)
                        offset = _rng(self.seed, self.split, ordinal, event["event_id"] + ":cf").randrange(1, size)
                        chars[position] = chr(base + (ord(chars[position]) - base + offset) % size)
                        code = "".join(chars)
                    bindings.append(dict(event_id=event["event_id"], entity=entity,
                                         attribute=attribute, code=code))
                event["content"] = _content(entity, code, attribute, row["source_language"], self.source_layout,
                                            _rng(self.seed, self.split, ordinal, event["event_id"] + ":layout"))
            # Candidate order is original source order, never the answer order.
            ids = [event["event_id"] for event in sources if event["event_id"] in {first_id, second_id}]
            template = row["probes"][0]
            probes = []
            for number, binding in enumerate(bindings):
                probe = copy.deepcopy(template)
                probe.update(probe_id=f"p{number + 1}",
                             query=_question(binding["entity"], binding["attribute"], position if indexed else None,
                                             row["query_language"], self.split),
                             target=binding["code"][position] if indexed else binding["code"],
                             support_groups=[[binding["event_id"]]])
                binding["probe_id"] = probe["probe_id"]
                probes.append(probe)
            row["probes"] = probes
            row["annotations"].update(variant=variant, fixed_memory_candidate_ids=ids,
                                      pair_bindings=bindings, pair_kind="attribute" if attribute_pair else "entity",
                                      source_layout=self.source_layout,
                                      support_positions=[i for i, e in enumerate(sources) if e["event_id"] in ids])
            row["annotations"]["negative_kinds"] = [v for v in row["annotations"]["negative_kinds"]
                                                       if v["event_id"] not in ids]
            row["episode_id"] = row["episode_id"].rsplit("-v", 1)[0] + f"-v{variant}"
        row["data_revision"] = FIXED_PAIR_REVISION
        for name in ("episode_id", "world_id", "counterfactual_group_id"):
            row[name] = "v7-" + row[name].removeprefix("v6-")
        return row
