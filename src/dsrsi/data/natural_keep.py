"""Versioned V5 fact curriculum plus long non-memory continuation distillation.

Procedural passages are independent of the pinned evaluation corpus. They
provide causal continuation positions for base-logit KL, not unique hard
answers to an open-ended writing request.
"""

import hashlib
import random

from .indexed_digits import IndexedDigitEpisodeGenerator

AB_NATURAL_KEEP_REVISION = "ab-v6-natural-keep"


def continuation_task(seed, split, ordinal, language):
    key = f"natural-keep-v1:{seed}:{split}:{ordinal}"
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
    if language == "zh":
        place = rng.choice(("街角的图书馆", "车站旁的小公园", "社区的旧书店", "河边的展览馆"))
        person = rng.choice(("林老师", "陈阿姨", "周先生", "王同学"))
        time = rng.choice(("一个初秋的下午", "雨后的清晨", "周末的傍晚", "春天的一个早晨"))
        prefix = (f"{time}，{person}走进{place}。窗边的光线很柔和，外面的声音渐渐安静下来。"
                  "几个人围坐在桌旁，认真地听着一位老人讲述过去的事情。")
        target = ("老人停下来喝了一口水，随后又慢慢地讲下去。有人把刚才听到的细节记在纸上，"
                  "也有人提出新的问题。大家并没有急着得出结论，而是把不同的经历放在一起比较。"
                  "谈话结束时，他们收拾好桌上的东西，约定下次带来更多材料。")
        instruction = rng.choice(("请自然地接着写下面的文本，不要添加解释或标题：",
                                  "延续下面这段文字，直接给出续写内容：",
                                  "接着写这段文字，保持自然的叙述："))
    elif language == "en":
        place = rng.choice(("the neighborhood library", "a small park near the station",
                            "an old bookshop", "the exhibition hall by the river"))
        person = rng.choice(("Alex", "Morgan", "Sam", "Taylor"))
        time = rng.choice(("On a quiet autumn afternoon", "Early in the morning after the rain",
                           "On a weekend evening", "One morning in spring"))
        prefix = (f"{time}, {person} walked into {place}. The light near the window was soft, "
                  "and the sounds outside gradually faded. A few people were sitting around a table, "
                  "listening carefully as an older visitor described events from many years ago.")
        target = ("The visitor paused for a sip of water and then continued the story. "
                  "One person wrote down a detail, while another asked a question. "
                  "Instead of rushing to a conclusion, the group compared their different experiences. "
                  "When the conversation ended, they put their materials away and agreed to bring more next time.")
        instruction = rng.choice(("Continue the text below naturally. Do not add explanations or headings:",
                                  "Continue the following passage. Give only the continuation:",
                                  "Keep writing this passage in the same natural style:"))
    else:
        raise ValueError("continuations require a concrete query language")
    return instruction + "\n\n" + prefix, target


class NaturalKeepEpisodeGenerator(IndexedDigitEpisodeGenerator):
    def episode(self, ordinal, *, variant=0, family=None, language=None):
        row = super().episode(ordinal, variant=variant, family=family, language=language)
        draw = int.from_bytes(hashlib.sha256(
            f"natural-keep-choice-v1:{self.seed}:{self.split}:{ordinal}".encode()).digest()[:8], "big")
        if row["family"] == "language_preservation" and draw % 2 == 0:
            query_language = row.get("query_language", row["language"])
            query, target = continuation_task(self.seed, self.split, ordinal, query_language)
            probe = row["probes"][0]
            probe.update(query=query, target=target)
            row["annotations"].update(natural_keep=True, keep_objective="base_kl_only_v1")
            # Include ordinary conversations without wall-clock metadata as
            # well as the original time-aware controls. Logical order remains.
            if (draw // 2) % 2 == 0:
                for event in row["source_events"]:
                    event.pop("time", None)
                probe.pop("query_time", None)
            row["annotations"]["keep_wall_time_known"] = "query_time" in probe
        row["data_revision"] = AB_NATURAL_KEEP_REVISION
        for name in ("episode_id", "world_id", "counterfactual_group_id"):
            row[name] = "v6-" + row[name].removeprefix("v5-")
        return row
