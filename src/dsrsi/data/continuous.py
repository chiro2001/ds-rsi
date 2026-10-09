"""Append native message frames without re-rendering already executed history."""

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import random

from dsrsi.harness.memory_tools import HarnessStore
from .generate import BASE_TIME
from .tools import tool_schemas

CONTINUOUS_DATA_REVISION = "continuous-tape-v1"


@dataclass
class MessageFrame:
    input_ids: list[int]
    labels: list[int]
    role: str
    content: str
    wall_time: float
    supervision_kind: str = "observation"
    read_at: int | None = None
    harness_after: dict | None = None
    history_id: str | None = None
    source_begin: int = 0
    source_end: int = 0
    total_targets: int = 0


class FrameCompiler:
    def __init__(self, compiler):
        self.compiler = compiler
        self.anchor = {"role": "user", "content": "__frame_anchor__"}
        self.anchor_ids = compiler.encode([self.anchor])

    def frame(self, message, *, wall_time, tools=False):
        c = self.compiler
        message = c.normalize_message(message)
        role = message["role"]
        labels, read_at, total = None, None, 0
        if role == "system":
            full = c.encode([message, self.anchor], tools=tool_schemas() if tools else None)
            if full[-len(self.anchor_ids):] != self.anchor_ids:
                raise ValueError("system frame is not separable from its native anchor")
            ids = full[:-len(self.anchor_ids)]
        elif role == "user":
            ids = c.encode([message])
        elif role == "assistant":
            prefix = c.encode([self.anchor], generation=True)
            full = c.encode([self.anchor, message])
            if full[:len(prefix)] != prefix:
                raise ValueError("assistant generation prefix is not additive")
            ids = full[len(self.anchor_ids):]
            begin = len(prefix) - len(self.anchor_ids)
            ends = [i for i in range(begin, len(ids)) if ids[i] == c.tokenizer.eos_token_id]
            if len(ends) != 1 or begin < 1:
                raise ValueError("assistant frame needs a visible prefix and exactly one end")
            end = ends[0]
            labels = [-100] * len(ids)
            labels[begin:end + 1] = ids[begin:end + 1]
            total, read_at = end - begin + 1, begin - 1
        elif role == "tool":
            preceding = [self.anchor, {"role": "assistant", "content": "__tool_anchor__"}]
            previous = c.encode(preceding)
            full = c.encode(preceding + [message])
            if full[:len(previous)] != previous:
                raise ValueError("tool observation frame rewrote its preceding native anchor")
            ids = full[len(previous):]
        else:
            raise ValueError("unsupported continuous role")
        if not ids:
            raise ValueError("empty continuous frame")
        return MessageFrame(ids, [-100] * len(ids) if labels is None else labels, role, message["content"],
                            wall_time, "tool" if message.get("tool_calls") else ("answer" if role == "assistant" else "observation"),
                            read_at=read_at, total_targets=total)


class ContinuousGenerator:
    def __init__(self, compiler, *, seed=20261010, split="train", include_tools=True, interference_tokens=2048):
        if split not in {"train", "dev", "test"} or interference_tokens < 0:
            raise ValueError("invalid continuous split/interference")
        self.frames = FrameCompiler(compiler)
        self.seed, self.split = seed, split
        self.include_tools, self.interference_tokens = include_tools, interference_tokens
        self.database_id = hashlib.sha256(f"{CONTINUOUS_DATA_REVISION}:{seed}:{split}".encode()).hexdigest()

    def plan(self, ordinal, *, life_id, journal, harness_reference, clock, startup=False, family=None):
        world, variant = divmod(ordinal, 2)
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"{CONTINUOUS_DATA_REVISION}:{self.seed}:{self.split}:{world}".encode()).digest(), "big"))
        language = rng.choices(("zh", "en"), (0.7, 0.3))[0]
        zh = language == "zh"
        families = ("fact", "task", "save", "history", "keep") if self.include_tools else ("fact", "task", "keep")
        weights = (35, 25, 15, 10, 15) if self.include_tools else (50, 35, 15)
        family = family or rng.choices(families, weights)[0]
        if family not in families:
            raise ValueError("unsupported continuous family")
        entity = f"{'青杉' if zh else 'Cedar'}-{rng.randrange(10000):04d}-{world}"
        code = f"{chr(65 + (rng.randrange(26) + variant) % 26)}{rng.randrange(10)}-{rng.randrange(1000):03d}"
        store = HarnessStore.from_state_dict(harness_reference, path=journal)
        frames, supports, last_results = [], [], []
        noise_tokens = 0
        when = BASE_TIME + timedelta(hours=world)
        def append(message, *, support=False, tools=False):
            nonlocal clock, when, last_results
            frame = self.frames.frame(message, wall_time=when.timestamp(), tools=tools)
            frame.source_begin, frame.source_end = clock, clock + len(frame.input_ids) - 1
            clock += len(frame.input_ids)
            frame.history_id = store.append_history(frame.role, frame.content or json.dumps(message, ensure_ascii=False, sort_keys=True), clock=clock)
            last_results = []
            if message.get("tool_calls"):
                for call in message["tool_calls"]:
                    function = call["function"]
                    # The action becomes visible only at this consumed frame's
                    # end. Tool observations are separate future input frames.
                    last_results.append(store.execute(function["name"], function["arguments"], clock=clock))
            frame.harness_after = store.state_dict()
            frames.append(frame)
            if support:
                supports.append({"history_id": frame.history_id, "source_begin": frame.source_begin,
                                 "source_end": frame.source_end})
            when += timedelta(minutes=1)
            return frame
        def call(name, arguments):
            frame = append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}]})
            result = last_results[0]
            append({"role": "tool", "name": name, "content": result})
            return result
        try:
            if startup:
                append({"role": "system", "content": "保持上下文接续。回答事实问题时只输出答案；工具返回是观察。" if zh else
                        "Continue the same context. Give only the answer to fact questions; tool results are observations."}, tools=self.include_tools)
            if family != "keep":
                if family == "task":
                    last = ("测实际样" if variant == 0 else "提交实验报告") if zh else (
                        "measure the real sample" if variant == 0 else "submit the experiment report")
                    append({"role": "user", "content": f"{entity}三步任务：校准传感器，测空白样，最后{last}。" if zh else
                            f"The three tasks for {entity} are calibrate the sensor, measure the blank sample, then {last}."}, support=True)
                    append({"role": "user", "content": f"{entity}前两步已经完成，第三步还没开始。" if zh else
                            f"The first two tasks for {entity} are complete; the third has not started."}, support=True)
                else:
                    text = f"{entity}入库记录。" if zh else f"Receipt record for {entity}. "
                    if family == "history":
                        text += ("逐项验收登记已完成。" * 25 if zh else "Inspection and inventory checks complete. " * 12)
                    text += f"本次校验码为 {code}。" if zh else f"The verification code is {code}."
                    append({"role": "user", "content": text}, support=True)
                while noise_tokens < self.interference_tokens:
                    number = rng.randrange(10000)
                    text = f"无关讨论{number}：明天交通安排暂定，午休使用绿色杯子，随后检查纸张。" if zh else (
                        f"Unrelated discussion {number}: travel plans are tentative; a green cup was used; paper stocks will be checked.")
                    noise_tokens += len(append({"role": "user", "content": text}).input_ids)
            if family == "fact":
                append({"role": "user", "content": f"现在找回{entity}的校验码，只回答校验码。" if zh else f"Recover the code for {entity}. Give only the code."})
                append({"role": "assistant", "content": code})
            elif family == "task":
                append({"role": "user", "content": f"现在继续{entity}，下一步是什么？只回答下一步。" if zh else f"Resume {entity}. Give only the next task."})
                append({"role": "assistant", "content": last})
            elif family == "save":
                key = f"{entity}校验码" if zh else f"{entity} verification code"
                append({"role": "user", "content": f"把{entity}的校验码保存为明文记忆，键名是{key}。" if zh else f"Save the {entity} code to explicit memory with key {key}."})
                call("memory.write", {"key": key, "text": code})
                append({"role": "assistant", "content": "已保存。" if zh else "Saved."})
            elif family == "history":
                append({"role": "user", "content": f"请用历史工具核实{entity}的原始入库记录，回答校验码。" if zh else f"Use history tools to verify the original {entity} receipt, then give its code."})
                results = call("history.search", {"query": entity})["results"]
                assert supports[0]["history_id"] in {r["id"] for r in results}
                call("history.open", {"id": supports[0]["history_id"]})
                append({"role": "assistant", "content": code})
            else:
                a, b = rng.randrange(10), rng.randrange(10)
                append({"role": "user", "content": f"{a}加{b}是多少？只回答结果。" if zh else f"What is {a} plus {b}? Give only the result."})
                append({"role": "assistant", "content": str(a + b)})
            return {"revision": CONTINUOUS_DATA_REVISION, "episode_id": f"{life_id}:case-{ordinal}",
                    "family": family, "language": language, "begin_clock": frames[0].source_begin,
                    "end_clock": clock, "noise_input_tokens": noise_tokens, "supports": supports,
                    "frames": [asdict(frame) for frame in frames]}
        finally:
            store.close()
