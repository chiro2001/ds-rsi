"""Deterministic ordinary-task keep samples, separate from held-out probes."""

import hashlib
import json
import random


def keep_task(seed, split, ordinal, language):
    rng = random.Random(int.from_bytes(hashlib.sha256(f"keep-general-v1:{seed}:{split}:{ordinal}".encode()).digest(), "big"))
    family = rng.choice(("arithmetic", "reading", "logic", "code", "format", "knowledge"))
    zh = language == "zh"
    if family == "arithmetic":
        a, b = rng.randrange(12, 90), rng.randrange(2, 12)
        operation = rng.choice(("+", "-", "*"))
        value = {"+": a + b, "-": a - b, "*": a * b}[operation]
        query = (f"计算{a}{operation}{b}，仅给出结果数字。" if zh else f"Calculate {a}{operation}{b}. Output just the numeric result.")
        target = str(value)
    elif family == "reading":
        identifier = f"V{rng.randrange(1000, 9999)}"
        query = (f"从下面记录抽取登记编号，只输出编号：交货窗口为周二，登记编号为{identifier}，接收人为小林。"
                 if zh else f"Extract just the registration ID from this note: delivery is Tuesday, registration ID is {identifier}, and the recipient is Sam.")
        target = identifier
    elif family == "logic":
        names = rng.sample(("A", "B", "C", "D", "E"), 3)
        query = (f"{names[0]}早于{names[1]}到达，{names[1]}早于{names[2]}到达。谁最先到达？只输出代号。"
                 if zh else f"{names[0]} arrives before {names[1]}, and {names[1]} before {names[2]}. Who arrives first? Output just the label.")
        target = names[0]
    elif family == "code":
        values = [rng.randrange(10) for _ in range(rng.randrange(2, 5))]
        operation = rng.choice(("sum", "len", "max"))
        target = str({"sum": sum(values), "len": len(values), "max": max(values)}[operation])
        expression = f"{operation}({values})"
        query = (f"求Python表达式{expression}的值，仅输出结果。" if zh else f"Evaluate the Python expression {expression}. Output only its value.")
    elif family == "format":
        value = rng.randrange(20)
        query = (f"请仅返回JSON：用键n存放数字{value}，不得附带说明。"
                 if zh else f"Return only JSON: store the number {value} under the key n. Do not add explanation.")
        target = json.dumps({"n": value}, separators=(",", ":"))
    else:
        facts = [("日本的首都", "the capital of Japan", "东京", "Tokyo"),
                 ("意大利的首都", "the capital of Italy", "罗马", "Rome"),
                 ("地球上最大的海洋", "the largest ocean on Earth", "太平洋", "Pacific Ocean"),
                 ("地球的天然卫星", "Earth's natural satellite", "月球", "Moon")]
        fact = rng.choice(facts)
        query = f"请给出{fact[0]}的名称，不输出解释。" if zh else f"Name {fact[1]}. Do not give an explanation."
        target = fact[2] if zh else fact[3]
    return {"family": family, "query": query, "target": target}
