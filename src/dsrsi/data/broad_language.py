"""Held-out natural-text corpus and generated broad language-retention probes.

The natural corpus is fetched from a pinned public source by a separate
script and stored outside Git.  Only the manifest (titles, revisions and
hashes) is committed.  Generated probes extend the fixed 36-query diagnostic
with deterministic arithmetic, reading, logic, code and format items; they
never participate in training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

CORPUS_REVISION = "broad-language-corpus-v1"
PROBE_REVISION = "language-retention-broad-v1"
LANGS = ("zh", "en")


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("fingerprints require text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    language: str
    title: str
    revision_id: int
    text: str

    def __post_init__(self):
        if not self.document_id or self.language not in LANGS or not self.title:
            raise ValueError("corpus documents require id, language and title")
        if not isinstance(self.revision_id, int) or self.revision_id < 1:
            raise ValueError("corpus documents require a pinned positive revision")
        if len(self.text.strip()) < 1:
            raise ValueError("corpus documents require nonempty text")

    @property
    def sha256(self) -> str:
        return text_sha256(self.text)


def load_corpus(path, manifest_path) -> list[CorpusDocument]:
    """Verify the manifest and corpus file, then return documents in file order."""
    manifest_path, path = Path(manifest_path), Path(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CORPUS_REVISION:
        raise ValueError("unexpected corpus manifest schema")
    expected_file = manifest.get("corpus_sha256")
    if not isinstance(expected_file, str) or len(expected_file) != 64:
        raise ValueError("manifest requires a corpus file fingerprint")
    actual_file = file_sha256(path)
    if actual_file != expected_file:
        raise ValueError("corpus file fingerprint differs from the manifest")
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest requires a nonempty document list")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(entries):
        raise ValueError("corpus file and manifest disagree on document count")
    documents = []
    seen = set()
    for entry, row in zip(entries, rows):
        if set(row) != {"document_id", "language", "title", "revision_id", "url", "license", "text"}:
            raise ValueError("corpus rows require the documented field set")
        document = CorpusDocument(row["document_id"], row["language"], row["title"],
                                  row["revision_id"], row["text"])
        if row["document_id"] in seen:
            raise ValueError("corpus document ids must be unique")
        seen.add(row["document_id"])
        if (entry.get("document_id") != document.document_id or entry.get("language") != document.language
                or entry.get("title") != document.title or entry.get("revision_id") != document.revision_id
                or entry.get("sha256") != document.sha256):
            raise ValueError("corpus row differs from its manifest entry")
        documents.append(document)
    return documents


def token_windows(tokens, *, window: int, limit: int) -> list[list[int]]:
    """Deterministic non-overlapping windows; drop the final partial window."""
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be at least two tokens")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("window limit must be positive")
    if len(tokens) < window:
        return []
    return [list(tokens[begin:begin + window]) for begin in range(0, len(tokens) - window + 1, window)][:limit]


def documents_for_language(documents, language: str) -> list[CorpusDocument]:
    if language not in LANGS:
        raise ValueError("unsupported corpus language")
    return [document for document in documents if document.language == language]


def select_windows(tokenizer, documents, *, window: int, budget: int):
    """Round-robin windows across documents so every document contributes.

    Returns deterministic ``(document, window_index, tokens)`` rows.  Documents
    shorter than one window are skipped rather than silently padded.
    """
    if budget < 1:
        raise ValueError("window budget must be positive")
    per_document = []
    for document in documents:
        tokens = tokenizer.encode(document.text, add_special_tokens=False)
        per_document.append((document,
                             token_windows(tokens, window=window, limit=1 << 20)))
    selected, depth = [], 0
    while len(selected) < budget:
        added = False
        for document, windows in per_document:
            if depth < len(windows) and len(selected) < budget:
                selected.append((document, depth, windows[depth]))
                added = True
        if not added:
            break
        depth += 1
    return selected


def _row(number: int, language: str, question: str, target: str) -> dict:
    from .generate import BASE_TIME, RESET
    key = f"{PROBE_REVISION}-{number}-{language}"
    return {"schema_version": "0.1", "status": "generated_training_episode", "split": "dev",
            "episode_id": key, "world_id": key, "family": "language_preservation", "language": language,
            "annotations": {"query_resets": list(RESET)},
            "source_encoding": "independent_events",
            "source_events": [{"event_id": "unrelated", "role": "user", "commit_boundary": True,
                "time": BASE_TIME.isoformat(), "content": ("白杨-941208的箱封码是J8-672。"
                     if language == "zh" else "The seal code for Poplar-941208 is J8-672.")}],
            "probes": [{"probe_id": "answer", "mode": "m_only", "query": question, "target": target,
                "query_time": BASE_TIME.replace(hour=18).isoformat(), "support_groups": [],
                "allow_tools": [], "write_enabled": False, "memory_transform": {}}]}


def generated_cases(seed: int = 20261011):
    """Deterministic (family, zh, en, answers) cases; regenerate only with care."""
    rng = random.Random(seed)
    cases = []

    def numeric(family, zh, en, value):
        cases.append((family, zh, en, [str(value), str(value)]))

    left, right = rng.randrange(12, 99), rng.randrange(3, 10)
    numeric("arithmetic", f"{left}乘以{right}是多少？只回答数字。",
            f"What is {left} times {right}? Give only the number.", left * right)
    first, second = rng.randrange(120, 989), rng.randrange(120, 989)
    numeric("arithmetic", f"{first}加上{second}是多少？只回答数字。",
            f"What is {first} plus {second}? Give only the number.", first + second)
    first, second = rng.randrange(300, 989), rng.randrange(101, 299)
    numeric("arithmetic", f"{first}减去{second}是多少？只回答数字。",
            f"What is {first} minus {second}? Give only the number.", first - second)
    divisor, quotient = rng.randrange(3, 10), rng.randrange(11, 40)
    numeric("arithmetic", f"{divisor * quotient}除以{divisor}是多少？只回答数字。",
            f"What is {divisor * quotient} divided by {divisor}? Give only the number.", quotient)
    base = rng.randrange(4, 25) * 4
    numeric("arithmetic", f"{base}的25%是多少？只回答数字。",
            f"What is 25% of {base}? Give only the number.", base // 4)
    while True:
        mean = rng.randrange(15, 70)
        first_delta, second_delta = rng.randrange(-9, 10), rng.randrange(-9, 10)
        values = [mean + first_delta, mean + second_delta, mean - first_delta - second_delta]
        if min(values) > 0 and all(value < 100 for value in values):
            break
    numeric("arithmetic", f"{values[0]}、{values[1]}和{values[2]}的平均数是多少？只回答数字。",
            f"What is the average of {values[0]}, {values[1]} and {values[2]}? Give only the number.",
            sum(values) // 3)
    base = rng.randrange(11, 25)
    numeric("arithmetic", f"{base}的平方是多少？只回答数字。",
            f"What is {base} squared? Give only the number.", base * base)
    dividend, modulus = rng.randrange(100, 999), rng.randrange(3, 10)
    numeric("arithmetic", f"{dividend}除以{modulus}的余数是多少？只回答数字。",
            f"What is the remainder when {dividend} is divided by {modulus}? Give only the number.",
            dividend % modulus)

    colors = [("蓝色", "blue"), ("红色", "red"), ("绿色", "green"), ("黑色", "black"), ("白色", "white")]
    color_zh, color_en = colors[rng.randrange(len(colors))]
    cases.append(("reading", f"读这句话：小周把{color_zh}杯子放在厨房。杯子是什么颜色？只回答颜色。",
                  f"Read this sentence: Alex put a {color_en} cup in the kitchen. "
                  "What color is the cup? Give only the color.", [color_zh, color_en]))
    weekdays = [("周一", "Monday"), ("周二", "Tuesday"), ("周三", "Wednesday"),
                ("周四", "Thursday"), ("周五", "Friday"), ("周六", "Saturday"), ("周日", "Sunday")]
    start = rng.randrange(6)
    end = rng.randrange(start + 1, 7)
    day_zh, day_en = weekdays[end]
    cases.append(("reading", f"读这句话：李琳{weekdays[start][0]}出发，{day_zh}返回。她哪天返回？只回答星期。",
                  f"Read this sentence: Lina departed on {weekdays[start][1]} and returned on {day_en}. "
                  "On which day did she return? Give only the day.", [day_zh, day_en]))
    order = rng.randrange(100, 999)
    cases.append(("reading", f"文本：订单编号是R{order}，交付地点是西门。请只提取订单编号。",
                  f"Text: The order number is R{order} and the delivery location is the west gate. "
                  "Extract only the order number.", [f"R{order}", f"R{order}"]))
    apples, pears = rng.randrange(2, 9), rng.randrange(2, 9)
    numeric("reading", f"果篮里有{apples}个苹果和{pears}个梨。共有几个水果？只回答数字。",
            f"A basket contains {apples} apples and {pears} pears. How many fruits are there in total? "
            "Give only the number.", apples + pears)
    serial = rng.randrange(1000, 9999)
    technician = [("王工", "Rowan"), ("刘工", "Priya"), ("陈工", "Mateo")][rng.randrange(3)]
    cases.append(("reading", f"设备S{serial}由{technician[0]}维护。谁维护S{serial}？只回答姓名。",
                  f"Device S{serial} is maintained by {technician[1]}. Who maintains S{serial}? "
                  "Give only the name.", [technician[0], technician[1]]))
    price, paid = rng.randrange(220, 990), rng.randrange(90, 200)
    numeric("reading", f"一件外套原价{price}元，打折后{paid}元。便宜了多少元？只回答数字。",
            f"A coat originally cost {price} yuan and now costs {paid} yuan. How many yuan cheaper is it? "
            "Give only the number.", price - paid)

    cases.append(("logic", "甲比乙高，乙比丙高。谁最高？只回答甲、乙或丙。",
                  "A is taller than B and B is taller than C. Who is tallest? Answer only A, B, or C.",
                  ["甲", "A"]))
    cases.append(("logic", "所有燕子都会飞。萌萌是一只燕子。萌萌会飞吗？只回答是或否。",
                  "All swallows can fly. Mimi is a swallow. Can Mimi fly? Answer only yes or no.",
                  ["是", "yes"]))
    cases.append(("logic", "没有猫会游泳。米米是一只猫。米米会游泳吗？只回答是或否。",
                  "No cat can swim. Mimi is a cat. Can Mimi swim? Answer only yes or no.",
                  ["否", "no"]))
    cases.append(("logic", "小雨最早到，小林第二，小周最后。谁第二到？只回答姓名。",
                  "Mia arrived first, Leo second, Zoe last. Who arrived second? Give only the name.",
                  ["小林", "Leo"]))

    numbers = [rng.randrange(2, 20) for _ in range(4)]
    numeric("code", f"Python表达式len([{','.join(map(str, numbers))}])的结果是什么？只回答数字。",
            f"What does the Python expression len([{','.join(map(str, numbers))}]) return? "
            "Give only the number.", len(numbers))
    numbers = [rng.randrange(2, 60) for _ in range(4)]
    numeric("code", f"Python表达式max([{','.join(map(str, numbers))}])的结果是什么？只回答数字。",
            f"What does max([{','.join(map(str, numbers))}]) return in Python? Give only the number.",
            max(numbers))
    cases.append(("code", '在Python中，"ab" * 3的结果是什么？只回答字符串内容。',
                  'What is the result of "ab" * 3 in Python? Give only the string content.',
                  ["ababab", "ababab"]))

    value = rng.randrange(10, 99)
    cases.append(("format", f"只输出一个JSON对象，键名x，值为数字{value}。不要其他文字。",
                  f"Output only a JSON object with key x and numeric value {value}. No other text.",
                  [f'{{"x":{value}}}', f'{{"x":{value}}}']))
    numbers = rng.sample(range(2, 60), 4)
    ordered = sorted(numbers)
    cases.append(("format", f"把{','.join(map(str, numbers))}按升序排列，只输出逗号分隔的数字。",
                  f"Sort {', '.join(map(str, numbers))} in ascending order. "
                  "Output only comma-separated numbers.",
                  [",".join(map(str, ordered)), ",".join(map(str, ordered))]))
    words = ["banana", "cat", "apple", "dog", "elderberry", "fig"]
    chosen = rng.sample(words, 4)
    cases.append(("format", f"把{','.join(chosen)}按字母顺序排列，只输出逗号分隔的单词。",
                  f"Sort {', '.join(chosen)} alphabetically. Output only comma-separated words.",
                  [",".join(sorted(chosen)), ",".join(sorted(chosen))]))
    return cases


def extended_probes():
    """The generated extension in the same episode schema as language_probes."""
    result = []
    for number, (family, zh, en, answers) in enumerate(generated_cases()):
        for language, question, target in zip(LANGS, (zh, en), answers):
            result.append((family, _row(number, language, question, target)))
    return result
