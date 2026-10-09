"""Held-out elementary instruction probes; never used as A/B training data."""

PROBE_REVISION = "language-retention-dev-v1"

# A small diagnostic across task types, not a broad language benchmark gate.
CASES = [
    ("knowledge", "法国的首都是哪里？只回答城市名。", "What is the capital of France? Give only the city name.", ["巴黎", "Paris"]),
    ("knowledge", "水的化学式是什么？只回答化学式。", "What is the chemical formula of water? Give only the formula.", ["H2O", "H2O"]),
    ("knowledge", "太阳系最大的行星是什么？只回答名称。", "Which planet is largest in our Solar System? Give only its name.", ["木星", "Jupiter"]),
    ("knowledge", "一小时有多少分钟？只回答数字。", "How many minutes are in one hour? Give only the number.", ["60", "60"]),
    ("arithmetic", "17乘以6是多少？只回答数字。", "What is 17 times 6? Give only the number.", ["102", "102"]),
    ("arithmetic", "144除以12是多少？只回答数字。", "What is 144 divided by 12? Give only the number.", ["12", "12"]),
    ("arithmetic", "80的25%是多少？只回答数字。", "What is 25% of 80? Give only the number.", ["20", "20"]),
    ("arithmetic", "(7+5)乘以3是多少？只回答数字。", "What is (7+5) times 3? Give only the number.", ["36", "36"]),
    ("reading", "读这句话：小周把蓝色杯子放在厨房。杯子是什么颜色？只回答颜色。", "Read this sentence: Alex put a blue cup in the kitchen. What color is the cup? Give only the color.", ["蓝色", "blue"]),
    ("reading", "读这句话：李琳周三出发，周五返回。她哪天返回？只回答星期。", "Read this sentence: Lina departed on Wednesday and returned on Friday. On which day did she return? Give only the day.", ["周五", "Friday"]),
    ("reading", "文本：订单编号是R614，交付地点是西门。请只提取订单编号。", "Text: The order number is R614 and the delivery location is the west gate. Extract only the order number.", ["R614", "R614"]),
    ("reading", "文本：果篮里有三个苹果和两个梨。共有几个水果？只回答数字。", "Text: A basket contains three apples and two pears. How many fruits are there in total? Give only the number.", ["5", "5"]),
    ("logic", "所有猫都是哺乳动物，米米是一只猫。米米是哺乳动物吗？只回答是或否。", "All cats are mammals, and Mimi is a cat. Is Mimi a mammal? Answer only yes or no.", ["是", "yes"]),
    ("logic", "甲比乙高，乙比丙高。谁最高？只回答甲、乙或丙。", "A is taller than B and B is taller than C. Who is tallest? Answer only A, B, or C.", ["甲", "A"]),
    ("code", "Python表达式len([2,4,6])的结果是什么？只回答数字。", "What does the Python expression len([2,4,6]) return? Give only the number.", ["3", "3"]),
    ("code", "Python中sum(range(4))的结果是什么？只回答数字。", "What does sum(range(4)) return in Python? Give only the number.", ["6", "6"]),
    ("format", "只输出一个JSON对象，键名x，值为数字7。不要其他文字。", "Output only a JSON object with key x and numeric value 7. No other text.", ['{"x":7}', '{"x":7}']),
    ("format", "把9、2、5按升序排列，只输出逗号分隔的数字。", "Sort 9,2,5 in ascending order. Output only comma-separated numbers.", ["2,5,9", "2,5,9"]),
]


def probes():
    from .generate import BASE_TIME, RESET
    result = []
    for number, (family, zh, en, answers) in enumerate(CASES):
        for lang, question, target in zip(("zh", "en"), (zh, en), answers):
            key = f"{PROBE_REVISION}-{number}-{lang}"
            row = {"schema_version": "0.1", "status": "generated_training_episode", "split": "dev",
                "episode_id": key, "world_id": key, "family": "language_preservation", "language": lang,
                "annotations": {"query_resets": list(RESET)},
                "source_encoding": "independent_events",
                "source_events": [{"event_id": "unrelated", "role": "user", "commit_boundary": True,
                    "time": BASE_TIME.isoformat(), "content": ("白杨-941208的箱封码是J8-672。"
                         if lang == "zh" else "The seal code for Poplar-941208 is J8-672.")}],
                "probes": [{"probe_id": "answer", "mode": "m_only", "query": question, "target": target,
                    "query_time": BASE_TIME.replace(hour=18).isoformat(), "support_groups": [], "allow_tools": [],
                    "write_enabled": False, "memory_transform": {}}]}
            result.append((family, row))
    return result


def matches(text, target, *, family):
    import json
    import re
    import unicodedata
    if family == "format" and target.startswith("{"):
        try:
            value = json.loads(text.strip())
            return value == json.loads(target) and type(value.get("x")) in (int, float)
        except (ValueError, AttributeError):
            return False
    def normalize(value):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold().rstrip("。.!！")
    aliases = {"周五": {"周五", "星期五"}, "蓝色": {"蓝色", "蓝"}, "yes": {"yes"}, "是": {"是"}}
    normalized = normalize(text)
    return normalized in {normalize(value) for value in aliases.get(target, {target})}
