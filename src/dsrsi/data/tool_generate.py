"""Deterministic tool demonstrations executed by the real durable harness."""

import hashlib
from pathlib import Path
import random
import tempfile

from dsrsi.harness.memory_tools import HarnessStore
from .generate import BASE_TIME

TOOL_DATA_REVISION = "tool-trajectories-v1"
TOOL_FAMILIES = ("write", "read", "update", "delete", "history", "task_save", "no_tool", "language_preservation")
TOOL_WEIGHTS = (15, 10, 10, 10, 15, 10, 10, 20)


class ToolEpisodeGenerator:
    def __init__(self, *, seed=20261010, split="train", journal_path=None):
        if split not in {"train", "dev", "test"}:
            raise ValueError("invalid tool data split")
        self.seed, self.split = seed, split
        self.journal_path = None if journal_path is None else Path(journal_path).resolve()
        self.database_id = hashlib.sha256(f"{TOOL_DATA_REVISION}:{seed}:{split}".encode()).hexdigest()
        if self.journal_path is not None:
            bootstrap = HarnessStore(self.journal_path, "__data_schema__", database_id=self.database_id)
            bootstrap.close()

    def _rng(self, ordinal):
        return random.Random(int.from_bytes(hashlib.sha256(
            f"{TOOL_DATA_REVISION}:{self.seed}:{self.split}:{ordinal}".encode()).digest(), "big"))

    def episode(self, ordinal, *, variant=0, family=None, language=None, select_one=True):
        if ordinal < 0 or variant not in (0, 1):
            raise ValueError("invalid tool world/variant")
        rng = self._rng(ordinal)
        family = family or rng.choices(TOOL_FAMILIES, TOOL_WEIGHTS)[0]
        language = language or rng.choices(("zh", "en"), (0.7, 0.3))[0]
        if family not in TOOL_FAMILIES or language not in {"zh", "en"}:
            raise ValueError("invalid tool task/language")
        if self.journal_path is None:
            with tempfile.TemporaryDirectory(prefix="dsrsi-tool-world-") as directory:
                return self._generate(Path(directory) / "journal.sqlite", ordinal, variant, family, language, rng, select_one)
        return self._generate(self.journal_path, ordinal, variant, family, language, rng, select_one)

    def _generate(self, path, ordinal, variant, family, language, rng, select_one):
        # Different fixtures of one ordinal get distinct lives. Replaying a life
        # starts at revision 0 so IDs do not depend on the current journal head.
        life_id = f"{self.split}-tool-{ordinal:08d}-{variant}-{family}-{language}"
        bootstrap = HarnessStore(path, life_id, database_id=self.database_id)
        bootstrap.close()
        store = HarnessStore(path, life_id, revision=0, expected_database_id=self.database_id)
        try:
            return self._trace(store, ordinal, variant, family, language, rng, select_one)
        finally:
            store.close()

    def _trace(self, store, ordinal, variant, family, language, rng, select_one):
        zh = language == "zh"
        entity = f"{'青杉' if zh else 'Cedar'}-{rng.randrange(10000):04d}-{ordinal}"
        key = f"{entity}出行偏好" if zh else f"{entity} travel preference"
        values = ("优先坐火车", "优先坐长途汽车") if zh else ("prefer trains", "prefer intercity buses")
        value, updated = values[variant], values[1 - variant]
        source = []
        trace = []
        clock = 0
        def advance():
            nonlocal clock
            clock += 1
            return clock
        def user(text):
            trace.append({"role": "user", "content": text})
            store.append_history("user", text, clock=advance())
        def reply(text):
            trace.append({"role": "assistant", "content": text})
            store.append_history("assistant", text, clock=advance())
        def call(name, arguments):
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}]}
            trace.append(message)
            import json
            store.append_history("assistant", json.dumps(message, ensure_ascii=False, sort_keys=True), clock=advance())
            result = store.execute(name, arguments, clock=advance())
            trace.append({"role": "tool", "name": name, "content": result})
            store.append_history("tool", json.dumps(result, ensure_ascii=False, sort_keys=True), clock=advance())
            return result

        if family in {"read", "update", "delete"}:
            old_text = f"过去已保存{key}：{value}。" if zh else f"Previously saved {key}: {value}."
            store.append_history("user", old_text, clock=advance())
            store.execute("memory.write", {"key": key, "text": value}, clock=advance())
        elif family == "history":
            code = f"{chr(65 + (rng.randrange(26) + variant) % 26)}{rng.randrange(10)}-{rng.randrange(1000):03d}"
            text = ((f"{entity}入库确认记录。" + "验收已完成，逐项检查设备与登记情况。" * 15 + f"最终校验码为 {code}。")
                    if zh else (f"Receipt confirmation for {entity}. " + "Inspection and inventory checks complete. " * 12
                                + f"The final verification code is {code}."))
            store.append_history("user", text, clock=advance())
            source.append({"event_id": "past_receipt", "role": "user", "content": text,
                           "time": BASE_TIME.isoformat(), "commit_boundary": True})
        elif family == "language_preservation":
            text = f"{entity}的设备存放在东侧柜。" if zh else f"The equipment for {entity} is stored in the east cabinet."
            store.append_history("user", text, clock=advance())
            source.append({"event_id": "irrelevant_device", "role": "user", "content": text,
                           "time": BASE_TIME.isoformat(), "commit_boundary": True})
        initial = store.state_dict()
        if family == "write":
            user(f"请把{key}记为：{value}。" if zh else f"Save {key} as: {value}.")
            call("memory.write", {"key": key, "text": value})
            reply(f"已保存：{value}。" if zh else f"Saved: {value}.")
        elif family == "read":
            user(f"读一下明文记忆里的{key}。" if zh else f"Read {key} from explicit memory.")
            result = call("memory.read", {"key": key})
            reply(result["record"]["text"])
        elif family in {"update", "delete"}:
            user((f"把明文记忆中的{key}改成{updated}。" if family == "update" else f"删除明文记忆中的{key}。")
                 if zh else (f"Change {key} in explicit memory to {updated}." if family == "update"
                             else f"Delete {key} from explicit memory."))
            record = call("memory.read", {"key": key})["record"]
            arguments = {"record_id": record["record_id"]}
            if family == "update":
                arguments["text"] = updated
            call(f"memory.{family}", arguments)
            reply((f"已更新：{updated}。" if family == "update" else "已删除这条明文记忆。")
                  if zh else (f"Updated: {updated}." if family == "update" else "Deleted that explicit memory record."))
        elif family == "history":
            user(f"查一下{entity}那次入库的校验码。" if zh else f"Look up the verification code from the {entity} receipt.")
            query = f"{entity} 入库" if zh else f"{entity} receipt"
            result = call("history.search", {"query": query})
            # The original observation is the earliest matching record; the
            # current question is also journaled but contains no hidden value.
            identifier = min(result["results"], key=lambda r: r["clock"])["id"]
            assert all(code not in r["summary"] for r in result["results"])
            opened = call("history.open", {"id": identifier})
            assert opened["found"] and code in opened["text"]
            reply(code)
        elif family == "task_save":
            key = f"{entity}待办" if zh else f"{entity} pending task"
            pending = ("先测空白样，再测实际样" if variant == 0 else "先更换电池，再校准传感器") if zh else (
                "measure the blank sample, then the real sample" if variant == 0 else "replace batteries, then calibrate the sensor")
            user(f"{entity}现在暂停，下次继续时还要{pending}。" if zh else f"Pause {entity}; when we resume, we still need to {pending}.")
            call("memory.write", {"key": key, "text": pending})
            reply(f"已保存待办：{pending}。" if zh else f"Saved the pending task: {pending}.")
        else:
            a, b = rng.randrange(10), rng.randrange(10)
            user(f"{a}加{b}是多少？只回答结果。" if zh else f"What is {a} plus {b}? Give only the result.")
            reply(str(a + b))
        final = store.state_dict()
        responses = [m for m in trace if m["role"] == "assistant"]
        row = {"schema_version": "0.1", "status": "generated_training_episode", "data_revision": TOOL_DATA_REVISION,
               "episode_id": store.life_id, "world_id": f"{self.split}-tool-w{ordinal:08d}", "split": self.split,
               "family": family, "language": language, "source_encoding": "independent_events",
               "source_events": source, "trajectory": trace, "annotations": {
                   "supervised_roles": ["assistant"], "tool_observations_are_not_labels": True,
                   "expected_final_explicit_store": {k: dict(v) for k, v in store.memory.items()},
                   "stateless_seed": self.seed, "ordinal": ordinal, "variant": variant,
                   "deletion_scope": "explicit_memory_only"}}
        if select_one:
            row["training_response_index"] = rng.randrange(len(responses))
        if family == "history":
            row["query_setup"] = {"reset": ["kv", "gdn", "conv", "pending_writer"],
                                  "memory_transform": {"drop_events": ["past_receipt"]}, "retain_history": True}
        if self.journal_path is not None:
            row["initial_harness"] = initial
            row["final_harness"] = final
        return row
