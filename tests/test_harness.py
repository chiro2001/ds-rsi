import pytest

from dsrsi.harness import HarnessStore, parse_native_tool_calls


def test_explicit_memory_crud_and_deletion_scope(tmp_path):
    store = HarnessStore(tmp_path / "journal.sqlite", "life")
    history = store.append_history("user", "请记住我优先坐火车。", clock=1)
    created = store.execute("memory.write", {"key": "出行偏好", "text": "优先坐火车"}, clock=2)
    identifier = created["record_id"]
    store.execute("memory.update", {"record_id": identifier, "text": "优先坐汽车"}, clock=3)
    assert store.execute("memory.read", {"key": "出行偏好"}, clock=3)["record"]["text"] == "优先坐汽车"
    store.execute("memory.delete", {"record_id": identifier}, clock=4)
    assert not store.execute("memory.read", {"key": "出行偏好"}, clock=4)["found"]
    assert store.execute("history.open", {"id": history}, clock=4)["found"]
    store.close()


def test_restore_revision_hides_future_and_branches_without_destroying_it(tmp_path):
    path = tmp_path / "journal.sqlite"
    original = HarnessStore(path, "life")
    first = original.append_history("user", "青杉仓的入库校验码是 Q7-492。", clock=1)
    original.execute("memory.write", {"key": "编号", "text": "Q7-492"}, clock=2)
    snapshot = original.state_dict()
    future = original.append_history("user", "青杉仓编号改为 R2-815。", clock=3)
    original.execute("memory.update", {"record_id": original.memory["编号"]["record_id"], "text": "R2-815"}, clock=4)
    restored = HarnessStore.from_state_dict(snapshot)
    assert restored.execute("memory.read", {"key": "编号"}, clock=2)["record"]["text"] == "Q7-492"
    assert not restored.execute("history.open", {"id": future}, clock=2)["found"]
    assert restored.execute("history.open", {"id": first}, clock=2)["found"]
    branch = restored.append_history("user", "青杉仓的新分支采用 T4-681。", clock=3)
    assert not original.execute("history.open", {"id": branch}, clock=4)["found"]
    assert restored.execute("history.open", {"id": branch}, clock=3)["found"]


def test_replayed_actions_have_stable_ids_and_revision(tmp_path):
    store = HarnessStore(tmp_path / "journal.sqlite", "life")
    snapshot = store.state_dict()
    a = store.execute("memory.write", {"key": "计划", "text": "明天采样"}, clock=2)
    rev = store.revision
    restored = HarnessStore.from_state_dict(snapshot)
    b = restored.execute("memory.write", {"key": "计划", "text": "明天采样"}, clock=2)
    assert a == b and restored.revision == rev


def test_history_search_supports_chinese_and_life_isolation(tmp_path):
    path = tmp_path / "journal.sqlite"
    a, b = HarnessStore(path, "a"), HarnessStore(path, "b")
    a.append_history("user", "青杉仓本次入库校验码为 Q7-492。", clock=2)
    b.append_history("user", "青杉仓本次入库校验码为 SECRET。", clock=2)
    results = a.execute("history.search", {"query": "青杉仓 校验码"}, clock=3)["results"]
    assert len(results) == 1 and "Q7-492" in results[0]["summary"]
    assert "SECRET" not in str(results)


def test_missing_journal_bad_schema_and_future_context_are_rejected(tmp_path):
    store = HarnessStore(tmp_path / "journal.sqlite", "life")
    store.append_history("user", "旧经历", clock=5)
    with pytest.raises(ValueError, match="after the query"):
        store.execute("history.search", {"query": "旧经历"}, clock=4)
    with pytest.raises(ValueError, match="schema"):
        store.execute("memory.write", {"key": "x", "text": "y", "unknown": "bad"}, clock=5)
    snapshot = store.state_dict()
    snapshot["path"] = str(tmp_path / "missing.sqlite")
    with pytest.raises(FileNotFoundError):
        HarnessStore.from_state_dict(snapshot)


def test_native_function_tag_parser_matches_tool_schema():
    text = "<tool_call>\n<function=memory.write>\n<parameter=key>\n出行偏好\n</parameter>\n<parameter=text>\n优先坐火车\n</parameter>\n</function>\n</tool_call>"
    assert parse_native_tool_calls(text) == [{"name": "memory.write", "arguments": {"key": "出行偏好", "text": "优先坐火车"}}]
    with pytest.raises(ValueError, match="unknown"):
        parse_native_tool_calls(text.replace("memory.write", "shell.execute"))
    with pytest.raises(ValueError, match="malformed"):
        parse_native_tool_calls(text.replace("</function>", ""))
