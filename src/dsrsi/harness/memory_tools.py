"""Append-only, branchable SQLite tool state; explicit deletion only affects E."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from dsrsi.data.tools import TOOL_PROPERTIES


class HarnessStore:
    def __init__(self, path, life_id, *, revision=None, expected_database_id=None):
        self.path = Path(path).resolve()
        if revision is not None and not self.path.is_file():
            raise FileNotFoundError("restoring a harness requires its durable journal")
        if not life_id:
            raise ValueError("life id is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.life_id = life_id
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                revision INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                parent_revision INTEGER NOT NULL, life_id TEXT NOT NULL, kind TEXT NOT NULL,
                logical_clock INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS heads (life_id TEXT PRIMARY KEY, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS history (
                event_id TEXT PRIMARY KEY, revision INTEGER UNIQUE NOT NULL, life_id TEXT NOT NULL,
                logical_clock INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(text, tokenize='trigram');
        """)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO meta VALUES ('database_id', ?)", (uuid.uuid4().hex,))
        self.database_id = self.connection.execute("SELECT value FROM meta WHERE name='database_id'").fetchone()[0]
        if expected_database_id is not None and expected_database_id != self.database_id:
            raise ValueError("harness journal identity differs from snapshot")
        if revision is None:
            head = self.connection.execute("SELECT revision FROM heads WHERE life_id=?", (life_id,)).fetchone()
            revision = 0 if head is None else head[0]
        self.revision = revision
        self.memory = {}
        self.last_clock = 0
        self._replay()

    def _ancestors(self):
        if self.revision == 0:
            return []
        rows = self.connection.execute("""
            WITH RECURSIVE chain AS (
              SELECT * FROM events WHERE revision=?
              UNION ALL SELECT e.* FROM events e JOIN chain c ON e.revision=c.parent_revision
            ) SELECT * FROM chain ORDER BY revision
        """, (self.revision,)).fetchall()
        if not rows or any(row["life_id"] != self.life_id for row in rows):
            raise ValueError("harness revision is missing or belongs to another life")
        return rows

    def _replay(self):
        for row in self._ancestors():
            payload = json.loads(row["payload"])
            self.last_clock = max(self.last_clock, row["logical_clock"])
            if row["kind"] in {"memory.write", "memory.update"}:
                self.memory[payload["key"]] = payload
            elif row["kind"] == "memory.delete":
                self.memory.pop(payload["key"], None)

    def _parent_id(self):
        if self.revision == 0:
            return "root"
        return self.connection.execute("SELECT event_id FROM events WHERE revision=?", (self.revision,)).fetchone()[0]

    def _append(self, kind, payload, clock):
        if not isinstance(clock, int) or clock < self.last_clock:
            raise ValueError("tool events must advance the visible logical clock")
        canonical = json.dumps({"life_id": self.life_id, "parent": self._parent_id(), "kind": kind,
                                "clock": clock, "payload": payload}, ensure_ascii=False, sort_keys=True)
        event_id = hashlib.sha256(canonical.encode()).hexdigest()
        with self.connection:
            existing = self.connection.execute("SELECT revision FROM events WHERE event_id=?", (event_id,)).fetchone()
            if existing is None:
                cursor = self.connection.execute(
                    "INSERT INTO events(event_id,parent_revision,life_id,kind,logical_clock,payload) VALUES (?,?,?,?,?,?)",
                    (event_id, self.revision, self.life_id, kind, clock, json.dumps(payload, ensure_ascii=False, sort_keys=True)))
                revision = cursor.lastrowid
                if kind == "history":
                    self.connection.execute("INSERT INTO history VALUES (?,?,?,?,?,?)",
                                            (event_id, revision, self.life_id, clock, payload["role"], payload["text"]))
                    self.connection.execute("INSERT INTO history_fts(rowid,text) VALUES (?,?)", (revision, payload["text"]))
            else:
                revision = existing[0]
            self.connection.execute("INSERT INTO heads VALUES (?,?) ON CONFLICT(life_id) DO UPDATE SET revision=excluded.revision",
                                    (self.life_id, revision))
        self.revision = revision
        self.last_clock = clock
        return event_id

    def append_history(self, role, text, *, clock):
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(text, str):
            raise ValueError("invalid history observation")
        return "h_" + self._append("history", {"role": role, "text": text}, clock)

    def execute(self, name, arguments, *, clock):
        if name not in TOOL_PROPERTIES or not isinstance(arguments, dict):
            raise ValueError("unknown/unstructured tool call")
        if set(arguments) != set(TOOL_PROPERTIES[name]) or any(not isinstance(v, str) for v in arguments.values()):
            raise ValueError("tool arguments do not match schema")
        if clock < self.last_clock:
            raise ValueError("tool context contains observations after the query clock")
        if name == "memory.read":
            record = self.memory.get(arguments["key"])
            return {"found": record is not None, "record": None if record is None else dict(record)}
        if name == "memory.write":
            key, text = arguments["key"], arguments["text"]
            existing = self.memory.get(key)
            identifier = existing["record_id"] if existing else "m_" + hashlib.sha256(
                f"{self.database_id}:{self.life_id}:{self._parent_id()}:{key}:{text}:{clock}".encode()).hexdigest()
            payload = {"record_id": identifier, "key": key, "text": text, "updated_clock": clock}
            self._append(name, payload, clock)
            self.memory[key] = payload
            return {"ok": True, "record_id": identifier}
        if name in {"memory.update", "memory.delete"}:
            record = next((r for r in self.memory.values() if r["record_id"] == arguments["record_id"]), None)
            if record is None:
                return {"ok": False, "error": "record_not_found"}
            if name == "memory.update":
                payload = {**record, "text": arguments["text"], "updated_clock": clock}
                self._append(name, payload, clock)
                self.memory[record["key"]] = payload
                return {"ok": True, "record_id": record["record_id"]}
            self._append(name, {"key": record["key"], "record_id": record["record_id"]}, clock)
            del self.memory[record["key"]]
            return {"ok": True, "deleted_record_id": record["record_id"]}
        visible = {row["revision"] for row in self._ancestors() if row["kind"] == "history" and row["logical_clock"] <= clock}
        if name == "history.open":
            identifier = arguments["id"]
            row = self.connection.execute("SELECT * FROM history WHERE event_id=?", (identifier.removeprefix("h_"),)).fetchone()
            if row is None or row["revision"] not in visible:
                return {"found": False}
            return {"found": True, "id": identifier, "role": row["role"], "text": row["text"], "clock": row["logical_clock"]}
        terms = arguments["query"].split()
        if not terms or not visible:
            return {"results": []}
        long_terms = [term for term in terms if len(term) >= 3]
        if long_terms:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in long_terms)
            rows = self.connection.execute("""
                SELECT h.*, bm25(history_fts) AS relevance FROM history_fts
                JOIN history h ON h.revision=history_fts.rowid
                WHERE history_fts MATCH ? AND h.life_id=? AND h.logical_clock<=?
                ORDER BY relevance, h.logical_clock DESC
            """, (expression, self.life_id, clock)).fetchall()
        else:
            rows = self.connection.execute("SELECT * FROM history WHERE life_id=? AND logical_clock<=? ORDER BY logical_clock DESC",
                                           (self.life_id, clock)).fetchall()
            rows = [row for row in rows if any(term.casefold() in row["text"].casefold() for term in terms)]
        rows = [row for row in rows if row["revision"] in visible][:8]
        return {"results": [{"id": "h_" + row["event_id"], "summary": row["text"][:160], "clock": row["logical_clock"]} for row in rows]}

    def state_dict(self):
        return {"schema_version": 1, "path": str(self.path), "database_id": self.database_id,
                "life_id": self.life_id, "revision": self.revision}

    @classmethod
    def from_state_dict(cls, data, *, path=None):
        if data["schema_version"] != 1:
            raise ValueError("unsupported harness snapshot")
        return cls(path or data["path"], data["life_id"], revision=data["revision"], expected_database_id=data["database_id"])

    def close(self):
        self.connection.close()


def parse_native_tool_calls(text):
    pattern = r"<tool_call>\s*<function=([A-Za-z0-9_.]+)>(.*?)</function>\s*</tool_call>"
    matches = list(re.finditer(pattern, text, re.S))
    if text.count("<tool_call>") != len(matches):
        raise ValueError("malformed native tool call")
    result = []
    for match in matches:
        name, body = match.group(1), match.group(2)
        if name not in TOOL_PROPERTIES:
            raise ValueError("unknown native tool name")
        parameters = list(re.finditer(r"<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>", body, re.S))
        remainder = re.sub(r"<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>", "", body, flags=re.S).strip()
        if remainder or len({p.group(1) for p in parameters}) != len(parameters):
            raise ValueError("malformed or duplicate native parameters")
        arguments = {p.group(1): p.group(2).strip() for p in parameters}
        if set(arguments) != set(TOOL_PROPERTIES[name]):
            raise ValueError("native tool parameters differ from schema")
        result.append({"name": name, "arguments": arguments})
    return result
