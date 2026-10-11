"""Atomic, fingerprinted progress for deterministic read-only evaluations."""

import hashlib
import json
from pathlib import Path
import signal

from dsrsi.utils.atomic import atomic_json


class EvaluationPaused(Exception):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class EvaluationJournal:
    def __init__(self, path, identity, jobs, *, resume=False, stop_after_items=None):
        self.path = Path(path)
        self.identity = json.loads(json.dumps(identity))
        self.jobs = list(jobs)
        if len(set(self.jobs)) != len(self.jobs) or any(not isinstance(key, str) for key in self.jobs):
            raise ValueError("evaluation jobs require unique string keys")
        if stop_after_items is not None and stop_after_items < 1:
            raise ValueError("stop-after-items must be positive")
        self.stop_after_items = stop_after_items
        self.pause_requested, self.visited = False, 0
        if resume:
            state = self.read(self.path)
            if state["identity"] != self.identity or state["jobs"] != self.jobs:
                raise ValueError("evaluation identity/job plan mismatch")
            self.results = state["results"]
        else:
            if self.path.exists():
                raise ValueError("evaluation journal exists; explicit --resume is required")
            self.results = []
            self._save("running")

    @staticmethod
    def read(path):
        state = json.loads(Path(path).read_text())
        checksum = state.pop("sha256", None)
        if checksum != digest(state) or state.get("schema") != "deterministic-evaluation-journal-v1":
            raise ValueError("evaluation journal checksum/schema mismatch")
        jobs, results = state["jobs"], state["results"]
        if (state["status"] not in {"running", "paused", "completed"}
                or len(set(jobs)) != len(jobs) or len(results) > len(jobs)
                or [item["key"] for item in results] != jobs[:len(results)]
                or (state["status"] == "completed" and len(results) != len(jobs))):
            raise ValueError("evaluation journal progress is invalid")
        return state

    def _save(self, status):
        state = {"schema": "deterministic-evaluation-journal-v1", "identity": self.identity,
                 "jobs": self.jobs, "results": self.results, "status": status}
        atomic_json(self.path, {**state, "sha256": digest(state)})

    def install_signal_handlers(self):
        def request_pause(_signum, _frame):
            self.pause_requested = True
        for name in ("SIGINT", "SIGTERM", "SIGUSR2"):
            signum = getattr(signal, name, None)
            if signum is not None:
                signal.signal(signum, request_pause)

    def _check_pause(self):
        if self.pause_requested or (self.stop_after_items is not None
                                    and len(self.results) >= self.stop_after_items):
            self._save("paused")
            raise EvaluationPaused(f"evaluation paused with {len(self.results)}/{len(self.jobs)} items committed")

    def evaluate(self, key, function):
        if self.visited >= len(self.jobs) or key != self.jobs[self.visited]:
            raise ValueError("evaluation job order differs from the pinned plan")
        index = self.visited
        self.visited += 1
        if index < len(self.results):
            return self.results[index]["value"]
        self._check_pause()
        value = function()
        item = json.loads(json.dumps({"key": key, "value": value}, allow_nan=False))
        self.results.append(item)
        self._save("running")
        self._check_pause()
        return item["value"]

    def finish(self):
        if self.visited != len(self.jobs) or len(self.results) != len(self.jobs):
            raise ValueError("cannot complete a partial evaluation")
        self._save("completed")
