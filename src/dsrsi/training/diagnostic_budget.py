"""Conservative on-disk admission for bounded single-process GPU diagnostics."""

import fcntl
from pathlib import Path
import uuid

from dsrsi.evaluation.journal import digest
from dsrsi.utils.atomic import atomic_json
import json


class DiagnosticBudgetExceeded(Exception):
    pass


class DiagnosticBudget:
    def __init__(self, path, limit=150000):
        self.path, self.limit = Path(path), limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(dict(version="bounded-diagnostic-ledger-v1", student_limit=limit, entries=[]))
        if self.read()["student_limit"] != limit:
            raise ValueError("diagnostic budget limit changed")

    def read(self):
        value=json.loads(self.path.read_text())
        checksum=value.pop("sha256")
        if checksum!=digest(value) or value["version"]!="bounded-diagnostic-ledger-v1":
            raise ValueError("diagnostic ledger checksum/version differs")
        return value

    def _write(self,value):
        atomic_json(self.path,dict(value,sha256=digest(value)))

    def reserve(self, *, run, kind, student, teacher=0):
        if min(student,teacher)<0:
            raise ValueError("negative diagnostic cost")
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=self.read()
            used=sum(e["student"] for e in value["entries"])
            if used+student>self.limit:
                raise DiagnosticBudgetExceeded("shared SHINE diagnostic student budget exhausted")
            key=uuid.uuid4().hex
            value["entries"].append(dict(id=key,run=run,kind=kind,student=student,teacher=teacher,status="reserved"))
            self._write(value)
            return key

    def commit(self,key,*,student,teacher=0):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=self.read()
            entry=next(e for e in value["entries"] if e["id"]==key)
            if (entry["status"]!="reserved" or not 0<=student<=entry["student"]
                    or not 0<=teacher<=entry["teacher"]):
                raise ValueError("actual diagnostic cost exceeds reservation or was already committed")
            entry.update(student=student,teacher=teacher,status="committed")
            self._write(value)
