"""Tensor memory store: causal visibility, stream isolation, explicit compaction."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
import torch.nn.functional as F

from dsrsi.training.checkpoint import cpu_tree
from dsrsi.training.losses import routing_scores


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    stream_id: str
    slots: torch.Tensor
    keys: torch.Tensor
    source_begin: int
    source_end: int
    visible_after: int
    codec_version: str = "v0"
    source_role: str = "user"
    source_kind: str = "observation"
    wall_time: float | None = None
    level: int = 0
    child_refs: tuple[str, ...] = ()
    history_ref: str | None = None

    def __post_init__(self):
        if not self.record_id or not self.stream_id or not self.codec_version:
            raise ValueError("record, stream and codec identities are required")
        if self.slots.ndim != 2 or self.keys.ndim != 2 or self.slots.shape[0] < 1 or self.keys.shape[0] < 1:
            raise ValueError("record slots/keys must be nonempty matrices")
        if self.source_begin < 0 or self.source_end < self.source_begin or self.visible_after <= self.source_end:
            raise ValueError("record visibility must follow its completed source")
        if self.level < 0:
            raise ValueError("invalid compaction level")

    @property
    def tensor_bytes(self):
        return self.slots.numel() * self.slots.element_size() + self.keys.numel() * self.keys.element_size()

    def state_dict(self):
        return cpu_tree({field.name: getattr(self, field.name) for field in fields(self)})


class MemoryStore:
    def __init__(self, *, memory_dim=256, key_dim=128, codec_version="v0", max_tensor_bytes=64 << 20):
        if min(memory_dim, key_dim, max_tensor_bytes) < 1:
            raise ValueError("memory dimensions and byte budget must be positive")
        self.memory_dim = memory_dim
        self.key_dim = key_dim
        self.codec_version = codec_version
        self.max_tensor_bytes = max_tensor_bytes
        self.records: dict[str, MemoryRecord] = {}

    @property
    def tensor_bytes(self):
        return sum(record.tensor_bytes for record in self.records.values())

    def _validate(self, record):
        if record.slots.shape[-1] != self.memory_dim or record.keys.shape[-1] != self.key_dim:
            raise ValueError("record dimensions differ from store")
        if record.codec_version != self.codec_version:
            raise ValueError("memory codec mismatch requires an explicit migration")
        if record.record_id in self.records:
            raise ValueError("record id already exists")

    def append(self, record: MemoryRecord):
        self._validate(record)
        if self.tensor_bytes + record.tensor_bytes > self.max_tensor_bytes:
            raise BudgetExceeded("compact or explicitly remove records before appending")
        self.records[record.record_id] = record

    def drop(self, record_id: str):
        return self.records.pop(record_id)

    def compact(self, parent: MemoryRecord, children: list[str]):
        self._validate(parent)
        if not children or len(set(children)) != len(children):
            raise ValueError("compaction children must be distinct and nonempty")
        old = [self.records[key] for key in children]
        if set(parent.child_refs) != set(children):
            raise ValueError("parent must retain child references")
        if any(r.stream_id != parent.stream_id for r in old):
            raise ValueError("cannot compact across streams")
        if (parent.source_begin > min(r.source_begin for r in old)
                or parent.source_end < max(r.source_end for r in old)
                or parent.visible_after < max(r.visible_after for r in old)
                or parent.level <= max(r.level for r in old)):
            raise ValueError("parent source range, visibility or level is invalid")
        resulting_bytes = self.tensor_bytes - sum(r.tensor_bytes for r in old) + parent.tensor_bytes
        if resulting_bytes > self.max_tensor_bytes:
            raise BudgetExceeded("compaction does not meet the configured byte budget")
        # Commit only after all validation: no partial child deletion on failure.
        for key in children:
            del self.records[key]
        self.records[parent.record_id] = parent

    def eligible(self, *, stream_id: str, query_position: int):
        return [record for record in self.records.values()
                if record.stream_id == stream_id and record.visible_after <= query_position]

    def retrieve(self, query: torch.Tensor, *, stream_id: str, query_position: int, topk=8):
        if query.ndim != 1 or query.shape[0] != self.key_dim or topk < 1:
            raise ValueError("invalid query/key width or top-k")
        records = self.eligible(stream_id=stream_id, query_position=query_position)
        if not records:
            return []
        # Runtime exact search. Training recomputes differentiable address keys
        # with the current encoder rather than relying on stale cached graphs.
        if len({r.keys.shape[0] for r in records}) != 1:
            raise ValueError("cached keys must share address count")
        keys = torch.stack([r.keys.detach().to(query.device) for r in records])
        scores = routing_scores(query.detach()[None], keys)[0]
        # Stable ties preserve insertion order, including after snapshot restore.
        selected = torch.argsort(scores, descending=True, stable=True)[:topk].tolist()
        return [records[index] for index in selected]

    def state_dict(self):
        return {"memory_dim": self.memory_dim, "key_dim": self.key_dim,
                "codec_version": self.codec_version, "max_tensor_bytes": self.max_tensor_bytes,
                "records": [r.state_dict() for r in self.records.values()]}

    @classmethod
    def from_state_dict(cls, state, *, device="cpu"):
        store = cls(**{key: state[key] for key in ("memory_dim", "key_dim", "codec_version", "max_tensor_bytes")})
        for item in state["records"]:
            item = dict(item)
            item["slots"] = item["slots"].to(device).clone()
            item["keys"] = item["keys"].to(device).clone()
            item["child_refs"] = tuple(item["child_refs"])
            store.append(MemoryRecord(**item))
        return store


def record_features(record, query_position, wall_time=None):
    roles = {"system": 0, "user": 1, "assistant": 2, "tool": 3}
    kinds = {"observation": 0, "model": 1, "tool": 2, "inference": 3, "summary": 4}
    age = max(query_position - record.source_end, 0)
    known_wall = wall_time is not None and record.wall_time is not None
    wall_age = max(wall_time - record.wall_time, 0) if known_wall else 0
    return [math.log1p(age) / 16, math.log1p(wall_age) / 16, record.level / 8,
            1.0, roles.get(record.source_role, 1) / 3, kinds.get(record.source_kind, 0) / 4,
            float(known_wall), float(record.level > 0)]


def pack_records(batches: list[list[MemoryRecord]], *, memory_dim, query_positions, stream_ids, device, dtype, wall_times=None):
    """Pad ragged records with cat/pad/stack, retaining Writer's tensor graph."""
    if len(batches) != len(query_positions) or len(batches) != len(stream_ids):
        raise ValueError("one logical query position and stream id are required per batch row")
    if wall_times is None:
        wall_times = [None] * len(batches)
    if len(wall_times) != len(batches):
        raise ValueError("one optional wall time is required per batch row")
    rows, features = [], []
    for records, position, stream, wall_time in zip(batches, query_positions, stream_ids, wall_times):
        if any(record.visible_after > position for record in records):
            raise ValueError("cannot pack future memory")
        if any(record.stream_id != stream for record in records):
            raise ValueError("cannot pack another stream into a reader row")
        if any(record.slots.shape[-1] != memory_dim for record in records):
            raise ValueError("memory width mismatch")
        rows.append(torch.cat([r.slots.to(device=device, dtype=dtype) for r in records]) if records
                    else torch.empty(0, memory_dim, device=device, dtype=dtype))
        features.append(torch.tensor([record_features(r, position, wall_time) for r in records for _ in range(r.slots.shape[0])],
                                     device=device, dtype=torch.float32).reshape(-1, 8))
    maximum = max((row.shape[0] for row in rows), default=0)
    if not rows:
        raise ValueError("empty batch")
    slots = torch.stack([F.pad(row, (0, 0, 0, maximum - row.shape[0])) for row in rows])
    feature_tensor = torch.stack([F.pad(row, (0, 0, 0, maximum - row.shape[0])) for row in features])
    lengths = torch.tensor([row.shape[0] for row in rows], device=device)
    valid = torch.arange(maximum, device=device)[None] < lengths[:, None]
    return slots, valid, feature_tensor
