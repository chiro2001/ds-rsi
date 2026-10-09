"""One continuously evolving life, with bounded numerical context and latent M."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from dsrsi.memory.store import MemoryRecord, MemoryStore
from dsrsi.training.checkpoint import cpu_tree
from .streaming_cache import StreamingCache


@dataclass
class LifeState:
    life_id: str
    cache: StreamingCache
    memory: MemoryStore
    writer_stride: int = 256
    retrieval_interval: int = 64
    pending: torch.Tensor | None = None
    pending_begin: int | None = None
    pending_end: int | None = None
    pending_metadata: dict | None = None
    active_records: list | None = None
    lookup_position: int | None = None
    lookup_wall_time: float | None = None
    next_record: int = 0
    execution_quantum: int | None = 1

    def __post_init__(self):
        if not self.life_id or min(self.writer_stride, self.retrieval_interval) < 1:
            raise ValueError("life id and positive update intervals are required")
        if self.execution_quantum is not None and self.execution_quantum < 1:
            raise ValueError("execution quantum must be positive")
        if self.active_records is None:
            self.active_records = []

    def fork(self):
        store = MemoryStore(memory_dim=self.memory.memory_dim, key_dim=self.memory.key_dim,
                            codec_version=self.memory.codec_version, max_tensor_bytes=self.memory.max_tensor_bytes)
        store.records = dict(self.memory.records)
        return replace(self, cache=self.cache.fork(), memory=store,
                       pending_metadata=None if self.pending_metadata is None else dict(self.pending_metadata),
                       active_records=list(self.active_records))

    def detached(self):
        other = self.fork()
        other.cache = self.cache.detached()
        other.pending = None if self.pending is None else self.pending.detach()
        def detached_record(record):
            return replace(record, slots=record.slots.detach(), keys=record.keys.detach())
        other.memory.records = {key: detached_record(value) for key, value in self.memory.records.items()}
        other.active_records = [detached_record(record) for record in self.active_records]
        return other

    def state_dict(self):
        return {"life_id": self.life_id, "cache": self.cache.state_dict(), "memory": self.memory.state_dict(),
                "writer_stride": self.writer_stride, "retrieval_interval": self.retrieval_interval,
                "pending": cpu_tree(self.pending), "pending_begin": self.pending_begin,
                "pending_end": self.pending_end, "pending_metadata": self.pending_metadata,
                # Active task memories can outlive a compacted leaf until the next
                # refresh; their separate payload is bounded by retrieval top-k.
                "active_records": [record.state_dict() for record in self.active_records],
                "lookup_position": self.lookup_position, "lookup_wall_time": self.lookup_wall_time,
                "next_record": self.next_record, "execution_quantum": self.execution_quantum}

    @classmethod
    def from_state_dict(cls, data, *, device="cpu"):
        active = []
        for item in data["active_records"]:
            item = dict(item)
            item["slots"] = item["slots"].to(device).clone()
            item["keys"] = item["keys"].to(device).clone()
            item["child_refs"] = tuple(item["child_refs"])
            active.append(MemoryRecord(**item))
        pending = None if data["pending"] is None else data["pending"].to(device).clone()
        state = cls(data["life_id"], StreamingCache.from_state_dict(data["cache"], device=device),
                    MemoryStore.from_state_dict(data["memory"], device=device), data["writer_stride"],
                    data["retrieval_interval"], pending, data["pending_begin"], data["pending_end"],
                    data["pending_metadata"], active, data["lookup_position"], data["lookup_wall_time"],
                    data["next_record"], data.get("execution_quantum", 1))
        if pending is not None and (pending.ndim != 3 or pending.shape[0] != 1
                                    or pending.shape[1] >= state.writer_stride
                                    or state.pending_begin is None or state.pending_end is None):
            raise ValueError("invalid pending write snapshot")
        if len(active) > 8 or any(record.stream_id != state.life_id for record in active):
            raise ValueError("invalid active memory snapshot")
        return state
