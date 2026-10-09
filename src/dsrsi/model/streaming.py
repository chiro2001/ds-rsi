"""Canonical read/write/rebase scheduling independent of external input chunks."""

from dataclasses import dataclass

import torch

from dsrsi.memory.store import BudgetExceeded, MemoryRecord, MemoryStore
from dsrsi.state.life_state import LifeState
from dsrsi.state.streaming_cache import StreamingCache


@dataclass
class StreamOutput:
    hidden: torch.Tensor
    state: LifeState


class StreamExecutor:
    def __init__(self, model):
        self.model = model

    def new_life(self, life_id, *, window=2048, rebase_interval=4096, writer_stride=256,
                 retrieval_interval=64, memory_tensor_budget=64 << 20, execution_quantum=1):
        cfg = self.model.memory_config
        return LifeState(life_id, StreamingCache(self.model.base.config.layer_types, window=window,
                                                rebase_interval=rebase_interval),
                         MemoryStore(memory_dim=cfg.memory_dim, key_dim=cfg.key_dim,
                                     max_tensor_bytes=memory_tensor_budget), writer_stride, retrieval_interval,
                         execution_quantum=execution_quantum)

    def _id(self, state):
        value = f"{state.life_id}:r{state.next_record}"
        state.next_record += 1
        return value

    def _fit_record(self, state, record):
        if record.tensor_bytes > state.memory.max_tensor_bytes:
            raise BudgetExceeded("new record alone exceeds memory tensor budget")
        while state.memory.tensor_bytes + record.tensor_bytes > state.memory.max_tensor_bytes:
            children = list(state.memory.records.values())[:4]
            if len(children) < 2:
                raise BudgetExceeded("insufficient children for semantic compaction")
            source = torch.cat([child.slots for child in children], dim=0)[None]
            with self.model._memory_context(source):
                slots = self.model.consolidator(source)
                keys = self.model.address.encode_keys(slots)
            parent = MemoryRecord(self._id(state), state.life_id, slots[0], keys[0],
                                  min(r.source_begin for r in children), max(r.source_end for r in children),
                                  state.cache.clock, codec_version=state.memory.codec_version,
                                  source_kind="summary", level=max(r.level for r in children) + 1,
                                  child_refs=tuple(r.record_id for r in children))
            old_bytes = sum(child.tensor_bytes for child in children)
            if parent.tensor_bytes >= old_bytes:
                raise BudgetExceeded("configured compressor does not reduce the selected payload")
            state.memory.compact(parent, [child.record_id for child in children])
        state.memory.append(record)

    def flush(self, state):
        if state.pending is None:
            return state
        updated = state.fork()
        with self.model._memory_context(updated.pending):
            slots = self.model.writer(updated.pending)
            keys = self.model.address.encode_keys(slots)
        metadata = updated.pending_metadata or {}
        record = MemoryRecord(self._id(updated), updated.life_id, slots[0], keys[0],
                              updated.pending_begin, updated.pending_end, updated.cache.clock,
                              codec_version=updated.memory.codec_version,
                              source_role=metadata.get("role", "user"), source_kind=metadata.get("kind", "observation"),
                              wall_time=metadata.get("wall_time"))
        self._fit_record(updated, record)
        updated.pending = None
        updated.pending_begin = updated.pending_end = None
        updated.pending_metadata = None
        return updated

    def forward(self, input_ids, state, *, write=True, memory_enabled=True, force_read_at=(),
                flush=False, role="user", kind="observation", wall_time=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("one nonempty stream per life is required")
        if any(not 0 <= position < input_ids.shape[1] for position in force_read_at):
            raise ValueError("forced reads must be visible positions in this input")
        updated = state.fork()
        metadata = {"role": role, "kind": kind, "wall_time": wall_time}
        if updated.pending is not None and (not write or updated.pending_metadata != metadata):
            updated = self.flush(updated)
        force = set(force_read_at)
        offset = 0
        outputs = []
        while offset < input_ids.shape[1]:
            clock = updated.cache.clock
            refresh = clock % updated.retrieval_interval == 0 or offset in force or updated.lookup_position is None
            limits = [input_ids.shape[1] - offset,
                      updated.retrieval_interval - clock % updated.retrieval_interval]
            if updated.execution_quantum is not None:
                limits.append(updated.execution_quantum)
            rebase = updated.cache.until_rebase()
            if rebase is not None:
                limits.append(rebase)
            later_forced = [position - offset for position in force if position > offset]
            if later_forced:
                limits.append(min(later_forced))
            if write:
                pending_tokens = 0 if updated.pending is None else updated.pending.shape[1]
                limits.append(updated.writer_stride - pending_tokens)
            length = min(limits)

            def selector(features, position):
                with self.model._memory_context(features):
                    query = self.model.address.encode_query(features)[0]
                return updated.memory.retrieve(query, stream_id=updated.life_id, query_position=position, topk=8)

            result = self.model.forward_cached(
                input_ids[:, offset:offset + length], updated.cache,
                records=updated.active_records, selector=selector if refresh else None,
                stream_id=updated.life_id, memory_enabled=memory_enabled,
                lookup_position=clock if refresh else updated.lookup_position,
                lookup_wall_time=wall_time if refresh else updated.lookup_wall_time)
            updated.cache = result.cache
            if refresh:
                updated.active_records = result.selected_records
                updated.lookup_position = clock
                updated.lookup_wall_time = wall_time
            outputs.append(result.hidden)
            if write:
                if updated.pending is None:
                    updated.pending = result.write_features
                    updated.pending_begin = clock
                    updated.pending_metadata = metadata
                else:
                    updated.pending = torch.cat([updated.pending, result.write_features], dim=1)
                updated.pending_end = updated.cache.clock - 1
                if updated.pending.shape[1] == updated.writer_stride:
                    updated = self.flush(updated)
            offset += length
        if flush:
            updated = self.flush(updated)
        return StreamOutput(torch.cat(outputs, dim=1), updated)
