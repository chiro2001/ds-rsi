"""Native Qwen3.5 text backbone with causal latent-memory insertion.

This is the uncached M-only training path. Continuous state and bounded-cache
execution are implemented separately; no inference-cache mutation occurs here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding,
    create_causal_mask, create_recurrent_attention_mask,
)

from dsrsi.memory.modules import AddressEncoder, Consolidator, MemoryConfig, MemoryReader, MemoryWriter, TimeSourceEncoder
from dsrsi.memory.store import MemoryRecord, pack_records, record_features
from dsrsi.training.losses import LossSum, response_cross_entropy_from_hidden


@dataclass
class MemoryModelOutput:
    hidden: torch.Tensor
    query_features: torch.Tensor
    loss: LossSum | None = None
    logits: torch.Tensor | None = None


@dataclass
class CachedModelOutput:
    hidden: torch.Tensor
    write_features: torch.Tensor
    cache: object
    selected_records: list


class MemoryQwen(nn.Module):
    def __init__(self, backbone, head, *, memory_config=None, writer_after=11, reader_before=(12, 20)):
        super().__init__()
        self.base = backbone
        self.head = head
        self.base.requires_grad_(False)
        self.head.requires_grad_(False)
        cfg = memory_config or MemoryConfig(hidden_size=backbone.config.hidden_size)
        if cfg.hidden_size != backbone.config.hidden_size:
            raise ValueError("memory and backbone hidden widths differ")
        if (not 0 <= writer_after < len(backbone.layers) or not reader_before
                or len(set(reader_before)) != len(reader_before)
                or any(not writer_after < index < len(backbone.layers) for index in reader_before)):
            raise ValueError("read sites must be distinct existing layers after the write feature site")
        self.memory_config = cfg
        self.writer_after = writer_after
        self.reader_before = tuple(reader_before)
        self.writer = MemoryWriter(cfg)
        self.readers = nn.ModuleDict({str(index): MemoryReader(cfg) for index in reader_before})
        self.address = AddressEncoder(cfg)
        self.consolidator = Consolidator(cfg)
        self.time_source = TimeSourceEncoder(cfg)
        self.base.eval()
        self.head.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.head.eval()
        return self

    def frozen_auxiliary_parameters(self):
        """Frozen added modules, including learned LoRA, are absent from base."""
        return {name: p for name, p in self.named_parameters()
                if not p.requires_grad and (not name.startswith(("base.", "head.")) or ".lora_" in name)}

    def consolidate_records(self, children, *, record_id, visible_after=None):
        """Compress completed records using their own time/source metadata.

        The feature anchor is independent of a future query or answer. The
        optional commit time controls visibility, never compressor features.
        """
        children = list(children)
        if len(children) < 2 or len({r.record_id for r in children}) != len(children):
            raise ValueError("compression requires at least two distinct completed records")
        stream = children[0].stream_id
        codec = children[0].codec_version
        if any(r.stream_id != stream or r.codec_version != codec for r in children):
            raise ValueError("compression cannot mix streams or codecs")
        anchor = max(r.source_end for r in children) + 1
        earliest_commit = max(r.visible_after for r in children)
        visibility = earliest_commit if visible_after is None else visible_after
        if visibility < earliest_commit:
            raise ValueError("summary visibility cannot precede any completed child")
        known_walls = [r.wall_time for r in children if r.wall_time is not None]
        wall_anchor = max(known_walls) if known_walls else None
        slots, valid, features = pack_records([children], memory_dim=self.memory_config.memory_dim,
            query_positions=[earliest_commit], stream_ids=[stream], device=children[0].slots.device,
            dtype=children[0].slots.dtype, wall_times=[wall_anchor])
        # Eligibility follows commit visibility; content ages follow source
        # time. A delayed child summary must not move the recursive anchor.
        features = torch.tensor([[record_features(r, anchor, wall_anchor)
                                  for r in children for _ in range(r.slots.shape[0])]],
                                device=slots.device, dtype=torch.float32)
        with self._memory_context(slots):
            summaries = self.consolidator(slots + self.time_source(features), valid)
            keys = self.address.encode_keys(summaries)
        return MemoryRecord(record_id, stream, summaries[0], keys[0],
            min(r.source_begin for r in children), max(r.source_end for r in children), visibility,
            codec_version=codec, source_kind="summary", level=max(r.level for r in children) + 1,
            child_refs=tuple(r.record_id for r in children),
            wall_time=wall_anchor if len(known_walls) == len(children) else None)

    def memory_architecture_identity(self):
        config = asdict(self.memory_config)
        if not self.memory_config.writer_anchor_weight:
            config.pop("writer_anchor_weight")
        result = {"memory_config": config, "writer_after": self.writer_after,
                  "reader_before": list(self.reader_before)}
        if self.memory_config.writer_anchor_weight:
            result["writer_anchor_version"] = "position_pool_v1"
        return result

    @staticmethod
    def _memory_context(hidden):
        if hidden.dtype in (torch.bfloat16, torch.float16):
            return torch.autocast(device_type=hidden.device.type, dtype=hidden.dtype)
        return nullcontext()

    def _prepare(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        hidden = self.base.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(4, input_ids.shape[0], -1)
        text_positions = positions[0]
        kwargs = {"config": self.base.config, "inputs_embeds": hidden,
                  "attention_mask": attention_mask, "past_key_values": None,
                  "position_ids": text_positions}
        masks = {"full_attention": create_causal_mask(**kwargs),
                 "linear_attention": create_recurrent_attention_mask(**kwargs)}
        rotary = self.base.rotary_emb(hidden, positions[1:])
        return hidden, text_positions, masks, rotary

    def _layer(self, index, hidden, positions, masks, rotary):
        return self.base.layers[index](hidden, position_embeddings=rotary,
                                       attention_mask=masks[self.base.config.layer_types[index]],
                                       position_ids=positions, past_key_values=None, use_cache=False)

    def source_features(self, input_ids, attention_mask=None):
        # Frozen source encoding may omit its graph; Writer below must not.
        with torch.no_grad():
            hidden, positions, masks, rotary = self._prepare(input_ids, attention_mask)
            for index in range(self.writer_after + 1):
                hidden = self._layer(index, hidden, positions, masks, rotary)
        return hidden

    def write_records(self, input_ids, *, stream_ids, logical_begins, record_ids,
                      attention_mask=None, wall_times=None, codec_version="v0"):
        batch, length = input_ids.shape
        if not all(len(values) == batch for values in (stream_ids, logical_begins, record_ids)):
            raise ValueError("one source identity and logical range is required per row")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        lengths = attention_mask.sum(-1)
        expected = torch.arange(length, device=input_ids.device)[None] < lengths[:, None]
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError("sources require prefix-valid padding")
        features = self.source_features(input_ids, attention_mask)
        with self._memory_context(features):
            slots = self.writer(features, attention_mask.bool())
            keys = self.address.encode_keys(slots)
        wall_times = [None] * batch if wall_times is None else wall_times
        if len(wall_times) != batch:
            raise ValueError("one optional wall time is required per source")
        records = []
        for row in range(batch):
            begin, end = logical_begins[row], logical_begins[row] + int(lengths[row]) - 1
            records.append(MemoryRecord(record_ids[row], stream_ids[row], slots[row], keys[row],
                                        begin, end, end + 1, codec_version=codec_version, wall_time=wall_times[row]))
        return records

    def forward(self, input_ids, *, records=None, query_indices=None, logical_begins=None,
                stream_ids=None, attention_mask=None, labels=None, memory_enabled=True,
                return_logits=False, wall_times=None):
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if query_indices is None:
            if labels is not None:
                raise ValueError("teacher-forcing requires the explicit answer-free retrieval index")
            query_indices = attention_mask.sum(-1) - 1
        query_indices = torch.as_tensor(query_indices, device=input_ids.device, dtype=torch.long)
        if query_indices.shape != (batch,) or torch.any(query_indices < 0) or torch.any(query_indices >= length):
            raise ValueError("invalid query indices")
        if not attention_mask[torch.arange(batch, device=input_ids.device), query_indices].all():
            raise ValueError("query cannot use padding")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels and input shape differ")
            before_or_at_query = torch.arange(length, device=input_ids.device)[None] <= query_indices[:, None]
            if torch.any((labels != -100) & before_or_at_query):
                raise ValueError("retrieval index sees supervised answer tokens")
        records = records or [[] for _ in range(batch)]
        if len(records) != batch or any(len(row) > 8 for row in records):
            raise ValueError("provided candidate batches must contain at most 8 records per row")
        has_memory = memory_enabled and any(records)
        if has_memory and (logical_begins is None or stream_ids is None
                           or len(logical_begins) != batch or len(stream_ids) != batch):
            raise ValueError("memory reads require explicit life ids and logical positions")
        hidden, positions, masks, rotary = self._prepare(input_ids, attention_mask)
        query_features = None
        packed = None
        for index in range(len(self.base.layers)):
            if index in self.reader_before and has_memory:
                if packed is None:
                    query_positions = [logical_begins[row] + int(query_indices[row]) for row in range(batch)]
                    slots, valid, time_features = pack_records(
                        records, memory_dim=self.memory_config.memory_dim, query_positions=query_positions,
                        stream_ids=stream_ids, device=hidden.device, dtype=hidden.dtype, wall_times=wall_times)
                    read_positions = torch.arange(length, device=hidden.device)[None] >= query_indices[:, None]
                    mask = valid[:, None] & (read_positions & attention_mask.bool())[:, :, None]
                    with self._memory_context(hidden):
                        features = self.time_source(time_features)
                    packed = slots, mask, features
                with self._memory_context(hidden):
                    hidden = self.readers[str(index)](hidden, packed[0], packed[1], slot_features=packed[2])
            hidden = self._layer(index, hidden, positions, masks, rotary)
            if index == self.writer_after:
                query_features = hidden[torch.arange(batch, device=hidden.device), query_indices]
        hidden = self.base.norm(hidden)
        loss = response_cross_entropy_from_hidden(hidden, self.head.weight, labels) if labels is not None else None
        logits = self.head(hidden) if return_logits else None
        return MemoryModelOutput(hidden, query_features, loss, logits)

    def forward_cached(self, input_ids, cache, *, records=None, selector=None, stream_id=None,
                       memory_enabled=True, lookup_position=None, lookup_wall_time=None):
        """One causal span of one life; the executor handles canonical boundaries."""
        from .streaming_layers import run_layer
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("each streaming cache represents one nonempty life")
        if tuple(self.base.config.layer_types) != cache.layer_types:
            raise ValueError("cache layer topology differs from backbone")
        until_rebase = cache.until_rebase()
        if until_rebase is not None and input_ids.shape[1] > until_rebase:
            raise ValueError("executor must split at the canonical rebase boundary")
        updated = cache.fork()
        updated.rebase_at_boundary()
        hidden = self.base.embed_tokens(input_ids)
        write_features = None
        selected = list(records or [])
        packed = None
        for index in range(len(self.base.layers)):
            if index in self.reader_before and memory_enabled and selected:
                if stream_id is None or len(selected) > 8:
                    raise ValueError("stream reads require a life id and bounded candidates")
                if packed is None:
                    position = updated.clock if lookup_position is None else lookup_position
                    slots, valid, metadata = pack_records(
                        [selected], memory_dim=self.memory_config.memory_dim,
                        query_positions=[position], stream_ids=[stream_id], device=hidden.device,
                        dtype=hidden.dtype, wall_times=[lookup_wall_time])
                    with self._memory_context(hidden):
                        metadata = self.time_source(metadata)
                    packed = slots, valid, metadata
                with self._memory_context(hidden):
                    hidden = self.readers[str(index)](hidden, packed[0], packed[1], slot_features=packed[2])
            hidden = run_layer(self.base, index, hidden, updated)
            if index == self.writer_after:
                write_features = hidden
                if selector is not None and memory_enabled:
                    selected = list(selector(hidden[:, 0], updated.clock))
        updated.finish_span(input_ids.shape[1])
        return CachedModelOutput(self.base.norm(hidden), write_features, updated, selected)

    @classmethod
    def from_local_checkpoint(cls, path, *, device="cpu", memory_config=None, memory_architecture=None):
        """Load only language tensors; retain original embedding/output tying."""
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        config = Qwen3_5TextConfig(**raw.get("text_config", raw))
        config._attn_implementation = "eager"
        prefix = "model.language_model."
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        tensors = {}
        for filename in sorted(set(index.values())):
            with safe_open(path / filename, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if key.startswith(prefix):
                        tensors[key[len(prefix):]] = reader.get_tensor(key)
        with torch.device("meta"):
            base = Qwen3_5TextModel(config)
            head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        base.load_state_dict(tensors, strict=True, assign=True)
        # Nonpersistent rotary buffers were constructed on meta and are absent
        # from the weight file. Recreate only this parameter-free component.
        base.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cpu")
        if not config.tie_word_embeddings:
            raise ValueError("this loader requires the selected checkpoint's tied text head")
        head.weight = base.embed_tokens.weight
        layout = {}
        if memory_architecture is not None:
            if memory_config is not None:
                raise ValueError("explicit memory config and checkpoint architecture cannot be combined")
            memory_config = MemoryConfig(**memory_architecture["memory_config"])
            layout = {"writer_after": memory_architecture["writer_after"],
                      "reader_before": tuple(memory_architecture["reader_before"])}
        model = cls(base, head, memory_config=memory_config, **layout)
        if memory_architecture is not None and model.memory_architecture_identity() != memory_architecture:
            raise ValueError("unknown or inconsistent checkpoint memory architecture")
        return model.to(device)
