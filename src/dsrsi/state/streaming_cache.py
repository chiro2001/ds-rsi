"""Functional hybrid cache with unrotated keys and an independent logical clock."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dsrsi.training.checkpoint import cpu_tree


@dataclass
class LayerState:
    keys: torch.Tensor | None = None  # normalized, unrotated K
    values: torch.Tensor | None = None
    kv_start: int = 0
    conv_states: list = field(default_factory=lambda: [None])
    recurrent_states: list = field(default_factory=lambda: [None])


class StreamingCache:
    def __init__(self, layer_types, *, window=2048, rebase_interval=4096):
        if window is not None and window < 1:
            raise ValueError("window must be positive; None is only an unbounded diagnostic")
        if rebase_interval is not None and rebase_interval < 1:
            raise ValueError("rebase interval must be positive")
        self.layer_types = tuple(layer_types)
        self.layers = [LayerState() for _ in layer_types]
        self.window = window
        self.rebase_interval = rebase_interval
        self.clock = 0
        self.position_origin = 0

    def fork(self):
        other = StreamingCache(self.layer_types, window=self.window, rebase_interval=self.rebase_interval)
        other.clock, other.position_origin = self.clock, self.position_origin
        for index, state in enumerate(self.layers):
            # Native single-token causal_conv1d_update mutates its input. Clone
            # only that small buffer; recurrent delta-rule state is functional.
            conv = [None if t is None else t.clone() for t in state.conv_states]
            other.layers[index] = LayerState(state.keys, state.values, state.kv_start,
                                             conv, list(state.recurrent_states))
        return other

    def has_previous_state(self, layer_idx, state_idx=0):
        state = self.layers[layer_idx]
        return state.conv_states[state_idx] is not None and state.recurrent_states[state_idx] is not None

    def update_conv_state(self, value, layer_idx, state_idx=0, **kwargs):
        self.layers[layer_idx].conv_states[state_idx] = value
        return value

    def update_recurrent_state(self, value, layer_idx, state_idx=0, **kwargs):
        self.layers[layer_idx].recurrent_states[state_idx] = value
        return value

    def append_unrotated(self, key, value, layer_idx):
        if self.layer_types[layer_idx] != "full_attention":
            raise ValueError("KV appended to a recurrent layer")
        state = self.layers[layer_idx]
        if state.keys is None:
            state.keys, state.values, state.kv_start = key, value, self.clock
        else:
            if state.kv_start + state.keys.shape[-2] != self.clock:
                raise ValueError("KV cache logical range is discontinuous")
            state.keys = torch.cat([state.keys, key], dim=-2)
            state.values = torch.cat([state.values, value], dim=-2)
        return state.keys, state.values, state.kv_start

    def rebase_at_boundary(self):
        if self.rebase_interval is not None:
            self.position_origin = (self.clock // self.rebase_interval) * self.rebase_interval

    def until_rebase(self):
        if self.rebase_interval is None:
            return None
        return self.rebase_interval - self.clock % self.rebase_interval

    def finish_span(self, length):
        if length < 1:
            raise ValueError("empty span")
        self.clock += length
        for state in self.layers:
            if state.keys is not None and self.window is not None and state.keys.shape[-2] > self.window:
                removed = state.keys.shape[-2] - self.window
                state.keys = state.keys[..., -self.window:, :]
                state.values = state.values[..., -self.window:, :]
                state.kv_start += removed

    def detached(self):
        other = self.fork()
        for state in other.layers:
            state.keys = None if state.keys is None else state.keys.detach()
            state.values = None if state.values is None else state.values.detach()
            state.conv_states = [None if t is None else t.detach() for t in state.conv_states]
            state.recurrent_states = [None if t is None else t.detach() for t in state.recurrent_states]
        return other

    def state_dict(self):
        return {"layer_types": list(self.layer_types), "window": self.window,
                "rebase_interval": self.rebase_interval, "clock": self.clock,
                "position_origin": self.position_origin,
                "layers": [cpu_tree({"keys": s.keys, "values": s.values, "kv_start": s.kv_start,
                                     "conv_states": s.conv_states, "recurrent_states": s.recurrent_states})
                           for s in self.layers]}

    @classmethod
    def from_state_dict(cls, data, *, device="cpu"):
        cache = cls(data["layer_types"], window=data["window"], rebase_interval=data["rebase_interval"])
        cache.clock, cache.position_origin = data["clock"], data["position_origin"]
        if len(data["layers"]) != len(cache.layers) or not 0 <= cache.position_origin <= cache.clock:
            raise ValueError("invalid streaming cache snapshot")
        def moved(value):
            return None if value is None else value.to(device).clone()
        for index, item in enumerate(data["layers"]):
            state = LayerState(moved(item["keys"]), moved(item["values"]), item["kv_start"],
                               [moved(t) for t in item["conv_states"]], [moved(t) for t in item["recurrent_states"]])
            if state.keys is not None:
                if state.values is None or state.values.shape != state.keys.shape:
                    raise ValueError("invalid KV snapshot shape")
                if state.kv_start + state.keys.shape[-2] != cache.clock:
                    raise ValueError("snapshot KV range does not end at logical clock")
                if cache.window is not None and state.keys.shape[-2] > cache.window:
                    raise ValueError("snapshot exceeds the configured KV window")
            cache.layers[index] = state
        return cache

    @property
    def tensor_bytes(self):
        tensors = [value for state in self.layers for value in
                   [state.keys, state.values, *state.conv_states, *state.recurrent_states] if value is not None]
        return sum(t.numel() * t.element_size() for t in tensors)
