"""Small SHINE-inspired source-only writer with differentiable generated LoRA.

This is a separate M-only architecture. It does not load the old slots or claim
compatibility with the published SHINE checkpoints or continuous life state.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .qwen35_memory import MemoryQwen, create_causal_mask, create_recurrent_attention_mask
from .streaming_layers import run_layer
from dsrsi.state.streaming_cache import StreamingCache
from dsrsi.training.losses import response_cross_entropy_from_hidden


SHINE_VERSION = "source-memory-queries-axial-m2p-down-v1"


@dataclass(frozen=True)
class ShineConfig:
    layers: tuple[int, ...] = (3, 7, 11, 15, 19, 23)
    rank: int = 4
    width: int = 256
    write_rank: int = 4
    scale: float = 1.0

    def __post_init__(self):
        if (not self.layers or tuple(sorted(set(self.layers))) != self.layers
                or min(self.layers) < 0 or min(self.rank, self.width, self.write_rank) < 1
                or not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("invalid SHINE configuration")


@dataclass
class GeneratedMemory:
    record_id: str
    life_id: str
    factors: dict[int, tuple[torch.Tensor, torch.Tensor]]

    def state(self):
        return dict(record_id=self.record_id, life_id=self.life_id,
                    factors={str(k): tuple(t.detach().cpu().clone() for t in v)
                             for k, v in self.factors.items()})

    @classmethod
    def from_state(cls, value, device):
        return cls(value["record_id"], value["life_id"],
                   {int(k): tuple(t.to(device) for t in v) for k, v in value["factors"].items()})

    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.factors.values() for t in pair)


class DynamicDown(nn.Module):
    """Source-only shared adapter; read-time factors are ordinary graph tensors."""

    def __init__(self, original, write_rank, scale):
        super().__init__()
        self.original = original.requires_grad_(False)
        self.write_a = nn.Parameter(torch.empty(write_rank, original.in_features, device=original.weight.device))
        self.write_b = nn.Parameter(torch.zeros(original.out_features, write_rank, device=original.weight.device))
        nn.init.kaiming_uniform_(self.write_a, a=math.sqrt(5))
        self.mode, self.active, self.scale = "off", (), scale

    @property
    def weight(self):
        return self.original.weight

    def forward(self, hidden):
        out = self.original(hidden)
        pairs = ((self.write_a, self.write_b),) if self.mode == "write" else self.active
        if not pairs:
            return out
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = None
            for a, b in pairs:
                term = F.linear(F.linear(hidden.float(), a.float()), b.float())
                delta = term if delta is None else delta + term
        return out + (delta * self.scale).to(out.dtype)


class EagerAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.heads=4 if width%4==0 else 1
        self.qkv=nn.Linear(width,3*width)
        self.out=nn.Linear(width,width)

    def forward(self,x):
        b,t,w=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,w//self.heads).permute(2,0,3,1,4).unbind(0)
        probability=((q@k.transpose(-1,-2))/math.sqrt(w//self.heads)).softmax(-1)
        return self.out((probability@v).transpose(1,2).reshape(b,t,w))


class AxialBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.token_norm,self.layer_norm,self.ffn_norm=[nn.LayerNorm(width) for _ in range(3)]
        self.tokens,self.layers=EagerAttention(width),EagerAttention(width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))

    def forward(self,x):
        x=x+self.tokens(self.token_norm(x))
        y=x.transpose(0,1)
        y=y+self.layers(self.layer_norm(y))
        x=y.transpose(0,1)
        return x+self.ffn(self.ffn_norm(x))


class ShineQwen(nn.Module):
    def __init__(self, language, config=ShineConfig()):
        super().__init__()
        self.config = config
        self.base, self.head = language.base, language.head
        self.base.requires_grad_(False)
        self.head.requires_grad_(False)
        if max(config.layers) >= len(self.base.layers):
            raise ValueError("SHINE target layer is absent")
        self._sites = {}
        dimensions = set()
        for index in config.layers:
            parent = self.base.layers[index].mlp
            original = parent.down_proj
            if not isinstance(original, nn.Linear):
                raise ValueError("SHINE needs unmodified down projections")
            dimensions.add((original.in_features, original.out_features))
            site = DynamicDown(original, config.write_rank, config.scale)
            parent.down_proj = site
            self._sites[index] = site
        if len(dimensions) != 1:
            raise ValueError("shared M2P heads require identical target dimensions")
        self.input_width, self.output_width = next(iter(dimensions))
        hidden = self.base.config.hidden_size
        self.memory_tokens=math.ceil(config.rank*(self.input_width+self.output_width)/hidden)
        self.memory_queries = nn.Parameter(torch.randn(self.memory_tokens, hidden) * 0.02)
        self.layer_embeddings = nn.Parameter(torch.randn(len(config.layers), config.width) * 0.02)
        self.token_embeddings = nn.Parameter(torch.randn(self.memory_tokens,config.width)*0.02)
        self.project = nn.Linear(hidden, config.width)
        self.m2p = nn.Sequential(AxialBlock(config.width),AxialBlock(config.width))
        self.back=nn.Linear(config.width,hidden)
        self.a_head = nn.Linear(hidden, hidden)
        self.b_head = nn.Linear(hidden, hidden)
        nn.init.normal_(self.a_head.weight, std=0.02 / math.sqrt(hidden))
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)
        self.train()

    @classmethod
    def from_local_checkpoint(cls, path, *, device="cpu", config=ShineConfig()):
        language = MemoryQwen.from_local_checkpoint(path, device=device)
        return cls(language, config).to(device)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.head.eval()
        return self

    def architecture_identity(self):
        return dict(version=SHINE_VERSION, config=asdict(self.config),
                    target_input=self.input_width, target_output=self.output_width,
                    memory_tokens=self.memory_tokens,m2p="two-bidirectional-axial-blocks-eager",
                    factor_dtype="float32", generated_factor_parameters_per_memory=
                    len(self.config.layers) * self.config.rank * (self.input_width + self.output_width))

    def frozen_auxiliary_parameters(self):
        return {}

    @contextmanager
    def _scope(self, mode, memories=(), life_id=None):
        if mode not in {"off", "write", "read"}:
            raise ValueError("unknown SHINE scope")
        memories = tuple(memories)
        if (mode != "read" and memories or len(memories) > 8
                or memories and (not life_id or any(m.life_id != life_id for m in memories))
                or len({m.record_id for m in memories}) != len(memories)):
            raise ValueError("invalid/cross-life/duplicate SHINE memory")
        for memory in memories:
            if set(memory.factors) != set(self.config.layers):
                raise ValueError("generated target topology differs")
            for a, b in memory.factors.values():
                if a.shape != (self.config.rank, self.input_width) or b.shape != (self.output_width, self.config.rank):
                    raise ValueError("generated factor dimensions differ")
        saved = {k: (s.mode, s.active) for k, s in self._sites.items()}
        try:
            for index, site in self._sites.items():
                site.mode = mode
                site.active = tuple(m.factors[index] for m in memories)
            yield
        finally:
            for index, site in self._sites.items():
                site.mode, site.active = saved[index]

    def _run(self, embeddings, *, collect=False):
        batch, length, _ = embeddings.shape
        positions = torch.arange(length, device=embeddings.device)[None, None].expand(4, batch, -1)
        kwargs = dict(config=self.base.config, inputs_embeds=embeddings, attention_mask=None,
                      past_key_values=None, position_ids=positions[0])
        masks = dict(full_attention=create_causal_mask(**kwargs),
                     linear_attention=create_recurrent_attention_mask(**kwargs))
        rotary = self.base.rotary_emb(embeddings, positions[1:])
        hidden, states = embeddings, []
        for index, layer in enumerate(self.base.layers):
            hidden = layer(hidden, position_embeddings=rotary,
                           attention_mask=masks[self.base.config.layer_types[index]],
                           position_ids=positions[0], past_key_values=None, use_cache=False)
            if collect and index in self._sites:
                states.append(hidden[:, -self.memory_tokens:])
        return self.base.norm(hidden), states

    def encode(self, source_ids, *, record_id, life_id):
        if source_ids.ndim != 2 or source_ids.shape[0] != 1 or source_ids.shape[1] < 1 or not record_id or not life_id:
            raise ValueError("SHINE source must be one identified nonempty observed context")
        embeddings = self.base.embed_tokens(source_ids)
        tokens = self.memory_queries.to(embeddings.dtype)[None]
        with self._scope("write"):
            _, states = self._run(torch.cat((embeddings, tokens), dim=1), collect=True)
        factors = {}
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            raw=torch.stack([state[0].float() for state in states])
            z=self.project(F.layer_norm(raw,(raw.shape[-1],)))
            z=z+self.layer_embeddings[:,None]+self.token_embeddings[None]
            z=self.m2p(z)
            enriched=F.layer_norm(raw+self.back(z),(raw.shape[-1],))
            av,bv=self.a_head(enriched),self.b_head(enriched)
            na=self.config.rank*self.input_width
            nb=self.config.rank*self.output_width
            for ordinal,index in enumerate(self.config.layers):
                a=av[ordinal].reshape(-1)[:na].reshape(self.config.rank,self.input_width)
                b=bv[ordinal].reshape(-1)[na:na+nb].reshape(self.config.rank,self.output_width).transpose(0,1).contiguous()
                factors[index]=(a,b)
        return GeneratedMemory(record_id, life_id, factors)

    def forward(self, input_ids, *, memories=(), life_id=None, labels=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("SHINE query must be one nonempty life")
        with self._scope("read", memories, life_id):
            hidden, _ = self._run(self.base.embed_tokens(input_ids))
        loss = response_cross_entropy_from_hidden(hidden, self.head.weight, labels) if labels is not None else None
        return hidden, loss

    def cached(self, input_ids, cache, *, memories=(), life_id=None):
        if input_ids.shape != (1, 1) or tuple(self.base.config.layer_types) != cache.layer_types:
            raise ValueError("SHINE canonical generation requires one token and matching topology")
        updated = cache.fork()
        updated.rebase_at_boundary()
        hidden = self.base.embed_tokens(input_ids)
        with self._scope("read", memories, life_id):
            for index in range(len(self.base.layers)):
                hidden = run_layer(self.base, index, hidden, updated)
        updated.finish_span(1)
        return self.base.norm(hidden), updated

    @torch.no_grad()
    def generate(self, prompt_ids, *, memories=(), life_id=None, eos_token_id, max_new_tokens=16):
        if not prompt_ids or max_new_tokens < 1:
            raise ValueError("SHINE decoding requires a bounded answer-free prompt")
        device = self.base.embed_tokens.weight.device
        cache = StreamingCache(self.base.config.layer_types, window=2048, rebase_interval=None)
        for token in prompt_ids:
            hidden, cache = self.cached(torch.tensor([[token]], device=device), cache,
                                        memories=memories, life_id=life_id)
        generated, stopped = [], False
        for _ in range(max_new_tokens):
            token = int(self.head(hidden[:, -1]).argmax(-1).item())
            if token == eos_token_id:
                stopped = True
                break
            generated.append(token)
            hidden, cache = self.cached(torch.tensor([[token]], device=device), cache,
                                        memories=memories, life_id=life_id)
        return dict(token_ids=generated, stopped_on_end=stopped,
                    input_tokens=len(prompt_ids) + len(generated))
