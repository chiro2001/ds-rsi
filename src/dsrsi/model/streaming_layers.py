"""Use native weights/attention math while caching normalized, unrotated K."""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, eager_attention_forward


def _rope(base, hidden, logical_positions, origin):
    positions = (logical_positions - origin).view(1, 1, -1).expand(3, hidden.shape[0], -1)
    return base.rotary_emb(hidden, positions)


def full_attention(base, module, hidden, cache, layer_index):
    batch, length, _ = hidden.shape
    shape = (batch, length, -1, module.head_dim)
    query, gate = torch.chunk(module.q_proj(hidden).view(batch, length, -1, module.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(batch, length, -1)
    query = module.q_norm(query.reshape(shape)).transpose(1, 2)
    raw_key = module.k_norm(module.k_proj(hidden).reshape(shape)).transpose(1, 2)
    value = module.v_proj(hidden).reshape(shape).transpose(1, 2)
    raw_keys, values, start = cache.append_unrotated(raw_key, value, layer_index)
    q_positions = torch.arange(cache.clock, cache.clock + length, device=hidden.device)
    k_positions = torch.arange(start, cache.clock + length, device=hidden.device)
    q_cos, q_sin = _rope(base, hidden, q_positions, cache.position_origin)
    k_cos, k_sin = _rope(base, hidden, k_positions, cache.position_origin)
    query = apply_rotary_pos_emb(query, query, q_cos, q_sin)[0]
    keys = apply_rotary_pos_emb(raw_keys, raw_keys, k_cos, k_sin)[0]
    allowed = k_positions[None] <= q_positions[:, None]
    if cache.window is not None:
        allowed = allowed & (k_positions[None] >= q_positions[:, None] - cache.window + 1)
    mask = torch.zeros(length, keys.shape[-2], device=hidden.device, dtype=hidden.dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]
    output, _ = eager_attention_forward(module, query, keys, values, mask,
                                        scaling=module.scaling, dropout=0.0)
    output = output.reshape(batch, length, -1) * gate.sigmoid()
    return module.o_proj(output)


def run_layer(base, index, hidden, cache):
    layer = base.layers[index]
    if layer.block_type == "linear_attention":
        # No padding: one continuous life per cache, with explicit external
        # batching of independent states rather than advancing through pad.
        return layer(hidden, position_embeddings=None, attention_mask=None,
                     position_ids=None, past_key_values=cache, use_cache=True)
    residual = hidden
    hidden = residual + full_attention(base, layer.self_attn, layer.input_layernorm(hidden), cache, index)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))
