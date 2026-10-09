import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from dsrsi.memory import MemoryConfig
from dsrsi.model.qwen35_memory import MemoryQwen


def tiny():
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
                             linear_num_key_heads=2, linear_num_value_heads=2,
                             layer_types=["linear_attention"] * 3 + ["full_attention"],
                             pad_token_id=0, eos_token_id=1,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                              "partial_rotary_factor": 1.0,
                                              "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    config._attn_implementation = "eager"
    base = Qwen3_5TextModel(config).eval()
    head = nn.Linear(16, 32, bias=False)
    head.weight = base.embed_tokens.weight
    memory = MemoryConfig(hidden_size=16, memory_dim=8, slots=3, heads=2,
                          intermediate=16, key_dim=4, summary_slots=4)
    return MemoryQwen(base, head, memory_config=memory, writer_after=1, reader_before=(2, 3))


def test_disabled_memory_matches_native_qwen_text_forward():
    model = tiny()
    inputs = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        want = model.base(inputs, use_cache=False).last_hidden_state
        got = model(inputs, memory_enabled=False).hidden
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_future_query_loss_reaches_writer_through_frozen_native_gdn_backbone():
    model = tiny().train()
    records = model.write_records(torch.tensor([[2, 3, 4, 5]]), stream_ids=["life"],
                                  logical_begins=[0], record_ids=["source"])
    inputs = torch.tensor([[7, 8, 9, 10, 1]])
    labels = torch.tensor([[-100, -100, -100, 10, 1]])
    out = model(inputs, records=[records], query_indices=[2], logical_begins=[4], stream_ids=["life"], labels=labels)
    out.loss.mean().backward()
    assert sum(float(p.grad.square().sum()) for p in model.writer.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert out.query_features.shape == (1, 16)


def test_memory_lookup_does_not_change_prefix_before_its_query_position():
    model = tiny().eval()
    records = model.write_records(torch.tensor([[2, 3, 4, 5]]), stream_ids=["life"],
                                  logical_begins=[0], record_ids=["source"])
    inputs = torch.tensor([[7, 8, 9, 10, 1]])
    with torch.no_grad():
        base = model(inputs, memory_enabled=False).hidden
        read = model(inputs, records=[records], query_indices=[2], logical_begins=[4], stream_ids=["life"]).hidden
    assert torch.equal(read[:, :2], base[:, :2])
    assert not torch.equal(read[:, 2:], base[:, 2:])


def test_teacher_forced_answer_cannot_be_used_for_retrieval():
    model = tiny()
    with pytest.raises(ValueError, match="supervised answer"):
        model(torch.tensor([[2, 3, 4]]), query_indices=[2], labels=torch.tensor([[-100, -100, 4]]))
