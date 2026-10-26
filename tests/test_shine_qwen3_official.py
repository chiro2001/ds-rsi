"""Run with the isolated 4.57.1 overlay; ordinary Qwen3.5 CI skips this module."""

import copy

import pytest
import torch

transformers=pytest.importorskip('transformers')
if transformers.__version__!='4.57.1':pytest.skip('official SHINE requires isolated Transformers 4.57.1',allow_module_level=True)

from dsrsi.model.shine_qwen3 import OfficialShineQwen3,per_layer_factor_size
from dsrsi.vendor.shine.LoraQwen import LoraQwen3ForCausalLM,Qwen3Config
from dsrsi.training.checkpoint import Progress,capture_training_state,restore_training_state
from dsrsi.training.topology import require_equal


def tiny():
    c=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=96,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,pad_token_id=0,eos_token_id=1)
    c.num_mem_token=per_layer_factor_size(c,2)//c.hidden_size;c._attn_implementation='eager'
    base=LoraQwen3ForCausalLM(c);base.reset_mem_tokens()
    return OfficialShineQwen3(base,rank=2,write_rank=4)


def backward(m,scale=1.):
    memory=m.encode(torch.tensor([[3,4,5]]))
    loss=m.answer(torch.tensor([[6,7,8,1]]),torch.tensor([[-100,-100,8,1]]),memory=memory)
    (loss*scale).backward()
    return loss


def test_disabled_upstream_model_matches_native_hf_logits():
    torch.manual_seed(3);m=tiny()
    from transformers import Qwen3ForCausalLM
    native=Qwen3ForCausalLM(copy.deepcopy(m.language.config))
    missing,unexpected=native.load_state_dict(m.language.state_dict(),strict=False)
    assert not missing and unexpected==['model.mem_tokens']
    ids=torch.tensor([[3,4,5,6]])
    torch.testing.assert_close(m.language(ids,ignore_mem_token=True,use_cache=False).logits,
                               native(ids,use_cache=False).logits,rtol=0,atol=0)


def test_direct_upstream_reshape_keeps_gradient_to_writer_memory_and_m2p():
    torch.manual_seed(5);m=tiny();backward(m)
    assert m.language.model.mem_tokens.grad.abs().sum()>0
    assert m.network.metanetwork.layer_pe.grad.abs().sum()>0
    assert sum(p.grad.abs().sum() for n,p in m.writer.items() if n.endswith('_B'))>0
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    with torch.no_grad():
        output=m.language(torch.tensor([[3,4,5]]),loradict=m._write,use_cache=False)
        flat=m.network.metanetwork(output.memory_states)
        generated=m.language.generate_lora_dict(m.rank,.001,flat)
        encoded=m.encode(torch.tensor([[3,4,5]]))
    for i in generated:
        for family in generated[i]:
            for projection in generated[i][family]:
                for factor in ['A','B']:
                    assert torch.equal(generated[i][family][projection][factor],encoded[i][family][projection][factor])


def test_full_adam_and_pending_gradient_restore_preserves_next_update():
    torch.manual_seed(7);m=tiny();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=5e-5,foreach=False)
    backward(m);opt.step();opt.zero_grad(set_to_none=True);backward(m,.5)
    p=capture_training_state(m,opt,progress=Progress('SHINE_qwen3_pretrain',1,3,1,1234,0),
                            data_cursor={'next_ordinal':3},life_state={})
    # Real runs reload the same fingerprinted pretrained base. Recreate the same
    # frozen random base here; full checkpoints intentionally omit base weights.
    torch.manual_seed(7)
    restored=tiny();other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=5e-5,foreach=False)
    restore_training_state(p,restored,other)
    a=backward(m,.5);opt.step();opt.zero_grad(set_to_none=True)
    b=backward(restored,.5);other.step();other.zero_grad(set_to_none=True)
    assert torch.equal(a,b)
    progress=Progress('SHINE_qwen3_pretrain',2,4,0,1500,0)
    left=capture_training_state(m,opt,progress=progress,data_cursor={'next_ordinal':4},life_state={})
    right=capture_training_state(restored,other,progress=progress,data_cursor={'next_ordinal':4},life_state={})
    require_equal(left,right,'official_full_state')


def test_actual_small_model_factor_geometry_matches_official_configuration():
    c=Qwen3Config(hidden_size=1024,intermediate_size=3072,num_hidden_layers=28,
                  num_attention_heads=16,num_key_value_heads=8,head_dim=128)
    assert per_layer_factor_size(c,8)==180224
    assert per_layer_factor_size(c,8)//c.hidden_size==176
    assert per_layer_factor_size(c,8)*28==5046272
