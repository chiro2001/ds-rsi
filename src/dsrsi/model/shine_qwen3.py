"""Qwen3 SHINE wrapper around pinned, numerically unchanged upstream modules.

Use the isolated Transformers 4.57.1 overlay. This is independent of the Qwen3.5
pilot and never loads its adapters. Only the full-state/checkpoint interface and
registration of upstream shared LoRA tensors are new.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import yaml

from dsrsi.vendor.shine.LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
from dsrsi.vendor.shine.metanetwork_family import Metanetwork


SHINE_QWEN3_VERSION = 'official-qwen3-r8-metalora128-axial4-rl-v1'


class AttributeDict(dict):
    """The mapping/attribute interface used by the upstream OmegaConf config."""
    __getattr__=dict.__getitem__


def network_config(config, memory_tokens, *, rank=8):
    raw=yaml.safe_load((Path(__file__).parents[1]/'vendor/shine/Qwen3-0.6B.yaml').read_text())
    transformer=deepcopy(raw['metanetwork']['transformer_cfg'])
    # The official small-model YAML already uses 1024/8/2048. Tiny CPU tests
    # change only these dimensions; actual 0.6B runs match its values exactly.
    for key in ['encoder_cfg','couple_encoder_cfg']:
        transformer[key].update(d_model=config.hidden_size,
                                nhead=8 if config.hidden_size%8==0 else 1,
                                dim_feedforward=2*config.hidden_size)
        transformer[key]=AttributeDict(transformer[key])
    return SimpleNamespace(num_layers=config.num_hidden_layers,num_mem_token=memory_tokens,
        hidden_size=config.hidden_size,model=SimpleNamespace(lora_r=rank),
        optim=SimpleNamespace(adapter_reg=0.),metanetwork=SimpleNamespace(type='transformer',method='rl',
            transformer_cfg=SimpleNamespace(**transformer)))


def per_layer_factor_size(config,rank):
    head=config.head_dim
    q=head*config.num_attention_heads
    kv=head*config.num_key_value_heads
    return rank*(7*config.hidden_size+2*q+2*kv+3*config.intermediate_size)


class OfficialShineQwen3(nn.Module):
    def __init__(self,metamodel,*,rank=8,write_rank=128):
        super().__init__()
        self.rank,self.write_rank=rank,write_rank
        cfg=metamodel.config
        size=per_layer_factor_size(cfg,rank)
        if size%cfg.hidden_size or cfg.num_mem_token!=size//cfg.hidden_size:
            raise ValueError('memory states must exactly cover generated factors')
        if metamodel.lora_params_numel(rank)!=size*cfg.num_hidden_layers:
            raise ValueError('upstream factor layout differs from shape calculation')
        self.memory_tokens=cfg.num_mem_token
        cfg.use_cache=False
        self.network=Metanetwork(metamodel,network_config(cfg,self.memory_tokens,rank=rank),
                                 size*cfg.num_hidden_layers)
        self.network.metamodel.requires_grad_(False)
        self.network.metamodel.model.mem_tokens.requires_grad_(True)
        self.writer=nn.ParameterDict()
        original=metamodel.init_lora_dict(write_rank,scale=self.network.scale,device=next(metamodel.parameters()).device)
        def register(tree,path=()):
            if isinstance(tree,dict):return {k:register(v,(*path,str(k))) for k,v in tree.items()}
            if tree is None:return None
            key='_'.join(path);value=nn.Parameter(tree.detach());self.writer[key]=value
            return value
        self._write=register(original)

    @classmethod
    def from_local_checkpoint(cls,path,*,device='cpu',rank=8,write_rank=128,tokenizer=None):
        config=Qwen3Config.from_pretrained(path,local_files_only=True)
        n=per_layer_factor_size(config,rank)
        if n%config.hidden_size:raise ValueError('non-integral memory token count')
        config.num_mem_token=n//config.hidden_size
        config._attn_implementation='eager'
        # Upstream loads in FP32 and disables AMP. Match it on the 5080.
        model=LoraQwen3ForCausalLM.from_pretrained(path,config=config,torch_dtype=torch.float32,
                                                 local_files_only=True,attn_implementation='eager')
        model.reset_mem_tokens()
        if tokenizer is not None:model.resize_token_embeddings(len(tokenizer))
        return cls(model.to(device),rank=rank,write_rank=write_rank).to(device)

    @property
    def language(self):return self.network.metamodel

    def architecture_identity(self):
        cfg=self.language.config
        return dict(version=SHINE_QWEN3_VERSION,rank=self.rank,write_rank=self.write_rank,
            hidden=cfg.hidden_size,layers=cfg.num_hidden_layers,memory_tokens=self.memory_tokens,
            scale=self.network.scale,method='rl',m2p_layers=4,heads=8,
            feedforward=cfg.hidden_size*2,projection='all q,k,v,o,gate,up,down at every layer',
            factors_per_memory=per_layer_factor_size(cfg,self.rank)*cfg.num_hidden_layers,
            dtype='float32',attention='eager',compile=False)

    def frozen_auxiliary_parameters(self):return {}

    def encode(self,source,*,checkpoint=False):
        return self.network.generate_lora_dict(source,torch.ones_like(source),self._write,
            use_gradient_checkpoint=checkpoint)

    def answer(self,ids,labels,*,memory=None,checkpoint=False):
        return self.language(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,
            loradict=memory,ignore_mem_token=True,use_cache=False,
            use_gradient_checkpoint=checkpoint).loss

    @torch.no_grad()
    def generate(self,prompt,*,memory=None,max_new_tokens=64,eos_token_id=None):
        ids=torch.tensor([prompt],device=next(self.parameters()).device)
        result=self.language.generate(input_ids=ids,attention_mask=torch.ones_like(ids),loradict=memory,
            ignore_mem_token=True,use_cache=True,do_sample=False,max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,pad_token_id=self.language.config.pad_token_id)
        answer=result[0,len(prompt):].tolist()
        return dict(token_ids=answer,input_tokens=len(prompt)+max(0,len(answer)-1),
                    stopped_on_end=bool(answer and answer[-1]==eos_token_id))
