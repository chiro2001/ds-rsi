import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.model.lora import LoRAConfig, LoRALinear, install_lora, lora_disabled
from dsrsi.state.streaming_cache import StreamingCache


def test_zero_initialized_lora_matches_original_and_never_adapts_writer_features():
    torch.manual_seed(31)
    model = tiny()
    ids = torch.tensor([[3, 4, 5, 6]])
    reference = model(ids, memory_enabled=False).hidden.detach()
    source = model.source_features(ids).clone()
    layout = install_lora(model, LoRAConfig(rank=2, alpha=4))
    assert layout["start_layer"] == 2
    assert all(name.startswith(("base.layers.2.", "base.layers.3.")) for name in layout["targets"])
    assert torch.equal(reference, model(ids, memory_enabled=False).hidden)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_b.data.normal_(std=0.1)
    assert torch.equal(source, model.source_features(ids))
    assert not torch.equal(reference, model(ids, memory_enabled=False).hidden)
    with lora_disabled(model):
        assert torch.equal(reference, model(ids, memory_enabled=False).hidden)
        with lora_disabled(model):
            assert torch.equal(reference, model(ids, memory_enabled=False).hidden)
    assert all(m.enabled for m in model.modules() if isinstance(m, LoRALinear))


def test_lora_has_a_real_answer_gradient_with_empty_memory_and_cached_execution():
    model = tiny()
    install_lora(model, LoRAConfig(rank=2))
    ids = torch.tensor([[3, 4, 5, 6, 1]])
    labels = torch.tensor([[-100, -100, -100, 6, 1]])
    model(ids, labels=labels, query_indices=[2], memory_enabled=False).loss.mean().backward()
    assert sum(float(m.lora_b.grad.square().sum()) for m in model.modules() if isinstance(m, LoRALinear)) > 0
    assert all(m.original.weight.grad is None for m in model.modules() if isinstance(m, LoRALinear))
    model.zero_grad(set_to_none=True)
    cache = StreamingCache(model.base.config.layer_types, window=3, rebase_interval=3)
    outputs = []
    for index in range(ids.shape[1]):
        output = model.forward_cached(ids[:, index:index + 1], cache, memory_enabled=False)
        cache = output.cache
        outputs.append(output.hidden)
    torch.cat(outputs, 1).float().square().mean().backward()
    assert sum(float(m.lora_b.grad.square().sum()) for m in model.modules() if isinstance(m, LoRALinear)) > 0


def test_lora_refuses_early_sites_and_duplicate_installation():
    model = tiny()
    with pytest.raises(ValueError, match="follow Writer"):
        install_lora(model, LoRAConfig(start_layer=1))
    install_lora(model, LoRAConfig(rank=2))
    with pytest.raises(ValueError, match="already installed"):
        install_lora(model)
