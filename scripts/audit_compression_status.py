#!/usr/bin/env python3
"""Audit the D-stage compression protocol against the current implementation.

Read-only. Re-verifies the mechanisms that already exist (time/source
conditioned compaction, atomic tight-budget replacement, frozen leaf gradient
isolation) and records what the D curriculum still needs. Production
dimensions are used for byte accounting; the executable checks run on a tiny
model that shares the same code path. It never loads Qwen3.5-0.8B and never
touches a training run.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from dsrsi.memory import MemoryConfig
from dsrsi.memory.store import MemoryRecord, MemoryStore
from dsrsi.model.qwen35_memory import MemoryQwen

ROOT = Path(__file__).resolve().parents[1]
TINY_SLOTS, TINY_DIM, TINY_KEYS, TINY_KEY_DIM, TINY_SUMMARY_SLOTS = 3, 8, 2, 4, 4
PRODUCTION = {"slots": 8, "memory_dim": 256, "keys": 2, "key_dim": 128, "summary_slots": 16}


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
    memory = MemoryConfig(hidden_size=16, memory_dim=TINY_DIM, slots=TINY_SLOTS, heads=2,
                          intermediate=16, key_dim=TINY_KEY_DIM, summary_slots=TINY_SUMMARY_SLOTS)
    return MemoryQwen(base, head, memory_config=memory, writer_after=1, reader_before=(2, 3))


def tiny_leaf(model, index):
    slots = torch.randn(TINY_SLOTS, TINY_DIM)
    keys = model.address.encode_keys(slots[None])[0]
    return MemoryRecord(f"leaf-{index}", "life", slots, keys, index * 4, index * 4 + 3, index * 4 + 4,
                        wall_time=100.0 + index)


def curriculum_chain(model, depth):
    """3d+1 leaves compressed as four leaves then 'old summary + three leaves'."""
    cursor = {"index": 0}

    def leaf():
        record = tiny_leaf(model, cursor["index"])
        cursor["index"] += 1
        return record

    leaves = [leaf() for _ in range(4)]
    summary = model.consolidate_records(leaves, record_id="summary-1")
    for level in range(2, depth + 1):
        summary = model.consolidate_records([summary] + [leaf() for _ in range(3)],
                                            record_id=f"summary-{level}")
    consumed = cursor["index"]
    leaf_bytes = sum(tiny_leaf(model, 0).tensor_bytes for _ in range(consumed))
    return {"depth": depth, "leaves_consumed": consumed, "summary_level": summary.level,
            "summary_source": [summary.source_begin, summary.source_end],
            "summary_bytes": summary.tensor_bytes, "leaf_total_bytes": leaf_bytes,
            "reduction": round(1 - summary.tensor_bytes / leaf_bytes, 4),
            "child_refs": len(summary.child_refs)}


def production_sizes():
    def record(base_floats, slots):
        slots_t = torch.zeros(slots, PRODUCTION["memory_dim"], dtype=torch.bfloat16)
        keys_t = torch.zeros(PRODUCTION["keys"], PRODUCTION["key_dim"], dtype=torch.bfloat16)
        return MemoryRecord("x", "life", slots_t, keys_t, 0, slots, slots + 1).tensor_bytes
    leaf = record(None, PRODUCTION["slots"])
    summary = record(None, PRODUCTION["summary_slots"])
    return {"leaf_bytes_bf16": leaf, "summary_bytes_bf16": summary,
            "four_leaf_reduction": round(1 - summary / (4 * leaf), 4),
            "two_leaf_reduction": round(1 - summary / (2 * leaf), 4),
            "summary_over_leaf": round(summary / leaf, 4),
            "leaves_at_depth_8": 3 * 8 + 1, "summary_bytes_at_depth_8": summary}


def source_scan():
    def has(relative, needle):
        path = ROOT / relative
        return needle in path.read_text() if path.exists() else False

    frozen = {name: has(name, "consolidator.requires_grad_(False)") for name in (
        "src/dsrsi/training/episode_objective.py", "src/dsrsi/training/context_runner.py",
        "src/dsrsi/training/tool_runner.py", "src/dsrsi/evaluation/checkpoint.py")}
    data_modules = sorted(p.name for p in (ROOT / "src/dsrsi/data").glob("*.py"))
    scripts = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if p.name != Path(__file__).name)
    return {
        "consolidator_frozen_in_ab_c_and_loaders": all(frozen.values()),
        "consolidator_frozen_by_file": frozen,
        "compression_data_module_present": any("compress" in name or "summar" in name for name in data_modules),
        "compression_data_modules": data_modules,
        "compression_objective_present": has("src/dsrsi/training/compression_objective.py", "class CompressionObjective"),
        "ab_objective_rejects_merge_events": has("src/dsrsi/training/episode_objective.py",
                                                 "A/B cannot silently train a compression transform"),
        "m_only_evaluator_rejects_merge_events": has("src/dsrsi/evaluation/m_only.py",
                                                     "compression evaluation requires its explicit"),
        "compression_runner_present": any(name.startswith(("train_compress", "train_summar")) for name in scripts),
        "compression_scripts": [name for name in scripts if "compress" in name or "summar" in name],
        "ab_runner_stages": "A\", \"B\"" in (ROOT / "scripts/train_ab.py").read_text(),
        "evaluation_loader_blocks_non_ab": has("src/dsrsi/evaluation/checkpoint.py",
                                               "evaluation loader supports explicit A/B checkpoints"),
        "config_stage_d_declared": has("configs/training_protocol_v0.yaml", "train_consolidator_first"),
    }


def invariants():
    torch.manual_seed(7)
    model = tiny().eval()
    children = [tiny_leaf(model, index) for index in range(4)]
    with torch.no_grad():
        parent = model.consolidate_records(children, record_id="parent", visible_after=20)
        later = model.consolidate_records(children, record_id="later", visible_after=4000)
        changed = model.consolidate_records([replace(r, source_kind="tool", source_role="tool") for r in children],
                                            record_id="changed", visible_after=20)
        unknown = model.consolidate_records([children[0], replace(children[1], wall_time=None)],
                                            record_id="unknown")
        reversed_order = model.consolidate_records(list(reversed(children)), record_id="reversed", visible_after=20)
    order_delta = float((parent.slots - reversed_order.slots).abs().max() /
                        parent.slots.abs().max().clamp_min(1e-6))
    rejections = {}
    for name, call in {
            "future_visibility": lambda: model.consolidate_records(children, record_id="bad", visible_after=15),
            "foreign_stream": lambda: model.consolidate_records([children[0], replace(children[1], stream_id="other")],
                                                               record_id="bad"),
            "single_child": lambda: model.consolidate_records(children[:1], record_id="bad")}.items():
        try:
            call()
            rejections[name] = False
        except ValueError:
            rejections[name] = True

    # Stored keys are frozen at write time; retrieve() never re-encodes them.
    store = MemoryStore(memory_dim=TINY_DIM, key_dim=TINY_KEY_DIM, max_tensor_bytes=1 << 20)
    for record in children:
        store.append(record)
    query = F.normalize(torch.randn(TINY_KEY_DIM), dim=-1)
    before = [r.record_id for r in store.retrieve(query, stream_id="life", query_position=100, topk=2)]
    with torch.no_grad():
        model.address.key.weight.zero_()
    after = [r.record_id for r in store.retrieve(query, stream_id="life", query_position=100, topk=2)]
    with torch.no_grad():
        reencoded = [record.record_id for record in sorted(
            children, key=lambda r: -float(torch.dot(query, model.address.encode_keys(r.slots[None])[0][0])))[:2]]

    train_model = tiny().train()
    train_model.requires_grad_(False)
    train_model.consolidator.requires_grad_(True)
    summary = train_model.consolidate_records(children, record_id="summary")
    labels = torch.tensor([[-100, -100, -100, 10, 1]])
    out = train_model(torch.tensor([[7, 8, 9, 10, 1]]), records=[[summary]], query_indices=[2],
                      logical_begins=[16], stream_ids=["life"], labels=labels)
    out.loss.mean().backward()
    consolidator_grad = sum(float(p.grad.square().sum()) for p in train_model.consolidator.parameters()
                            if p.grad is not None)
    foreign_grads = [name for name, p in train_model.named_parameters()
                     if not name.startswith("consolidator.") and p.grad is not None]

    return {
        "future_commit_time_changes_content": not torch.equal(parent.slots, later.slots),
        "source_metadata_changes_content": not torch.equal(parent.slots, changed.slots),
        "unknown_wall_time_reported_known": unknown.wall_time is not None,
        "child_order_relative_delta": order_delta,
        "child_order_material_delta": order_delta > 1e-3,
        "child_order_bitwise_difference_only_float_noise": not torch.equal(parent.slots, reversed_order.slots),
        "rejections": rejections,
        "summary_answer_ce_reaches_consolidator": consolidator_grad > 0,
        "consolidator_grad_norm_sq": round(consolidator_grad, 6),
        "non_consolidator_parameter_grads": foreign_grads,
        "cached_keys_ignore_address_key_update": before == after,
        "reencoded_selection_after_update": reencoded,
        "cached_selection_before_update": before,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output")
    args = parser.parse_args()
    model = tiny().eval()
    chains = [curriculum_chain(model, depth) for depth in args.depth]
    production_config = MemoryConfig(hidden_size=1024, memory_dim=PRODUCTION["memory_dim"],
                                     slots=PRODUCTION["slots"], key_dim=PRODUCTION["key_dim"],
                                     summary_slots=PRODUCTION["summary_slots"])
    from dsrsi.memory.modules import Consolidator
    scan = source_scan()
    missing = []
    if not scan["compression_data_module_present"]:
        missing.append("D 内容课程尚未实现：没有生成 3d+1 叶链、支撑限定在第一组四叶的数据模块")
    if not scan["compression_objective_present"]:
        missing.append("D 目标函数尚未实现：压缩后回答 CE 与经自由生成筛查的叶教师 top32+tail KL 都无执行路径")
    report = {
        "status": "compression_protocol_not_trained",
        "capability_gate_passed": False,
        "consolidator_parameters_tiny": sum(p.numel() for p in model.consolidator.parameters()),
        "consolidator_parameters_production": sum(p.numel() for p in Consolidator(production_config).parameters()),
        "source_scan": scan,
        "production_sizes": production_sizes(),
        "curriculum_chain": chains,
        "invariants": invariants(),
        "runtime_compaction_policy": "StreamExecutor._fit_record compacts the first four stored records in "
                                     "insertion order; if that cannot reduce, it merges incoming with them",
        "open_requirements": missing + [
            "D 训练器尚未实现：train_ab.py 只接受 A/B；恢复/控制/账本与 stage D 未接入",
            "摘要寻址尚未适配：地址模块没有叶/摘要混合池覆盖训练",
            "压缩评测尚未实现：m_only 明确拒绝 merge_events，没有禁用/反事实 payload、递归深度分项和保留率评测",
            "D token 预算未定：configs/training_protocol_v0.yaml 仍写 pending_C_results",
        ],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
