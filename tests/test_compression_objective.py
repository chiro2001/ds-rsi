"""D-stage objective: summary-only answers, screened leaf teacher, chain depth."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.compression import compression_plan
from dsrsi.data.revisions import make_generator, COMPRESSION_REVISION
from dsrsi.memory import MemoryConfig
from dsrsi.model.qwen35_memory import MemoryQwen
from dsrsi.training.compression_objective import CompressionObjective, TeacherScreen
from test_compiler import BoundaryTokenizer

# BoundaryTokenizer maps each character to ord(char)+3; the fixture keeps the
# full-width punctuation the generated Chinese queries use.
VOCAB = 65536


class DecodingTokenizer(BoundaryTokenizer):
    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join(chr(token - 3) for token in tokens if token >= 3)


def tiny():
    config = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
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
    head = nn.Linear(16, VOCAB, bias=False)
    head.weight = base.embed_tokens.weight
    memory = MemoryConfig(hidden_size=16, memory_dim=8, slots=3, heads=2,
                          intermediate=16, key_dim=4, summary_slots=4)
    return MemoryQwen(base, head, memory_config=memory, writer_after=1, reader_before=(2, 3))


@pytest.fixture
def toolkit():
    torch.manual_seed(7)
    compiler = EpisodeCompiler(DecodingTokenizer(), max_query_tokens=10000)
    generator = make_generator(revision=COMPRESSION_REVISION, seed=5, split="train")
    return compiler, generator


def frozen_model():
    model = tiny().train()
    model.requires_grad_(False)
    model.consolidator.requires_grad_(True)
    return model


@pytest.mark.parametrize("world,depth", [(0, 1), (1, 2), (2, 4), (3, 8)])
def test_each_depth_builds_its_real_chain_and_backpropagates_only_the_consolidator(toolkit, world, depth):
    compiler, generator = toolkit
    row = generator.episode(world)
    assert compression_plan(row)["depth"] == depth
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.0, teacher_screening="none")
    out = objective(row)
    out.loss.backward()
    assert out.depth == depth and out.chain_levels == depth
    assert out.kl_weighted == 0.0 and out.teacher_screened is False
    assert out.supervised_tokens > 0
    assert sum(float(p.grad.square().sum()) for p in model.consolidator.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("consolidator."))


def test_teacher_reads_exactly_the_first_four_leaves(toolkit):
    compiler, generator = toolkit
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.2)
    for world in range(4):
        row = generator.episode(world)
        plan = compression_plan(row)
        records = objective.teacher_records(row)
        assert [record.record_id for record in records] == plan["leaves"][:4]
        assert all(record.level == 0 for record in records)


def test_kl_is_gated_on_the_screened_free_generation(toolkit, monkeypatch):
    from dsrsi.training import compression_objective as module
    compiler, generator = toolkit
    row = generator.episode(3)  # depth 8, support still in the first four
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.2)
    monkeypatch.setattr(objective, "_teacher_free_generation",
                        lambda *args, **kwargs: TeacherScreen("NOT-THE-TARGET", 10, [], False))
    refused = objective(row)
    assert refused.teacher_screened is False and refused.kl_loss == 0.0
    assert refused.screened_text == "NOT-THE-TARGET"
    assert refused.teacher_input_tokens == 10

    captured = []
    original_kl = module.topk_tail_kl
    def capture_kl(student, teacher):
        assert student.requires_grad and not teacher.requires_grad
        value = original_kl(student, teacher)
        captured.append(value)
        return value
    monkeypatch.setattr(module, "topk_tail_kl", capture_kl)
    monkeypatch.setattr(objective, "_teacher_free_generation",
                        lambda *args, **kwargs: TeacherScreen(row["probes"][0]["target"], 10, [], True))
    accepted = objective(row)
    assert accepted.teacher_screened is True and accepted.kl_loss > 0
    assert accepted.teacher_input_tokens == 10 + len(compiler.compile(row).responses[0].input_ids)
    # CE also has gradients: verify the KL term itself reaches the compressor.
    grads = torch.autograd.grad(captured[0], tuple(model.consolidator.parameters()),
                                retain_graph=True, allow_unused=True)
    assert sum(float(g.square().sum()) for g in grads if g is not None) > 0
    accepted.loss.backward()
    assert sum(float(p.grad.square().sum()) for p in model.consolidator.parameters() if p.grad is not None) > 0


def test_screening_uses_the_cached_answer_free_prefix_and_recorded_path(toolkit):
    compiler, generator = toolkit
    row = generator.episode(0)
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.2)
    consumed = []
    original = model.forward_cached
    def observe(input_ids, *args, **kwargs):
        assert input_ids.shape[1] == 1
        consumed.append(input_ids.numel())
        return original(input_ids, *args, **kwargs)
    model.forward_cached = observe
    out = objective(row)
    # A random tiny model cannot hit the target, but the screening must have run
    # through the cached path and reported which prefill semantics it used.
    assert out.screening_path == "token"
    assert out.teacher_screened is False
    assert isinstance(out.screened_text, str)
    assert out.teacher_input_tokens == sum(consumed)
    episode = compiler.compile(row)
    assert out.student_input_tokens == sum(len(s.input_ids) for s in episode.source) + len(episode.responses[0].input_ids)


def test_screening_excludes_supervised_answer_and_counts_eos_and_budget_paths(toolkit, monkeypatch):
    from dataclasses import replace
    compiler, generator = toolkit
    row = generator.episode(0)
    model = frozen_model()
    objective = CompressionObjective(model, compiler, max_screen_tokens=3)
    response = compiler.compile(row).responses[0]
    records = objective.teacher_records(row)
    poisoned = replace(response, input_ids=response.prompt_ids + [9, 9, 1], labels=[])
    def fixed_head(token):
        def forward(hidden):
            logits = torch.zeros(hidden.shape[0], VOCAB)
            logits[:, token] = 1
            return logits
        return forward
    monkeypatch.setattr(model.head, "forward", fixed_head(compiler.tokenizer.eos_token_id))
    ended = objective._teacher_free_generation(response, records, row["episode_id"])
    assert ended.stopped_on_end and ended.token_ids == []
    assert ended.input_tokens == response.prompt_length
    monkeypatch.setattr(model.head, "forward", fixed_head(ord("X") + 3))
    full = objective._teacher_free_generation(response, records, row["episode_id"])
    other = objective._teacher_free_generation(poisoned, records, row["episode_id"])
    assert full == other
    assert full.text == "XXX" and not full.stopped_on_end
    assert full.input_tokens == response.prompt_length + 2


def test_objective_freezes_the_codec_and_rejects_noncanonical_teacher_screening(toolkit):
    compiler, _ = toolkit
    model = tiny()
    CompressionObjective(model, compiler)
    assert all(p.requires_grad == name.startswith("consolidator.") for name, p in model.named_parameters())
    with pytest.raises(ValueError, match="canonical token"):
        CompressionObjective(model, compiler, screening_prefill="chunked")


def test_teacher_only_screening_never_reads_the_answer_into_memory(toolkit):
    compiler, generator = toolkit
    row = generator.episode(1)
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.0, teacher_screening="none")
    seen = {}
    original = model.consolidate_records

    def observed(children, *, record_id, visible_after=None):
        seen[record_id] = [child.record_id for child in children]
        return original(children, record_id=record_id, visible_after=visible_after)

    model.consolidate_records = observed
    objective(row)
    assert list(seen) == [f"{row['episode_id']}:summary-1", f"{row['episode_id']}:summary-2"]
    assert seen[f"{row['episode_id']}:summary-1"] == [e["event_id"] for e in row["source_events"][:4]]
    assert seen[f"{row['episode_id']}:summary-2"] == [
        f"{row['episode_id']}:summary-1", "neg-3", "neg-4", "neg-5"]


def test_compression_rows_must_not_declare_transforms_or_write_answers(toolkit):
    compiler, generator = toolkit
    model = frozen_model()
    objective = CompressionObjective(model, compiler, kl_weight=0.0, teacher_screening="none")
    row = generator.episode(0)
    row["probes"][0]["memory_transform"] = {"merge_events": ["support"]}
    with pytest.raises(ValueError, match="transforms"):
        objective(row)
    row = generator.episode(0)
    row["probes"][0]["write_enabled"] = True
    with pytest.raises(ValueError, match="must not write"):
        objective(row)
    row = generator.episode(1)
    row["annotations"]["support_event"] = "neg-3"
    row["probes"][0]["support_groups"] = [["neg-3"]]
    with pytest.raises(ValueError, match="first four"):
        objective(row)
