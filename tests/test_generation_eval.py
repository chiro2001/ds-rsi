from dataclasses import replace

import pytest
import torch

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_episode_objective import SmallVocabularyTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.data.generate import EpisodeGenerator
from dsrsi.evaluation.m_only import MOnlyEvaluator


class DecodingTokenizer(SmallVocabularyTokenizer):
    def decode(self, tokens, **kwargs):
        return ",".join(str(t) for t in tokens)


def build():
    torch.manual_seed(44)
    model = tiny().eval()
    compiler = EpisodeCompiler(DecodingTokenizer(), max_query_tokens=1000)
    row = EpisodeGenerator(split="dev", distractors=2).episode(1, family="single_fact", language="en")
    episode = compiler.compile(row)
    evaluator = MOnlyEvaluator(model, compiler, max_new_tokens=3)
    with torch.no_grad():
        records = evaluator.source_records(episode)
    return evaluator, row, episode, records


def test_free_generation_does_not_consume_teacher_answer_or_expected_value():
    evaluator, row, episode, records = build()
    response = episode.responses[0]
    a = evaluator.generate(response, records, life_id=episode.episode_id, expected="first-label", mode="oracle")
    # Poison all supervised token IDs/labels; prompt and source stay unchanged.
    poisoned = replace(response, input_ids=response.prompt_ids + [30, 30, 1],
                       labels=[-100] * response.prompt_length + [30, 30, 1])
    b = evaluator.generate(poisoned, records, life_id=episode.episode_id, expected="different-label", mode="oracle")
    assert a.token_ids == b.token_ids and a.text == b.text


def test_modes_are_independent_and_off_has_no_selected_records():
    evaluator, row, episode, records = build()
    response = episode.responses[0]
    oracle = evaluator.generate(response, records, life_id=episode.episode_id, expected="x", mode="oracle")
    off = evaluator.generate(response, records, life_id=episode.episode_id, expected="x", mode="off")
    again = evaluator.generate(response, records, life_id=episode.episode_id, expected="x", mode="oracle")
    assert oracle.token_ids == again.token_ids
    assert off.selected_records == []
    assert len(oracle.token_ids) <= 3


def test_train_worlds_are_not_allowed_as_heldout_evaluation():
    evaluator, row, episode, records = build()
    row["split"] = "train"
    with pytest.raises(ValueError, match="training worlds"):
        evaluator.episode(row)


def test_content_shuffle_preserves_actual_addresses_and_replaces_payloads():
    evaluator, row, episode, records = build()
    response = episode.responses[0]
    captured = []
    original = evaluator.model.forward_cached
    def capture(*args, **kwargs):
        if kwargs.get("records"):
            captured.append(kwargs["records"])
        return original(*args, **kwargs)
    evaluator.model.forward_cached = capture
    actual = evaluator.generate(response, records, life_id=episode.episode_id, expected="x", mode="actual")
    original_payloads = {r.record_id: r.slots.clone() for r in records}
    captured.clear()
    shuffled = evaluator.generate(response, records, life_id=episode.episode_id, expected="x", mode="shuffled")
    assert actual.selected_records == shuffled.selected_records
    assert captured
    assert any(not torch.equal(original_payloads[r.record_id], r.slots) for r in captured[0])
    assert all(torch.equal(original_payloads[r.record_id], r.slots) for r in records)


def test_retrieval_only_is_answer_free_and_matches_generation_addresses():
    evaluator, row, episode, records = build()
    response = episode.responses[0]
    before = evaluator.retrieval(response, records, life_id=episode.episode_id)
    poisoned = replace(response, input_ids=response.prompt_ids + [31, 31, 1],
                       labels=[-100] * response.prompt_length + [31, 31, 1])
    after = evaluator.retrieval(poisoned, records, life_id=episode.episode_id)
    generation = evaluator.generate(response, records, life_id=episode.episode_id, expected="unused", mode="actual")
    assert before == after
    assert before["selected_records"] == generation.selected_records
    assert before["support_coverage"] == generation.actual_support_coverage
    assert before["evidence_available"] and before["required_groups"] == 1


def test_foreign_payload_removes_all_original_selected_content_but_preserves_addresses():
    evaluator, row, episode, records = build()
    foreign = [replace(r, stream_id="other-life", slots=r.slots + 100) for r in records]
    original = evaluator.model.forward_cached
    captured = []
    def capture(*args, **kwargs):
        if kwargs.get("records"):
            captured.append(kwargs["records"])
        return original(*args, **kwargs)
    evaluator.model.forward_cached = capture
    actual = evaluator.generate(episode.responses[0], records, life_id=episode.episode_id, expected="x")
    captured.clear()
    altered = evaluator.generate(episode.responses[0], records, life_id=episode.episode_id, expected="x",
                                 mode="foreign", replacement_records=foreign)
    assert actual.selected_records == altered.selected_records
    originals = {r.record_id: r for r in records}
    for r in captured[0]:
        assert torch.equal(r.slots, originals[r.record_id].slots + 100)
        assert torch.equal(r.keys, originals[r.record_id].keys) and r.stream_id == episode.episode_id
    with pytest.raises(ValueError, match="independent"):
        evaluator.generate(episode.responses[0], records, life_id=episode.episode_id, expected="x",
                           mode="foreign", replacement_records=records)


def test_evaluation_weights_loader_rejects_wrong_base_without_changing_weights(tmp_path):
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.training.checkpoint import CheckpointError
    evaluator, _, _, _ = build()
    trainer = Trainer(evaluator.model, evaluator.compiler, tmp_path, TrainConfig(distractors=2), base_identity="base")
    trainer.run(max_micro_steps=1, install_signals=False)
    model = tiny()
    before = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
    with pytest.raises(CheckpointError, match="base differs"):
        load_ab_weights(model, trainer.store, base_identity="different-base")
    for name, p in model.named_parameters():
        if name in before:
            assert torch.equal(before[name], p)


def test_evaluation_rejects_divergent_rank_weights_even_with_valid_checksums(tmp_path):
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.training.checkpoint import CheckpointStore, CheckpointError, capture_training_state
    evaluator, _, _, _ = build()
    trainer = Trainer(evaluator.model, evaluator.compiler, tmp_path / "original", TrainConfig(distractors=2),
                      base_identity="base")
    trainer.run(max_micro_steps=1, install_signals=False)
    store = CheckpointStore(tmp_path / "divergent")
    pending = store.begin(trainer.progress)
    for rank in (0, 1):
        state = capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
                                      data_cursor=trainer.cursor, life_state=trainer.life, rank=rank, world_size=2)
        if rank == 1:
            next(iter(state["weights"].values())).add_(1)
        store.save_rank(pending, state)
    store.commit(pending, world_size=2, identity=trainer.identity, progress=trainer.progress)
    with pytest.raises(CheckpointError, match="rank 1 weights differ"):
        load_ab_weights(tiny(), store, base_identity="base")


def test_memory_architecture_mismatch_is_rejected_even_when_tensor_shapes_match(tmp_path):
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.evaluation.checkpoint import load_ab_weights
    from dsrsi.training.checkpoint import CheckpointError
    evaluator, _, _, _ = build()
    trainer = Trainer(evaluator.model, evaluator.compiler, tmp_path, TrainConfig(distractors=2), base_identity="base")
    trainer.run(max_micro_steps=1, install_signals=False)
    changed = tiny()
    changed.memory_config = replace(changed.memory_config, heads=1)
    for module in changed.modules():
        if hasattr(module, "heads"):
            module.heads = 1
    with pytest.raises(CheckpointError, match="architecture differs"):
        load_ab_weights(changed, trainer.store, base_identity="base")
