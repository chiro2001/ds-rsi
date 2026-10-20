from dataclasses import replace
import json
import math

import pytest
import torch

from dsrsi.training.reader_support_objective import (ReaderSupportObjective, capture_reader_support,
                                                     support_attention_loss)
from dsrsi.training.episode_objective import EpisodeObjective
from dsrsi.training.memory_trainable import apply_memory_trainable_policy, FREEZE_READER_GATES_V1
from test_candidate_curriculum import build
from test_memory_trainable import frozen_config


@pytest.mark.parametrize("with_features", [False, True])
def test_support_mass_matches_actual_bfloat16_reader_attention_and_slot_permutations(monkeypatch, with_features):
    from dsrsi.memory.modules import MemoryConfig, MemoryReader
    import dsrsi.memory.modules as modules
    torch.manual_seed(19)
    reader = MemoryReader(MemoryConfig(hidden_size=8, memory_dim=8, heads=2))
    h, slots = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    valid = torch.tensor([[[False] * 4, [True, True, True, False], [True, True, True, False]]])
    features = torch.randn_like(slots) if with_features else None
    slots[0, 3] = torch.nan
    if features is not None:
        features[0, 3] = torch.nan
    positive = torch.tensor([[False, False, True, False]])
    prediction = torch.tensor([[False, True, True]])
    actual = []
    original = modules._attention

    def observe(q, k, v, mask, heads):
        assert q.dtype == k.dtype == torch.bfloat16
        qh = q.reshape(1, 3, 2, 4).transpose(1, 2)
        kh = k.reshape(1, 4, 2, 4).transpose(1, 2)
        with torch.autocast("cpu", enabled=False):
            score = qh.float() @ kh.float().transpose(-1, -2) / 2
            # Only the two supervised positions have valid memory.
            p = score[:, :, 1:].masked_fill(~mask[:, None, 1:], -torch.inf).softmax(-1)
            actual.append(p[..., 2].mean(1))
        return original(q, k, v, mask, heads)

    monkeypatch.setattr(modules, "_attention", observe)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plain = reader(h, slots, valid, slot_features=features)
        loss, mass = support_attention_loss(reader, h, slots, valid, features, positive, prediction)
        expected = actual[0]
        assert torch.allclose(loss, -expected.log().mean(), atol=1e-6, rtol=1e-6)
        assert torch.allclose(mass, expected.mean(), atol=1e-6, rtol=1e-6)
        order = [2, 0, 3, 1]
        swapped, _ = support_attention_loss(reader, h, slots[:, order], valid[:, :, order],
                                             None if features is None else features[:, order],
                                             positive[:, order], prediction)
        assert torch.allclose(swapped, loss, atol=1e-6, rtol=1e-6)
        assert torch.equal(plain, reader(h, slots, valid, slot_features=features))


def test_support_loss_gradient_step_increases_true_record_probability():
    from dsrsi.memory.modules import MemoryConfig, MemoryReader
    torch.manual_seed(13)
    reader = MemoryReader(MemoryConfig(hidden_size=4, memory_dim=4, heads=1))
    hidden, slots = torch.randn(1, 1, 4), torch.randn(1, 2, 4)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    support, prediction = torch.tensor([[False, True]]), torch.tensor([[True]])
    optimizer = torch.optim.SGD(list(reader.q.parameters()) + list(reader.k.parameters()), lr=.01)
    loss, before = support_attention_loss(reader, hidden, slots, valid, None, support, prediction)
    loss.backward()
    optimizer.step()
    _, after = support_attention_loss(reader, hidden, slots, valid, None, support, prediction)
    assert after > before
    with pytest.raises(ValueError, match="no valid supporting"):
        support_attention_loss(reader, hidden, slots, torch.tensor([[[True, False]]]), None,
                               support, prediction)


def objective_pair(count, weight, *, keep=False, reuse=True):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.data.revisions import make_generator
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    generator = make_generator(revision="ab-v6-natural-keep", seed=17, distractors=2)
    family = "language_preservation" if keep else "single_fact"
    row = generator.episode(0, family=family, language="en")
    opposite = generator.episode(0, variant=1, family=family, language="en")
    models, results = [], []
    for cls in (EpisodeObjective, ReaderSupportObjective):
        torch.manual_seed(23)
        m = model()
        apply_memory_trainable_policy(m, FREEZE_READER_GATES_V1)
        options = dict(stage="A", memory_candidates=count, counterfactual_weight=1,
                       keep_weight=1, reuse_frozen_prefix=reuse)
        if cls is ReaderSupportObjective:
            options["reader_support_weight"] = weight
        objective = cls(m, compiler, **options)
        results.append(objective(row, counterfactual_row=None if keep else opposite))
        models.append(m)
    return models, results


def test_auxiliary_only_adds_supervision_to_the_two_actual_answer_readers():
    models, (base, result) = objective_pair(2, 1)
    assert base.counterfactual_loss > 0
    assert result.reader_support_readers == len(models[1].readers) == 2
    assert result.reader_support_loss > 0 and 0 < result.reader_support_mass <= 1
    assert torch.allclose(result.loss, base.loss + result.reader_support_loss)
    for field in ("answer_loss", "counterfactual_loss", "student_tokens", "teacher_tokens", "supervised_tokens"):
        assert getattr(base, field) == getattr(result, field), field
    assert not hasattr(models[1], "_reader_support_capture_active")
    assert not models[1]._forward_hooks and not models[1]._forward_pre_hooks
    assert all(not reader._forward_hooks for reader in models[1].readers.values())


@pytest.mark.parametrize("count,weight,keep", [(2, 0, False), (1, 1, False), (2, 1, True)])
def test_zero_weight_single_record_and_keep_retain_original_loss_and_all_gradients(count, weight, keep):
    models, (a, b) = objective_pair(count, weight, keep=keep)
    assert torch.equal(a.loss, b.loss) and b.reader_support_readers == 0
    a.loss.backward()
    b.loss.backward()
    for (name, p), (other_name, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == other_name and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name


def test_scope_restores_existing_hooks_after_exception(monkeypatch, tmp_path):
    trainer = build(tmp_path, cfg=frozen_config())
    original = trainer.model.forward

    def fail(*args, **kwargs):
        if kwargs.get("labels") is not None:
            raise RuntimeError("fixture failure")
        return original(*args, **kwargs)

    keeper = next(iter(trainer.model.readers.values())).register_forward_hook(lambda *args: None)
    counts = [len(r._forward_hooks) for r in trainer.model.readers.values()]
    monkeypatch.setattr(trainer.model, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with capture_reader_support(trainer.model, {"unused"}, end_token_id=1):
            trainer._forward_episode(6)
    assert [len(r._forward_hooks) for r in trainer.model.readers.values()] == counts
    assert not trainer.model._forward_hooks and not trainer.model._forward_pre_hooks
    assert not hasattr(trainer.model, "_reader_support_capture_active")
    keeper.remove()


def test_actual_supervision_positions_precede_answer_tokens_and_exclude_eos(monkeypatch, tmp_path):
    import dsrsi.training.reader_support_objective as module
    trainer = build(tmp_path, cfg=replace(frozen_config(), reader_support_weight=1))
    predictions, labels = [], []
    original = module.support_attention_loss

    def observe(reader, hidden, slots, valid, features, positive, prediction):
        predictions.append(prediction.detach().clone())
        return original(reader, hidden, slots, valid, features, positive, prediction)

    def capture_labels(_model, _args, kwargs):
        if kwargs.get("labels") is not None:
            labels.append(kwargs["labels"].detach().clone())

    monkeypatch.setattr(module, "support_attention_loss", observe)
    handle = trainer.model.register_forward_pre_hook(capture_labels, with_kwargs=True)
    try:
        _, _, result = trainer._forward_episode(6)
    finally:
        handle.remove()
    assert result.reader_support_readers == 2 and len(predictions) == 2 and len(labels) == 1
    answer = (labels[0][0] != -100).nonzero().flatten().tolist()
    assert labels[0][0, answer[-1]] == trainer.compiler.tokenizer.eos_token_id
    for mask in predictions:
        assert mask[0].nonzero().flatten().tolist() == [position - 1 for position in answer[:-1]]


def test_only_objective_change_creates_new_run_and_cli_retains_weight(tmp_path):
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path
    source = build(tmp_path / "source", cfg=frozen_config())
    source.run(max_micro_steps=6, install_signals=False)
    pointer = (source.store.root / "latest.json").read_bytes()
    cfg = replace(frozen_config(), reader_support_weight=1)
    target = build(tmp_path / "target", cfg=cfg, initialize_from=source.run_dir)
    change = target.identity["initialization"]["reader_support_objective_change"]
    assert change["source_weight"] == 0 and change["target_weight"] == 1
    assert target.progress.micro_step == 0 and not target.optimizer.state
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, target.model.state_dict()[name]), name
    assert (source.store.root / "latest.json").read_bytes() == pointer
    path = Path(__file__).resolve().parents[1] / "scripts/verify_cuda_resume.py"
    spec = importlib.util.spec_from_file_location("verify_reader_support_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.train_arguments("base", "run", asdict(cfg), 240)
    assert float(args[args.index("--reader-support-weight") + 1]) == 1


def test_support_course_complete_resume_prefix_reference_and_identity(tmp_path):
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    from dsrsi.training.topology import require_equal
    cfg = replace(frozen_config(), reader_support_weight=1)
    a = build(tmp_path / "continuous", cfg=cfg)
    a.run(max_micro_steps=24, install_signals=False)
    b = build(tmp_path / "interrupted", cfg=cfg)
    b.run(max_micro_steps=5, install_signals=False)
    c = build(tmp_path / "interrupted", cfg=cfg, resume=True)
    c.run(max_micro_steps=24, install_signals=False)
    reference = build(tmp_path / "reference", cfg=replace(cfg, reuse_frozen_prefix=False))
    reference.run(max_micro_steps=24, install_signals=False)
    def state(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress,
                                      data_cursor=t.cursor, life_state=t.life)
    require_equal(state(a), state(c), "support-complete-resume")
    require_equal(state(a), state(reference), "support-prefix-reference")
    rows = [r for line in (a.run_dir / "events.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("event") == "micro"]
    for row in rows:
        active = row["family"] == "single_fact" and row["memory_candidates_used"] == 2
        assert row["reader_support_readers"] == (2 if active else 0)
    assert any(r["reader_support_loss"] > 0 for r in rows)
    with pytest.raises(CheckpointError, match="identity"):
        build(tmp_path / "interrupted", cfg=frozen_config(), resume=True)


@pytest.mark.parametrize("weight", [-1, math.inf, math.nan])
def test_invalid_support_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(frozen_config(), reader_support_weight=weight)
