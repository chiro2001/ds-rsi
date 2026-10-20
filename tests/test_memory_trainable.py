from dataclasses import replace

import pytest
import torch

from dsrsi.training.memory_trainable import (FREEZE_READER_GATES_V1, reader_gate_parameters,
                                             source_memory_trainable_policy)
from test_candidate_curriculum import build, config


def frozen_config():
    return replace(config(), memory_trainable_policy=FREEZE_READER_GATES_V1)


def test_policy_transition_keeps_values_and_only_removes_reader_gate_gradients(tmp_path):
    source = build(tmp_path / "source")
    source.run(max_micro_steps=0, install_signals=False)
    target = build(tmp_path / "target", cfg=frozen_config(), initialize_from=source.run_dir)
    names = set(reader_gate_parameters(target.model))
    assert len(names) == 6
    for name, p in target.model.named_parameters():
        assert torch.equal(p, dict(source.model.named_parameters())[name]), name
        if name in names:
            assert not p.requires_grad
    assert sum(p.requires_grad for p in target.model.parameters()) == 28
    assert target.identity["initialization"]["memory_trainable_policy_change"]["source"] is None
    assert set(target.identity["initialization"]["memory_trainable_policy_change"]["frozen_parameter_names"]) == names
    _, _, a = source._forward_episode(0)
    _, _, b = target._forward_episode(0)
    assert torch.equal(a.loss, b.loss)
    a.loss.backward()
    b.loss.backward()
    original = dict(source.model.named_parameters())
    for name, p in target.model.named_parameters():
        if name in names:
            assert p.grad is None
        elif p.requires_grad:
            assert (p.grad is None) == (original[name].grad is None)
            if p.grad is not None:
                assert torch.equal(p.grad, original[name].grad), name


def test_frozen_reader_gates_remain_complete_across_resume_and_prefix_reference(tmp_path):
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    from dsrsi.training.topology import require_equal
    a = build(tmp_path / "continuous", cfg=frozen_config())
    initial = {name: p.detach().clone() for name, p in reader_gate_parameters(a.model).items()}
    a.run(max_micro_steps=18, install_signals=False)
    b = build(tmp_path / "interrupted", cfg=frozen_config())
    b.run(max_micro_steps=5, install_signals=False)
    assert b.progress.accumulation_index == 5
    c = build(tmp_path / "interrupted", cfg=frozen_config(), resume=True)
    c.run(max_micro_steps=18, install_signals=False)
    reference = build(tmp_path / "reference", cfg=replace(frozen_config(), reuse_frozen_prefix=False))
    reference.run(max_micro_steps=18, install_signals=False)
    def state(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress,
                                      data_cursor=t.cursor, life_state=t.life)
    payload = state(a)
    require_equal(payload, state(c), "frozen-gate-complete-resume")
    require_equal(payload, state(reference), "frozen-gate-prefix-reference")
    assert len(payload["weights"]) == len(payload["optimizer"]["state"]["state"]) == 28
    for name, p in reader_gate_parameters(a.model).items():
        assert torch.equal(p, initial[name]) and torch.equal(payload["frozen_auxiliary"][name], initial[name])
        assert name not in payload["weights"] and name not in payload["gradients"]
    with pytest.raises(CheckpointError, match="identity"):
        build(tmp_path / "interrupted", cfg=config(), resume=True)


def test_evaluation_loader_loads_frozen_auxiliary_gates_and_restores_caller_flags(tmp_path):
    from dsrsi.evaluation.checkpoint import load_ab_weights
    a = build(tmp_path / "frozen", cfg=frozen_config())
    a.run(max_micro_steps=7, install_signals=False)
    evaluator = build(tmp_path / "plain")
    flags = {n: p.requires_grad for n, p in reader_gate_parameters(evaluator.model).items()}
    load_ab_weights(evaluator.model, a.store, base_identity="candidate-course-tiny")
    for name, p in a.model.state_dict().items():
        assert torch.equal(p, evaluator.model.state_dict()[name]), name
    assert {n: p.requires_grad for n, p in reader_gate_parameters(evaluator.model).items()} == flags
    with pytest.raises(RuntimeError, match="fixture failure"):
        with source_memory_trainable_policy(evaluator.model, FREEZE_READER_GATES_V1):
            assert all(not p.requires_grad for p in reader_gate_parameters(evaluator.model).values())
            raise RuntimeError("fixture failure")
    assert {n: p.requires_grad for n, p in reader_gate_parameters(evaluator.model).items()} == flags


@pytest.mark.parametrize("changes", [{"stage": "B"}, {"memory_candidate_curriculum": None},
                                    {"memory_trainable_policy": "unknown"}])
def test_freezing_requires_an_explicit_supported_new_course(changes):
    with pytest.raises(ValueError):
        replace(frozen_config(), **changes)


def test_resume_cli_retains_the_trainable_policy():
    import importlib.util
    from dataclasses import asdict
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/verify_cuda_resume.py"
    spec = importlib.util.spec_from_file_location("verify_frozen_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.train_arguments("base", "run", asdict(frozen_config()), 240)
    assert args[args.index("--memory-trainable-policy") + 1] == FREEZE_READER_GATES_V1
