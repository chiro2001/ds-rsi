from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from dsrsi.training.checkpoint import CheckpointError, capture_training_state, CheckpointStore
from dsrsi.training.execution import enable_scripted_gdn
from dsrsi.training.topology import require_equal


def test_scripted_training_and_pending_gradient_resume_match_original_and_pin_execution(tmp_path, monkeypatch):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.model import gdn_execution
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=2000)
    config = TrainConfig(data_revision="ab-v6-natural-keep", student_token_budget=100000,
                         gradient_accumulation=6, distractors=2, seed=17, curriculum="facts",
                         memory_candidates=1, counterfactual_weight=1, keep_fraction=0.4, keep_weight=1,
                         learning_rate=0.00025, reuse_frozen_prefix=True)
    def make(name, scripted=False, resume=False):
        return Trainer(model(), compiler, tmp_path / name,
                       replace(config, gdn_inverse="scripted_rows" if scripted else None),
                       base_identity="gdn-tiny", resume=resume)
    def snapshot(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress,
                                      data_cursor=t.cursor, life_state=t.life)
    baseline = make("baseline")
    baseline.run(max_micro_steps=8, install_signals=False)
    optimized = make("optimized", True)
    optimized.run(max_micro_steps=8, install_signals=False)
    part = make("part", True)
    part.run(max_micro_steps=3, install_signals=False)
    assert part.progress.accumulation_index == 3
    resumed = make("part", True, True)
    resumed.run(max_micro_steps=8, install_signals=False)
    require_equal(snapshot(baseline), snapshot(optimized), "original-vs-scripted")
    require_equal(snapshot(optimized), snapshot(resumed), "scripted-recovery")
    assert "gdn_inverse" not in baseline.identity["train_config"] and "gdn_execution" not in baseline.identity
    assert resumed.identity["gdn_execution"]["summation_order_changed"] is False
    assert not hasattr(resumed.model, "_experimental_gdn_active")
    with pytest.raises(CheckpointError, match="identity"):
        make("part", False, True)
    original = gdn_execution.gdn_execution_identity
    def changed_identity():
        return {**original(), "reference_source_sha256": "different-HF-backend"}
    monkeypatch.setattr(gdn_execution, "gdn_execution_identity", changed_identity)
    with pytest.raises(CheckpointError, match="identity"):
        make("part", True, True)


def test_execution_scope_restores_backend_when_forward_fails(tmp_path):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import TrainConfig, Trainer
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    t = Trainer(model(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=2000), tmp_path,
                TrainConfig(distractors=2, gdn_inverse="scripted_rows"), base_identity="gdn-tiny")
    def failed(*args, **kwargs):
        assert t.model._experimental_gdn_active
        raise RuntimeError("episode failed")
    t.objective = failed
    with pytest.raises(RuntimeError, match="episode failed"):
        t.run(max_micro_steps=1, install_signals=False)
    assert not hasattr(t.model, "_experimental_gdn_active")
    assert all(layer.linear_attn.chunk_gated_delta_rule is torch_chunk_gated_delta_rule
               for layer in t.model.base.layers if hasattr(layer, "linear_attn"))


def test_execution_migration_preserves_each_payload_controls_and_source_pointer(tmp_path):
    from test_topology import make_source
    from dsrsi.training.control import ControlChannel
    source = make_source(tmp_path / "source")
    directory, manifest = source.load_committed()
    pointer = (source.root / "latest.json").read_bytes()
    ControlChannel(tmp_path / "source").request("pause")
    report = enable_scripted_gdn(tmp_path / "source", tmp_path / "target", dry_run=True)
    assert report["dry_run"] and not (tmp_path / "target").exists()
    report = enable_scripted_gdn(tmp_path / "source", tmp_path / "target")
    assert report["exact_payload_match"] and not report["optimizer_and_counters_reset"]
    target = CheckpointStore(tmp_path / "target/checkpoints")
    new_directory, new_manifest = target.load_committed()
    assert new_manifest["identity"]["train_config"]["gdn_inverse"] == "scripted_rows"
    assert not (tmp_path / "target/control").exists()
    assert (tmp_path / "source/control/request.json").exists()
    for rank in range(manifest["world_size"]):
        require_equal(source.load_rank(directory, manifest, rank=rank),
                      target.load_rank(new_directory, new_manifest, rank=rank), f"rank{rank}")
    assert (source.root / "latest.json").read_bytes() == pointer
    with pytest.raises(CheckpointError, match="already"):
        enable_scripted_gdn(tmp_path / "target", tmp_path / "again")


@pytest.mark.parametrize("status,accumulation", [("checkpoint", 0), ("stopped", 2)])
def test_gdn_migration_requires_stopped_empty_accumulation(tmp_path, status, accumulation):
    from test_topology import make_source
    make_source(tmp_path / "source", status=status, accumulation_index=accumulation)
    with pytest.raises(CheckpointError, match="boundary"):
        enable_scripted_gdn(tmp_path / "source", tmp_path / "target")


def gdn_worker(rank, root, scripted, resume, micro_steps):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.distributed import CollectiveContext
    from dsrsi.training.runner import TrainConfig, Trainer
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
        init_method="file://" + str(Path(root) / f"rendezvous-{micro_steps}"), timeout_seconds=60)
    try:
        config = TrainConfig(data_revision="ab-v6-natural-keep", student_token_budget=100000,
                             gradient_accumulation=6, distractors=2, seed=17, curriculum="facts",
                             memory_candidates=1, counterfactual_weight=1, keep_fraction=0.4, keep_weight=1,
                             learning_rate=0.00025, reuse_frozen_prefix=True,
                             gdn_inverse="scripted_rows" if scripted else None)
        t = Trainer(model(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=2000),
                    Path(root) / "run", config, base_identity="gdn-tiny", resume=resume, collectives=context)
        assert t.run(max_micro_steps=micro_steps, install_signals=False) == "stopped"
        state = capture_training_state(t.model, t.optimizer, progress=t.progress, data_cursor=t.cursor,
                                       life_state=t.life, rank=rank, world_size=2)
        torch.save(state, Path(root) / f"state-{rank}.pt")
    finally:
        context.close()


def test_migrated_two_rank_run_retains_prefix_reuse_and_resumes_exactly(tmp_path):
    from dsrsi.training.execution import restore_original_gdn
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    mp.spawn(gdn_worker, args=(str(source), False, False, 6), nprocs=2, join=True)
    enable_scripted_gdn(source / "run", target / "run")
    manifest = CheckpointStore(target / "run/checkpoints").load_committed()[1]
    assert manifest["identity"]["train_config"]["reuse_frozen_prefix"] is True
    mp.spawn(gdn_worker, args=(str(source), False, True, 14), nprocs=2, join=True)
    mp.spawn(gdn_worker, args=(str(target), True, True, 14), nprocs=2, join=True)
    for rank in range(2):
        require_equal(torch.load(source / f"state-{rank}.pt", weights_only=True),
                      torch.load(target / f"state-{rank}.pt", weights_only=True), f"rank{rank}")
    # Finish the current accumulation before rolling back the execution mode.
    mp.spawn(gdn_worker, args=(str(source), False, True, 18), nprocs=2, join=True)
    mp.spawn(gdn_worker, args=(str(target), True, True, 18), nprocs=2, join=True)
    restored = tmp_path / "restored"
    restore_original_gdn(target / "run", restored / "run")
    manifest = CheckpointStore(restored / "run/checkpoints").load_committed()[1]
    assert "gdn_inverse" not in manifest["identity"]["train_config"]
    assert "gdn_execution" not in manifest["identity"]
    assert manifest["identity"]["train_config"]["reuse_frozen_prefix"] is True
    mp.spawn(gdn_worker, args=(str(source), False, True, 22), nprocs=2, join=True)
    mp.spawn(gdn_worker, args=(str(restored), False, True, 22), nprocs=2, join=True)
    for rank in range(2):
        require_equal(torch.load(source / f"state-{rank}.pt", weights_only=True),
                      torch.load(restored / f"state-{rank}.pt", weights_only=True), f"restored-rank{rank}")


def test_rollback_requires_a_scripted_source_and_preserves_every_payload(tmp_path):
    from test_topology import make_source
    from dsrsi.training.execution import restore_original_gdn
    make_source(tmp_path / "source")
    with pytest.raises(CheckpointError, match="enabled scripted source"):
        restore_original_gdn(tmp_path / "source", tmp_path / "no-op")
    enable_scripted_gdn(tmp_path / "source", tmp_path / "scripted")
    source = CheckpointStore(tmp_path / "scripted/checkpoints")
    directory, manifest = source.load_committed()
    pointer = (source.root / "latest.json").read_bytes()
    report = restore_original_gdn(tmp_path / "scripted", tmp_path / "restored")
    assert report["exact_payload_match"] and not report["optimizer_and_counters_reset"]
    restored = CheckpointStore(tmp_path / "restored/checkpoints")
    dest, new_manifest = restored.load_committed()
    assert "gdn_execution" not in new_manifest["identity"]
    for rank in range(manifest["world_size"]):
        require_equal(source.load_rank(directory, manifest, rank=rank),
                      restored.load_rank(dest, new_manifest, rank=rank), f"rollback-rank{rank}")
    assert (source.root / "latest.json").read_bytes() == pointer


@pytest.mark.parametrize("method", ["doubling", "blocks", "solve"])
def test_drifting_inverse_candidates_are_not_production_choices(method):
    from dsrsi.training.runner import TrainConfig
    with pytest.raises(ValueError, match="original-order"):
        TrainConfig(gdn_inverse=method)
