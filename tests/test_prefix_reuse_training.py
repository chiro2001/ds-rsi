from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp


def test_optimized_training_and_pending_gradient_recovery_match_original(tmp_path):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.checkpoint import capture_training_state, CheckpointError
    from dsrsi.training.topology import require_equal
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    config = TrainConfig(data_revision="ab-v5-indexed-digits", student_token_budget=100000,
                         gradient_accumulation=6, distractors=2, seed=17, curriculum="facts",
                         memory_candidates=1, counterfactual_weight=1, keep_fraction=0.4, keep_weight=0.5)
    def make(name, optimized=False, resume=False):
        torch.manual_seed(23)
        return Trainer(model(), compiler, tmp_path / name, replace(config, reuse_frozen_prefix=optimized),
                       base_identity="prefix-tiny", resume=resume)
    baseline = make("baseline")
    baseline.run(max_micro_steps=8, install_signals=False)
    optimized = make("optimized", True)
    optimized.run(max_micro_steps=8, install_signals=False)
    partial = make("partial", True)
    partial.run(max_micro_steps=3, install_signals=False)
    assert partial.progress.accumulation_index == 3
    restored = make("partial", True, True)
    restored.run(max_micro_steps=8, install_signals=False)
    def snapshot(t):
        return capture_training_state(t.model, t.optimizer, progress=t.progress, data_cursor=t.cursor, life_state=t.life)
    require_equal(snapshot(baseline), snapshot(optimized), "baseline-vs-optimized")
    require_equal(snapshot(optimized), snapshot(restored), "optimized-recovery")
    with pytest.raises(CheckpointError, match="identity"):
        make("partial", False, True)
    assert "reuse_frozen_prefix" not in baseline.identity["train_config"]
    assert optimized.identity["train_config"]["reuse_frozen_prefix"] is True


def test_execution_migration_preserves_every_payload_and_source_pointer(tmp_path):
    from test_topology import make_source
    from dsrsi.training.execution import enable_prefix_reuse
    from dsrsi.training.checkpoint import CheckpointStore, CheckpointError
    from dsrsi.training.topology import require_equal
    source = make_source(tmp_path / "source")
    old_dir, old_manifest = source.load_committed()
    pointer = (source.root / "latest.json").read_bytes()
    report = enable_prefix_reuse(tmp_path / "source", tmp_path / "target")
    assert report["exact_payload_match"] is True
    target = CheckpointStore(tmp_path / "target/checkpoints")
    new_dir, manifest = target.load_committed()
    assert manifest["identity"]["train_config"]["reuse_frozen_prefix"] is True
    assert manifest["progress"] == old_manifest["progress"]
    for rank in range(old_manifest["world_size"]):
        require_equal(source.load_rank(old_dir, old_manifest, rank=rank), target.load_rank(new_dir, manifest, rank=rank), f"rank{rank}")
    assert (source.root / "latest.json").read_bytes() == pointer
    with pytest.raises(CheckpointError, match="already"):
        enable_prefix_reuse(tmp_path / "target", tmp_path / "again")


@pytest.mark.parametrize("status,accumulation", [("checkpoint", 0), ("stopped", 2)])
def test_execution_migration_rejects_active_or_pending_gradient_states(tmp_path, status, accumulation):
    from test_topology import make_source
    from dsrsi.training.execution import enable_prefix_reuse
    from dsrsi.training.checkpoint import CheckpointError
    make_source(tmp_path / "source", status=status, accumulation_index=accumulation)
    with pytest.raises(CheckpointError, match="boundary"):
        enable_prefix_reuse(tmp_path / "source", tmp_path / "target")


def migration_worker(rank, root, optimized, resume, micro_steps):
    from test_lookup_gates import model
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.distributed import CollectiveContext
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.checkpoint import capture_training_state
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
        init_method="file://" + str(Path(root) / f"rendezvous-{micro_steps}"), timeout_seconds=60)
    try:
        torch.manual_seed(23)
        config = TrainConfig(data_revision="ab-v5-indexed-digits", student_token_budget=100000,
                             gradient_accumulation=6, distractors=2, seed=17, curriculum="facts",
                             memory_candidates=1, counterfactual_weight=1, keep_fraction=0.4,
                             keep_weight=0.5, reuse_frozen_prefix=optimized)
        trainer = Trainer(model(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000),
                          Path(root) / "run", config, base_identity="prefix-tiny",
                          collectives=context, resume=resume)
        assert trainer.run(max_micro_steps=micro_steps, install_signals=False) == "stopped"
        state = capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
            data_cursor=trainer.cursor, life_state=trainer.life, rank=rank, world_size=2)
        torch.save(state, Path(root) / f"state-{rank}.pt")
        if optimized:
            assert trainer.identity["execution_migrations"]
    finally:
        context.close()


def test_migrated_two_rank_run_resumes_with_exact_original_training_state(tmp_path):
    from dsrsi.training.execution import enable_prefix_reuse
    from dsrsi.training.topology import require_equal
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    mp.spawn(migration_worker, args=(str(source), False, False, 6), nprocs=2, join=True)
    enable_prefix_reuse(source / "run", target / "run")
    mp.spawn(migration_worker, args=(str(source), False, True, 14), nprocs=2, join=True)
    mp.spawn(migration_worker, args=(str(target), True, True, 14), nprocs=2, join=True)
    for rank in range(2):
        baseline = torch.load(source / f"state-{rank}.pt", weights_only=True)
        optimized = torch.load(target / f"state-{rank}.pt", weights_only=True)
        require_equal(baseline, optimized, f"rank{rank}")
