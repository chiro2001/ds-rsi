from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from dsrsi.data.revisions import AB_DATA_REVISION, data_identity
from dsrsi.training.checkpoint import CheckpointError, CheckpointStore, Progress, capture_training_state
from dsrsi.training.topology import reduce_ab_replicas, require_equal


def make_source(path, mutation=None, status="stopped", stage="A", accumulation_index=0):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress = Progress(stage, micro_step=4, optimizer_step=1, student_tokens=300, teacher_tokens=40,
                        accumulation_index=accumulation_index)
    identity = {"base": "toy", "data": data_identity(AB_DATA_REVISION),
                "execution_semantics": "ab-episode-v4-query-weighted-coordinated-gradients",
                "train_config": {"data_revision": AB_DATA_REVISION, "gradient_accumulation": 4}}
    store = CheckpointStore(path / "checkpoints")
    pending = store.begin(progress)
    for rank in range(4):
        torch.manual_seed(30 + rank)
        payload = capture_training_state(model, optimizer, rank=rank, world_size=4, progress=progress,
            data_cursor={"next_ordinal": 16 + rank, "schema_version": 1},
            life_state={"mode": "m_only", "streams": {}, "harness_refs": {}})
        if mutation and rank == 3:
            mutation(payload)
        store.save_rank(pending, payload)
    store.commit(pending, world_size=4, identity=identity, progress=progress, status=status)
    return store


def test_conversion_preserves_optimizer_rng_and_consumed_tokens_without_touching_source(tmp_path):
    source = make_source(tmp_path / "source")
    latest = (source.root / "latest.json").read_bytes()
    directory, manifest = source.load_committed()
    dry = reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3, dry_run=True)
    assert dry["next_global_ordinal"] == 16 and not (tmp_path / "target").exists()
    report = reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3)
    target = CheckpointStore(tmp_path / "target/checkpoints")
    converted, got_manifest = target.load_committed()
    assert got_manifest["progress"] == manifest["progress"]
    assert got_manifest["identity"]["topology_migrations"][0]["same_trajectory_as_original_topology"] is False
    assert report["inactive_source_rng_ranks"] == [3]
    assert report["future_global_episodes_per_optimizer_step"] == 12
    for rank in range(3):
        wanted = source.load_rank(directory, manifest, rank=rank)
        wanted["world_size"] = 3
        require_equal(wanted, target.load_rank(converted, got_manifest, rank=rank), "state")
    assert (source.root / "latest.json").read_bytes() == latest
    assert (directory / "rank_00003.pt").exists()


def test_repeated_reduction_keeps_frontier_that_is_not_divisible_by_current_world_size(tmp_path):
    make_source(tmp_path / "source")
    reduce_ab_replicas(tmp_path / "source", tmp_path / "three", world_size=3)
    # Frontier16 is a valid migrated three-rank boundary, with cursors16,17,18.
    # It need not be a multiple of the new world size.
    report = reduce_ab_replicas(tmp_path / "three", tmp_path / "two", world_size=2)
    assert report["next_global_ordinal"] == 16
    store = CheckpointStore(tmp_path / "two/checkpoints")
    directory, manifest = store.load_committed()
    assert len(manifest["identity"]["topology_migrations"]) == 2
    assert [store.load_rank(directory, manifest, rank=r)["data_cursor"]["next_ordinal"] for r in range(2)] == [16, 17]


def test_explicit_accumulation_change_keeps_global_batch_and_all_optimizer_state(tmp_path):
    make_source(tmp_path / "source")
    reduce_ab_replicas(tmp_path / "source", tmp_path / "three", world_size=3)
    source = CheckpointStore(tmp_path / "three/checkpoints")
    old_dir, old_manifest = source.load_committed()
    old_pointer = (source.root / "latest.json").read_bytes()
    report = reduce_ab_replicas(tmp_path / "three", tmp_path / "two", world_size=2, gradient_accumulation=6)
    target = CheckpointStore(tmp_path / "two/checkpoints")
    directory, manifest = target.load_committed()
    assert report["global_episode_batch_preserved"] is True
    assert report["future_global_episodes_per_optimizer_step"] == report["source_global_episodes_per_optimizer_step"] == 12
    assert manifest["identity"]["train_config"]["gradient_accumulation"] == 6
    assert old_manifest["identity"]["train_config"]["gradient_accumulation"] == 4
    for rank in range(2):
        wanted = source.load_rank(old_dir, old_manifest, rank=rank)
        wanted["world_size"] = 2
        require_equal(wanted, target.load_rank(directory, manifest, rank=rank), "state")
    assert (source.root / "latest.json").read_bytes() == old_pointer


@pytest.mark.parametrize("accumulation", [0, -1, 1.5, True])
def test_bad_accumulation_change_is_rejected(tmp_path, accumulation):
    make_source(tmp_path / "source")
    with pytest.raises(CheckpointError, match="positive integer"):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=2, gradient_accumulation=accumulation)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["data_cursor"].update(next_ordinal=33), "cursors"),
    (lambda p: p["life_state"]["streams"].update(life="persisted"), "persistent"),
    (lambda p: p["gradients"].update(weight=torch.ones_like(p["weights"]["weight"])), "gradients"),
    (lambda p: p["optimizer"]["state"]["param_groups"][0].update(lr=0.2), "optimizer"),
    (lambda p: p["weights"]["weight"].add_(1), "weights"),
])
def test_conversion_refuses_state_loss_or_unsynchronized_replicas(tmp_path, mutation, error):
    make_source(tmp_path / "source", mutation=mutation)
    with pytest.raises(CheckpointError, match=error):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3)
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("size", [0, 1, 4, 5])
def test_conversion_refuses_unsupported_topology(tmp_path, size):
    make_source(tmp_path / "source")
    with pytest.raises(CheckpointError, match="reduction"):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=size)


def test_conversion_requires_stopped_optimizer_boundary_and_new_run(tmp_path):
    make_source(tmp_path / "source", status="checkpoint")
    with pytest.raises(CheckpointError, match="boundary"):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3)
    (tmp_path / "target").mkdir()
    with pytest.raises(CheckpointError, match="new target"):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3)


@pytest.mark.parametrize("stage,accumulation,error", [("C_context", 0, "stateless"), ("A", 2, "boundary")])
def test_conversion_refuses_stateful_stages_and_partial_accumulation(tmp_path, stage, accumulation, error):
    make_source(tmp_path / "source", stage=stage, accumulation_index=accumulation)
    with pytest.raises(CheckpointError, match=error):
        reduce_ab_replicas(tmp_path / "source", tmp_path / "target", world_size=3)


def trainer_worker(rank, root, size, resumed):
    from test_qwen_bridge import tiny
    from test_episode_objective import SmallVocabularyTokenizer
    from dsrsi.data.compiler import EpisodeCompiler
    from dsrsi.training.runner import TrainConfig, Trainer
    from dsrsi.training.distributed import CollectiveContext
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=size,
        init_method="file://" + str(Path(root) / "rendezvous"), timeout_seconds=60)
    try:
        def make(resume):
            torch.manual_seed(19)
            return Trainer(tiny(), EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000),
                Path(root) / "run", TrainConfig(student_token_budget=100000, distractors=2, seed=17),
                base_identity="topology-tiny", resume=resume, collectives=context)
        trainer = make(size == 3)
        if size == 4:
            assert trainer.run(max_micro_steps=4, install_signals=False) == "stopped"
        else:
            assert trainer.cursor["next_ordinal"] == 16 + rank
            if resumed:
                assert trainer.run(max_micro_steps=7, install_signals=False) == "stopped"
                assert trainer.progress.accumulation_index == 3
                trainer = make(True)
            assert trainer.run(max_micro_steps=9, install_signals=False) == "stopped"
            assert trainer.cursor["next_ordinal"] == 31 + rank
            state = capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
                data_cursor=trainer.cursor, life_state=trainer.life, rank=rank, world_size=size)
            torch.save(state, Path(root) / f"result-{rank}.pt")
    finally:
        context.close()


def test_four_to_three_rank_actual_training_and_partial_accumulation_recovery(tmp_path):
    pytest.importorskip("transformers")
    root = tmp_path / "source"
    root.mkdir()
    mp.spawn(trainer_worker, args=(str(root), 4, False), nprocs=4, join=True)
    for name, resumed in (("full", False), ("resumed", True)):
        target = tmp_path / name
        reduce_ab_replicas(root / "run", target / "run", world_size=3)
        mp.spawn(trainer_worker, args=(str(target), 3, resumed), nprocs=3, join=True)
    for rank in range(3):
        full = torch.load(tmp_path / "full" / f"result-{rank}.pt", weights_only=True)
        resumed = torch.load(tmp_path / "resumed" / f"result-{rank}.pt", weights_only=True)
        require_equal(full, resumed, "training")
