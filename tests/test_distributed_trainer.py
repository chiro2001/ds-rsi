from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_episode_objective import SmallVocabularyTokenizer
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.training.distributed import CollectiveContext
from dsrsi.training.runner import TrainConfig, Trainer
from dsrsi.evaluation.checkpoint import load_ab_weights


def worker(rank, root, rendezvous, mode):
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo",
                                            init_method="file://" + rendezvous, rank=rank, world_size=2,
                                            timeout_seconds=30)
    try:
        def make(resume=False):
            torch.manual_seed(137)
            model = tiny()
            compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
            config = TrainConfig(student_token_budget=100000, gradient_accumulation=4,
                                 distractors=2, seed=17, save_every_micro=100)
            return Trainer(model, compiler, Path(root) / "run", config,
                           base_identity="distributed-native-tiny", resume=resume, collectives=context)
        trainer = make()
        if mode == "query_weighted":
            from dsrsi.training.episode_objective import ObjectiveOutput
            parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
            def synthetic(row, **kwargs):
                loss = parameter.sum() * (rank + 1)
                return ObjectiveOutput(loss, 1, 0, 0, rank + 1, 0.0, 0.0, 0.0, rank + 1, rank + 1)
            trainer.objective = synthetic
            assert trainer.run(max_micro_steps=3, install_signals=False) == "stopped"
            context.synchronize_gradients(p for p in trainer.model.parameters() if p.requires_grad)
            # (1 query * grad1 + 2 queries * grad2) / 3, accumulated
            # for three of four microbatches = 5/3 * 3/4 = 1.25.
            assert torch.equal(parameter.grad, torch.full_like(parameter, 1.25))
            assert all(p.grad is None for p in trainer.model.parameters() if p.requires_grad and p is not parameter)
            return
        if mode in {"forward_error", "backward_error", "optimizer_error", "nonfinite_gradient", "restore_error"}:
            assert trainer.run(max_micro_steps=1, install_signals=False) == "stopped"
            previous = (trainer.store.root / "latest.json").read_bytes()
            if mode == "restore_error":
                from dsrsi.training.checkpoint import CheckpointStore
                original_load = CheckpointStore.load_latest
                def load(*args, **kwargs):
                    if rank == 1:
                        raise OSError("injected rank1 restore failure")
                    return original_load(*args, **kwargs)
                CheckpointStore.load_latest = load
            try:
                trainer = make(resume=True)
                if rank == 1:
                    if mode == "forward_error":
                        def fail(*args, **kwargs):
                            raise FloatingPointError("injected objective failure")
                        trainer.objective = fail
                    elif mode in {"backward_error", "nonfinite_gradient"}:
                        parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
                        def hook(grad):
                            if mode == "backward_error":
                                raise RuntimeError("injected backward failure")
                            return grad * float("nan")
                        parameter.register_hook(hook)
                    elif mode == "optimizer_error":
                        def fail():
                            raise RuntimeError("injected optimizer failure")
                        trainer.optimizer.step = fail
                trainer.run(max_micro_steps=5, install_signals=False)
            except RuntimeError as exc:
                phase = {"forward_error": "episode forward", "backward_error": "episode backward",
                         "optimizer_error": "optimizer step", "nonfinite_gradient": "clip gradients",
                         "restore_error": "initialize trainer"}[mode]
                assert phase in str(exc), str(exc)
                assert (trainer.store.root / "latest.json").read_bytes() == previous
            else:
                raise AssertionError("failed local operation did not propagate")
            return
        if mode == "resumed":
            assert trainer.run(max_micro_steps=3, install_signals=False) == "stopped"
            trainer = make(resume=True)
        if mode == "worker_pause":
            original = trainer.objective
            calls = 0
            def inject(row, **kwargs):
                nonlocal calls
                result = original(row, **kwargs)
                calls += 1
                if rank == 1 and calls == 2:
                    trainer.control._signal_action = "pause"
                return result
            trainer.objective = inject
            status = trainer.run(max_micro_steps=5, install_signals=False)
            assert status == "paused" and trainer.progress.micro_step == 2
        elif mode == "save_error":
            if rank == 1:
                def failed_save(*args):
                    raise OSError("injected rank1 IO failure")
                trainer.store.save_rank = failed_save
            try:
                trainer.run(max_micro_steps=1, install_signals=False)
            except RuntimeError as exc:
                assert "rank 1" in str(exc)
            else:
                raise AssertionError("rank failure did not propagate")
            assert not (trainer.store.root / "latest.json").exists()
        else:
            status = trainer.run(max_micro_steps=5, install_signals=False)
            assert status == "stopped"
        # All-rank equality after sync, and per-rank local gradients when the
        # pending group has not synchronized yet, are verified by the parent.
        torch.save({"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "gradients": {name: p.grad for name, p in trainer.model.named_parameters() if p.requires_grad},
                    "progress": asdict(trainer.progress), "cursor": trainer.cursor},
                   Path(root) / f"rank_{rank}.pt")
    finally:
        context.close()


def launch(path, mode):
    path.mkdir()
    mp.spawn(worker, args=(str(path), str(path / "rendezvous"), mode), nprocs=2, join=True)
    return [torch.load(path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_two_rank_resume_matches_uninterrupted_with_local_pending_gradients(tmp_path):
    full = launch(tmp_path / "full", "full")
    resumed = launch(tmp_path / "resumed", "resumed")
    for a, b in zip(full, resumed):
        equal(a, b)
    assert full[0]["progress"] == full[1]["progress"]
    assert full[0]["cursor"]["next_ordinal"] == 10
    assert full[1]["cursor"]["next_ordinal"] == 11
    equal(full[0]["model"], full[1]["model"])
    equal(full[0]["optimizer"], full[1]["optimizer"])
    manifest = next((tmp_path / "resumed" / "run" / "checkpoints").glob("micro_000000000005_*/manifest.json"))
    data = json.loads(manifest.read_text())
    assert data["world_size"] == 2 and len(data["files"]) == 2
    # Read-only evaluation can load both ranks without constructing an optimizer
    # or restoring training RNG. Pending gradients are allowed to differ.
    from dsrsi.training.checkpoint import CheckpointStore
    torch.manual_seed(137)
    model = tiny()
    rng = torch.get_rng_state().clone()
    _, loaded = load_ab_weights(model, CheckpointStore(tmp_path / "resumed" / "run" / "checkpoints"),
                                base_identity="distributed-native-tiny")
    assert loaded["world_size"] == 2
    equal(model.state_dict(), resumed[0]["model"])
    assert torch.equal(rng, torch.get_rng_state())


def test_worker_signal_pauses_all_ranks_and_commits_every_rank(tmp_path):
    results = launch(tmp_path / "pause", "worker_pause")
    assert all(r["progress"]["micro_step"] == 2 and r["progress"]["accumulation_index"] == 2 for r in results)
    root = tmp_path / "pause" / "run" / "checkpoints"
    latest = json.loads((root / "latest.json").read_text())
    manifest = json.loads((root / latest["directory"] / "manifest.json").read_text())
    assert manifest["status"] == "paused" and len(manifest["files"]) == 2


def test_rank_save_failure_propagates_without_a_false_latest_checkpoint(tmp_path):
    launch(tmp_path / "failure", "save_error")


@pytest.mark.parametrize("mode", ["forward_error", "backward_error", "optimizer_error", "nonfinite_gradient", "restore_error"])
def test_local_failures_exit_every_rank_and_preserve_last_committed_state(tmp_path, mode):
    path = tmp_path / mode
    path.mkdir()
    mp.spawn(worker, args=(str(path), str(path / "rendezvous"), mode), nprocs=2, join=True)


def test_global_query_weighting_and_unused_gradient_synchronization(tmp_path):
    path = tmp_path / "query_weighted"
    path.mkdir()
    mp.spawn(worker, args=(str(path), str(path / "rendezvous"), "query_weighted"), nprocs=2, join=True)
