from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from test_tool_training import CompactTokenizer
from test_distributed_trainer import equal
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.training.checkpoint import capture_training_state
from dsrsi.training.distributed import CollectiveContext
from dsrsi.training.tool_runner import ToolTrainer, ToolTrainConfig


def worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
                                           init_method="file://" + rendezvous, timeout_seconds=60)
    try:
        def make(directory, resume=False):
            torch.manual_seed(137)
            config = ToolTrainConfig(student_token_budget=100000, seed=17, lora_rank=2, replay_distractors=2,
                                     max_query_tokens=4096, allow_untrained_memory=True, memory_replay_fraction=0.4)
            return ToolTrainer(tiny(), EpisodeCompiler(CompactTokenizer(), max_query_tokens=config.max_query_tokens),
                               Path(root) / directory, config, base_identity="two-rank-tools", collectives=context, resume=resume)
        def state(trainer):
            payload = capture_training_state(trainer.model, trainer.optimizer, progress=trainer.progress,
                data_cursor=trainer.cursor, life_state=trainer.life, device="cpu", rank=rank, world_size=2)
            for reference in payload["life_state"]["harness_refs"].values():
                reference["path"] = "<per-rank-journal>"
            return payload
        full = make("full")
        assert full.run(max_micro_steps=5, install_signals=False) == "stopped"
        reference = state(full)
        part = make("resumed")
        original = part._forward_episode
        calls = 0
        def pause(ordinal):
            nonlocal calls
            result = original(ordinal)
            calls += 1
            if rank == 1 and calls == 3:
                part.control._signal_action = "pause"
            return result
        part._forward_episode = pause
        assert part.run(max_micro_steps=5, install_signals=False) == "paused"
        resumed = make("resumed", resume=True)
        assert resumed.run(max_micro_steps=5, install_signals=False) == "stopped"
        equal(reference, state(resumed))
        torch.save({"model": resumed.model.state_dict(), "progress": asdict(resumed.progress)}, Path(root) / f"rank_{rank}.pt")
    finally:
        context.close()


def test_two_rank_tool_and_memory_replay_recovery_has_identical_complete_state(tmp_path):
    mp.spawn(worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2, join=True)
    a, b = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]
    equal(a, b)
