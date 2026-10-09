import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("transformers")
from test_context_trainer import deeper, state
from test_tool_training import CompactTokenizer
from test_distributed_trainer import equal
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.training.context_runner import ContextTrainConfig, ContextTrainer
from dsrsi.training.distributed import CollectiveContext


def worker(rank, root, rendezvous):
    from pathlib import Path
    torch.set_num_threads(1)
    context = CollectiveContext.initialize(device="cpu", backend="gloo", rank=rank, world_size=2,
                                           init_method="file://" + rendezvous, timeout_seconds=60)
    try:
        def make(directory, resume=False):
            torch.manual_seed(83)
            config = ContextTrainConfig(student_token_budget=100000, gradient_accumulation=3, seed=17, lora_rank=2,
                unroll_tokens=7, window=5, rebase_interval=8, writer_stride=64, retrieval_interval=3,
                interference_tokens=0, include_tools=False, allow_untrained_memory=True)
            return ContextTrainer(deeper(), EpisodeCompiler(CompactTokenizer()), Path(root) / directory, config,
                                  base_identity="two-rank-deep-context", resume=resume, collectives=context)
        full = make("full")
        assert full.run(max_micro_steps=35, install_signals=False) == "stopped"
        reference = state(full)
        part = make("resumed")
        original = part._forward_episode
        calls = 0
        def pause(ordinal):
            nonlocal calls
            result = original(ordinal)
            calls += 1
            if rank == 1 and calls == 17:
                part.control._signal_action = "pause"
            return result
        part._forward_episode = pause
        assert part.run(max_micro_steps=35, install_signals=False) == "paused"
        resumed = make("resumed", resume=True)
        assert resumed.run(max_micro_steps=35, install_signals=False) == "stopped"
        equal(reference, state(resumed))
        assert resumed.progress.optimizer_step > 0
        torch.save({"weights": {name: p.detach() for name, p in resumed.model.named_parameters() if p.requires_grad},
                    "progress": resumed.progress.__dict__}, Path(root) / f"rank_{rank}.pt")
    finally:
        context.close()


def test_distributed_observation_and_supervised_chunks_restore_each_life_exactly(tmp_path):
    mp.spawn(worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2, join=True)
    a, b = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]
    equal(a, b)
