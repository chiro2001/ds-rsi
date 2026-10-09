"""Recoverable C-tools event training, sharing all-rank control with A/B."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path

import torch

from dsrsi.data.tool_generate import ToolEpisodeGenerator, TOOL_DATA_REVISION
from dsrsi.data.generate import EpisodeGenerator
from dsrsi.harness.memory_tools import HarnessStore
from dsrsi.model.lora import LoRAConfig, install_lora
from .checkpoint import CheckpointStore, Progress, file_sha256, restore_training_state
from .control import ControlChannel
from .episode_objective import EpisodeObjective
from .runner import Trainer
from .tool_objective import ToolObjective


@dataclass(frozen=True)
class ToolTrainConfig:
    stage: str = "C_tools"
    student_token_budget: int = 1_000_000
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20261010
    save_every_micro: int = 100
    lora_rank: int = 8
    lora_alpha: float = 16.0
    keep_weight: float = 0.2
    max_query_tokens: int = 2048
    allow_untrained_memory: bool = False
    memory_replay_fraction: float = 0.25
    replay_distractors: int = 15

    def __post_init__(self):
        if self.stage != "C_tools" or min(self.student_token_budget, self.gradient_accumulation,
                                          self.save_every_micro, self.max_query_tokens) < 1:
            raise ValueError("invalid C-tools stage/budget/accumulation")
        if self.learning_rate <= 0 or self.keep_weight <= 0 or self.gradient_clip <= 0:
            raise ValueError("invalid C-tools optimizer or language-preservation weight")
        if not 0 <= self.memory_replay_fraction <= 1 or self.replay_distractors < 1:
            raise ValueError("invalid C-tools memory replay mixture")
        LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = ("data/tool_generate.py", "data/generate.py", "data/compiler.py", "data/tools.py", "harness/memory_tools.py",
             "model/lora.py", "model/qwen35_memory.py", "memory/modules.py", "training/losses.py",
             "training/tool_objective.py", "training/tool_runner.py", "training/runner.py",
             "training/distributed.py", "training/checkpoint.py", "evaluation/checkpoint.py")
    return {"files": {name: file_sha256(root / name) for name in files},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}}


class ToolTrainer(Trainer):
    def _initialize(self, base_identity, resume, initialize_from):
        model, config = self.model, self.config
        if resume and initialize_from is not None:
            raise ValueError("resume and stage initialization are distinct operations")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = CheckpointStore(self.run_dir / "checkpoints")
        self.control = ControlChannel(self.run_dir)
        initialization = None
        if resume:
            _, manifest = self.store.load_committed()
            initialization = manifest["identity"].get("initialization")
        elif initialize_from is not None:
            from dsrsi.evaluation.checkpoint import load_ab_weights
            source_store = CheckpointStore(Path(initialize_from) / "checkpoints")
            committed = source_store.load_committed()
            if committed[1]["progress"]["stage"] != "B" or self.run_dir.resolve() == Path(initialize_from).resolve():
                raise ValueError("C-tools initialization requires a B checkpoint and a new run directory")
            source, manifest = load_ab_weights(model, source_store, base_identity=base_identity, committed=committed)
            initialization = {"transition": "B-to-C-tools-reset-training-state", "checkpoint": source.name,
                              "manifest_sha256": file_sha256(source / "manifest.json"),
                              "identity": manifest["identity"], "progress": manifest["progress"]}
        elif not config.allow_untrained_memory:
            raise ValueError("initialize from B, or explicitly allow untrained memory for a tool-only experiment")
        model.address.requires_grad_(True)
        model.consolidator.requires_grad_(False)
        layout = install_lora(model, LoRAConfig(rank=config.lora_rank, alpha=config.lora_alpha))
        if self.compiler.max_query_tokens != config.max_query_tokens:
            raise ValueError("compiler budget must match the checkpointed C-tools configuration")
        self.objective = ToolObjective(model, self.compiler, keep_weight=config.keep_weight)
        self.replay_objective = EpisodeObjective(model, self.compiler, stage="B")
        self.replay_generator = EpisodeGenerator(seed=config.seed, distractors=config.replay_distractors)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=config.learning_rate, weight_decay=config.weight_decay,
                                           foreach=False, fused=False)
        self.optimizer.zero_grad(set_to_none=True)
        journal = self.run_dir / "harness" / f"rank_{self.rank:05d}.sqlite"
        if resume and not journal.is_file():
            raise FileNotFoundError("C-tools resume requires the durable per-rank harness journal")
        self.generator = ToolEpisodeGenerator(seed=config.seed, journal_path=journal)
        self.identity = {"base": base_identity, "train_config": asdict(config),
                         "execution_semantics": "c-tools-v1-single-assistant-event-coordinated-gradients",
                         "data_revision": TOOL_DATA_REVISION, "lora": layout,
                         "memory_architecture": model.memory_architecture_identity(),
                         "implementation": implementation_identity()}
        if initialization is not None:
            self.identity["initialization"] = initialization
        self.progress = Progress(config.stage)
        self.cursor = {"next_ordinal": self.rank, "schema_version": 1}
        self.life = {"mode": "tool-sft-and-m-only-replay", "streams": {}, "harness_refs": {}}
        if resume:
            payload, _ = self.store.load_latest(identity=self.identity, rank=self.rank, world_size=self.world_size)
            self.progress, self.cursor, self.life = restore_training_state(payload, model, self.optimizer, device=self.device)
            for reference in self.life["harness_refs"].values():
                restored = HarnessStore.from_state_dict(reference, path=journal)
                restored.close()
        elif (self.store.root / "latest.json").exists():
            raise FileExistsError("run already has a checkpoint; use explicit resume")
        self.last_checkpoint = None

    def _forward_episode(self, ordinal):
        world, variant = divmod(ordinal, 2)
        draw = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{world}:c-tools-mix".encode()).digest()[:8], "big") / 2**64
        if draw < self.config.memory_replay_fraction:
            row = self.replay_generator.episode(world, variant=variant)
            return row, True, self.replay_objective(row, oracle_include=True)
        row = self.generator.episode(world, variant=variant)
        result = self.objective(row)
        self.life["harness_refs"] = {"last_teacher_episode": row["final_harness"]}
        return row, None, result
