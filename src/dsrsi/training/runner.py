"""Recoverable single-rank A/B executor; multi-rank orchestration is separate."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dsrsi.data.generate import EpisodeGenerator
from .checkpoint import CheckpointStore, Progress, capture_training_state, restore_training_state
from .control import ControlChannel
from .episode_objective import EpisodeObjective


@dataclass(frozen=True)
class TrainConfig:
    stage: str = "A"
    student_token_budget: int = 30_000_000
    gradient_accumulation: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20261010
    distractors: int = 7
    save_every_micro: int = 100

    def __post_init__(self):
        if self.stage not in {"A", "B"} or min(self.student_token_budget, self.gradient_accumulation) < 1:
            raise ValueError("invalid training stage/budget/accumulation")
        if self.learning_rate <= 0 or self.distractors < 1 or self.save_every_micro < 1:
            raise ValueError("invalid optimizer/distractor/checkpoint configuration")


class Trainer:
    def __init__(self, model, compiler, run_dir, config, *, base_identity, resume=False):
        self.model, self.compiler, self.config = model, compiler, config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = CheckpointStore(self.run_dir / "checkpoints")
        self.control = ControlChannel(self.run_dir)
        self.device = model.base.embed_tokens.weight.device
        self.objective = EpisodeObjective(model, compiler, stage=config.stage)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=config.learning_rate, weight_decay=config.weight_decay,
                                           foreach=False, fused=False)
        self.optimizer.zero_grad(set_to_none=True)
        self.generator = EpisodeGenerator(seed=config.seed, distractors=config.distractors)
        self.identity = {"base": base_identity, "train_config": asdict(config),
                         "execution_semantics": "ab-episode-v2-explicit-optimizer-backend"}
        self.progress = Progress(config.stage)
        self.cursor = {"next_ordinal": 0, "schema_version": 1}
        self.life = {"mode": "m_only", "streams": {}, "harness_refs": {}}
        if resume:
            payload, _ = self.store.load_latest(identity=self.identity)
            self.progress, self.cursor, self.life = restore_training_state(
                payload, model, self.optimizer, device=self.device)
        elif (self.store.root / "latest.json").exists():
            raise FileExistsError("run already has a checkpoint; use explicit resume")
        self.last_checkpoint = None

    def _save(self, status):
        started = time.monotonic()
        pending = self.store.begin(self.progress)
        state = capture_training_state(self.model, self.optimizer, progress=self.progress,
                                       data_cursor=self.cursor, life_state=self.life, device=self.device)
        self.store.save_rank(pending, state)
        final = self.store.commit(pending, world_size=1, identity=self.identity,
                                  progress=self.progress, status=status,
                                  required_student_tokens=self.config.student_token_budget if status == "completed" else None)
        self.last_checkpoint = final
        self._log({"event": "checkpoint", "status": status, "path": str(final),
                   "save_seconds": time.monotonic() - started})
        return final

    def _log(self, data):
        with (self.run_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({**data, "progress": asdict(self.progress)}, ensure_ascii=False) + "\n")

    def _optimizer_step(self, partial=False):
        count = self.progress.accumulation_index
        if count == 0:
            return
        if partial:
            factor = self.config.gradient_accumulation / count
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(factor)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.progress = replace(self.progress, optimizer_step=self.progress.optimizer_step + 1, accumulation_index=0)

    def _control_at_boundary(self):
        request = self.control.poll()
        if request is None:
            return None
        status = {"checkpoint": "checkpoint", "pause": "paused", "stop": "stopped"}[request.action]
        checkpoint = self._save(status)
        self.control.acknowledge(request, checkpoint)
        return status if request.action != "checkpoint" else None

    def run(self, *, max_micro_steps=None, install_signals=True):
        if install_signals:
            self.control.install_signal_handlers()
        self.model.train()
        self._log({"event": "ready", "pid": os.getpid(), "device": str(self.device)})
        status = self._control_at_boundary()
        if status:
            return status
        while self.progress.student_tokens < self.config.student_token_budget:
            if max_micro_steps is not None and self.progress.micro_step >= max_micro_steps:
                self._save("stopped")
                return "stopped"
            started = time.monotonic()
            ordinal = self.cursor["next_ordinal"]
            world, variant = divmod(ordinal, 2)
            # Pair members share task kind; only source outcomes/values differ.
            mix = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{world}:mix".encode()).digest()[:8], "big") % 100
            keep = mix >= (75 if self.config.stage == "A" else 70)
            row = self.generator.episode(world, variant=variant, family="language_preservation" if keep else None)
            if self.config.stage == "A":
                oracle = True
            else:
                # Do not anneal simply because token consumption advanced.
                # Held-out actual retrieval coverage must authorize that later.
                probability = 0.8
                draw = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{ordinal}:oracle".encode()).digest()[:8], "big") / 2**64
                oracle = draw < probability
            result = self.objective(row, oracle_include=oracle)
            (result.loss / self.config.gradient_accumulation).backward()
            self.cursor["next_ordinal"] += 1
            self.progress = replace(self.progress, micro_step=self.progress.micro_step + 1,
                                    accumulation_index=self.progress.accumulation_index + 1,
                                    student_tokens=self.progress.student_tokens + result.student_tokens,
                                    teacher_tokens=self.progress.teacher_tokens + result.teacher_tokens)
            if self.progress.accumulation_index == self.config.gradient_accumulation:
                self._optimizer_step()
            self._log({"event": "micro", "episode_id": row["episode_id"], "family": row["family"],
                       "loss": float(result.loss.detach()), "answer_loss": result.answer_loss,
                       "route_loss": result.route_loss, "keep_kl": result.keep_kl,
                       "student_input_tokens": result.student_tokens, "teacher_input_tokens": result.teacher_tokens,
                       "padding_positions": result.padding_positions, "supervised_tokens": result.supervised_tokens,
                       "covered_queries": result.covered_queries, "queries": result.queries,
                       "oracle_included": oracle, "micro_seconds": time.monotonic() - started})
            # No autograd graphs survive this boundary. Completed gradients are
            # saved even before an accumulation group has reached its optimizer step.
            del result
            status = self._control_at_boundary()
            if status:
                return status
            if self.progress.micro_step % self.config.save_every_micro == 0:
                self._save("checkpoint")
        self._optimizer_step(partial=True)
        self._save("completed")
        return "completed"
