"""Recoverable A/B executor with coordinated rank boundaries and gradients."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dsrsi.data.revisions import AB_DATA_REVISION, AB_REVISIONS, LEGACY_AB_REVISION, data_identity, make_generator
from .checkpoint import (CheckpointStore, Progress, capture_training_state, restore_training_state, file_sha256,
                         frozen_auxiliary_parameters)
from .control import ControlChannel
from .episode_objective import EpisodeObjective
from .distributed import CollectiveContext


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
    data_revision: str = AB_DATA_REVISION
    curriculum: str = "mixed"
    memory_candidates: int = 8
    counterfactual_weight: float = 0.0
    counterfactual_margin: float = 2.0
    keep_fraction: float | None = None
    keep_weight: float = 0.2

    def __post_init__(self):
        if self.stage not in {"A", "B"} or min(self.student_token_budget, self.gradient_accumulation) < 1:
            raise ValueError("invalid training stage/budget/accumulation")
        self._validate_common()
        if self.data_revision not in AB_REVISIONS or self.curriculum not in ("mixed", "facts", "locations"):
            raise ValueError("unknown data revision/curriculum")
        if self.data_revision == LEGACY_AB_REVISION and self.curriculum != "mixed":
            raise ValueError("legacy A/B data requires its unchanged mixed curriculum")
        if not 1 <= self.memory_candidates <= 8 or min(self.counterfactual_weight, self.counterfactual_margin) < 0:
            raise ValueError("invalid candidate/contrastive curriculum")
        if self.data_revision == LEGACY_AB_REVISION and (self.memory_candidates != 8 or self.counterfactual_weight):
            raise ValueError("legacy replay requires its original candidate and loss semantics")
        if (self.keep_fraction is not None and not 0 <= self.keep_fraction <= 1) or self.keep_weight <= 0:
            raise ValueError("invalid language keep mixture/weight")
        if self.data_revision == LEGACY_AB_REVISION and (self.keep_fraction is not None or self.keep_weight != 0.2):
            raise ValueError("legacy replay requires its original keep semantics")

    def _validate_common(self):
        if min(self.student_token_budget, self.gradient_accumulation) < 1:
            raise ValueError("invalid training budget/accumulation")
        if self.learning_rate <= 0 or self.distractors < 1 or self.save_every_micro < 1:
            raise ValueError("invalid optimizer/distractor/checkpoint configuration")


class Trainer:
    def __init__(self, model, compiler, run_dir, config, *, base_identity, resume=False, collectives=None, initialize_from=None):
        self.model, self.compiler, self.config = model, compiler, config
        self.run_dir = Path(run_dir)
        self.device = model.base.embed_tokens.weight.device
        self.collectives = collectives or CollectiveContext(device=self.device, enabled=False)
        self.rank, self.world_size = self.collectives.rank, self.collectives.world_size
        self.collectives.run_local("initialize trainer", lambda: self._initialize(base_identity, resume, initialize_from))
        descriptor = {"identity": self.identity, "resume": resume, "progress": asdict(self.progress),
                      "parameters": [[name, list(p.shape), str(p.dtype)] for name, p in model.named_parameters()
                                     if p.requires_grad]}
        reference = self.collectives.broadcast_json(descriptor if self.rank == 0 else None)
        self.collectives.require_all_success(None if descriptor == reference else "rank trainer identity/layout differs")
        if not resume:
            parameters = [p for p in model.parameters() if p.requires_grad]
            parameters += list(frozen_auxiliary_parameters(model).values())
            self.collectives.broadcast_parameters(parameters)

    def _initialize(self, base_identity, resume, initialize_from):
        model, compiler, config = self.model, self.compiler, self.config
        if resume and initialize_from is not None:
            raise ValueError("resume and stage initialization are distinct operations")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = CheckpointStore(self.run_dir / "checkpoints")
        self.control = ControlChannel(self.run_dir)
        initialization = None
        if initialize_from is not None:
            from dsrsi.evaluation.checkpoint import load_ab_weights
            source_store = CheckpointStore(Path(initialize_from) / "checkpoints")
            committed = source_store.load_committed()
            if config.stage not in ("A", "B") or committed[1]["progress"]["stage"] != "A":
                raise ValueError("explicit initialization supports A-to-B or a versioned A-to-A curriculum change")
            source_config = committed[1]["identity"]["train_config"]
            if config.stage == "A" and (source_config.get("data_revision", LEGACY_AB_REVISION),
                                       source_config.get("curriculum", "mixed"), source_config.get("memory_candidates", 8),
                                       source_config.get("counterfactual_weight", 0.0), source_config.get("keep_fraction"),
                                       source_config.get("keep_weight", 0.2)) == (
                                           config.data_revision, config.curriculum, config.memory_candidates, config.counterfactual_weight,
                                           config.keep_fraction, config.keep_weight):
                raise ValueError("A-to-A initialization must explicitly change data revision or curriculum")
            if self.run_dir.resolve() == Path(initialize_from).resolve():
                raise ValueError("stage transition requires a new run directory")
            source, manifest = load_ab_weights(model, source_store, base_identity=base_identity, committed=committed)
            initialization = {"transition": f"A-to-{config.stage}-reset-training-state", "checkpoint": source.name,
                              "manifest_sha256": file_sha256(source / "manifest.json"),
                              "identity": manifest["identity"], "progress": manifest["progress"]}
        elif resume:
            # Resume pins lineage and never follows an advancing source latest.
            initialization = self.store.load_committed()[1]["identity"].get("initialization")
        self.objective = EpisodeObjective(model, compiler, stage=config.stage, memory_candidates=config.memory_candidates,
            counterfactual_weight=config.counterfactual_weight, counterfactual_margin=config.counterfactual_margin,
            keep_weight=config.keep_weight)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=config.learning_rate, weight_decay=config.weight_decay,
                                           foreach=False, fused=False)
        self.optimizer.zero_grad(set_to_none=True)
        self.generator = make_generator(revision=config.data_revision, seed=config.seed, distractors=config.distractors)
        config_identity = asdict(config)
        if config.keep_fraction is None:
            config_identity.pop("keep_fraction")
        if config.keep_weight == 0.2:
            config_identity.pop("keep_weight")
        if config.memory_candidates == 8:
            config_identity.pop("memory_candidates")
        if not config.counterfactual_weight:
            config_identity.pop("counterfactual_weight")
            config_identity.pop("counterfactual_margin")
        if config.data_revision == LEGACY_AB_REVISION:
            # Legacy replay is explicit and retains the original identity and
            # generator. Never reinterpret old ordinals with new data.
            config_identity.pop("data_revision")
            config_identity.pop("curriculum")
        self.identity = {"base": base_identity, "train_config": config_identity,
                         "execution_semantics": "ab-episode-v2-explicit-optimizer-backend"}
        if config.data_revision != LEGACY_AB_REVISION:
            self.identity["data"] = data_identity(config.data_revision)
        if self.world_size > 1:
            self.identity["execution_semantics"] = "ab-episode-v4-query-weighted-coordinated-gradients"
        if not resume or "memory_architecture" in self.store.load_committed()[1]["identity"]:
            self.identity["memory_architecture"] = model.memory_architecture_identity()
        if initialization is not None:
            self.identity["initialization"] = initialization
        self.progress = Progress(config.stage)
        self.cursor = {"next_ordinal": self.rank, "schema_version": 1}
        self.life = {"mode": "m_only", "streams": {}, "harness_refs": {}}
        if resume:
            payload, _ = self.store.load_latest(identity=self.identity, rank=self.rank, world_size=self.world_size)
            self.progress, self.cursor, self.life = restore_training_state(
                payload, model, self.optimizer, device=self.device)
        elif (self.store.root / "latest.json").exists():
            raise FileExistsError("run already has a checkpoint; use explicit resume")
        self.last_checkpoint = None

    def _save(self, status):
        started = time.monotonic()
        result = None
        if self.rank == 0:
            try:
                result = {"directory": str(self.store.begin(self.progress))}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        result = self.collectives.broadcast_json(result)
        if "error" in result:
            raise RuntimeError(result["error"])
        pending = Path(result["directory"])
        error = None
        try:
            state = capture_training_state(self.model, self.optimizer, progress=self.progress,
                                           data_cursor=self.cursor, life_state=self.life, device=self.device,
                                           rank=self.rank, world_size=self.world_size)
            self.store.save_rank(pending, state)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.collectives.require_all_success(error)
        result = None
        if self.rank == 0:
            try:
                result = {"directory": str(self.store.commit(
                    pending, world_size=self.world_size, identity=self.identity,
                    progress=self.progress, status=status,
                    required_student_tokens=self.config.student_token_budget if status == "completed" else None))}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        result = self.collectives.broadcast_json(result)
        if "error" in result:
            raise RuntimeError(result["error"])
        final = Path(result["directory"])
        self.last_checkpoint = final
        self._log({"event": "checkpoint", "status": status, "path": str(final),
                   "save_seconds": time.monotonic() - started})
        return final

    def _log(self, data):
        filename = "events.jsonl" if self.world_size == 1 else f"events_rank_{self.rank:05d}.jsonl"
        def write():
            with (self.run_dir / filename).open("a") as stream:
                stream.write(json.dumps({**data, "progress": asdict(self.progress)}, ensure_ascii=False) + "\n")
        self.collectives.run_local("write event log", write)

    def _optimizer_step(self, partial=False):
        count = self.progress.accumulation_index
        if count == 0:
            return
        def rescale():
            if partial:
                factor = self.config.gradient_accumulation / count
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(factor)
        self.collectives.run_local("rescale partial gradients", rescale)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.collectives.synchronize_gradients(trainable)
        self.collectives.run_local("clip gradients", lambda: torch.nn.utils.clip_grad_norm_(
            trainable, self.config.gradient_clip, error_if_nonfinite=True))
        self.collectives.run_local("optimizer step", self.optimizer.step)
        self.collectives.run_local("clear gradients", lambda: self.optimizer.zero_grad(set_to_none=True))
        self.progress = replace(self.progress, optimizer_step=self.progress.optimizer_step + 1, accumulation_index=0)

    def _control_at_boundary(self):
        request = self.collectives.control(self.collectives.run_local("poll control", self.control.poll))
        if request is None:
            return None
        status = {"checkpoint": "checkpoint", "pause": "paused", "stop": "stopped"}[request.action]
        checkpoint = self._save(status)
        self.collectives.run_local("acknowledge control", lambda: self.control.acknowledge(request, checkpoint))
        return status if request.action != "checkpoint" else None

    def _forward_episode(self, ordinal):
        world, variant = divmod(ordinal, 2)
        mix = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{world}:mix".encode()).digest()[:8], "big") % 100
        fraction = self.config.keep_fraction
        keep = mix >= ((75 if self.config.stage == "A" else 70) if fraction is None else 100 * (1 - fraction))
        family = {"mixed": None, "facts": "single_fact", "locations": "paraphrase_coreference"}[self.config.curriculum]
        row = self.generator.episode(world, variant=variant, family="language_preservation" if keep else family)
        if self.config.stage == "A":
            oracle = True
        else:
            draw = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{ordinal}:oracle".encode()).digest()[:8], "big") / 2**64
            oracle = draw < 0.8
        kwargs = {}
        if self.config.counterfactual_weight and row["family"] not in ("language_preservation", "missing_evidence"):
            kwargs["counterfactual_row"] = self.generator.episode(world, variant=1 - variant,
                family="language_preservation" if keep else family)
        return row, oracle, self.objective(row, oracle_include=oracle, **kwargs)

    def run(self, *, max_micro_steps=None, install_signals=True):
        def prepare():
            if install_signals:
                self.control.install_signal_handlers()
            self.model.train()
        self.collectives.run_local("prepare training", prepare)
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
            row, oracle, result = self.collectives.run_local("episode forward", lambda: self._forward_episode(ordinal))
            student_tokens, teacher_tokens, global_queries = self.collectives.totals(
                result.student_tokens, result.teacher_tokens, result.queries)
            # synchronize_gradients averages ranks. This factor makes that
            # average equal the mean over all valid queries, rather than giving
            # a one-query rank the same weight as a multi-query rank.
            weight = self.world_size * result.queries / global_queries if global_queries else 0.0
            self.collectives.run_local("episode backward", lambda: (
                result.loss * weight / self.config.gradient_accumulation).backward() if result.queries else None)
            self.collectives.run_local("commit numerical state", self._finish_micro)
            self.cursor["next_ordinal"] += self.world_size
            self.progress = replace(self.progress, micro_step=self.progress.micro_step + 1,
                                    accumulation_index=self.progress.accumulation_index + int(global_queries > 0),
                                    student_tokens=self.progress.student_tokens + student_tokens,
                                    teacher_tokens=self.progress.teacher_tokens + teacher_tokens)
            if self.progress.accumulation_index == self.config.gradient_accumulation:
                self._optimizer_step()
            self._log({"event": "micro", "episode_id": row["episode_id"], "family": row["family"],
                       "loss": float(result.loss.detach()), "answer_loss": result.answer_loss,
                       "route_loss": result.route_loss, "keep_kl": result.keep_kl,
                       "counterfactual_loss": result.counterfactual_loss,
                       "student_input_tokens": result.student_tokens, "teacher_input_tokens": result.teacher_tokens,
                       "padding_positions": result.padding_positions, "supervised_tokens": result.supervised_tokens,
                       "covered_queries": result.covered_queries, "queries": result.queries,
                       "global_queries": global_queries,
                       "tool_events": result.tool_events, "reply_events": result.reply_events, "tool_loss": result.tool_loss,
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

    def _finish_micro(self):
        """Stateful trainers detach numerical state after a successful backward."""
