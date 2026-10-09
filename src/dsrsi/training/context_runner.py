"""TBPTT on one continuous life per rank, including M and a separate teacher."""

from dataclasses import asdict, dataclass
import importlib.metadata
from pathlib import Path

import torch
import torch.nn.functional as F

from dsrsi.data.continuous import ContinuousGenerator, CONTINUOUS_DATA_REVISION
from dsrsi.harness.memory_tools import HarnessStore
from dsrsi.model.lora import LoRAConfig, install_lora, lora_disabled
from dsrsi.model.streaming import StreamExecutor
from dsrsi.state.life_state import LifeState
from .checkpoint import CheckpointStore, Progress, file_sha256, restore_training_state
from .control import ControlChannel
from .episode_objective import ObjectiveOutput
from .losses import required_support_loss, routing_scores, topk_tail_kl
from .runner import Trainer


@dataclass(frozen=True)
class ContextTrainConfig:
    stage: str = "C_context"
    student_token_budget: int = 1_000_000
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20261010
    save_every_micro: int = 100
    lora_rank: int = 8
    lora_alpha: float = 16.0
    unroll_tokens: int = 32
    window: int = 2048
    rebase_interval: int = 4096
    writer_stride: int = 256
    retrieval_interval: int = 64
    memory_tensor_budget: int = 64 << 20
    interference_tokens: int = 2048
    include_tools: bool = True
    keep_weight: float = 0.2
    route_weight: float = 0.05
    allow_untrained_memory: bool = False

    def __post_init__(self):
        if self.stage != "C_context" or min(self.student_token_budget, self.gradient_accumulation, self.save_every_micro,
                self.unroll_tokens, self.window, self.rebase_interval, self.writer_stride,
                self.retrieval_interval, self.memory_tensor_budget) < 1 or self.interference_tokens < 0:
            raise ValueError("invalid continuous configuration")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.keep_weight <= 0 or self.route_weight < 0:
            raise ValueError("invalid continuous optimizer/objective configuration")
        LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


def configure_context_trainable(model):
    # Frozen Writer and key projections keep stored old records compatible with
    # the current reader/index. Query projection can learn against fixed keys.
    model.writer.requires_grad_(False)
    model.address.requires_grad_(False)
    model.address.query.requires_grad_(True)
    model.consolidator.requires_grad_(False)
    model.readers.requires_grad_(True)
    model.time_source.requires_grad_(True)


def context_implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = ("data/continuous.py", "data/compiler.py", "data/tools.py", "harness/memory_tools.py",
             "model/lora.py", "model/qwen35_memory.py", "model/streaming.py", "model/streaming_layers.py",
             "memory/modules.py", "memory/store.py", "state/life_state.py", "state/streaming_cache.py",
             "training/losses.py", "training/context_runner.py", "training/runner.py", "training/distributed.py",
             "training/checkpoint.py", "evaluation/checkpoint.py")
    return {"files": {name: file_sha256(root / name) for name in files},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}}


class ContextTrainer(Trainer):
    def _initialize(self, base_identity, resume, initialize_from):
        model, config = self.model, self.config
        if resume and initialize_from is not None:
            raise ValueError("resume and stage initialization are distinct operations")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = CheckpointStore(self.run_dir / "checkpoints")
        self.control = ControlChannel(self.run_dir)
        initialization, layout = None, None
        if resume:
            initialization = self.store.load_committed()[1]["identity"].get("initialization")
        elif initialize_from is not None:
            from dsrsi.evaluation.checkpoint import load_ab_weights, load_c_tool_weights
            source_store = CheckpointStore(Path(initialize_from) / "checkpoints")
            committed = source_store.load_committed()
            stage = committed[1]["progress"]["stage"]
            if stage not in {"B", "C_tools"} or self.run_dir.resolve() == Path(initialize_from).resolve():
                raise ValueError("continuous initialization requires B/C-tools and a new run")
            loader = load_ab_weights if stage == "B" else load_c_tool_weights
            source, manifest = loader(model, source_store, base_identity=base_identity, committed=committed)
            if stage == "C_tools":
                layout = model.lora_layout
                if layout["rank"] != config.lora_rank or layout["alpha"] != config.lora_alpha:
                    raise ValueError("continuous LoRA config must match the initialized adapters")
            initialization = {"transition": f"{stage}-to-C-context-reset-training-state", "checkpoint": source.name,
                              "manifest_sha256": file_sha256(source / "manifest.json"),
                              "identity": manifest["identity"], "progress": manifest["progress"]}
        elif not config.allow_untrained_memory:
            raise ValueError("initialize from validated memory, or explicitly allow an untrained-memory experiment")
        if layout is None:
            layout = install_lora(model, LoRAConfig(rank=config.lora_rank, alpha=config.lora_alpha))
        configure_context_trainable(model)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=config.learning_rate, weight_decay=config.weight_decay,
                                           foreach=False, fused=False)
        self.optimizer.zero_grad(set_to_none=True)
        self.generator = ContinuousGenerator(self.compiler, seed=config.seed, include_tools=config.include_tools,
                                             interference_tokens=config.interference_tokens)
        self.journal = self.run_dir / "harness" / f"rank_{self.rank:05d}.sqlite"
        life_id = f"train-context-rank-{self.rank}"
        if resume and not self.journal.is_file():
            raise FileNotFoundError("continuous resume requires its durable harness journal")
        store = HarnessStore(self.journal, life_id, database_id=self.generator.database_id)
        self.harness_reference = store.state_dict()
        store.close()
        self.executor = StreamExecutor(model)
        self.stream = self.executor.new_life(life_id, window=config.window, rebase_interval=config.rebase_interval,
            writer_stride=config.writer_stride, retrieval_interval=config.retrieval_interval,
            memory_tensor_budget=config.memory_tensor_budget)
        self.teacher = self.executor.new_life(life_id + ":teacher", window=config.window,
                                              rebase_interval=config.rebase_interval)
        self.identity = {"base": base_identity, "train_config": asdict(config), "lora": layout,
                         "memory_architecture": model.memory_architecture_identity(),
                         "data_revision": CONTINUOUS_DATA_REVISION,
                         "execution_semantics": "continuous-v1-next-target-event-fractions-tbptt",
                         "implementation": context_implementation_identity()}
        if initialization is not None:
            self.identity["initialization"] = initialization
        self.progress = Progress(config.stage)
        self.cursor = {"next_ordinal": self.rank, "next_case_ordinal": self.rank, "schema_version": 1,
                       "started": False, "case": None, "frame_index": 0, "frame_offset": 0}
        self.life = {"mode": "continuous-tbptt", "streams": {}, "harness_refs": {}}
        if resume:
            payload, _ = self.store.load_latest(identity=self.identity, rank=self.rank, world_size=self.world_size)
            self.progress, self.cursor, self.life = restore_training_state(payload, model, self.optimizer, device=self.device)
            self.stream = LifeState.from_state_dict(self.life["streams"]["student"], device=self.device)
            self.teacher = LifeState.from_state_dict(self.life["streams"]["teacher"], device=self.device)
            self.harness_reference = self.life["harness_refs"]["visible"]
            store = HarnessStore.from_state_dict(self.harness_reference, path=self.journal)
            if store.last_clock > self.stream.cache.clock:
                raise ValueError("restored harness contains observations after the model clock")
            store.close()
        elif (self.store.root / "latest.json").exists():
            raise FileExistsError("run already has a checkpoint; use explicit resume")
        self.last_checkpoint = None

    def _save(self, status):
        def snapshot():
            self.life["streams"] = {"student": self.stream.state_dict(), "teacher": self.teacher.state_dict()}
            self.life["harness_refs"] = {"visible": self.harness_reference}
        self.collectives.run_local("snapshot continuous life", snapshot)
        return super()._save(status)

    def _finish_micro(self):
        self.stream, self.teacher = self.stream.detached(), self.teacher.detached()

    def _ensure_case(self):
        if self.cursor["case"] is None:
            self.cursor["case"] = self.generator.plan(self.cursor["next_case_ordinal"], life_id=self.stream.life_id,
                journal=self.journal, harness_reference=self.harness_reference, clock=self.stream.cache.clock,
                startup=not self.cursor["started"])
            self.cursor["started"] = True
            self.cursor["frame_index"] = self.cursor["frame_offset"] = 0

    def _forward_episode(self, ordinal):
        self._ensure_case()
        case, config = self.cursor["case"], self.config
        remaining = config.unroll_tokens
        outputs, teachers, targets, events, routes = [], [], [], [], []
        covered = 0
        source_begin = min((s["source_begin"] for s in case["supports"]), default=0)
        while remaining and self.cursor["frame_index"] < len(case["frames"]):
            index, offset = self.cursor["frame_index"], self.cursor["frame_offset"]
            frame = case["frames"][index]
            count = min(remaining, len(frame["input_ids"]) - offset)
            ids = torch.tensor([frame["input_ids"][offset:offset + count]], device=self.device)
            force = [] if frame["read_at"] is None or not offset <= frame["read_at"] < offset + count else [frame["read_at"] - offset]
            expected_query_position = frame["source_begin"] + frame["read_at"] if frame["read_at"] is not None else -1
            def observe(query, pool, position, selected):
                nonlocal covered
                if position != expected_query_position or not case["supports"]:
                    return
                groups = [[r.record_id for r in pool if r.history_ref == s["history_id"]
                           and r.source_begin <= s["source_begin"] and r.source_end >= s["source_end"]]
                          for s in case["supports"]]
                if any(not group for group in groups):
                    return  # Partial/compacted records are not presumed sufficient.
                selected_ids = {r.record_id for r in selected}
                covered += int(all(selected_ids.intersection(group) for group in groups))
                positives = torch.tensor([[[r.record_id in group for r in pool] for group in groups]],
                                         device=self.device, dtype=torch.bool)
                negatives = torch.tensor([[r.source_end < source_begin for r in pool]], device=self.device)
                eligible = negatives | positives.any(1)
                keys = torch.stack([r.keys.detach().to(self.device) for r in pool])
                scores = routing_scores(query[None], keys)
                routes.append(required_support_loss(scores, positives, eligible, negatives).mean())
            output = self.executor.forward(ids, self.stream, write=frame["role"] in {"user", "tool"},
                force_read_at=force, flush=offset + count == len(frame["input_ids"]), role=frame["role"],
                kind="tool" if frame["role"] == "tool" else "observation", wall_time=frame["wall_time"],
                history_ref=frame["history_id"], retrieval_observer=observe)
            self.stream = output.state
            with torch.no_grad(), lora_disabled(self.model):
                teacher = self.executor.forward(ids, self.teacher, write=False, memory_enabled=False)
            self.teacher = teacher.state
            outputs.append(output.hidden)
            teachers.append(teacher.hidden)
            for position in range(offset, offset + count):
                if position + 1 < len(frame["input_ids"]):
                    next_frame, next_pos, next_index = frame, position + 1, index
                elif index + 1 < len(case["frames"]):
                    next_frame, next_pos, next_index = case["frames"][index + 1], 0, index + 1
                else:
                    next_frame, next_pos, next_index = frame, None, index
                target = -100 if next_pos is None else next_frame["labels"][next_pos]
                targets.append(target)
                events.append(next_index)
            offset += count
            remaining -= count
            if offset == len(frame["input_ids"]):
                self.harness_reference = frame["harness_after"]
                self.cursor["frame_index"], self.cursor["frame_offset"] = index + 1, 0
            else:
                self.cursor["frame_offset"] = offset
        hidden, teacher_hidden = torch.cat(outputs, 1), torch.cat(teachers, 1)
        labels = torch.tensor(targets, device=self.device)
        event_ids = torch.tensor(events, device=self.device)
        zero = hidden.reshape(-1)[:0].float().sum()
        values, kls, tool_values = [], [], []
        event_numbers = event_ids[labels != -100].unique().tolist()
        for event in event_numbers:
            mask = (labels != -100) & (event_ids == event)
            frame = case["frames"][event]
            logits = self.model.head(hidden[0, mask])
            # Each chunk contributes its fraction of the complete assistant event.
            ce = F.cross_entropy(logits.float(), labels[mask], reduction="sum") / frame["total_targets"]
            value = ce
            if case["family"] == "keep":
                with torch.no_grad():
                    teacher_logits = self.model.head(teacher_hidden[0, mask])
                kl = topk_tail_kl(logits, teacher_logits) * (int(mask.sum()) / frame["total_targets"])
                value = config.keep_weight * (ce + kl)
                kls.append(kl)
            values.append(value)
            if frame["supervision_kind"] == "tool":
                tool_values.append(ce)
        queries = len(event_numbers)
        answer = torch.stack(values).sum() / queries if values else zero
        route = torch.stack(routes).mean() if routes else zero
        loss = answer + config.route_weight * route
        if not torch.isfinite(hidden).all() or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite continuous hidden state/objective")
        count = hidden.shape[1]
        row = {"episode_id": case["episode_id"], "family": case["family"]}
        if self.cursor["frame_index"] == len(case["frames"]):
            self.cursor["case"] = None
            self.cursor["next_case_ordinal"] += self.world_size
        return row, False, ObjectiveOutput(loss, count, count, 0, int((labels != -100).sum()),
            float(answer.detach()), float(route.detach()), float(torch.stack(kls).sum().detach()) if kls else 0.0,
            covered, queries, tool_events=len(tool_values), reply_events=queries - len(tool_values),
            tool_loss=float(torch.stack(tool_values).mean().detach()) if tool_values else 0.0)
