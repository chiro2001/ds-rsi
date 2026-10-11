"""D-stage content objective: train the Consolidator through summary answers.

Protocol: docs/training-protocol-v0.md section 8 and
docs/semantic-compression.md. A depth-d episode has 3d+1 leaves whose answer
bearing support sits in the first group of four. The student answers from the
final summary alone; the teacher reads the uncompressed first four leaves and
is used for top32+tail KL only after its own answer-free free generation has
been screened as exactly correct.

The Consolidator never receives the question, the answer tokens or any later
leaf as compression input: the chain is built first, then the answer is scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dsrsi.data.compiler import collate_responses
from dsrsi.data.compression import compression_plan
from dsrsi.evaluation.m_only import normalized_answer
from dsrsi.state.streaming_cache import StreamingCache
from dsrsi.training.losses import topk_tail_kl


@dataclass
class CompressionOutput:
    loss: torch.Tensor
    answer_loss: float
    kl_loss: float
    kl_weighted: float
    teacher_screened: bool
    depth: int
    chain_levels: int
    student_input_tokens: int
    teacher_input_tokens: int
    padding_positions: int
    supervised_tokens: int
    screened_text: str
    screening_path: str


@dataclass
class TeacherScreen:
    text: str
    input_tokens: int
    token_ids: list[int]
    stopped_on_end: bool


class CompressionObjective:
    """One compression episode -> a Consolidator-only training loss."""

    def __init__(self, model, compiler, *, kl_weight=0.2, teacher_screening="free_generation",
                 screening_prefill="token", max_screen_tokens=16):
        if kl_weight < 0:
            raise ValueError("kl weight cannot be negative")
        if teacher_screening not in {"free_generation", "none"}:
            raise ValueError("teacher screening must be free_generation or none")
        if screening_prefill != "token":
            raise ValueError("teacher screening requires the canonical token prefill")
        if max_screen_tokens < 1:
            raise ValueError("screening budget must be positive")
        self.model, self.compiler = model, compiler
        self.kl_weight = kl_weight
        self.teacher_screening = teacher_screening
        self.screening_prefill = screening_prefill
        self.max_screen_tokens = max_screen_tokens
        # Content training must not silently update the already trained codec.
        model.requires_grad_(False)
        model.consolidator.requires_grad_(True)

    def _write_leaves(self, episode):
        """Frozen Writer pass; the chain must not need leaf gradients."""
        device = self.model.base.embed_tokens.weight.device
        records = []
        for span in episode.source:
            source = torch.tensor([span.input_ids], device=device)
            records.extend(self.model.write_records(
                source, stream_ids=[episode.episode_id], logical_begins=[span.logical_begin],
                record_ids=[span.event_id], wall_times=[span.wall_time]))
        return records, sum(len(span.input_ids) for span in episode.source)

    def _chain(self, row, episode, leaves):
        """Fold 3d+1 leaves into one summary along the declared group layout."""
        plan = compression_plan(row)
        by_id = {record.record_id: record for record in leaves}
        missing = [name for name in plan["leaves"] if name not in by_id]
        if missing:
            raise ValueError(f"compiled episode is missing compression leaves: {missing}")
        groups = plan["groups"]
        summary = self.model.consolidate_records([by_id[name] for name in plan["leaves"][groups[0][0]:groups[0][1]]],
                                                record_id=f"{episode.episode_id}:summary-1")
        levels = 1
        for index, (begin, end) in enumerate(groups[1:], start=2):
            fresh = [by_id[name] for name in plan["leaves"][begin:end]]
            summary = self.model.consolidate_records([summary] + fresh,
                                                     record_id=f"{episode.episode_id}:summary-{index}")
            levels += 1
        if levels != plan["depth"]:
            raise ValueError("compression chain length differs from the declared depth")
        return summary, plan, levels

    def teacher_records(self, row, episode=None):
        """The screened teacher reads exactly the first four uncompressed leaves."""
        episode = episode or self.compiler.compile(row)
        plan = compression_plan(row)
        first = set(plan["leaves"][:4])
        records = self._write_leaves(episode)[0]
        return [record for record in records if record.record_id in first]

    def _teacher_free_generation(self, response, teacher_leaves, life_id):
        """Greedy answer-free decode with the leaf teacher, deployment semantics.

        The last prompt token is forwarded with memory enabled at the lookup
        position exactly like the M-only evaluator; the answer-free prefix before
        it runs with memory disabled. Parallel BF16 GDN prefill cannot stand in
        for this canonical per-token path when deciding whether to enable KL.
        """
        device = self.model.base.embed_tokens.weight.device
        prompt = torch.tensor([response.prompt_ids], device=device)
        if not prompt.shape[1]:
            raise ValueError("empty answer-free screening prefix")
        cache = StreamingCache(self.model.base.config.layer_types, window=2048, rebase_interval=None)
        cache.clock = response.logical_begin
        cache.position_origin = response.logical_begin
        with torch.no_grad():
            for index in range(prompt.shape[1] - 1):
                cache = self.model.forward_cached(prompt[:, index:index + 1], cache,
                                                  memory_enabled=False).cache
            lookup_position = response.logical_begin + prompt.shape[1] - 1
            output = self.model.forward_cached(prompt[:, -1:], cache, records=teacher_leaves,
                                               stream_id=life_id, lookup_position=lookup_position,
                                               lookup_wall_time=response.wall_time)
            generated = []
            consumed = prompt.shape[1]
            stopped = False
            for index in range(self.max_screen_tokens):
                token = int(self.model.head(output.hidden[:, -1]).argmax(-1).item())
                if token == self.compiler.tokenizer.eos_token_id:
                    stopped = True
                    break
                generated.append(token)
                if index + 1 == self.max_screen_tokens:
                    break
                output = self.model.forward_cached(torch.tensor([[token]], device=device), output.cache,
                                                   records=teacher_leaves, stream_id=life_id,
                                                   lookup_position=lookup_position,
                                                   lookup_wall_time=response.wall_time)
                consumed += 1
        return TeacherScreen(self.compiler.tokenizer.decode(generated, skip_special_tokens=True),
                             consumed, generated, stopped)

    def __call__(self, row):
        episode = self.compiler.compile(row)
        if len(episode.responses) != 1:
            raise ValueError("compression episodes supervise exactly one answer")
        if episode.source_encoding != "independent_events":
            raise ValueError("compression episodes require independent source encoding")
        if any(len(span.input_ids) > 256 for span in episode.source):
            raise ValueError("compression leaf longer than the Writer span requires explicit support mapping")
        response = episode.responses[0]
        if response.write_enabled:
            raise ValueError("compression answers must never be written back into M")
        if any(response.memory_transform.get(name) for name in ("drop_events", "merge_events")):
            raise ValueError("compression rows declare their chain in annotations, not in transforms")
        plan = compression_plan(row)
        if plan["support_group"] != 0:
            raise ValueError("compression support must live in the first four leaves")
        if response.event_id != plan["support_probe_id"] or response.support_groups != [[plan["support_event"]]]:
            raise ValueError("compression supervision must match its declared support probe")
        device = self.model.base.embed_tokens.weight.device
        with torch.no_grad():
            leaves, source_tokens = self._write_leaves(episode)
        summary, plan, levels = self._chain(row, episode, leaves)

        batch = collate_responses([response], pad_token_id=self.compiler.tokenizer.pad_token_id, device=device)
        student = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
                             records=[[summary]], query_indices=batch["retrieval_query_indices"],
                             logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                             wall_times=[response.wall_time])
        answer = student.loss.mean()
        loss = answer
        kl_value, weighted, screened, text = 0.0, 0.0, False, ""
        teacher_tokens = 0
        if self.kl_weight and self.teacher_screening == "free_generation":
            teacher_leaves = [record for record in leaves if record.record_id in set(plan["leaves"][:4])]
            screen = self._teacher_free_generation(response, teacher_leaves, episode.episode_id)
            text = screen.text
            teacher_tokens += screen.input_tokens
            screened = normalized_answer(text) == normalized_answer(row["probes"][0]["target"])
            if screened:
                mask = batch["labels"][:, 1:] != -100
                # Only the teacher is detached. The student head and KL must
                # remain in the Consolidator's autograd graph.
                with torch.no_grad():
                    teacher = self.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                         records=[teacher_leaves], query_indices=batch["retrieval_query_indices"],
                                         logical_begins=[response.logical_begin],
                                         stream_ids=[episode.episode_id], wall_times=[response.wall_time])
                    teacher_logits = self.model.head(teacher.hidden[:, :-1][mask])
                kl = topk_tail_kl(self.model.head(student.hidden[:, :-1][mask]), teacher_logits)
                kl_value = float(kl.detach())
                weighted = self.kl_weight * kl_value
                loss = loss + self.kl_weight * kl
                teacher_tokens += len(response.input_ids)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite compression objective")
        supervised = int((batch["labels"] != -100).sum())
        return CompressionOutput(loss, float(answer.detach()), kl_value, weighted, screened, plan["depth"],
                                 levels, source_tokens + len(response.input_ids), teacher_tokens, 0, supervised, text,
                                 self.screening_prefill if self.kl_weight and self.teacher_screening == "free_generation" else "none")
