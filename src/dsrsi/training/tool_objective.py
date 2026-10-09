"""Stage C tool event CE and no-tool language preservation through real LoRA."""

from dataclasses import replace

import torch

from dsrsi.data.compiler import collate_responses
from dsrsi.model.lora import LoRALinear, lora_disabled
from .episode_objective import ObjectiveOutput
from .losses import topk_tail_kl


class ToolObjective:
    def __init__(self, model, compiler, *, keep_weight=0.2):
        if not any(isinstance(m, LoRALinear) for m in model.modules()):
            raise ValueError("empty-memory tool training needs installed, trainable language adapters")
        self.model, self.compiler, self.keep_weight = model, compiler, keep_weight

    def __call__(self, row):
        episode = self.compiler.compile(row)
        if any(r.mode != "tools" or r.write_enabled for r in episode.responses):
            raise ValueError("tool SFT requires assistant-only tool trajectory events")
        if "training_response_index" not in row:
            raise ValueError("one selected assistant event is the recoverable C-tools microbatch")
        index = row["training_response_index"]
        if not 0 <= index < len(episode.responses):
            raise ValueError("invalid selected assistant event")
        response = episode.responses[index]
        device = self.model.base.embed_tokens.weight.device
        records = []
        source_tokens = 0
        if episode.family == "language_preservation":
            for span in episode.source:
                encoded = self.model.write_records(torch.tensor([span.input_ids], device=device),
                                                   stream_ids=[episode.episode_id], logical_begins=[span.logical_begin],
                                                   record_ids=[span.event_id], wall_times=[span.wall_time])
                records += [replace(r, slots=r.slots.detach(), keys=r.keys.detach()) for r in encoded]
                source_tokens += len(span.input_ids)
        batch = collate_responses([response], pad_token_id=self.compiler.tokenizer.pad_token_id, device=device)
        output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
                            query_indices=batch["retrieval_query_indices"], memory_enabled=bool(records), records=[records],
                            logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                            wall_times=[response.wall_time])
        answer = output.loss.mean()
        loss, kl, teacher_tokens = answer, None, 0
        if episode.family == "language_preservation":
            with torch.no_grad(), lora_disabled(self.model):
                teacher = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], memory_enabled=False).hidden
            mask = batch["labels"][:, 1:] != -100
            kl = topk_tail_kl(self.model.head(output.hidden[:, :-1][mask]), self.model.head(teacher[:, :-1][mask]))
            loss = self.keep_weight * (answer + kl)
            teacher_tokens = len(response.input_ids)
        if not loss.requires_grad or not torch.isfinite(loss):
            raise FloatingPointError("tool objective has no trainable path or is nonfinite")
        tool = response.supervision_kind == "tool"
        return ObjectiveOutput(loss, len(response.input_ids) + source_tokens, teacher_tokens, 0,
                               sum(label != -100 for label in response.labels), float(answer.detach()),
                               0.0, 0.0 if kl is None else float(kl.detach()), 0, 1,
                               tool_events=int(tool), reply_events=int(not tool),
                               tool_loss=float(answer.detach()) if tool else 0.0)
