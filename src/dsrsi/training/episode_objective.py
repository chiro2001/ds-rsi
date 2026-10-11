"""A/B episode objective with actual-source accounting and answer-free routing."""

from dataclasses import dataclass, replace
import copy

import torch
import torch.nn.functional as F

from dsrsi.data.compiler import collate_responses
from dsrsi.training.losses import required_support_loss, routing_scores, topk_tail_kl


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    student_tokens: int
    teacher_tokens: int
    padding_positions: int
    supervised_tokens: int
    answer_loss: float
    route_loss: float
    keep_kl: float
    covered_queries: int
    queries: int
    tool_events: int = 0
    reply_events: int = 0
    tool_loss: float = 0.0
    counterfactual_loss: float = 0.0


class EpisodeObjective:
    def __init__(self, model, compiler, *, stage="A", route_weight=0.05, keep_weight=0.2,
                 memory_candidates=8, counterfactual_weight=0.0, counterfactual_margin=2.0,
                 reuse_frozen_prefix=False):
        if stage not in {"A", "B"}:
            raise ValueError("this objective explicitly implements A/B; C/D have different state semantics")
        self.model, self.compiler, self.stage = model, compiler, stage
        self.route_weight, self.keep_weight = route_weight, keep_weight
        if not 1 <= memory_candidates <= 8 or min(counterfactual_weight, counterfactual_margin) < 0:
            raise ValueError("invalid memory candidate limit or counterfactual loss")
        self.memory_candidates = memory_candidates
        self.counterfactual_weight, self.counterfactual_margin = counterfactual_weight, counterfactual_margin
        self.reuse_frozen_prefix = reuse_frozen_prefix
        model.address.requires_grad_(stage == "B")
        model.consolidator.requires_grad_(False)

    def _sources(self, episode):
        if episode.source_encoding != "independent_events" or any(r.mode != "m_only" for r in episode.responses):
            raise ValueError("A/B controls require independently encoded sources and M-only queries")
        if not episode.source or any(len(span.input_ids) > 256 for span in episode.source):
            raise ValueError("A/B source controls need 1..256 tokens per explicitly labeled event")
        device = self.model.base.embed_tokens.weight.device
        maximum = max(len(span.input_ids) for span in episode.source)
        source_ids = torch.full((len(episode.source), maximum), self.compiler.tokenizer.pad_token_id,
                                device=device, dtype=torch.long)
        source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
        for index, span in enumerate(episode.source):
            source_ids[index, :len(span.input_ids)] = torch.tensor(span.input_ids, device=device)
            source_mask[index, :len(span.input_ids)] = True
        records = self.model.write_records(source_ids, attention_mask=source_mask,
                                          stream_ids=[episode.episode_id] * len(episode.source),
                                          logical_begins=[span.logical_begin for span in episode.source],
                                          record_ids=[span.event_id for span in episode.source],
                                          wall_times=[span.wall_time for span in episode.source])
        return records, sum(len(span.input_ids) for span in episode.source), source_ids.numel() - int(source_mask.sum())

    def __call__(self, row, *, oracle_include=True, counterfactual_row=None):
        if self.reuse_frozen_prefix:
            from dsrsi.model.frozen_prefix_reuse import frozen_prefix_reuse
            with frozen_prefix_reuse(self.model):
                return self._run_episode(row, oracle_include=oracle_include, counterfactual_row=counterfactual_row)
        return self._run_episode(row, oracle_include=oracle_include, counterfactual_row=counterfactual_row)

    def _run_episode(self, row, *, oracle_include=True, counterfactual_row=None):
        episode = self.compiler.compile(row)
        device = self.model.base.embed_tokens.weight.device
        records, total_tokens, padding = self._sources(episode)
        is_keep = episode.family == "language_preservation"
        if is_keep:
            records = [replace(record, slots=record.slots.detach(), keys=record.keys.detach()) for record in records]
        counter_episode = None
        counter_records = None
        if self.counterfactual_weight and not is_keep and episode.family != "missing_evidence":
            if counterfactual_row is None or any(row[key] != counterfactual_row[key] for key in
                                                ("world_id", "counterfactual_group_id", "family", "language", "split")):
                raise ValueError("counterfactual loss requires the same world's opposite evidence variant")
            if row["annotations"]["variant"] == counterfactual_row["annotations"]["variant"]:
                raise ValueError("counterfactual loss requires an independent opposite variant")
            counter_episode = self.compiler.compile(counterfactual_row)
            counter_records, counter_tokens, counter_padding = self._sources(counter_episode)
            total_tokens += counter_tokens
            padding += counter_padding
        teacher_tokens = 0
        supervised = 0
        values, answers, routes, kls = [], [], [], []
        covered = 0
        contrastive = []
        for response_index, original_response in enumerate(episode.responses):
            response = original_response
            if response.memory_transform.get("merge_events"):
                raise ValueError("A/B cannot silently train a compression transform")
            dropped = set(response.memory_transform.get("drop_events", []))
            pool = [record for record in records if record.record_id not in dropped]
            if not pool:
                raise ValueError("A/B missing-evidence and keep controls need distractor payload to train readers")
            groups = [set(group) for group in response.support_groups]
            available_groups = [group for group in groups if any(r.record_id in group for r in pool)]
            scores = None
            route = None
            if self.stage == "B" and groups and len(available_groups) == len(groups):
                prompt = torch.tensor([response.prompt_ids], device=device)
                features = self.model.source_features(prompt)[:, response.retrieval_query_index]
                with self.model._memory_context(features):
                    query = self.model.address.encode_query(features)
                    keys = self.model.address.encode_keys(torch.stack([record.slots for record in pool]))
                # Keep scoring in FP32 outside autocast, including its gradients.
                scores = routing_scores(query, keys)
                positives = torch.tensor([[[record.record_id in group for record in pool] for group in groups]],
                                         device=device, dtype=torch.bool)
                route = required_support_loss(scores, positives, torch.ones_like(scores, dtype=torch.bool)).mean()
                total_tokens += len(response.prompt_ids)
            ordering = list(range(len(pool))) if scores is None else torch.argsort(scores[0].detach(), descending=True, stable=True).tolist()
            if oracle_include and available_groups:
                positive_indices = [index for index, record in enumerate(pool) if any(record.record_id in group for group in available_groups)]
                ordering = positive_indices + [index for index in ordering if index not in positive_indices]
            selected = [pool[index] for index in ordering[:self.memory_candidates]]
            selected_ids = {record.record_id for record in selected}
            sufficient = all(selected_ids & group for group in groups)
            covered += int(sufficient)
            if groups and not sufficient:
                fallback_row = copy.deepcopy(row)
                target_probe = next(probe for probe in fallback_row["probes"] if probe["probe_id"] == response.event_id)
                query_language = row.get("query_language", row["language"])
                target_probe["target"] = ("The available memory does not contain enough evidence to determine that."
                                          if query_language == "en" else "当前可用记忆没有足够信息确定答案。")
                response = next(item for item in self.compiler.compile(fallback_row).responses if item.event_id == response.event_id)
            batch = collate_responses([response], pad_token_id=self.compiler.tokenizer.pad_token_id, device=device)
            output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
                                records=[selected], query_indices=batch["retrieval_query_indices"],
                                logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                                wall_times=[response.wall_time])
            answer = output.loss.mean()
            loss = answer
            kl = None
            if is_keep:
                with torch.no_grad():
                    teacher = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], memory_enabled=False).hidden
                mask = batch["labels"][:, 1:] != -100
                teacher_logits = self.model.head(teacher[:, :-1][mask])
                student_logits = self.model.head(output.hidden[:, :-1][mask])
                kl = topk_tail_kl(student_logits, teacher_logits)
                loss = self.keep_weight * (answer + kl)
                teacher_tokens += len(response.input_ids)
            if route is not None:
                loss = loss + self.route_weight * route
            contrast = None
            if counter_records is not None and groups and sufficient:
                opposite = next(r for r in counter_episode.responses if r.event_id == response.event_id)
                if response.prompt_ids != opposite.prompt_ids:
                    raise ValueError("counterfactual questions differ; do not train from mismatched worlds")
                actual_suffix = response.input_ids[response.prompt_length:]
                other_suffix = opposite.input_ids[opposite.prompt_length:]
                difference = next((i for i, (a, b) in enumerate(zip(actual_suffix, other_suffix)) if a != b), None)
                if difference is not None:
                    # Up to this point both completions have the same prefix.
                    # The distinguishing target token is scored, never fed in.
                    prefix_ids = response.input_ids[:response.prompt_length + difference]
                    prefix = torch.tensor([prefix_ids], device=device)
                    by_id = {r.record_id: r for r in counter_records}
                    replacements = [replace(r, slots=by_id[r.record_id].slots) for r in selected]
                    logits = []
                    for payload in (selected, replacements):
                        state = self.model(prefix, records=[payload], query_indices=[response.retrieval_query_index],
                            logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                            wall_times=[response.wall_time])
                        logits.append(F.log_softmax(self.model.head(state.hidden[:, -1]).float(), dim=-1)
                                      [0, actual_suffix[difference]])
                    contrast = F.softplus(self.counterfactual_margin - (logits[0] - logits[1]))
                    loss = loss + self.counterfactual_weight * contrast
                    total_tokens += 2 * len(prefix_ids)
            values.append(loss)
            answers.append(float(answer.detach()))
            routes.append(0.0 if route is None else float(route.detach()))
            kls.append(0.0 if kl is None else float(kl.detach()))
            contrastive.append(0.0 if contrast is None else float(contrast.detach()))
            total_tokens += len(response.input_ids)
            supervised += sum(token != -100 for token in response.labels)
        n = len(values)
        loss = torch.stack(values).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite episode objective")
        return ObjectiveOutput(loss, total_tokens, teacher_tokens,
                               padding, supervised,
                               sum(answers) / n, sum(routes) / n, sum(kls) / n, covered, n,
                               counterfactual_loss=sum(contrastive) / n)
