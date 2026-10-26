"""A fixed candidate pair, multiple questions, and gated plaintext-teacher KL."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from dsrsi.data.compiler import collate_responses
from dsrsi.evaluation.fixed_records import evidence_teacher_response, generate_fixed_records
from .episode_objective import EpisodeObjective, ObjectiveOutput


EVIDENCE_DISTILLATION_VERSION = "fixed-pair-evidence-top32-tail-v1"


def sparse_teacher_kl(student_logits, indices, log_probabilities, log_tail):
    """KL over teacher top-k categories and one exact remaining-mass bucket."""
    student = student_logits.float()
    if (student.ndim != 2 or indices.shape != log_probabilities.shape or indices.ndim != 2
            or indices.shape[0] != student.shape[0] or log_tail.shape != (student.shape[0], 1)
            or not 0 < indices.shape[1] < student.shape[-1]):
        raise ValueError("sparse teacher/student dimensions differ")
    indices = indices.to(device=student.device, dtype=torch.long)
    p_top = log_probabilities.detach().to(device=student.device, dtype=torch.float32)
    p_tail = log_tail.detach().to(device=student.device, dtype=torch.float32)
    mask = torch.zeros_like(student, dtype=torch.bool).scatter(1, indices, True)
    norm = torch.logsumexp(student, dim=-1, keepdim=True)
    q_top = student.gather(1, indices) - norm
    q_tail = torch.logsumexp(student.masked_fill(mask, -torch.inf), dim=-1, keepdim=True) - norm
    return ((p_top.exp() * (p_top - q_top)).sum(-1)
            + (p_tail.exp() * (p_tail - q_tail)).squeeze(-1)).mean()


def exact_teacher_match(result, response, eos_token_id):
    answer = result["answer"]
    return (answer["stopped_on_end"] and answer["token_ids"] + [eos_token_id]
            == response.input_ids[response.prompt_length:]
            and len(result["distributions"]) == len(response.input_ids) - response.prompt_length)


def _statistics(result, *, evidence_loss=0.0, attempted=0, accepted=0, selected=None):
    result.evidence_loss = evidence_loss
    result.evidence_teacher_attempted = attempted
    result.evidence_teacher_accepted = accepted
    result.fixed_candidates_per_query = selected or []
    return result


class FixedPairDistillationObjective(EpisodeObjective):
    def __init__(self, *args, evidence_distillation_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.stage != "A" or self.memory_candidates not in (1, 2)
                or not 0 <= evidence_distillation_weight < float("inf")):
            raise ValueError("fixed-pair distillation requires A with one/two candidates and finite weight")
        self.evidence_distillation_weight = evidence_distillation_weight

    def _teacher(self, row, probe):
        response = evidence_teacher_response(self.compiler, row, probe)
        result = generate_fixed_records(self.model, self.compiler, response, [], life_id=row["episode_id"],
                                        expected=probe["target"], collect_distribution=True)
        accepted = exact_teacher_match(result, response, self.compiler.tokenizer.eos_token_id)
        return result, accepted

    def _run_episode(self, row, *, oracle_include=True, counterfactual_row=None):
        if row["family"] == "language_preservation":
            return _statistics(super()._run_episode(row, oracle_include=oracle_include,
                                                    counterfactual_row=counterfactual_row))
        if row["family"] != "single_fact" or not oracle_include:
            raise ValueError("fixed-pair A requires explicitly available fact evidence")
        episode = self.compiler.compile(row)
        ids = row["annotations"]["fixed_memory_candidate_ids"]
        if len(ids) != len(set(ids)) or len(ids) != 2 or len(episode.responses) != 2:
            raise ValueError("a fixed pair requires two distinct records and two questions")
        records, total, padding = self._sources(episode)
        by_id = {r.record_id: r for r in records}
        if not set(ids) <= by_id.keys():
            raise ValueError("fixed candidates are absent from the written records")
        counter_records, counter_episode = None, None
        if self.counterfactual_weight:
            if (counterfactual_row is None or any(row[k] != counterfactual_row[k] for k in
                    ("world_id", "counterfactual_group_id", "family", "language", "split"))
                    or row["annotations"]["variant"] == counterfactual_row["annotations"]["variant"]
                    or ids != counterfactual_row["annotations"]["fixed_memory_candidate_ids"]):
                raise ValueError("fixed-pair contrast requires the matched opposite world")
            counter_episode = self.compiler.compile(counterfactual_row)
            counter_records, cost, pad = self._sources(counter_episode)
            total += cost
            padding += pad
        device = self.model.base.embed_tokens.weight.device
        losses, answers, contrasts, evidence_losses, chosen = [], [], [], [], []
        attempted = accepted = teacher_tokens = supervised = 0
        for response, probe in zip(episode.responses, row["probes"]):
            if response.event_id != probe["probe_id"] or len(response.support_groups) != 1:
                raise ValueError("fixed pair response/probe identity differs")
            selected_ids = ids if self.memory_candidates == 2 else [i for i in ids if i in response.support_groups[0]]
            if (len(selected_ids) != self.memory_candidates
                    or not set(selected_ids).intersection(response.support_groups[0])):
                raise ValueError("fixed pair is missing the required support")
            selected = [by_id[i] for i in selected_ids]
            chosen.append(list(selected_ids))
            teacher = None
            eligible = False
            if self.evidence_distillation_weight and self.memory_candidates == 2:
                teacher, eligible = self._teacher(row, probe)
                teacher_tokens += teacher["input_tokens"]
                attempted += 1
                accepted += int(eligible)
            batch = collate_responses([response], pad_token_id=self.compiler.tokenizer.pad_token_id, device=device)
            output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
                                records=[selected], query_indices=batch["retrieval_query_indices"],
                                logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                                wall_times=[response.wall_time])
            answer = output.loss.mean()
            loss = answer
            kl = answer.detach().new_zeros(())
            if eligible:
                parts = teacher["distributions"]
                indices, probabilities, tail = [torch.cat([part[i] for part in parts], dim=0) for i in range(3)]
                mask = batch["labels"][:, 1:] != -100
                logits = self.model.head(output.hidden[:, :-1][mask])
                kl = sparse_teacher_kl(logits, indices, probabilities, tail)
                loss = loss + self.evidence_distillation_weight * kl
            contrast = answer.detach().new_zeros(())
            if counter_records is not None:
                opposite = next(r for r in counter_episode.responses if r.event_id == response.event_id)
                if opposite.prompt_ids != response.prompt_ids:
                    raise ValueError("counterfactual questions differ")
                actual = response.input_ids[response.prompt_length:]
                other = opposite.input_ids[opposite.prompt_length:]
                difference = next((i for i, (a, b) in enumerate(zip(actual, other)) if a != b), None)
                if difference is not None:
                    prefix_ids = response.input_ids[:response.prompt_length + difference]
                    prefix = torch.tensor([prefix_ids], device=device)
                    replacements = {r.record_id: r for r in counter_records}
                    foreign = [replace(r, slots=replacements[r.record_id].slots) for r in selected]
                    probabilities = []
                    for payload in (selected, foreign):
                        state = self.model(prefix, records=[payload], query_indices=[response.retrieval_query_index],
                                           logical_begins=[response.logical_begin], stream_ids=[episode.episode_id],
                                           wall_times=[response.wall_time])
                        probabilities.append(F.log_softmax(self.model.head(state.hidden[:, -1]).float(), -1)
                                             [0, actual[difference]])
                    contrast = F.softplus(self.counterfactual_margin - (probabilities[0] - probabilities[1]))
                    loss = loss + self.counterfactual_weight * contrast
                    total += 2 * len(prefix_ids)
            total += len(response.input_ids)
            supervised += sum(t != -100 for t in response.labels)
            losses.append(loss)
            answers.append(float(answer.detach()))
            contrasts.append(float(contrast.detach()))
            evidence_losses.append(float(kl.detach()))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite fixed-pair objective")
        n = len(losses)
        result = ObjectiveOutput(loss, total, teacher_tokens, padding, supervised,
                                 sum(answers) / n, 0.0, 0.0, n, n,
                                 counterfactual_loss=sum(contrasts) / n)
        return _statistics(result, evidence_loss=sum(evidence_losses) / n, attempted=attempted,
                           accepted=accepted, selected=chosen)
