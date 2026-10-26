"""Greedy canonical decoding with an explicit, question-independent record list."""

from dataclasses import asdict
import copy

import torch

from .m_only import GeneratedAnswer, MOnlyEvaluator, normalized_answer


def evidence_teacher_response(compiler, row, probe, *, scope="support"):
    if scope not in {"support", "pair"}:
        raise ValueError("unknown teacher evidence scope")
    ids = (set(sum(probe["support_groups"], [])) if scope == "support" else
           set(row["annotations"]["fixed_memory_candidate_ids"]))
    evidence = [event["content"] for event in row["source_events"] if event["event_id"] in ids]
    if len(evidence) != len(ids) or not evidence:
        raise ValueError("teacher evidence IDs are missing")
    query = ("请只根据下面的记录回答问题，不要添加解释。\n记录：\n" if row["query_language"] == "zh" else
             "Answer using only the following records. Do not add explanations.\nRecords:\n")
    query += "\n".join(evidence) + "\n\n" + probe["query"]
    teacher_row = copy.deepcopy(row)
    teacher_row["probes"] = [dict(probe, query=query)]
    return compiler.compile(teacher_row).responses[0]


def sparse_distribution(logits, topk=32):
    logits = logits.float()
    k = min(topk, logits.shape[-1] - 1)
    if k < 1:
        raise ValueError("teacher distribution requires at least two vocabulary entries")
    indices = logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
    norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    return (indices.detach(), (logits.gather(-1, indices) - norm).detach(),
            (torch.logsumexp(logits.masked_fill(mask, -torch.inf), dim=-1, keepdim=True) - norm).detach())


@torch.no_grad()
def generate_fixed_records(model, compiler, response, records, *, life_id, expected, max_new_tokens=16,
                           collect_distribution=False):
    if max_new_tokens < 1 or response.mode != "m_only" or response.write_enabled:
        raise ValueError("fixed-record decoding requires a bounded read-only M-only query")
    device = model.base.embed_tokens.weight.device
    # No support IDs, answer tokens, or scoring fields participate in routing.
    prompt, cache = MOnlyEvaluator(model, compiler)._encode_prefix(response)
    lookup = cache.clock
    output = model.forward_cached(prompt[:, -1:], cache, records=records, stream_id=life_id,
                                  memory_enabled=bool(records), lookup_position=lookup,
                                  lookup_wall_time=response.wall_time)
    generated, distributions = [], []
    stopped = False
    for _ in range(max_new_tokens):
        logits = model.head(output.hidden[:, -1])
        if collect_distribution:
            distributions.append(tuple(v.cpu() for v in sparse_distribution(logits)))
        token = int(logits.argmax(-1).item())
        if token == compiler.tokenizer.eos_token_id:
            stopped = True
            break
        generated.append(token)
        output = model.forward_cached(torch.tensor([[token]], device=device), output.cache, records=records,
                                      stream_id=life_id, memory_enabled=bool(records), lookup_position=lookup,
                                      lookup_wall_time=response.wall_time)
    text = compiler.tokenizer.decode(generated, skip_special_tokens=True)
    selected = [r.record_id for r in records]
    answer = GeneratedAnswer(response.event_id, "fixed" if records else "off", text, generated, stopped,
                             expected, normalized_answer(text) == normalized_answer(expected),
                             all(set(selected).intersection(g) for g in response.support_groups), selected)
    return dict(answer=asdict(answer), input_tokens=len(response.prompt_ids) + len(generated),
                distributions=distributions)
