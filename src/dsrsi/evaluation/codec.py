"""Teacher-forced, position-specific diagnostics, never a recall capability gate."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from .m_only import MOnlyEvaluator


def distinguishing_token(response, opposite):
    if response.prompt_ids != opposite.prompt_ids:
        raise ValueError("counterfactual diagnostic requires identical questions")
    a = response.input_ids[response.prompt_length:]
    b = opposite.input_ids[opposite.prompt_length:]
    difference = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if difference is None:
        raise ValueError("counterfactual diagnostic requires a distinguishing token")
    return difference, a, b


def swapped_payloads(records, opposite_records):
    by_id = {r.record_id: r for r in opposite_records}
    if any(r.record_id not in by_id for r in records):
        raise ValueError("counterfactual payload lacks the original record identity")
    return [replace(r, slots=by_id[r.record_id].slots) for r in records]


@torch.no_grad()
def native_logits(model, response, records, *, life_id, reader_trace=None):
    device = model.base.embed_tokens.weight.device
    targets = response.input_ids[response.prompt_length:]
    positions = list(range(response.prompt_length - 1, len(response.input_ids) - 1))
    handles = []
    if reader_trace is not None:
        def capture(name):
            def hook(module, args, output):
                hidden = args[0]
                gate = module.gate(module.h_norm(hidden)).sigmoid().float()
                relative = ((output.float() - hidden.float()).square().mean(-1).sqrt()
                            / hidden.float().square().mean(-1).sqrt().clamp_min(1e-12))
                reader_trace[name] = {"alpha": float(module.alpha),
                    "gate": gate[0, positions, 0].tolist(),
                    "relative_residual_rms": relative[0, positions].tolist()}
            return hook
        handles = [module.register_forward_hook(capture(name)) for name, module in model.readers.items()]
    try:
        ids = torch.tensor([response.input_ids[:-1]], device=device)
        state = model(ids, records=[records], query_indices=[response.retrieval_query_index],
            logical_begins=[response.logical_begin], stream_ids=[life_id], wall_times=[response.wall_time])
        logits = model.head(state.hidden[0, response.prompt_length - 1:]).float()
        if len(logits) != len(targets):
            raise ValueError("native diagnostic prediction alignment differs from targets")
        return logits
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def canonical_logits(model, compiler, response, records, *, life_id, prefix=None):
    evaluator = MOnlyEvaluator(model, compiler, memory_candidates=len(records))
    state, selected, _, _, lookup = evaluator._prompt_forward(response, records, life_id=life_id,
                                                             mode="oracle", prefix=prefix)
    device = model.base.embed_tokens.weight.device
    targets = response.input_ids[response.prompt_length:]
    logits = []
    for index, target in enumerate(targets):
        # Score token t before consuming it; subsequent targets are absent.
        logits.append(model.head(state.hidden[0, -1]).float())
        if index + 1 < len(targets):
            state = model.forward_cached(torch.tensor([[target]], device=device), state.cache,
                records=selected, stream_id=life_id, lookup_position=lookup, lookup_wall_time=response.wall_time)
    return torch.stack(logits)


def summarize_logits(logits, targets, tokenizer):
    target = torch.tensor(targets, device=logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    predictions = logits.argmax(-1).tolist()
    return {"target_token_ids": list(targets), "target_tokens": [tokenizer.decode([x]) for x in targets],
        "predicted_token_ids": predictions, "predicted_tokens": [tokenizer.decode([x]) for x in predictions],
        "correct": [x == y for x, y in zip(predictions, targets)],
        "target_nll": (-logp.gather(1, target[:, None]).squeeze(1)).tolist()}


def payload_stats(actual, opposite):
    a, b = actual.slots.float(), opposite.slots.float()
    matrix = F.normalize(a, dim=-1) @ F.normalize(a, dim=-1).T
    off_diagonal = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    return {"relative_change_rms": float((a - b).square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)),
        "paired_slot_cosine_mean": float(F.cosine_similarity(a, b, dim=-1).mean()),
        "within_record_slot_cosine_mean": float(matrix[off_diagonal].mean()) if len(a) > 1 else None}
