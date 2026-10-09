"""Generate tool calls, execute observations, then score; never roll out labels."""

import json

import torch

from dsrsi.data.tools import tool_schemas
from dsrsi.harness.memory_tools import HarnessStore, parse_native_tool_calls
from dsrsi.state.streaming_cache import StreamingCache
from .m_only import normalized_answer


class ToolEvaluator:
    def __init__(self, model, compiler, *, max_new_tokens=128, max_turns=4, parallel_prefill_diagnostic=False):
        if min(max_new_tokens, max_turns) < 1:
            raise ValueError("positive generation and tool-turn budgets are required")
        self.model, self.compiler = model, compiler
        self.max_new_tokens, self.max_turns = max_new_tokens, max_turns
        self.parallel_prefill_diagnostic = parallel_prefill_diagnostic

    @torch.no_grad()
    def generate(self, messages):
        normalized = [self.compiler.normalize_message(message) for message in messages]
        ids = self.compiler.encode(normalized, generation=True, tools=tool_schemas())
        if not ids or len(ids) > self.compiler.max_query_tokens:
            raise ValueError("tool prompt exceeds its budget; do not truncate observations")
        device = self.model.base.embed_tokens.weight.device
        inputs = torch.tensor([ids], device=device)
        cache = StreamingCache(self.model.base.config.layer_types, window=2048, rebase_interval=None)
        if self.parallel_prefill_diagnostic:
            output = self.model.forward_cached(inputs, cache, memory_enabled=False)
        else:
            for index in range(inputs.shape[1]):
                output = self.model.forward_cached(inputs[:, index:index + 1], cache, memory_enabled=False)
                cache = output.cache
        tokens, stopped = [], False
        for _ in range(self.max_new_tokens):
            token = int(self.model.head(output.hidden[:, -1]).argmax(-1).item())
            if token == self.compiler.tokenizer.eos_token_id:
                stopped = True
                break
            tokens.append(token)
            output = self.model.forward_cached(torch.tensor([[token]], device=device), output.cache, memory_enabled=False)
        # Native <tool_call> markers may be special tokens. Preserve them for
        # schema validation instead of silently stripping the generated action.
        return self.compiler.tokenizer.decode(tokens, skip_special_tokens=False), stopped, len(ids)

    @torch.no_grad()
    def episode(self, row):
        if row.get("split") == "train" or "initial_harness" not in row:
            raise ValueError("tool evaluation requires held-out worlds and a durable initial harness snapshot")
        self.model.eval()
        store = HarnessStore.from_state_dict(row["initial_harness"])
        try:
            # Only the new user request is exposed initially. Source observations
            # remain in the archive/E and must be obtained through real tools.
            request = next(message for message in row["trajectory"] if message["role"] == "user")
            messages = [self.compiler.normalize_message(request)]
            clock = store.last_clock + 1
            store.append_history("user", request["content"], clock=clock)
            predictions, observations, generations = [], [], []
            error, final, stop_reason = None, "", "turn_budget"
            for _ in range(self.max_turns):
                try:
                    text, ended, prompt_tokens = self.generate(messages)
                except ValueError as exc:
                    error, stop_reason = str(exc), "prompt_budget_or_invalid_prefix"
                    break
                generations.append({"text": text, "stopped_on_end": ended, "prompt_tokens": prompt_tokens})
                if not ended:
                    stop_reason = "generation_budget"
                    final = text
                    break
                try:
                    calls = parse_native_tool_calls(text)
                except ValueError as exc:
                    error = str(exc)
                    stop_reason = "invalid_action"
                    break
                if not calls:
                    final = text
                    stop_reason = "final_reply"
                    messages.append({"role": "assistant", "content": text})
                    clock += 1
                    store.append_history("assistant", text, clock=clock)
                    break
                assistant = {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": call} for call in calls]}
                messages.append(assistant)
                clock += 1
                store.append_history("assistant", json.dumps(assistant, ensure_ascii=False, sort_keys=True), clock=clock)
                for call in calls:
                    clock += 1
                    result = store.execute(call["name"], call["arguments"], clock=clock)
                    observations.append({"name": call["name"], "result": result})
                    predictions.append(call)
                    messages.append({"role": "tool", "name": call["name"], "content": result})
                    clock += 1
                    store.append_history("tool", json.dumps(result, ensure_ascii=False, sort_keys=True), clock=clock)
            # Labels and expected state enter only after the model rollout.
            expected_calls = [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                              for message in row["trajectory"] for call in message.get("tool_calls", [])]
            expected_reply = next(message["content"] for message in reversed(row["trajectory"])
                                  if message["role"] == "assistant" and not message.get("tool_calls"))
            expected_store = {k: value["text"] for k, value in row["annotations"]["expected_final_explicit_store"].items()}
            actual_store = {k: value["text"] for k, value in store.memory.items()}
            calls_match = predictions == expected_calls
            state_match = expected_store == actual_store
            answer_match = normalized_answer(final) == normalized_answer(expected_reply)
            mutation_names = {"memory.write", "memory.update", "memory.delete"}
            expected_mutations = [call for call in expected_calls if call["name"] in mutation_names]
            actual_mutations = [call for call in predictions if call["name"] in mutation_names]
            if expected_mutations:
                # Extra verification reads can be useful. Separate successful
                # requested state changes from exact demonstration imitation.
                operation_success = (expected_mutations == actual_mutations and state_match
                    and all(o["result"].get("ok", False) for o in observations if o["name"] in mutation_names))
            elif row["family"] == "read":
                operation_success = answer_match and any(o["name"] == "memory.read" and o["result"].get("found") for o in observations)
            elif row["family"] == "history":
                operation_success = answer_match and any(o["name"] == "history.open" and o["result"].get("found") for o in observations)
            else:
                operation_success = answer_match and not predictions
            return {"episode_id": row["episode_id"], "world_id": row["world_id"], "family": row["family"],
                    "language": row["language"], "predicted_calls": predictions, "expected_calls": expected_calls,
                    "observations": observations, "generations": generations, "final_text": final,
                    "expected_reply": expected_reply, "tool_call_exact_match": calls_match,
                    "explicit_state_match": state_match, "final_answer_exact_match": answer_match,
                    "requested_operation_success": operation_success,
                    "end_to_end_success": calls_match and state_match and answer_match and error is None
                        and all(g["stopped_on_end"] for g in generations),
                    "actual_explicit_store": actual_store, "error": error, "stop_reason": stop_reason}
        finally:
            store.close()
