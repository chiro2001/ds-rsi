"""Compile source and query inputs separately, with explicit causal read sites."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .tools import tool_schemas, TOOL_PROPERTIES

IGNORE_INDEX = -100
RESET_COMPONENTS = ("kv", "gdn", "conv", "pending_writer")


@dataclass
class SourceSpan:
    event_id: str
    input_ids: list[int]
    logical_begin: int
    logical_end: int
    role: str
    commit_boundary: bool
    wall_time: float | None = None
    reset_working_before: bool = False


@dataclass
class SupervisedSequence:
    event_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    retrieval_query_index: int
    memory_read_from: int
    logical_begin: int
    mode: str
    reset_components: list[str]
    support_groups: list[list[str]] = field(default_factory=list)
    memory_transform: dict = field(default_factory=dict)
    allow_tools: list[str] = field(default_factory=list)
    wall_time: float | None = None
    write_enabled: bool = False
    supervision_kind: str = "answer"

    @property
    def prompt_ids(self):
        """Generation input never contains the supervised completion."""
        return self.input_ids[:self.prompt_length]

    def reader_position_mask(self, device="cpu"):
        return torch.arange(len(self.input_ids), device=device) >= self.memory_read_from


@dataclass
class CompiledEpisode:
    episode_id: str
    family: str
    language: str
    source: list[SourceSpan]
    background: list[SourceSpan]
    responses: list[SupervisedSequence]
    split: str | None = None
    world_id: str | None = None
    counterfactual_group_id: str | None = None
    requires_query_update: bool = False
    source_encoding: str = "independent_events"
    query_setup: dict = field(default_factory=dict)

    def state_dict(self):
        return asdict(self)


def _wall_time(value):
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("wall times must contain an explicit timezone")
    return dt.timestamp()


class EpisodeCompiler:
    def __init__(self, tokenizer, *, max_query_tokens=4096, allow_preview=False):
        self.tokenizer = tokenizer
        self.max_query_tokens = max_query_tokens
        self.allow_preview = allow_preview
        if tokenizer.eos_token_id is None or tokenizer.eos_token_id == tokenizer.pad_token_id:
            raise ValueError("distinct end and padding tokens are required")

    def encode(self, messages, *, generation=False, tools=None):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=generation,
                                                  enable_thinking=False, tools=tools)
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def normalize_message(message):
        normalized = {key: message[key] for key in ("role", "content", "name", "tool_calls", "tool_call_id") if key in message}
        if normalized.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message role")
        content = normalized.get("content", "")
        if content is None and normalized["role"] == "assistant":
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if any(marker in content for marker in ("<|im_start|>", "<|im_end|>")):
            raise ValueError("raw message contains reserved chat boundary tokens")
        normalized["content"] = content
        for call in normalized.get("tool_calls", []):
            if normalized["role"] != "assistant" or call.get("type") != "function":
                raise ValueError("only assistant function calls are supported")
            function = call["function"]
            name, arguments = function["name"], function["arguments"]
            if name not in TOOL_PROPERTIES or not isinstance(arguments, dict):
                raise ValueError("unknown function or unstructured tool arguments")
            if set(arguments) != set(TOOL_PROPERTIES[name]) or any(not isinstance(v, str) for v in arguments.values()):
                raise ValueError("tool arguments do not satisfy the shared schema")
        return normalized

    def _sources(self, events, begin=0, prefix=None, previous_ids=None, tools=None, independent=False):
        messages = list(prefix or [])
        previous = list(previous_ids or [])
        output = []
        seen = set()
        cursor = begin
        for number, event in enumerate(events):
            if independent:
                messages, previous = [], []
            event_id = event.get("event_id", f"background-{number}")
            if event_id in seen:
                raise ValueError("duplicate source event id")
            seen.add(event_id)
            messages.append(self.normalize_message(event))
            full = self.encode(messages, tools=tools)
            if full[:len(previous)] != previous:
                raise ValueError("native template rewrote earlier source events")
            delta = full[len(previous):]
            if not delta:
                raise ValueError("empty encoded source event")
            output.append(SourceSpan(event_id, delta, cursor, cursor + len(delta) - 1,
                                     event["role"], bool(event.get("commit_boundary", False)),
                                     _wall_time(event.get("time")), independent))
            cursor += len(delta)
            previous = full
        return output, cursor, messages, previous

    def _response(self, prefix, answer, *, event_id, logical_begin, mode, reset, tools=None, past_ids=None):
        prompt = self.encode(prefix, generation=True, tools=tools)
        completed = self.encode(prefix + [answer], tools=tools)
        if not prompt or completed[:len(prompt)] != prompt:
            raise ValueError("assistant prefix is not additive in native template")
        if past_ids is not None:
            if prompt[:len(past_ids)] != past_ids or completed[:len(past_ids)] != past_ids:
                raise ValueError("streaming template rewrote cached source history")
            prompt = prompt[len(past_ids):]
            completed = completed[len(past_ids):]
        suffix = completed[len(prompt):]
        end_positions = [index for index, token in enumerate(suffix) if token == self.tokenizer.eos_token_id]
        if len(end_positions) != 1:
            raise ValueError("assistant target must contain exactly one end marker")
        end = len(prompt) + end_positions[0]
        input_ids = completed[:end + 1]
        if len(input_ids) > self.max_query_tokens:
            raise ValueError("query exceeds budget; refusing to truncate evidence or end token")
        labels = [IGNORE_INDEX] * len(prompt) + input_ids[len(prompt):]
        # Retrieval uses the last visible prompt state, never the last state of
        # a teacher-forced answer. Earlier prompt states must not see that lookup.
        query_index = len(prompt) - 1
        return SupervisedSequence(event_id, input_ids, labels, len(prompt), query_index,
                                  query_index, logical_begin, mode, list(reset))

    def compile(self, row):
        if row.get("status") == "semantic_preview" and not self.allow_preview:
            raise ValueError("semantic previews require explicit allow_preview")
        episode_id = row["episode_id"]
        if not episode_id:
            raise ValueError("episode identity is required")
        has_streaming = any(probe.get("mode") == "streaming" for probe in row.get("probes", []))
        source_encoding = row.get("source_encoding", "continuous" if has_streaming else "independent_events")
        if source_encoding not in {"continuous", "independent_events"}:
            raise ValueError("invalid source encoding policy")
        if has_streaming and source_encoding != "continuous":
            raise ValueError("streaming tasks require continuous source encoding")
        independent = source_encoding == "independent_events"
        source, cursor, messages, previous = self._sources(row.get("source_events", []), independent=independent)
        known_events = {span.event_id for span in source}
        background, cursor, messages, previous = self._sources(
            row.get("intervening_events", []), cursor, messages, previous, independent=independent)
        if known_events & {span.event_id for span in background}:
            raise ValueError("background/source event identities overlap")
        responses = []
        for probe in row.get("probes", []):
            mode = probe["mode"]
            if mode not in {"m_only", "streaming"}:
                raise ValueError("invalid probe mode")
            groups = probe.get("support_groups", [])
            if any(not group or not set(group) <= known_events for group in groups):
                raise ValueError("support annotations refer to unknown source events")
            reset = row.get("annotations", {}).get("query_resets", [])
            if mode == "m_only" and set(reset) != set(RESET_COMPONENTS):
                raise ValueError("M-only probes must clear all working states")
            if mode == "streaming" and reset:
                raise ValueError("streaming probes must preserve working states")
            if probe.get("write_enabled", False):
                raise ValueError("teacher-forced probes must not write answers into M")
            transforms = probe.get("memory_transform", {})
            for name in ("drop_events", "merge_events"):
                if not set(transforms.get(name, [])) <= known_events:
                    raise ValueError("memory transform refers to unknown events")
            if transforms.get("drop_events") and not independent:
                remaining = known_events - set(transforms["drop_events"])
                if remaining and row.get("annotations", {}).get("evidence_available") is False:
                    raise ValueError("continuous later states may retain dropped evidence; cannot label it unavailable")
            query_prefix = list(messages) if mode == "streaming" else []
            if probe.get("query_time") is not None:
                query_prefix.append({"role": "system", "content": f"当前时间：{probe['query_time']}"})
            query_prefix.append(self.normalize_message({"role": "user", "content": probe["query"]}))
            tools = tool_schemas(probe["allow_tools"]) if probe.get("allow_tools") else None
            response = self._response(query_prefix, self.normalize_message({"role": "assistant", "content": probe["target"]}),
                                      event_id=probe["probe_id"], logical_begin=cursor,
                                      mode=mode, reset=reset, tools=tools,
                                      past_ids=previous if mode == "streaming" else None)
            response.support_groups = groups
            response.memory_transform = transforms
            response.allow_tools = probe.get("allow_tools", [])
            response.wall_time = _wall_time(probe.get("query_time"))
            responses.append(response)
        trajectory = row.get("trajectory", [])
        query_setup = row.get("query_setup", {})
        if not set(query_setup.get("memory_transform", {}).get("drop_events", [])) <= known_events:
            raise ValueError("tool query setup drops unknown source evidence")
        if trajectory:
            tools = tool_schemas()
            history = []
            for index, message in enumerate(trajectory):
                message = self.normalize_message(message)
                if message["role"] == "assistant":
                    responses.append(self._response(history, message, event_id=f"assistant-{index}",
                                                     logical_begin=cursor, mode="tools", reset=[], tools=tools))
                    responses[-1].allow_tools = [tool["function"]["name"] for tool in tools]
                    responses[-1].supervision_kind = "tool" if message.get("tool_calls") else "answer"
                history.append(message)
        if not responses:
            raise ValueError("episode has no supervised assistant events")
        return CompiledEpisode(episode_id, row["family"], row["language"], source, background,
                               responses, row.get("split"), row.get("world_id"), row.get("counterfactual_group_id"),
                               bool(row.get("annotations", {}).get("requires_query_update", False)), source_encoding,
                               query_setup)


def collate_responses(responses, *, pad_token_id, device="cpu"):
    if not responses:
        raise ValueError("cannot collate empty responses")
    length = max(len(item.input_ids) for item in responses)
    input_ids = torch.full((len(responses), length), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    read_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, item in enumerate(responses):
        size = len(item.input_ids)
        input_ids[index, :size] = torch.tensor(item.input_ids, device=device)
        labels[index, :size] = torch.tensor(item.labels, device=device)
        attention_mask[index, :size] = True
        read_mask[index, item.memory_read_from:size] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
            "memory_read_mask": read_mask,
            "retrieval_query_indices": torch.tensor([item.retrieval_query_index for item in responses], device=device)}


def tokenizer_fingerprint(path):
    result = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
        file = Path(path) / name
        if file.is_file():
            result[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Compile episodes using the native Qwen template")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--allow-preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    compiler = EpisodeCompiler(tokenizer, allow_preview=args.allow_preview)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError("output exists; choose another path or use explicit --overwrite")
    seen = set()
    counts = {"episodes": 0, "responses": 0, "source_tokens": 0, "background_tokens": 0,
              "response_input_tokens": 0, "supervised_tokens": 0}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with Path(args.input).open() as source, os.fdopen(descriptor, "w") as destination:
            for line in source:
                row = json.loads(line)
                episode = compiler.compile(row)
                if episode.episode_id in seen:
                    raise ValueError("duplicate episode identity")
                seen.add(episode.episode_id)
                destination.write(json.dumps(episode.state_dict(), ensure_ascii=False) + "\n")
                counts["episodes"] += 1
                counts["responses"] += len(episode.responses)
                counts["source_tokens"] += sum(len(s.input_ids) for s in episode.source)
                counts["background_tokens"] += sum(len(s.input_ids) for s in episode.background)
                counts["response_input_tokens"] += sum(len(s.input_ids) for s in episode.responses)
                counts["supervised_tokens"] += sum(sum(t != IGNORE_INDEX for t in s.labels) for s in episode.responses)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest = {"format_version": 1, "tokenizer_fingerprint": tokenizer_fingerprint(args.tokenizer),
                "input_sha256": _file_hash(args.input), "output_sha256": _file_hash(output), "counts": counts,
                "semantic_preview": args.allow_preview, "actual_training_consumption": False}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
