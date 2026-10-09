import copy
import json
from pathlib import Path

import pytest

from dsrsi.data.compiler import EpisodeCompiler, collate_responses


class BoundaryTokenizer:
    """Deterministic fixture for protocol edge cases; real Qwen is checked separately."""
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = ""
        if kwargs.get("tools"):
            text += json.dumps(kwargs["tools"], sort_keys=True)
        for message in messages:
            text += message["role"] + ":" + message["content"]
            if message.get("tool_calls"):
                text += json.dumps(message["tool_calls"], sort_keys=True)
            text += "\x01\n"
        if add_generation_prompt:
            text += "assistant:"
        return text

    def encode(self, text, **kwargs):
        return [1 if char == "\x01" else ord(char) + 3 for char in text]


def example():
    return json.loads(Path("data/examples/memory_episodes.jsonl").read_text().splitlines()[0])


def test_query_prefix_excludes_source_answer_and_future_teacher_tokens():
    compiler = EpisodeCompiler(BoundaryTokenizer(), allow_preview=True)
    episode = compiler.compile(example())
    response = episode.responses[0]
    target_ids = compiler.tokenizer.encode("Q7-492")
    assert all(response.prompt_ids[i:i + len(target_ids)] != target_ids
               for i in range(len(response.prompt_ids)))
    assert response.retrieval_query_index == response.prompt_length - 1
    assert response.input_ids[response.retrieval_query_index + 1] == target_ids[0]
    assert response.labels[:response.prompt_length] == [-100] * response.prompt_length
    assert response.labels[-1] == compiler.tokenizer.eos_token_id
    assert response.input_ids[-1] == compiler.tokenizer.eos_token_id  # trailing newline removed
    mask = response.reader_position_mask()
    assert not mask[:response.retrieval_query_index].any()
    assert mask[response.retrieval_query_index:].all()


def test_independent_records_and_continuous_sources_are_explicit():
    rows = [json.loads(line) for line in Path("data/examples/memory_episodes.jsonl").read_text().splitlines()]
    compiler = EpisodeCompiler(BoundaryTokenizer(), allow_preview=True)
    missing = compiler.compile(next(row for row in rows if row["family"] == "missing_evidence"))
    assert missing.source_encoding == "independent_events"
    assert all(span.reset_working_before for span in missing.source)
    continuous = compiler.compile(next(row for row in rows if row["family"] == "streaming_task"))
    assert continuous.source_encoding == "continuous"
    assert not any(span.reset_working_before for span in continuous.source)
    assert not continuous.responses[0].reset_components


def test_continuous_sources_cannot_label_dropped_fact_as_definitely_missing():
    row = json.loads(next(line for line in Path("data/examples/memory_episodes.jsonl").read_text().splitlines()
                         if '"family":"missing_evidence"' in line))
    row["source_encoding"] = "continuous"
    with pytest.raises(ValueError, match="retain dropped"):
        EpisodeCompiler(BoundaryTokenizer(), allow_preview=True).compile(row)


def test_preview_unknown_support_and_teacher_writes_are_rejected():
    row = example()
    with pytest.raises(ValueError, match="preview"):
        EpisodeCompiler(BoundaryTokenizer()).compile(row)
    compiler = EpisodeCompiler(BoundaryTokenizer(), allow_preview=True)
    bad = copy.deepcopy(row)
    bad["probes"][0]["support_groups"] = [["future-source"]]
    with pytest.raises(ValueError, match="unknown source"):
        compiler.compile(bad)
    bad = copy.deepcopy(row)
    bad["probes"][0]["write_enabled"] = True
    with pytest.raises(ValueError, match="must not write"):
        compiler.compile(bad)


def test_long_targets_are_rejected_not_truncated():
    with pytest.raises(ValueError, match="refusing to truncate"):
        EpisodeCompiler(BoundaryTokenizer(), allow_preview=True, max_query_tokens=10).compile(example())


def test_collation_masks_padding_and_keeps_causal_reader_start():
    compiler = EpisodeCompiler(BoundaryTokenizer(), allow_preview=True)
    row = example()
    first = compiler.compile(row).responses[0]
    row["probes"][0]["query"] += " 多问一些详细信息。"
    second = compiler.compile(row).responses[0]
    batch = collate_responses([first, second], pad_token_id=0)
    assert not batch["attention_mask"][0, len(first.input_ids):].any()
    assert (batch["labels"][0, len(first.input_ids):] == -100).all()
    assert not batch["memory_read_mask"][0, len(first.input_ids):].any()
    assert not batch["memory_read_mask"][0, :first.retrieval_query_index].any()


def test_tool_results_are_prefix_observations_and_only_assistant_events_are_targets():
    row = json.loads(next(line for line in Path("data/examples/memory_episodes.jsonl").read_text().splitlines()
                         if '"family":"explicit_memory_tools"' in line))
    episode = EpisodeCompiler(BoundaryTokenizer(), allow_preview=True).compile(row)
    assert len(episode.responses) == sum(message["role"] == "assistant" for message in row["trajectory"])
    assert all(response.event_id.startswith("assistant-") for response in episode.responses)
    assert all(all(label == -100 for label in response.labels[:response.prompt_length]) for response in episode.responses)


def test_real_qwen_tokenizer_compiles_all_review_examples():
    directory = Path("artifacts/tokenizer/Qwen3.5-0.8B")
    if not (directory / "tokenizer.json").is_file():
        pytest.skip("local tokenizer-only artifact is not in Git; native check runs in integration environment")
    from transformers import AutoTokenizer
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(directory, local_files_only=True), allow_preview=True)
    rows = [json.loads(line) for line in Path("data/examples/memory_episodes.jsonl").read_text().splitlines()]
    compiled = [compiler.compile(row) for row in rows]
    assert sum(len(row.responses) for row in compiled) == 19
    for row in compiled:
        for response in row.responses:
            assert response.labels[-1] == compiler.tokenizer.eos_token_id
            assert response.retrieval_query_index < response.prompt_length
            assert not response.write_enabled
