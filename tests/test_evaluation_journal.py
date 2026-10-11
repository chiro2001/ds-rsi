import json

import pytest

from dsrsi.evaluation.journal import EvaluationJournal, EvaluationPaused


def test_evaluation_recovery_reuses_committed_results_and_matches_uninterrupted(tmp_path):
    jobs = ["nll/en/a", "nll/zh/b", "probe/en/c"]
    identity = {"checkpoint": "fixed", "corpus": "hash", "config": {"tokens": 16}}
    full = EvaluationJournal(tmp_path / "full.json", identity, jobs)
    for number, job in enumerate(jobs):
        full.evaluate(job, lambda: {"tokens": 16, "nll": number / 10})
    full.finish()
    part = EvaluationJournal(tmp_path / "part.json", identity, jobs, stop_after_items=1)
    with pytest.raises(EvaluationPaused, match="1/3"):
        part.evaluate(jobs[0], lambda: {"tokens": 16, "nll": 0.0})
    assert EvaluationJournal.read(part.path)["status"] == "paused"
    resumed = EvaluationJournal(part.path, identity, jobs, resume=True)
    calls = []
    for number, job in enumerate(jobs):
        def compute():
            calls.append(job)
            return {"tokens": 16, "nll": number / 10}
        resumed.evaluate(job, compute)
    resumed.finish()
    assert calls == jobs[1:]
    assert EvaluationJournal.read(part.path) == EvaluationJournal.read(full.path)


def test_evaluation_journal_rejects_checkpoint_config_corpus_and_plan_changes(tmp_path):
    path = tmp_path / "journal.json"
    identity = {"checkpoint": "fixed", "corpus": "abc", "config": {"tokens": 16}}
    EvaluationJournal(path, identity, ["one", "two"])
    with pytest.raises(ValueError, match="explicit --resume"):
        EvaluationJournal(path, identity, ["one", "two"])
    for changed in ({**identity, "checkpoint": "later"}, {**identity, "corpus": "xyz"},
                    {**identity, "config": {"tokens": 32}}):
        with pytest.raises(ValueError, match="identity/job plan"):
            EvaluationJournal(path, changed, ["one", "two"], resume=True)
    with pytest.raises(ValueError, match="identity/job plan"):
        EvaluationJournal(path, identity, ["two", "one"], resume=True)
    state = json.loads(path.read_text())
    state["status"] = "completed"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checksum"):
        EvaluationJournal(path, identity, ["one", "two"], resume=True)


def test_failed_item_is_replayed_and_partial_evaluation_cannot_complete(tmp_path):
    path = tmp_path / "journal.json"
    journal = EvaluationJournal(path, {"checkpoint": "fixed"}, ["one", "two"])
    journal.evaluate("one", lambda: 1)
    def fail():
        raise RuntimeError("interrupted compute")
    with pytest.raises(RuntimeError):
        journal.evaluate("two", fail)
    resumed = EvaluationJournal(path, {"checkpoint": "fixed"}, ["one", "two"], resume=True)
    with pytest.raises(ValueError, match="partial"):
        resumed.finish()
    assert resumed.evaluate("one", lambda: pytest.fail("committed item recomputed")) == 1
    assert resumed.evaluate("two", lambda: 2) == 2
    resumed.finish()


def test_signal_intent_saves_current_item_then_pauses(tmp_path):
    journal = EvaluationJournal(tmp_path / "journal.json", {}, ["one", "two"])
    def compute():
        journal.pause_requested = True
        return {"answer": "saved"}
    with pytest.raises(EvaluationPaused, match="1/2"):
        journal.evaluate("one", compute)
    state = EvaluationJournal.read(journal.path)
    assert state["results"] == [{"key": "one", "value": {"answer": "saved"}}]
    assert state["status"] == "paused"


def test_nll_results_are_identical_after_journal_recovery(tmp_path):
    import torch
    from dsrsi.evaluation.language_nll import ScoringSequence, score_sequence
    from test_qwen_bridge import tiny
    torch.set_num_threads(2)
    torch.manual_seed(5)
    model = tiny().eval()
    sequence = ScoringSequence([4, 5, 6, 7], [-100, 5, 6, 7], 0)
    reference = score_sequence(model, sequence, mode="off")
    path = tmp_path / "journal.json"
    part = EvaluationJournal(path, {}, ["base", "off"], stop_after_items=1)
    with pytest.raises(EvaluationPaused):
        part.evaluate("base", lambda: score_sequence(model, sequence, mode="base"))
    resumed = EvaluationJournal(path, {}, ["base", "off"], resume=True)
    assert resumed.evaluate("base", lambda: pytest.fail("must reuse committed item")) == reference
    assert resumed.evaluate("off", lambda: score_sequence(model, sequence, mode="off")) == reference
    resumed.finish()
