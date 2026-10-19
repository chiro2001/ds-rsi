import copy

import pytest
import torch

from dsrsi.data.revisions import AB_DATA_REVISION, data_identity
from dsrsi.training import learning_rate
from dsrsi.training.checkpoint import CheckpointError, CheckpointStore, Progress, capture_training_state
from dsrsi.training.topology import require_equal


def source_run(path, *, mutation=None, status="stopped", accumulation=0):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress = Progress("A", micro_step=12, optimizer_step=1, accumulation_index=accumulation,
                        student_tokens=500, teacher_tokens=30)
    store = CheckpointStore(path / "checkpoints")
    pending = store.begin(progress)
    payload = capture_training_state(model, optimizer, progress=progress,
        data_cursor={"schema_version": 1, "next_ordinal": 12},
        life_state={"mode": "m_only", "streams": {}, "harness_refs": {}})
    payload["rng"]["accelerator"] = {"kind": "cuda", "state": torch.tensor([1, 3], dtype=torch.uint8)}
    if mutation:
        mutation(payload)
    store.save_rank(pending, payload)
    identity = {"base": "fixture", "data": data_identity(AB_DATA_REVISION),
        "cuda_execution": {"version": "cuda-reference-gdn-eager-v1"},
        "execution_semantics": "ab-episode-v2-explicit-optimizer-backend",
        "train_config": {"learning_rate": 0.01, "gradient_accumulation": 12,
                         "student_token_budget": 100000, "data_revision": AB_DATA_REVISION}}
    store.commit(pending, world_size=1, identity=identity, progress=progress, status=status)
    return store


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setattr(learning_rate, "cuda_execution_identity", lambda device:
                        {"version": "cuda-reference-gdn-eager-v1"})


def test_rate_change_preserves_every_other_state_field_and_source(tmp_path):
    source = source_run(tmp_path / "source")
    pointer = (source.root / "latest.json").read_bytes()
    d, m = source.load_committed()
    original = source.load_rank(d, m, rank=0)
    dry = learning_rate.lower_learning_rate(tmp_path / "source", tmp_path / "target", learning_rate=0.005, dry_run=True)
    assert dry["target_learning_rate"] == 0.005 and not (tmp_path / "target").exists()
    report = learning_rate.lower_learning_rate(tmp_path / "source", tmp_path / "target", learning_rate=0.005)
    target = CheckpointStore(tmp_path / "target/checkpoints")
    converted, manifest = target.load_committed()
    actual = target.load_rank(converted, manifest, rank=0)
    assert manifest["identity"]["train_config"]["learning_rate"] == 0.005
    assert actual["optimizer"]["state"]["param_groups"][0]["lr"] == 0.005
    assert manifest["identity"]["train_config"]["student_token_budget"] == 100000
    expected = copy.deepcopy(original)
    expected["optimizer"]["state"]["param_groups"][0]["lr"] = 0.005
    require_equal(expected, actual, "retained training state")
    assert report["all_payload_fields_preserved_except_optimizer_group_lr"]
    assert (source.root / "latest.json").read_bytes() == pointer


@pytest.mark.parametrize("rate", [0, -1, 0.01, 0.02, float("nan"), float("inf"), True])
def test_only_finite_lower_rates_are_allowed(tmp_path, rate):
    source_run(tmp_path / "source")
    with pytest.raises(CheckpointError, match="strictly lower"):
        learning_rate.lower_learning_rate(tmp_path / "source", tmp_path / "target", learning_rate=rate)
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("options", [{"status": "checkpoint"}, {"accumulation": 3},
    {"mutation": lambda p: p["gradients"].update(weight=torch.zeros_like(p["weights"]["weight"]))},
    {"mutation": lambda p: p["life_state"]["streams"].update(active=True)},
    {"mutation": lambda p: p["scheduler"].update({"class": "scheduler", "state": {}})},
    {"mutation": lambda p: p["optimizer"]["state"]["param_groups"][0].update(lr=0.02)},
    {"mutation": lambda p: p["rng"].update(accelerator=None)}])
def test_migration_refuses_pending_or_unsupported_state(tmp_path, options):
    source_run(tmp_path / "source", **options)
    with pytest.raises(CheckpointError):
        learning_rate.lower_learning_rate(tmp_path / "source", tmp_path / "target", learning_rate=0.005)
    assert not (tmp_path / "target").exists()
