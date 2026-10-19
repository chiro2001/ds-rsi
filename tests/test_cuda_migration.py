import copy
import json

import pytest
import torch

from dsrsi.data.revisions import AB_DATA_REVISION, data_identity
from dsrsi.training import hardware
from dsrsi.training.checkpoint import CheckpointError, CheckpointStore, Progress, capture_training_state
from dsrsi.training.topology import require_equal


def source_run(path, *, mutation=None, status="stopped", accumulation=0, stage="A"):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress = Progress(stage, micro_step=6, optimizer_step=1, student_tokens=300,
                        teacher_tokens=40, accumulation_index=accumulation)
    identity = {"base": "fixture", "data": data_identity(AB_DATA_REVISION),
                "execution_semantics": "ab-episode-v4-query-weighted-coordinated-gradients",
                "train_config": {"stage": stage, "data_revision": AB_DATA_REVISION, "gradient_accumulation": 6}}
    store = CheckpointStore(path / "checkpoints")
    pending = store.begin(progress)
    for rank in range(2):
        torch.manual_seed(10 + rank)
        payload = capture_training_state(model, optimizer, progress=progress, rank=rank, world_size=2,
            data_cursor={"schema_version": 1, "next_ordinal": 12 + rank},
            life_state={"mode": "m_only", "streams": {}, "harness_refs": {}})
        payload["rng"]["accelerator"] = {"kind": "npu", "state": torch.tensor([rank], dtype=torch.uint8)}
        if mutation:
            mutation(payload)
        store.save_rank(pending, payload)
    store.commit(pending, world_size=2, identity=identity, progress=progress, status=status)
    return store


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_execution_identity", lambda device: {"version": "cuda-reference-fixture"})
    monkeypatch.setattr(hardware, "new_cuda_rng", lambda device, seed:
                        {"kind": "cuda", "state": torch.tensor([7, 9], dtype=torch.uint8)})


def test_cuda_migration_preserves_all_portable_state_and_source(runtime, tmp_path):
    source = source_run(tmp_path / "source")
    old_pointer = (source.root / "latest.json").read_bytes()
    directory, manifest = source.load_committed()
    expected = source.load_rank(directory, manifest, rank=0)
    dry = hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target", dry_run=True)
    assert dry["target_accumulation"] == 12
    assert not (tmp_path / "target").exists()
    result = hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target", gradient_accumulation=12)
    target = CheckpointStore(tmp_path / "target/checkpoints")
    new_dir, got = target.load_committed()
    payload = target.load_rank(new_dir, got, rank=0)
    assert got["progress"] == manifest["progress"]
    assert got["identity"]["train_config"]["gradient_accumulation"] == 12
    assert got["identity"]["execution_semantics"] == "ab-episode-v2-explicit-optimizer-backend"
    assert payload["rng"]["accelerator"]["kind"] == "cuda"
    assert result["same_trajectory_as_source_claimed"] is False
    assert result["inactive_source_rng_ranks"] == [1]
    expected["world_size"] = 1
    expected["rng"]["accelerator"] = copy.deepcopy(payload["rng"]["accelerator"])
    require_equal(expected, payload, "converted")
    assert (source.root / "latest.json").read_bytes() == old_pointer
    assert source.load_rank(directory, manifest, rank=1)["rng"]["accelerator"]["kind"] == "npu"


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["data_cursor"].update(next_ordinal=100), "contiguous"),
    (lambda p: p["gradients"].update(weight=torch.zeros_like(p["weights"]["weight"])), "pending gradients"),
    (lambda p: p["life_state"]["streams"].update(active=True), "persistent life"),
    (lambda p: p["rng"].update(accelerator=None), "NPU RNG"),
    (lambda p: p["weights"]["weight"].add_(p["rank"]), "state differs"),
])
def test_cuda_migration_rejects_state_loss(runtime, tmp_path, mutation, error):
    source_run(tmp_path / "source", mutation=mutation)
    with pytest.raises(CheckpointError, match=error):
        hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target")
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("options", [{"status": "checkpoint"}, {"accumulation": 1}, {"stage": "C_context"}])
def test_cuda_migration_requires_a_stopped_stateless_boundary(runtime, tmp_path, options):
    source_run(tmp_path / "source", **options)
    with pytest.raises(CheckpointError):
        hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target")


@pytest.mark.parametrize("accumulation", [6, 0, 12.0, True])
def test_cuda_migration_requires_preserved_global_batch(runtime, tmp_path, accumulation):
    source_run(tmp_path / "source")
    with pytest.raises(CheckpointError, match="global episode batch"):
        hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target", gradient_accumulation=accumulation)


def test_cuda_migration_report_is_carried_through_resume_identity(runtime, tmp_path):
    source_run(tmp_path / "source")
    hardware.migrate_ab_to_cuda(tmp_path / "source", tmp_path / "target")
    report = json.loads((tmp_path / "target/hardware-migration.json").read_text())
    assert report["all_payload_fields_preserved_except_world_size_and_accelerator_rng"] is True
    store = CheckpointStore(tmp_path / "target/checkpoints")
    _, manifest = store.load_committed()
    assert manifest["identity"]["hardware_migrations"][0]["cuda_seed"] == 20261010
