"""Behavioral recovery tests: resumed optimization must match uninterrupted work."""

import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from dsrsi.training.checkpoint import (
    CheckpointError, CheckpointStore, Progress, capture_training_state,
    restore_training_state,
)

IDENTITY = {"base": "frozen-fixture-v1", "config": "accum4", "data": "seeded-episodes-v1"}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        with torch.no_grad():
            self.base.weight.fill_(0.2)
            self.base.bias.fill_(0.1)
        self.base.requires_grad_(False)
        self.adapter = nn.Linear(4, 2)
        self.dropout = nn.Dropout(0.3)
        self.register_buffer("observations", torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        self.observations.add_(1)
        return self.adapter(self.dropout(self.base(x)))


def build():
    model = Toy()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    return model, optimizer, scheduler


def micros(model, optimizer, scheduler, progress, cursor, life, count):
    losses = []
    samples = []
    for _ in range(count):
        x = torch.randn(3, 4) * float(np.random.uniform(0.5, 1.5))
        target = torch.randn(3, 2) + random.random()
        samples.append((cursor["next_episode"], x.clone(), target.clone()))
        loss = (model(x) - target).square().mean()
        (loss / 4).backward()
        losses.append(float(loss.detach()))
        accumulation = progress.accumulation_index + 1
        step = progress.optimizer_step
        if accumulation == 4:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation = 0
            step += 1
        progress = replace(progress, optimizer_step=step, micro_step=progress.micro_step + 1,
                           accumulation_index=accumulation, student_tokens=progress.student_tokens + 37)
        cursor["next_episode"] += 1
        life["clock"] += 37
        life["recurrent"] = life["recurrent"] + x.sum(0)
        life["memory"]["r0"] = target.detach().clone()
    return progress, losses, samples


def assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def fresh_state():
    return Progress("A"), {"next_episode": 0}, {"clock": 0, "recurrent": torch.zeros(4), "memory": {}}


def save(store, model, optimizer, progress, cursor, life, scheduler=None, world=1):
    pending = store.begin(progress)
    for rank in range(world):
        store.save_rank(pending, capture_training_state(
            model, optimizer, progress=progress, data_cursor={**cursor, "rank": rank},
            life_state=life, scheduler=scheduler, rank=rank, world_size=world,
        ))
    return store.commit(pending, world_size=world, identity=IDENTITY, progress=progress, status="paused")


def test_resume_mid_accumulation_matches_uninterrupted_training(tmp_path):
    seed(73)
    full_model, full_opt, full_schedule = build()
    full_progress, full_cursor, full_life = fresh_state()
    full_progress, full_losses, full_samples = micros(
        full_model, full_opt, full_schedule, full_progress, full_cursor, full_life, 11)

    seed(73)
    model, opt, schedule = build()
    progress, cursor, life = fresh_state()
    progress, first_losses, first_samples = micros(model, opt, schedule, progress, cursor, life, 6)
    assert progress.accumulation_index == 2
    assert any(p.grad is not None for p in model.adapter.parameters())
    store = CheckpointStore(tmp_path)
    save(store, model, opt, progress, cursor, life, schedule)
    payload, manifest = store.load_latest(identity=IDENTITY)
    assert manifest["status"] == "paused"
    assert not any(name.startswith("base.") for name in payload["weights"])
    assert all(p.device.type == "cpu" and not p.requires_grad for p in payload["weights"].values())

    # New process construction and unrelated random work must not change continuation.
    seed(909)
    resumed, resumed_opt, resumed_schedule = build()
    progress, cursor, life = restore_training_state(
        payload, resumed, resumed_opt, scheduler=resumed_schedule)
    progress, last_losses, last_samples = micros(
        resumed, resumed_opt, resumed_schedule, progress, cursor, life, 5)

    assert first_losses + last_losses == full_losses
    for got, want in zip(first_samples + last_samples, full_samples):
        assert_tree_equal(got, want)
    assert progress == full_progress
    cursor.pop("rank")
    assert_tree_equal(cursor, full_cursor)
    assert_tree_equal(life, full_life)
    assert_tree_equal(resumed.state_dict(), full_model.state_dict())
    assert_tree_equal(resumed_opt.state_dict(), full_opt.state_dict())
    assert_tree_equal(resumed_schedule.state_dict(), full_schedule.state_dict())
    for p, q in zip(resumed.adapter.parameters(), full_model.adapter.parameters()):
        assert_tree_equal(p.grad, q.grad)


def test_partial_multirank_save_never_replaces_latest(tmp_path):
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    store = CheckpointStore(tmp_path)
    first = save(store, model, opt, progress, cursor, life)
    latest = (tmp_path / "latest.json").read_bytes()
    later = replace(progress, micro_step=1)
    pending = store.begin(later)
    store.save_rank(pending, capture_training_state(
        model, opt, progress=later, data_cursor=cursor, life_state=life, world_size=2))
    with pytest.raises(CheckpointError, match="rank 1"):
        store.commit(pending, world_size=2, identity=IDENTITY, progress=later)
    assert (tmp_path / "latest.json").read_bytes() == latest
    assert first.is_dir()


def test_all_rank_inventory_is_validated_before_resume(tmp_path):
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    store = CheckpointStore(tmp_path)
    directory = save(store, model, opt, progress, cursor, life, world=2)
    (directory / "rank_00001.pt").unlink()
    with pytest.raises(CheckpointError, match="rank 1"):
        store.load_latest(identity=IDENTITY, world_size=2, rank=0)


def test_corruption_and_identity_changes_are_rejected(tmp_path):
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    store = CheckpointStore(tmp_path)
    directory = save(store, model, opt, progress, cursor, life)
    with pytest.raises(CheckpointError, match="identity"):
        store.load_latest(identity={**IDENTITY, "data": "changed"})
    with pytest.raises(CheckpointError, match="world_size"):
        store.load_latest(identity=IDENTITY, world_size=2)
    path = directory / "rank_00000.pt"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(CheckpointError, match="checksum"):
        store.load_latest(identity=IDENTITY)


def test_mismatched_optimizer_is_rejected_before_model_mutation():
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    payload = capture_training_state(model, opt, progress=progress, data_cursor=cursor, life_state=life)
    other, _, _ = build()
    before = copy.deepcopy(other.state_dict())
    other_opt = torch.optim.SGD(other.adapter.parameters(), lr=0.1)
    with pytest.raises(CheckpointError, match="optimizer class"):
        restore_training_state(payload, other, other_opt)
    assert_tree_equal(other.state_dict(), before)


def test_optimizer_parameter_order_is_not_silently_changed():
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    payload = capture_training_state(model, opt, progress=progress, data_cursor=cursor, life_state=life)
    other, _, _ = build()
    wrong = torch.optim.AdamW(list(other.adapter.parameters())[::-1], lr=0.01)
    with pytest.raises(CheckpointError, match="order"):
        restore_training_state(payload, other, wrong)


def test_completed_requires_actual_budget(tmp_path):
    model, opt, _ = build()
    progress, cursor, life = fresh_state()
    store = CheckpointStore(tmp_path)
    pending = store.begin(progress)
    store.save_rank(pending, capture_training_state(
        model, opt, progress=progress, data_cursor=cursor, life_state=life))
    with pytest.raises(CheckpointError, match="budget"):
        store.commit(pending, world_size=1, identity=IDENTITY, progress=progress,
                     status="completed", required_student_tokens=100)
    assert not (tmp_path / "latest.json").exists()


def test_sparse_accumulated_gradients_survive_checkpoint(tmp_path):
    model = nn.Embedding(9, 3, sparse=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.tensor([1, 1, 4])).sum().backward()
    want = model.weight.grad.to_dense().clone()
    progress = Progress("A", micro_step=1, accumulation_index=1)
    store = CheckpointStore(tmp_path)
    save(store, model, opt, progress, {}, {})
    payload, _ = store.load_latest(identity=IDENTITY)
    other = nn.Embedding(9, 3, sparse=True)
    other_opt = torch.optim.SGD(other.parameters(), lr=0.1)
    restore_training_state(payload, other, other_opt)
    assert other.weight.grad.is_sparse
    assert torch.equal(other.weight.grad.to_dense(), want)
