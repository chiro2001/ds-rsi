import pytest

from dsrsi.evaluation.pinned import load_pinned
from dsrsi.training.checkpoint import CheckpointError, CheckpointStore, Progress
from test_recovery import build, fresh_state, save


def committed_generation(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    model, optimizer, scheduler = build()
    progress, cursor, life = fresh_state()
    save(store, model, optimizer, progress, cursor, life, scheduler)
    return store, *store.load_committed()


def test_pinned_load_matches_latest_and_verifies_every_rank_file(tmp_path):
    store, directory, manifest = committed_generation(tmp_path)
    assert load_pinned(store, None) == (directory, manifest)
    assert load_pinned(store, directory.name) == (directory, manifest)
    rank = directory / "rank_00000.pt"
    payload = rank.read_bytes()
    rank.write_bytes(bytes([payload[0] ^ 0xFF]) + payload[1:])
    with pytest.raises(CheckpointError, match="payload checksum mismatch"):
        load_pinned(store, directory.name)
    rank.write_bytes(payload)
    assert load_pinned(store, directory.name) == (directory, manifest)


def test_pinned_load_rejects_unknown_pending_and_escaping_names(tmp_path):
    store, directory, _ = committed_generation(tmp_path)
    with pytest.raises(CheckpointError, match="committed generation directory"):
        load_pinned(store, "micro_000000000000_missing")
    pending = store.begin(Progress("A"))
    with pytest.raises(CheckpointError, match="committed generation directory"):
        load_pinned(store, pending.name)
    pending.rmdir()
    with pytest.raises(CheckpointError, match="direct child"):
        load_pinned(store, f"../{directory.name}")
