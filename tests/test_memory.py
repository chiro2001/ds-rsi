from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from dsrsi.memory import AddressEncoder, BudgetExceeded, Consolidator, MemoryConfig, MemoryReader, MemoryRecord, MemoryStore, MemoryWriter
from dsrsi.memory.store import pack_records
from dsrsi.training.losses import required_support_loss, routing_scores


def config():
    return MemoryConfig(hidden_size=16, memory_dim=8, slots=3, heads=2, intermediate=16,
                        key_dim=4, keys=2, summary_slots=4)


def record(name, *, stream="life-a", begin=0, end=3, slots=None, keys=None):
    return MemoryRecord(name, stream, torch.randn(2, 8) if slots is None else slots,
                        torch.randn(2, 4) if keys is None else keys, begin, end, end + 1)


def test_future_answer_gradient_reaches_writer_through_padded_record_batch():
    torch.manual_seed(4)
    c = config()
    writer, reader, address = MemoryWriter(c), MemoryReader(c), AddressEncoder(c)
    source = torch.randn(1, 7, c.hidden_size)
    slots = writer(source)
    keys = address.encode_keys(slots)
    r = record("source", slots=slots[0], keys=keys[0])
    packed, valid, _ = pack_records([[r], []], memory_dim=c.memory_dim,
                                    query_positions=[8, 8], stream_ids=["life-a", "life-a"],
                                    device="cpu", dtype=torch.float32)
    query = torch.randn(2, 2, c.hidden_size)
    out = reader(query, packed, valid)
    frozen_decoder = torch.randn(c.hidden_size, 5)
    loss = F.cross_entropy(out[:, -1] @ frozen_decoder, torch.tensor([1, 3]))
    loss.backward()
    assert sum(float(p.grad.square().sum()) for p in writer.parameters() if p.grad is not None) > 0
    assert sum(float(p.grad.square().sum()) for p in reader.parameters() if p.grad is not None) > 0
    assert torch.equal(out[1], query[1])  # all-masked row is an exact bypass


def test_writer_padding_isolation_and_no_empty_writes():
    writer = MemoryWriter(config())
    source = torch.randn(2, 5, 16)
    want = writer(source)
    padded = F.pad(source, (0, 0, 0, 3), value=float("nan"))
    valid = torch.arange(8)[None].expand(2, -1) < 5
    torch.testing.assert_close(writer(padded, valid), want)
    with pytest.raises(ValueError, match="nonempty"):
        writer(source, torch.zeros(2, 5, dtype=torch.bool))


def test_reader_cannot_use_future_slots_at_earlier_query():
    reader = MemoryReader(config())
    hidden = torch.randn(1, 2, 16)
    slots = torch.randn(1, 4, 8, requires_grad=True)
    valid = torch.tensor([[[True, True, False, False], [True, True, True, True]]])
    got = reader(hidden, slots, valid)
    changed = slots.detach().clone()
    changed[:, 2:] += 1000
    other = reader(hidden, changed, valid)
    assert torch.equal(got[:, 0], other[:, 0])
    got[:, 0].sum().backward()
    assert slots.grad[:, 2:].abs().sum() == 0


def test_empty_disabled_and_all_masked_reader_are_exact_identity():
    reader = MemoryReader(config())
    hidden = torch.randn(2, 4, 16)
    assert reader(hidden, None) is hidden
    assert reader(hidden, torch.empty(2, 0, 8)) is hidden
    assert reader(hidden, torch.randn(2, 3, 8), enabled=False) is hidden
    out = reader(hidden, torch.full((2, 3, 8), float("nan")), torch.zeros(2, 3, dtype=torch.bool))
    assert torch.equal(out, hidden)


def test_routing_updates_address_heads_without_changing_writer_or_base_features():
    c = config()
    writer, address = MemoryWriter(c), AddressEncoder(c)
    source = torch.randn(2, 5, 16, requires_grad=True)
    slots = writer(source)
    q_source = torch.randn(1, 16, requires_grad=True)
    q = address.encode_query(q_source)
    keys = address.encode_keys(slots)
    scores = routing_scores(q, keys)
    loss = required_support_loss(scores, torch.tensor([[[True, False]]]),
                                 torch.ones(1, 2, dtype=torch.bool)).mean()
    loss.backward()
    assert source.grad is None and q_source.grad is None
    assert all(p.grad is None for p in writer.parameters())
    assert sum(float(p.grad.square().sum()) for p in address.parameters() if p.grad is not None) > 0


def test_bf16_autocast_memory_path_is_finite_and_trainable():
    c = config()
    writer, reader, address = MemoryWriter(c), MemoryReader(c), AddressEncoder(c)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = writer(torch.randn(2, 8, 16))
        k = address.encode_keys(z)
        out = reader(torch.randn(2, 3, 16), z)
        loss = out.square().mean() + k.sum()
    assert torch.isfinite(out).all() and torch.isfinite(k).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in writer.parameters() if p.grad is not None)


def test_store_enforces_time_stream_and_codec():
    store = MemoryStore(memory_dim=8, key_dim=4)
    past = record("past")
    future = record("future", begin=10, end=13)
    foreign = record("other", stream="life-b")
    for r in [past, future, foreign]:
        store.append(r)
    assert store.retrieve(torch.randn(4), stream_id="life-a", query_position=4) == [past]
    assert store.retrieve(torch.randn(4), stream_id="life-a", query_position=3) == []
    with pytest.raises(ValueError, match="codec"):
        store.append(replace(record("new"), codec_version="incompatible"))
    with pytest.raises(ValueError, match="another stream"):
        pack_records([[foreign]], memory_dim=8, query_positions=[20], stream_ids=["life-a"],
                     device="cpu", dtype=torch.float32)


def test_compaction_is_transactional_and_releases_leaf_payloads():
    store = MemoryStore(memory_dim=8, key_dim=4, max_tensor_bytes=200)
    a, b = record("a"), record("b", begin=4, end=7)
    store.append(a)
    store.append(b)
    with pytest.raises(BudgetExceeded):
        store.append(record("overflow"))
    parent = replace(record("parent", begin=0, end=7), level=1, child_refs=("a", "b"), source_kind="summary")
    with pytest.raises(ValueError, match="source range"):
        store.compact(replace(parent, source_begin=4), ["a", "b"])
    assert set(store.records) == {"a", "b"}
    store.compact(parent, ["a", "b"])
    assert set(store.records) == {"parent"}
    assert store.tensor_bytes == parent.tensor_bytes


def test_snapshot_preserves_retrieval_ties_and_has_independent_numeric_state():
    store = MemoryStore(memory_dim=8, key_dim=4)
    key = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])
    for name in ["first", "second"]:
        store.append(record(name, keys=key.clone(), slots=torch.randn(2, 8, requires_grad=True) * 2))
    state = store.state_dict()
    restored = MemoryStore.from_state_dict(state)
    query = torch.tensor([1., 0, 0, 0])
    assert [r.record_id for r in restored.retrieve(query, stream_id="life-a", query_position=4)] == ["first", "second"]
    assert not restored.records["first"].slots.requires_grad
    restored.records["first"].slots.zero_()
    assert state["records"][0]["slots"].abs().sum() > 0


def test_wall_time_metadata_reaches_reader_features():
    r = replace(record("time"), wall_time=1000.0)
    _, _, features = pack_records([[r]], memory_dim=8, query_positions=[9], stream_ids=["life-a"],
                                  device="cpu", dtype=torch.float32, wall_times=[4600.0])
    assert (features[..., 1] > 0).all()
    assert (features[..., 6] == 1).all()


def test_consolidator_receives_future_reading_gradient():
    c = config()
    consolidate, reader = Consolidator(c), MemoryReader(c)
    children = torch.randn(2, 12, c.memory_dim)
    summary = consolidate(children)
    answer = reader(torch.randn(2, 1, c.hidden_size), summary)
    answer.square().mean().backward()
    assert sum(float(p.grad.square().sum()) for p in consolidate.parameters() if p.grad is not None) > 0


def test_modules_learn_to_store_and_read_unseen_random_bits():
    # A controlled module test, not evidence of Qwen language-model performance.
    torch.manual_seed(42)
    c = config()
    writer, reader = MemoryWriter(c), MemoryReader(c)
    optimizer = torch.optim.AdamW(list(writer.parameters()) + list(reader.parameters()), lr=0.02)
    query = torch.ones(32, 1, c.hidden_size)
    query[..., 0] = 0
    decoder = torch.zeros(c.hidden_size, 2)
    decoder[0] = torch.tensor([-1., 1.])
    for _ in range(100):
        bits = torch.randint(2, (32,))
        source = torch.zeros(32, 4, c.hidden_size)
        source[..., 0] = (2 * bits - 1)[:, None]
        out = reader(query, writer(source))[:, 0] @ decoder
        loss = F.cross_entropy(out, bits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bits = torch.randint(2, (128,))
    source = torch.zeros(128, 4, c.hidden_size)
    source[..., 0] = (2 * bits - 1)[:, None]
    q = query[:1].expand(128, -1, -1)
    with torch.no_grad():
        slots = writer(source)
        correct = (reader(q, slots)[:, 0] @ decoder).argmax(-1)
        shuffled = (reader(q, slots.roll(1, 0))[:, 0] @ decoder).argmax(-1)
    assert (correct == bits).float().mean() > 0.95
    assert (shuffled == bits).float().mean() < 0.7
