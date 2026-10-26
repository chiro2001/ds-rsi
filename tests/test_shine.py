import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from test_qwen_bridge import tiny
from dsrsi.model.shine import DynamicDown, GeneratedMemory, ShineConfig, ShineQwen, AxialBlock
from dsrsi.data.shine import shine_episode
from dsrsi.training.checkpoint import Progress, capture_training_state, restore_training_state
from dsrsi.training.topology import require_equal
from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.training.shine_pilot import objective, episode_token_cost, evaluate_fixture, fixture_rows
from dsrsi.training.diagnostic_budget import DiagnosticBudget, DiagnosticBudgetExceeded
from dsrsi.evaluation.journal import EvaluationJournal, EvaluationPaused


def model():
    return ShineQwen(tiny(), ShineConfig(layers=(0, 3), rank=2, width=8, write_rank=2))


def test_disabled_and_zero_generated_adapter_equal_original_base():
    torch.manual_seed(5)
    language = tiny()
    ids = torch.tensor([[2, 3, 4, 5]])
    before = language(ids, memory_enabled=False).hidden.detach()
    m = ShineQwen(language, ShineConfig(layers=(0, 3), rank=2, width=8, write_rank=2))
    memory = m.encode(ids, record_id="r1", life_id="life")
    assert torch.equal(before, m(ids)[0])
    assert torch.equal(before, m(ids, memories=[memory], life_id="life")[0])
    assert all(not p.requires_grad for n, p in m.named_parameters() if "original." in n)


def test_answer_gradient_reaches_m2p_memory_queries_and_shared_writer():
    torch.manual_seed(3)
    m = model()
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
    source = torch.tensor([[2, 3, 4]])
    query = torch.tensor([[5, 6, 7, 1]])
    labels = torch.tensor([[-100, -100, 7, 1]])
    for step in range(2):
        memory = m.encode(source, record_id="r", life_id="life")
        m(query, memories=[memory], life_id="life", labels=labels)[1].mean().backward()
        if step:
            assert m.memory_queries.grad.abs().sum() > 0
            assert m.project.weight.grad.abs().sum() > 0
            assert sum(s.write_b.grad.abs().sum() for s in m._sites.values()) > 0
            assert sum(p.grad.abs().sum() for p in m.m2p.parameters()) > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for n, p in m.named_parameters() if not p.requires_grad)


def test_source_encoding_is_independent_of_active_read_memory_and_scope_is_exception_safe():
    m = model()
    with torch.no_grad():
        m.b_head.weight.normal_(std=0.01)
    ids = torch.tensor([[2, 3, 4]])
    a = m.encode(ids, record_id="a", life_id="one")
    with m._scope("read", [a], "one"):
        b = m.encode(ids, record_id="b", life_id="two")
        assert all(s.mode == "read" for s in m._sites.values())
    for index in m.config.layers:
        for x, y in zip(a.factors[index], b.factors[index]):
            assert torch.equal(x, y)
    with pytest.raises(ValueError, match="cross-life"):
        m(ids, memories=[a], life_id="other")
    with pytest.raises(RuntimeError):
        with m._scope("read", [a], "one"):
            raise RuntimeError("test")
    assert all(s.mode == "off" and not s.active for s in m._sites.values())
    saved = GeneratedMemory.from_state(a.state(), "cpu")
    assert saved.payload_bytes() == a.payload_bytes()
    assert torch.equal(m(ids, memories=[a], life_id="one")[0], m(ids, memories=[saved], life_id="one")[0])


def test_composition_is_sum_of_independent_updates_without_factor_cross_terms():
    original = torch.nn.Linear(5, 3, bias=False)
    site = DynamicDown(original, 2, 1.0)
    pairs = tuple((torch.randn(2, 5, requires_grad=True), torch.randn(3, 2, requires_grad=True)) for _ in range(2))
    x = torch.randn(1, 4, 5)
    site.mode, site.active = "read", pairs
    want = original(x) + sum(F.linear(F.linear(x, a), b) for a, b in pairs)
    torch.testing.assert_close(site(x), want, rtol=0, atol=0)
    site(x).square().mean().backward()
    assert all(t.grad.abs().sum() > 0 for pair in pairs for t in pair)


def test_nonempty_accumulation_full_state_resume_matches_continuous_next_step():
    torch.manual_seed(7)
    m = model()
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.001)
    def step():
        memory = m.encode(torch.tensor([[2, 3]]), record_id="r", life_id="life")
        m(torch.tensor([[4, 5, 6, 1]]), memories=[memory], life_id="life",
          labels=torch.tensor([[-100, -100, 6, 1]]))[1].mean().backward()
    step()
    p = capture_training_state(m, optimizer, progress=Progress("SHINE_diagnostic", micro_step=1, accumulation_index=1),
                               data_cursor={"next_ordinal": 1}, life_state={})
    step()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    continuous = capture_training_state(m, optimizer, progress=Progress("SHINE_diagnostic", micro_step=2),
                                        data_cursor={"next_ordinal": 2}, life_state={})
    restore_training_state(p, m, optimizer)
    step()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    resumed = capture_training_state(m, optimizer, progress=Progress("SHINE_diagnostic", micro_step=2),
                                     data_cursor={"next_ordinal": 2}, life_state={})
    require_equal(continuous, resumed, "SHINE accumulated resume")


@pytest.mark.parametrize("task,count", [("U", 1), ("P", 1), ("C", 2)])
def test_packaging_variants_change_only_observed_content_not_questions_or_activity(task, count):
    a, b = [shine_episode(1, variant=v, task=task) for v in (0, 1)]
    assert len(a["source_events"]) == count and len(a["probes"]) == 2
    assert a == shine_episode(1, task=task)
    assert [p["query"] for p in a["probes"]] == [p["query"] for p in b["probes"]]
    assert a["annotations"]["fixed_memory_candidate_ids"] == b["annotations"]["fixed_memory_candidate_ids"]
    assert all(x["target"] != y["target"] for x, y in zip(a["probes"], b["probes"]))
    support = {e["event_id"] for e in a["source_events"]}
    assert all(set(sum(p["support_groups"], [])) <= support for p in a["probes"])


class SmallTokenizer:
    eos_token_id=1
    pad_token_id=0
    def apply_chat_template(self,messages,*,tokenize=False,add_generation_prompt=False,**kwargs):
        return "".join(m["role"]+":"+m["content"]+"<end>" for m in messages)+("assistant:" if add_generation_prompt else "")
    def encode(self,text,**kwargs):
        parts=text.split("<end>")
        output=[]
        for i,part in enumerate(parts):
            output.extend(2+ord(c)%29 for c in part)
            if i<len(parts)-1:output.append(1)
        return output
    def decode(self,ids,**kwargs):return "".join(str(x) for x in ids)


@pytest.mark.parametrize("task",["U","P","C"])
def test_actual_objective_counterfactual_cost_and_source_gradient(task):
    m=model()
    with torch.no_grad():m.b_head.weight.normal_(std=.01)
    compiler=EpisodeCompiler(SmallTokenizer())
    a,b=[shine_episode(1,variant=v,task=task) for v in (0,1)]
    loss,cost,stats=objective(m,compiler,a,counterfactual=b,cf_weight=1.)
    assert cost==episode_token_cost(m,compiler,a,1.)
    assert stats['queries']==2 and torch.isfinite(loss)
    loss.backward()
    assert m.memory_queries.grad.abs().sum()>0
    assert all(s.mode=='off' and not s.active for s in m._sites.values())


def test_shared_ledger_admission_accounts_failed_reserved_work_and_actual_settlement(tmp_path):
    ledger=DiagnosticBudget(tmp_path/'ledger.json',limit=10)
    a=ledger.reserve(run='a',kind='train',student=8,teacher=4)
    with pytest.raises(DiagnosticBudgetExceeded):ledger.reserve(run='b',kind='train',student=3)
    ledger.commit(a,student=5,teacher=2)
    b=ledger.reserve(run='b',kind='failed_work',student=5)
    assert ledger.read()['entries'][-1]['status']=='reserved'
    with pytest.raises(DiagnosticBudgetExceeded):ledger.reserve(run='c',kind='train',student=1)
    with pytest.raises(ValueError):ledger.commit(b,student=6)
    with pytest.raises(ValueError):DiagnosticBudget(tmp_path/'ledger.json',limit=11)


def test_plaintext_eval_does_not_encode_unused_student_memories_and_resumes_journal(tmp_path,monkeypatch):
    m=model()
    def no_source(*args,**kwargs):raise AssertionError('plaintext evaluation encoded student memory')
    monkeypatch.setattr(m,'encode',no_source)
    monkeypatch.setattr(m,'generate',lambda prompt_ids,**kwargs:dict(token_ids=[],stopped_on_end=True,input_tokens=len(prompt_ids)))
    rows=fixture_rows(worlds=1,task='U',seed=20261026)
    compiler=EpisodeCompiler(SmallTokenizer())
    p=tmp_path/'eval.journal.json'
    count=0
    def pause():
        nonlocal count
        count+=1
        return count==3
    rng=torch.get_rng_state().clone()
    with pytest.raises(EvaluationPaused):
        evaluate_fixture(m,compiler,rows,views=('plaintext','off'),journal_path=p,identity={'id':1},pause_check=pause)
    assert EvaluationJournal.read(p)['status']=='paused'
    assert torch.equal(torch.get_rng_state(),rng)
    result=evaluate_fixture(m,compiler,rows,views=('plaintext','off'),journal_path=p,identity={'id':1},resume=True)
    assert result['source_encoding_input_tokens']==result['student_input_tokens']==0
    assert result['teacher_input_tokens']>0 and EvaluationJournal.read(p)['status']=='completed'


def test_m2p_has_cross_layer_and_cross_memory_message_paths_and_sufficient_output_width():
    m=model()
    assert m.memory_tokens*m.base.config.hidden_size>=m.config.rank*(m.input_width+m.output_width)
    block=AxialBlock(8)
    states=torch.randn(2,6,8,requires_grad=True)
    block(states)[0,0].square().sum().backward()
    assert states.grad[1].abs().sum()>0
    assert states.grad[0,1:].abs().sum()>0
