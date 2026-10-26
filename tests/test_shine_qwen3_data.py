import pytest

from dsrsi.data.shine_qwen3 import labels_for,token_cost
from dsrsi.training.diagnostic_budget import DiagnosticBudget


def test_mask_supervises_answers_eos_and_newline_but_not_questions():
    class Tokenizer:
        def convert_tokens_to_ids(self,text):return {'<|im_start|>':1,'<|im_end|>':2,'assistant':3}[text]
    # Two turns. Tokens 8/9 are user text, 4 is newline, and 5/6 are answers.
    ids=[1,8,4,9,2,4,1,3,4,5,2,4,1,8,4,9,2,4,1,3,4,6,2,4]
    labels=labels_for(Tokenizer(),ids)
    assert [i for i,t in enumerate(labels) if t!=-100]==[9,10,11,21,22,23]
    assert [t for t in labels if t!=-100]==[5,2,4,6,2,4]


def test_cost_includes_memory_queries_and_explicit_rematerialization():
    record=dict(source_ids=[1]*25,input_ids=[2]*72)
    assert token_cost(record)==273
    assert token_cost(record,checkpoint=True)==546


def test_new_approved_ledger_cannot_reset_or_enlarge_old_ledger(tmp_path):
    old=DiagnosticBudget(tmp_path/'old.json');key=old.reserve(run='old',kind='test',student=149523)
    old.commit(key,student=149523)
    new=DiagnosticBudget(tmp_path/'new.json',limit=300000)
    key=new.reserve(run='new',kind='test',student=10);new.commit(key,student=10)
    assert sum(e['student'] for e in old.read()['entries'])==149523
    with pytest.raises(ValueError,match='limit changed'):DiagnosticBudget(tmp_path/'old.json',limit=300000)
