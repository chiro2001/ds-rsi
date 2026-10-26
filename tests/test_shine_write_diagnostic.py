"""Check content-control diagnostics against known synthetic outcomes."""

import importlib.util
from pathlib import Path

import torch


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_diagnostic_separates_changed_field_from_collateral_damage():
    rows=[]
    for context,expected,outputs in [(0,'Z1-346',['Z1-346','Z1-346']),
                                     (1,'Z1-376',['Z1-376','Z9-376'])]:
        for probe,text in zip(['p1','p2'],outputs):
            rows.append(dict(context_index=context,probe_id=probe,view='memory',expected=expected,
                             text=text,exact_match=text==expected,stopped_on_end=True))
    value=script('analyze_shine_write').output_summary(rows)
    assert value['exact']==3 and value['pair_both_exact']==1
    assert value['same_answer_across_questions']==1
    group=value['changed_character_groups']['suffix_digit']
    assert group['changed_character_correct']==4
    assert group['both_unchanged_characters_correct']==1
    assert group['output_changed_at_unchanged_character']==1


def test_encoding_summary_distinguishes_constant_from_varying_representations():
    summary=script('diagnose_shine_path').representation_summary
    constant=summary([torch.tensor([2.,3.])]*4)
    assert constant['centered_energy_fraction']==0 and constant['centered_effective_rank']==0
    assert constant['counterfactual_relative_distances']==[0.,0.]
    values=[torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.tensor([-1.,0.]),torch.tensor([0.,-1.])]
    varied=summary(values)
    assert varied['centered_energy_fraction']==1
    assert abs(varied['centered_effective_rank']-2)<1e-8
