import runpy
from pathlib import Path


helpers=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/evaluate_shine_qwen3_prompts.py'))


def test_unfinished_reasoning_cannot_count_as_final_answer():
    answer,unfinished=helpers['final_answer']('<think>the answer is R4-232')
    assert answer=='' and unfinished
    assert helpers['final_answer']('<think>reasoning</think>\nR4-232')==('R4-232',False)
    assert helpers['final_answer']('R4-232')==('R4-232',False)


def test_span_f1_and_exact_code_scoring_remain_separate():
    f1=helpers['span_f1']
    assert f1('The Thames Estuary.','Thames Estuary')==1
    assert f1('Thames','Thames Estuary')==2/3
    assert f1('','R4-232')==0
