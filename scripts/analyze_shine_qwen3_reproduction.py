#!/usr/bin/env python3
"""CPU-only score and accounting summary for the bounded Qwen3 SHINE run."""

import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',required=True)
    p.add_argument('--reports',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();reports=Path(args.reports)
    helper=runpy.run_path(str(Path(__file__).with_name('evaluate_shine_qwen3_prompts.py')))
    final_answer,span_f1=helper['final_answer'],helper['span_f1']
    normalize=lambda text:' '.join(text.split()).casefold()
    def load(name):return json.loads((reports/name).read_text())
    def summarize(rows):
        result={}
        for kind in ['natural','synthetic']:
            result[kind]={}
            for view in ['memory','off','foreign','plaintext']:
                records=[r for r in rows if r['kind']==kind and r['view']==view]
                answers=[final_answer(r['text']) for r in records]
                result[kind][view]=dict(queries=len(records),normalized_EM=sum(r['exact'] for r in records),
                    strict_string_and_EOS=sum(r['text'].strip()==r['expected'].strip() and r['stopped_on_end'] for r in records),
                    ended=sum(r['stopped_on_end'] for r in records),unfinished_think=sum(x[1] for x in answers),
                    final_answer_word_F1=statistics.mean(span_f1(a[0],r['expected']) for a,r in zip(answers,records)),
                    mean_gold_NLL=statistics.mean(r['gold_nll'] for r in records))
        synthetic=[r for r in rows if r['kind']=='synthetic']
        lookup={(r['row_id'],r['probe'],r['view']):r for r in synthetic}
        paired=[];follows=0;foreign_trials=0
        for row_id,probe,view in lookup:
            if view!='memory':continue
            base,v=row_id.rsplit('-v',1)
            other=f'{base}-v{1-int(v)}'
            if (other,probe,'memory') not in lookup:continue
            if v=='0':
                a,b=lookup[row_id,probe,'memory'],lookup[other,probe,'memory']
                paired.append(dict(world=base,probe=probe,both_exact=a['exact'] and b['exact'],
                    both_strict_and_EOS=a['text'].strip()==a['expected'].strip() and a['stopped_on_end'] and
                                        b['text'].strip()==b['expected'].strip() and b['stopped_on_end']))
            foreign=lookup[row_id,probe,'foreign'];expected=lookup[other,probe,'memory']['expected']
            follows+=normalize(foreign['text'])==normalize(expected);foreign_trials+=1
        result['synthetic_pair_control']=dict(pairs=len(paired),both_exact=sum(x['both_exact'] for x in paired),
            both_strict_and_EOS=sum(x['both_strict_and_EOS'] for x in paired),
            foreign_follows_swapped_content=follows,foreign_trials=foreign_trials,details=paired)
        return result
    initial=load('shine-qwen3-06b-initial-dev-20261026.json')
    pretrain=load('shine-qwen3-06b-pretrain-dev-20261026.json')
    train=load('shine-qwen3-06b-final-train-20261026.json')
    dev=load('shine-qwen3-06b-final-dev-20261026.json')
    common={(r['row_id'],r['probe']) for r in initial['records']}
    assert common=={(r['row_id'],r['probe']) for r in pretrain['records']}
    matched=[r for r in dev['records'] if (r['row_id'],r['probe']) in common]
    events=[json.loads(l) for l in Path(args.events).read_text().splitlines()];micros=[e for e in events if e['event']=='micro']
    assert len(micros)==652 and [e['progress']['micro_step'] for e in micros]==list(range(1,653))
    stages={}
    for stage in ['SHINE_qwen3_pretrain','SHINE_qwen3_qa']:
        rows=[e for e in micros if e['progress']['stage']==stage]
        stages[stage]=dict(micro_steps=len(rows),student_input_tokens=sum(e['student_input_tokens'] for e in rows),
            supervised_tokens=sum(e['supervised_tokens'] for e in rows),compute_seconds=sum(e['seconds'] for e in rows),
            first32_mean_loss=statistics.mean(e['loss'] for e in rows[:32]),last32_mean_loss=statistics.mean(e['loss'] for e in rows[-32:]),
            exposure_distribution=dict(Counter(Counter(e['row_id'] for e in rows).values())),
            trainable_parameters=114728960)
        stages[stage]['student_tokens_per_compute_second']=stages[stage]['student_input_tokens']/stages[stage]['compute_seconds']
    result=dict(stages=stages,initial_dev=summarize(initial['records']),pretrain_dev=summarize(pretrain['records']),
        final_train=summarize(train['records']),final_dev=summarize(dev['records']),
        final_dev_matched_initial_subset=summarize(matched),
        total_train_student=sum(e['student_input_tokens'] for e in micros),
        compute_seconds=sum(e['seconds'] for e in micros),
        peak_allocated_bytes=max(e['peak_allocated_bytes'] for e in micros),
        peak_reserved_bytes=max(e['peak_reserved_bytes'] for e in micros),
        metric_notes=['Normalized EM is the original raw-string, case/whitespace-normalized metric.',
                      'Strict string+EOS is additionally reported; it is not an exact-token benchmark.',
                      'Word F1 is SQuAD-style after extracting completed reasoning blocks; unfinished reasoning scores zero.',
                      'No M5 or paper-benchmark acceptance is implied by these selected small sets.'])
    out=Path(args.output)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(out)


if __name__=='__main__':main()
