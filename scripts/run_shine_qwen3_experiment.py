#!/usr/bin/env python3
"""Serial controller for the approved 300k/15GiB Qwen3 SHINE experiment.

Training always executes from the separately pinned complete snapshot. This
controller never edits a committed checkpoint, resets a ledger, or retries a
failed GPU job. Immutable source/byte-identical files may share hard links.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True)
    args=p.parse_args()
    root=Path('/mnt/e/ds-rsi/reproduction/shine-qwen3-06b');snapshot=Path(args.snapshot).resolve()
    assert snapshot.is_relative_to(root/'code')
    py=Path('/mnt/e/ds-rsi/envs/train-cu130-torch210/bin/python')
    sys.path[:0]=[str(root/'compat'),str(snapshot/'src')]
    import torch
    from dsrsi.training.checkpoint import CheckpointStore,file_sha256
    from dsrsi.training.diagnostic_budget import DiagnosticBudget
    from dsrsi.utils.atomic import atomic_json
    from dsrsi.data.shine_qwen3 import setup_tokenizer,chat,compile_messages
    ledger=DiagnosticBudget(root/'control/student-ledger.json',limit=300000)
    main_run=root/'runs/main-01';main_store=CheckpointStore(main_run/'checkpoints')
    plan=json.loads((root/'data/plan.json').read_text());corpus=json.loads((root/'data/corpus.json').read_text())
    tokenizer=setup_tokenizer(root/'model')
    def status(phase,**extra):
        value=dict(phase=phase,updated_unix=time.time(),student_used=sum(e['student'] for e in ledger.read()['entries']),**extra)
        atomic_json(root/'control/experiment-status.json',value);print(json.dumps(value),flush=True)
    def call(label,run,action,*extra):
        record=root/'control'/f'{label}-submit.json'
        assert not record.exists(),'submission already exists; inspect instead of repeating'
        cmd=[str(py),str(snapshot/'scripts/run_5080_checked.py'),'--code-root',str(snapshot),'--label',label,'--',
             'scripts/shine_qwen3_reproduce.py','--root',str(root),'--run-dir',str(run),
             '--diagnostic-ledger',str(ledger.path),'--action',action,'--resume',*map(str,extra)]
        with (root/'logs'/f'{label}.log').open('x') as log:
            worker=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(record,dict(pid=worker.pid,command=cmd))
        status(label,pid=worker.pid,run=str(run))
        code=worker.wait()
        if code:raise RuntimeError(f'{label} exited {code}; retained log/reservations, no automatic retry')
    def clone(source,parent,name):
        run=root/'runs'/name;assert not run.exists()
        target=run/'checkpoints'/source.name;target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():os.link(path,target/path.name)
        manifest=json.loads((source/'manifest.json').read_text())
        atomic_json(run/'checkpoints/latest.json',dict(directory=source.name,manifest_sha256=file_sha256(source/'manifest.json')))
        inherited=manifest['progress']['micro_step']
        events=[json.loads(line) for line in (parent/'events.jsonl').read_text().splitlines()]
        rows=[e for e in events if e['event']=='micro'][:inherited]
        assert len(rows)==inherited
        (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
        atomic_json(run/'origin.json',dict(source=str(source),manifest_sha256=file_sha256(source/'manifest.json'),
            inherited_student_tokens=manifest['progress']['student_tokens'],prefix_is_not_new_compute=True,immutable_hardlinks=True))
        return run
    d,m=main_store.load_committed();assert m['status']=='stopped' and m['progress']['micro_step']==4
    assert m['progress']['optimizer_step']==1 and m['progress']['accumulation_index']==0
    source_sha=m['files'][0]['sha256'];source_manifest_sha=file_sha256(d/'manifest.json')
    call('shine-qwen3-max-shape',main_run,'resource','--output',root/'resource-micro4.json')
    assert json.loads((root/'resource-micro4.json').read_text())['full_training_state_unchanged']
    reference=clone(d,main_run,'recovery-reference');resumed=clone(d,main_run,'recovery-resumed')
    call('shine-qwen3-recovery-reference',reference,'train','--max-micro-steps',10)
    call('shine-qwen3-recovery-pause7',resumed,'train','--max-micro-steps',10,'--pause-after',7)
    paused_store=CheckpointStore(resumed/'checkpoints');pd,pm=paused_store.load_committed()
    assert pm['status']=='paused' and pm['progress']['micro_step']==7 and pm['progress']['accumulation_index']==3
    call('shine-qwen3-recovery-resume10',resumed,'train','--max-micro-steps',10)
    rd,rm=CheckpointStore(reference/'checkpoints').load_committed();sd,sm=paused_store.load_committed()
    left=CheckpointStore(reference/'checkpoints').load_rank(rd,rm,rank=0)
    right=paused_store.load_rank(sd,sm,rank=0);count=0;raw_bytes=0
    def equal(a,b,path='payload'):
        nonlocal count,raw_bytes
        assert type(a)==type(b),path
        if isinstance(a,torch.Tensor):
            assert a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride() and a.storage_offset()==b.storage_offset(),path
            assert a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()==b.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),path
            count+=1;raw_bytes+=a.numel()*a.element_size()
        elif isinstance(a,dict):
            assert list(a)==list(b),path
            for k in a:equal(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
        else:assert a==b,path
    equal(left,right)
    def events(run):return [json.loads(l) for l in (run/'events.jsonl').read_text().splitlines() if json.loads(l)['event']=='micro']
    first,second=events(reference)[4:],events(resumed)[4:]
    assert len(first)==len(second)==6
    for a,b in zip(first,second):
        for key in ['row_id','task','loss','student_input_tokens','supervised_tokens','grad_norm','lr','progress']:assert a[key]==b[key],key
    assert file_sha256(d/m['files'][0]['file'])==source_sha and file_sha256(d/'manifest.json')==source_manifest_sha
    # Deduplicate only after proving whole-file byte equality. Manifest hashes
    # stay valid and both checkpoint paths retain identical complete contents.
    deduplicated=False
    if rm['files'][0]['sha256']==sm['files'][0]['sha256']:
        original=rd/rm['files'][0]['file'];duplicate=sd/sm['files'][0]['file']
        staging=duplicate.with_suffix('.link');os.link(original,staging);os.replace(staging,duplicate)
        assert file_sha256(duplicate)==sm['files'][0]['sha256'];deduplicated=True
    report=dict(source=d.name,source_manifest_sha256=source_manifest_sha,source_payload_sha256=source_sha,
        paused=pd.name,paused_progress=pm['progress'],reference=rd.name,resumed=sd.name,
        final_progress=sm['progress'],full_payload_equal=True,raw_tensor_count=count,raw_tensor_bytes=raw_bytes,
        all_events_equal=True,source_unchanged=True,identical_payload_storage_deduplicated=deduplicated,
        new_student_per_trajectory=sum(e['student_input_tokens'] for e in first))
    atomic_json(root/'recovery-report.json',report);del left,right
    def evaluation(run,label,split,contexts):
        output=root/'evaluations'/f'{label}.json'
        selected=[x for x in corpus['qa_'+split] if x['kind']=='natural'][:contexts//2]
        selected+=[x for x in corpus['qa_'+split] if x['kind']=='synthetic'][:contexts-contexts//2]
        upper=sum(len(tokenizer.encode(x['source'],add_special_tokens=False))+176 for x in selected)
        for row in selected:
            for q in row['queries']:
                for view in ['memory','off','foreign','plaintext']:
                    messages=[dict(role='user',content=q['question'])]
                    if view=='plaintext':messages.insert(0,dict(role='system',content='Answer only from this context, concisely.\n\n'+row['source']))
                    upper+=len(chat(tokenizer,messages,prompt=True))+32
                    upper+=len(compile_messages(tokenizer,row['source'],messages+[dict(role='assistant',content=q['target'])])['input_ids'])
        used=sum(e['student'] for e in ledger.read()['entries'])
        assert used+upper<=300000,f'{label} whole evaluation exceeds budget: {used}+{upper}'
        call(label,run,'evaluate','--output',output,'--split',split,'--contexts',contexts)
        if not output.exists():
            status('paused_during_evaluation',journal=str(output)+'.journal.json');raise SystemExit(0)
    initial_dirs=[p for p in (main_run/'checkpoints').glob('micro_000000000000_*') if not p.name.endswith('.pending')]
    assert len(initial_dirs)==1
    initial=clone(initial_dirs[0],main_run,'initial-eval')
    evaluation(initial,'shine-qwen3-initial-dev','dev',4)
    call('shine-qwen3-full-curriculum',main_run,'train','--max-micro-steps',plan['total_steps'])
    final,fm=main_store.load_committed()
    if fm['status']=='paused':status('paused_training',progress=fm['progress']);return
    assert fm['status']=='stopped' and fm['progress']['micro_step']==plan['total_steps']
    actual=events(main_run)[4:10]
    for a,b in zip(actual,first):
        for key in ['row_id','task','loss','student_input_tokens','supervised_tokens','grad_norm','lr','progress']:assert a[key]==b[key],key
    boundaries=[]
    for path in (main_run/'checkpoints').glob(f"micro_{plan['pretrain_steps']:012d}_*"):
        if path.name.endswith('.pending'):continue
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['progress']['stage']=='SHINE_qwen3_pretrain':boundaries.append(path)
    assert len(boundaries)==1
    pretrain=clone(boundaries[0],main_run,'pretrain-eval')
    evaluation(pretrain,'shine-qwen3-pretrain-dev','dev',4)
    evaluation(main_run,'shine-qwen3-final-train','train',8)
    evaluation(main_run,'shine-qwen3-final-dev','dev',8)
    used=sum(e['student'] for e in ledger.read()['entries'])
    assert used+6*(176+64)<300000
    call('shine-qwen3-write-benchmark',main_run,'benchmark','--output',root/'write-benchmark.json')
    assert file_sha256(final/fm['files'][0]['file'])==fm['files'][0]['sha256']
    status('finished',checkpoint=final.name,manifest_sha256=file_sha256(final/'manifest.json'),
           progress=fm['progress'],all_planned_evaluations_complete=True,recovery_passed=True)


if __name__=='__main__':main()
