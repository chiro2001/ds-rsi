#!/usr/bin/env python3
"""Frozen final-state controls for official system prompts and empty-think prefix.

The original raw-prompt evaluations remain intact. These controls never change
training weights, targets, thresholds, or the student's available source text.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import string
import sys


NO_CONTEXT='You are a helpful assistant. Answer the question concisely with short words or phrases. Answer the question directly and output nothing else. Never say you don\'t know the answer. Never enter think mode.'
WITH_CONTEXT='You are a helpful assistant, answer the questions based on the given context. Each answer must be directly extractable from the context (i.e., an exact span or minor paraphrase for fluency). Do not invent information. Answer the question directly and output nothing else. Never enter think mode.\n\nContext: '


def final_answer(text):
    if '<think>' in text:
        if '</think>' not in text:return '',True
        text=text.split('</think>',1)[1]
    return text.strip(),False


def span_f1(prediction,target):
    def tokens(text):
        text=''.join(c for c in text.casefold() if c not in string.punctuation)
        return re.sub(r'\b(a|an|the)\b',' ',text).split()
    p,t=tokens(prediction),tokens(target)
    if not p or not t:return float(p==t)
    common=sum((Counter(p)&Counter(t)).values())
    return 2*common/(len(p)+len(t))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['root','run-dir','diagnostic-ledger','output']:p.add_argument('--'+flag,required=True)
    p.add_argument('--expected-micro',type=int,default=652)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--binding-only',action='store_true')
    args=p.parse_args();root=Path(args.root).resolve();run=Path(args.run_dir).resolve();out=Path(args.output).resolve();ledger_path=Path(args.diagnostic_ledger).resolve()
    assert root==Path('/mnt/e/ds-rsi/reproduction/shine-qwen3-06b')
    assert all(x.is_relative_to(root) for x in [run,out,ledger_path])
    if out.exists() and not args.resume:raise ValueError('new pinned output required')
    sys.path.insert(0,str(root/'compat'))
    import torch
    import transformers
    from dsrsi.data.shine_qwen3 import setup_tokenizer,chat,normalized
    from dsrsi.model.shine_qwen3 import OfficialShineQwen3
    from dsrsi.evaluation.journal import EvaluationJournal,EvaluationPaused
    from dsrsi.training.checkpoint import CheckpointStore,file_sha256
    from dsrsi.training.diagnostic_budget import DiagnosticBudget
    from dsrsi.utils.atomic import atomic_json
    assert transformers.__version__=='4.57.1'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.mha.set_fastpath_enabled(False)
    store=CheckpointStore(run/'checkpoints');d,m=store.load_committed()
    assert m['world_size']==1 and m['progress']['micro_step']==args.expected_micro
    runtime=m['identity']['runtime']
    assert str(torch.__version__)==runtime['torch'] and torch.version.cuda==runtime['cuda']
    assert torch.cuda.get_device_name()==runtime['gpu']
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==runtime['cublas_workspace']
    assert file_sha256(Path(transformers.__file__))==runtime['transformers_init_sha256']
    code=Path(__file__).resolve().parents[1]
    for name,expected in m['identity']['implementation'].items():assert file_sha256(code/name)==expected,name
    for name,expected in m['identity']['base'].items():assert file_sha256(root/'model'/name)==expected,name
    assert file_sha256(root/'data/corpus.json')==m['identity']['corpus_sha256']
    tokenizer=setup_tokenizer(root/'model');torch.manual_seed(m['identity']['plan']['seed'])
    model=OfficialShineQwen3.from_local_checkpoint(root/'model',device='cuda',tokenizer=tokenizer)
    model.eval();payload=store.load_rank(d,m,rank=0);params={n:v for n,v in model.named_parameters() if v.requires_grad}
    assert set(params)==set(payload['weights'])
    with torch.no_grad():
        for n,v in params.items():
            assert v.shape==payload['weights'][n].shape and v.dtype==payload['weights'][n].dtype
            v.copy_(payload['weights'][n].to(v.device))
        for n,v in model.named_buffers():v.copy_(payload['buffers'][n].to(v.device))
    weights=payload['weights'];del payload
    corpus=json.loads((root/'data/corpus.json').read_text());rows=[x for x in corpus['qa_dev'] if x['kind']=='natural'][:2]
    rows+=[x for x in corpus['qa_dev'] if x['kind']=='synthetic'][:2]
    if args.binding_only:
        import time
        known=[x for x in corpus['qa_train'] if x['kind']=='synthetic' and x['variant']==0]
        a,b=known[:2];av,bv=a['queries'][0]['target'],b['queries'][0]['target']
        novel=av[:-1]+str((int(av[-1])+1)%10)
        assert novel not in {q['target'] for x in corpus['qa_train'] for q in x['queries']}
        assert a['source'].count(av)==1 and b['source'].count(bv)==1
        cases=[dict(name='known_A',row=a,source=a['source'],target=av),
               dict(name='known_entity_A_seen_value_B',row=a,source=a['source'].replace(av,bv),target=bv),
               dict(name='known_entity_B_seen_value_A',row=b,source=b['source'].replace(bv,av),target=av),
               dict(name='known_entity_A_novel_last_digit',row=a,source=a['source'].replace(av,novel),target=novel)]
        ledger=DiagnosticBudget(ledger_path,limit=300000)
        prompts=[[chat(tokenizer,[dict(role='user',content=q['question'])],prompt=True) for q in c['row']['queries']] for c in cases]
        sources=[tokenizer.encode(c['source'],add_special_tokens=False) for c in cases]
        upper=sum(len(x)+176 for x in sources)+sum(len(x)+32 for qs in prompts for x in qs)+len(prompts[1][0])+32
        assert sum(e['student'] for e in ledger.read()['entries'])+upper<=300000,'binding controls exceed remaining budget'
        memories=[];records=[]
        for c,ids,qs in zip(cases,sources,prompts,strict=True):
            key=ledger.reserve(run=str(out),kind='binding_probe_source',student=len(ids)+176)
            with torch.no_grad():mem=model.encode(torch.tensor([ids],device='cuda'))
            ledger.commit(key,student=len(ids)+176);memories.append(mem)
            for j,query in enumerate(qs):
                key=ledger.reserve(run=str(out),kind='binding_probe_query',student=len(query)+32)
                result=model.generate(query,memory=mem,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id)
                ledger.commit(key,student=result['input_tokens']);text=tokenizer.decode(result['token_ids'],skip_special_tokens=True)
                records.append(dict(case=c['name'],probe=j,source=c['source'],expected=c['target'],text=text,
                    exact=normalized(text)==normalized(c['target']),**result))
        def tensors(tree):
            if isinstance(tree,dict):return [t for v in tree.values() for t in tensors(v)]
            return [tree] if isinstance(tree,torch.Tensor) else []
        def transfer(tree,device):
            if isinstance(tree,dict):return {k:transfer(v,device) for k,v in tree.items()}
            return tree.detach().to(device).clone() if isinstance(tree,torch.Tensor) else tree
        payload_path=Path(str(out)+'.memory.pt');assert not payload_path.exists()
        torch.cuda.synchronize();start=time.perf_counter()
        exported=dict(checkpoint_manifest_sha256=file_sha256(d/'manifest.json'),case=cases[1]['name'],factors=transfer(memories[1],'cpu'))
        with payload_path.open('xb') as f:torch.save(exported,f);f.flush();os.fsync(f.fileno())
        export_seconds=time.perf_counter()-start
        start=time.perf_counter();loaded=torch.load(payload_path,map_location='cpu',weights_only=True)
        assert loaded['checkpoint_manifest_sha256']==file_sha256(d/'manifest.json')
        reloaded=transfer(loaded['factors'],'cuda');torch.cuda.synchronize();reload_seconds=time.perf_counter()-start
        key=ledger.reserve(run=str(out),kind='binding_payload_roundtrip_query',student=len(prompts[1][0])+32)
        repeated=model.generate(prompts[1][0],memory=reloaded,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id)
        ledger.commit(key,student=repeated['input_tokens'])
        live=next(r for r in records if r['case']==cases[1]['name'] and r['probe']==0)
        assert all(repeated[k]==live[k] for k in repeated),'payload roundtrip changed output'
        assert all(torch.equal(v.detach().cpu(),weights[n]) for n,v in params.items())
        assert file_sha256(d/m['files'][0]['file'])==m['files'][0]['sha256']
        atomic_json(out,dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),script_sha256=file_sha256(Path(__file__)),
            scope='Frozen familiar-entity/new-binding controls; two questions decoded independently, no answer in prompt',
            student_upper=upper,records=records,training_payload_unchanged=True,
            roundtrip=dict(factor_bytes=sum(t.numel()*t.element_size() for t in tensors(memories[1])),
                file_bytes=payload_path.stat().st_size,file_sha256=file_sha256(payload_path),
                export_cpu_and_fsync_seconds=export_seconds,reload_cpu_to_gpu_seconds=reload_seconds,
                generation_identical=True,example_correct=live['exact']),M5_passed=False))
        print(json.dumps(dict(binding_controls_complete=True,correct=sum(r['exact'] for r in records),queries=len(records))),flush=True)
        return
    views=['memory','off','foreign','plaintext'];styles=['official_system','empty_think']
    def prompt(row,q,style,view):
        system=WITH_CONTEXT+row['source'] if view=='plaintext' else NO_CONTEXT
        ids=chat(tokenizer,[dict(role='system',content=system),dict(role='user',content=q['question'])],prompt=True)
        if style=='empty_think':ids+=tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
        return ids
    upper=sum(len(tokenizer.encode(x['source'],add_special_tokens=False))+176 for x in rows)
    upper+=sum(len(prompt(row,q,s,v))+128 for row in rows for q in row['queries'] for s in styles for v in views)
    ledger=DiagnosticBudget(ledger_path,limit=300000);used=sum(e['student'] for e in ledger.read()['entries'])
    if not args.resume and used+upper>300000:raise ValueError(f'whole supplementary evaluation exceeds remaining budget: {used}+{upper}')
    jobs=[f'{i}/{j}/{s}/{v}' for i,row in enumerate(rows) for j,_ in enumerate(row['queries']) for s in styles for v in views]
    identity=dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),script_sha256=file_sha256(Path(__file__)),
        training_identity=m['identity'],styles=styles,max_new_tokens=128,contexts=[x['id'] for x in rows],
        scope='Supplementary final-state generation controls; all views receive the same content-free system instructions')
    journal=EvaluationJournal(str(out)+'.journal.json',identity,jobs,resume=args.resume);journal.install_signal_handlers()
    memories={};results=[]
    def memory(i):
        if i not in memories:
            ids=tokenizer.encode(rows[i]['source'],add_special_tokens=False);cost=len(ids)+176
            charge=ledger.reserve(run=str(out),kind='prompt_control_source',student=cost)
            with torch.no_grad():memories[i]=model.encode(torch.tensor([ids],device='cuda'))
            ledger.commit(charge,student=cost)
        return memories[i]
    try:
        for i,row in enumerate(rows):
            for j,q in enumerate(row['queries']):
                for style in styles:
                    for view in views:
                        def compute():
                            mem=memory(i) if view=='memory' else memory(i^1) if view=='foreign' else None
                            ids=prompt(row,q,style,view)
                            charge=ledger.reserve(run=str(out),kind='prompt_control_query',student=len(ids)+128)
                            result=model.generate(ids,memory=mem,max_new_tokens=128,eos_token_id=tokenizer.eos_token_id)
                            ledger.commit(charge,student=result['input_tokens'])
                            text=tokenizer.decode(result['token_ids'],skip_special_tokens=True);answer,unfinished=final_answer(text)
                            return dict(row_id=row['id'],kind=row['kind'],probe=j,style=style,view=view,
                                expected=q['target'],text=text,answer=answer,unfinished_think=unfinished,
                                raw_exact=normalized(text)==normalized(q['target']),exact=normalized(answer)==normalized(q['target']),
                                span_f1=span_f1(answer,q['target']),**result)
                        results.append(journal.evaluate(f'{i}/{j}/{style}/{view}',compute))
        journal.finish()
    except EvaluationPaused:
        print(json.dumps(dict(status='paused',items=len(journal.results))),flush=True);return
    assert all(torch.equal(v.detach().cpu(),weights[n]) for n,v in params.items())
    assert file_sha256(d/m['files'][0]['file'])==m['files'][0]['sha256']
    summary={}
    for kind in ['natural','synthetic']:
        summary[kind]={}
        for style in styles:
            summary[kind][style]={}
            for view in views:
                records=[r for r in results if r['kind']==kind and r['style']==style and r['view']==view]
                summary[kind][style][view]=dict(queries=len(records),correct=sum(r['exact'] for r in records),
                    raw_correct=sum(r['raw_exact'] for r in records),unfinished_think=sum(r['unfinished_think'] for r in records),
                    mean_span_f1=sum(r['span_f1'] for r in records)/len(records))
    atomic_json(out,dict(identity=identity,summary=summary,records=results,student_upper=upper,
        training_payload_unchanged=True,diagnostic_only=True,M5_passed=False))
    print(json.dumps(dict(summary=summary)),flush=True)


if __name__=='__main__':main()
