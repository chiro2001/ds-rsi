#!/usr/bin/env python3
"""Bounded official-Qwen3 SHINE reproduction, with full-state safe interruption."""

import argparse
import json
import os
from pathlib import Path
import sys


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--diagnostic-ledger',required=True)
    p.add_argument('--action',choices=['prepare','initialize','train','evaluate'],required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-micro-steps',type=int,default=10**9)
    p.add_argument('--pause-after',type=int)
    p.add_argument('--output')
    p.add_argument('--split',choices=['train','dev'],default='dev')
    p.add_argument('--contexts',type=int,default=8)
    return p.parse_args()


def main():
    args=arguments();root=Path(args.root).resolve();run=Path(args.run_dir).resolve();ledger_path=Path(args.diagnostic_ledger).resolve()
    authorized=Path('/mnt/e/ds-rsi/reproduction/shine-qwen3-06b')
    if root!=authorized or any(not p.is_relative_to(root) for p in [run,ledger_path]):
        raise ValueError('new reproduction files must remain in the authorized E: subtree')
    compat=root/'compat';sys.path.insert(0,str(compat))
    import hashlib
    import platform
    import random
    import shutil
    import time
    from dataclasses import asdict
    import numpy as np
    import torch
    import transformers
    from dsrsi.data.shine_qwen3 import (setup_tokenizer,prepare_corpus,plan_steps,training_record,
        token_cost,chat,compile_messages,normalized,digest)
    from dsrsi.evaluation.journal import EvaluationJournal,EvaluationPaused
    from dsrsi.model.shine_qwen3 import OfficialShineQwen3
    from dsrsi.training.checkpoint import (CheckpointStore,Progress,capture_training_state,
        restore_training_state,capture_rng,restore_rng,file_sha256)
    from dsrsi.training.control import ControlChannel
    from dsrsi.training.diagnostic_budget import DiagnosticBudget,DiagnosticBudgetExceeded
    from dsrsi.utils.atomic import atomic_json

    if transformers.__version__!='4.57.1' or not str(transformers.__file__).startswith(str(compat)):
        raise ValueError('isolated official Transformers 4.57.1 overlay required')
    base=root/'model';data_path=root/'data/corpus.json';plan_path=root/'data/plan.json'
    torch.set_num_threads(2);tokenizer=setup_tokenizer(base)
    if args.action=='prepare':
        if data_path.exists() or plan_path.exists():raise FileExistsError('prepared data already exist')
        corpus=prepare_corpus(root/'data/upstream-prefix.json',tokenizer)
        plan=plan_steps(corpus,tokenizer)
        plan.update(seed=20261026,lr=.00005,weight_decay=.01,warmup_steps=4,
            checkpoint=False,source_hash=file_sha256(root/'data/upstream-prefix.json'),
            corpus_hash=digest(corpus),total_budget=300000,disk_growth_limit_bytes=15*(1<<30),
            optimizer_transition='new Adam and linear schedule at empty pretrain-to-QA boundary; weights, RNG and ledger carried')
        atomic_json(data_path,corpus);atomic_json(plan_path,plan)
        atomic_json(root/'control/authorization.json',dict(plan=plan,created_unix=time.time(),
            baseline_windows_free_bytes=shutil.disk_usage(root).free,
            approved_student_budget=300000,approved_growth_bytes=15*(1<<30)))
        print(json.dumps(dict(event='prepared',plan=plan,counts={k:len(corpus[k]) for k in ['pretrain_train','pretrain_dev','qa_train','qa_dev']})),flush=True)
        return
    if not torch.cuda.is_available():raise RuntimeError('authorized 5080 CUDA required')
    if torch.cuda.mem_get_info()[0]<8*(1<<30):raise RuntimeError('need 8GiB available before loading')
    plan=json.loads(plan_path.read_text());corpus=json.loads(data_path.read_text())
    if digest(corpus)!=plan['corpus_hash']:raise ValueError('corpus changed')
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    model=OfficialShineQwen3.from_local_checkpoint(base,device='cuda',tokenizer=tokenizer)
    model.train();parameters=[p for p in model.parameters() if p.requires_grad]
    no_decay=('bias','LayerNorm.weight','layer_norm.weight','norm.weight','norm1','norm2')
    def make_optimizer(stage):
        groups=[]
        for decay in [True,False]:
            values=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('writer.')
                    and (not any(t in n for t in no_decay))==decay]
            groups.append(dict(params=values,weight_decay=plan['weight_decay'] if decay else 0.))
        groups.append(dict(params=list(model.writer.parameters()),weight_decay=plan['weight_decay']))
        values=[p for g in groups for p in g['params']]
        if len(values)!=len({id(p) for p in values}) or {id(p) for p in values}!={id(p) for p in parameters}:
            raise ValueError('every trainable parameter must occur exactly once in optimizer')
        opt=torch.optim.AdamW(groups,lr=plan['lr'],foreach=False)
        n=plan['pretrain_steps'] if stage=='pretrain' else plan['total_steps']-plan['pretrain_steps']
        scheduler=transformers.get_linear_schedule_with_warmup(opt,num_warmup_steps=plan['warmup_steps'],
                                                              num_training_steps=n//plan['grad_accum'])
        return opt,scheduler
    source_root=Path(__file__).resolve().parents[1]
    implementation_files=[source_root/p for p in ['scripts/shine_qwen3_reproduce.py','src/dsrsi/model/shine_qwen3.py',
        'src/dsrsi/data/shine_qwen3.py','src/dsrsi/training/checkpoint.py','src/dsrsi/training/control.py',
        'src/dsrsi/training/diagnostic_budget.py','src/dsrsi/evaluation/journal.py']]
    implementation_files+=sorted((source_root/'src/dsrsi/vendor/shine').glob('*'))
    identity=dict(plan=plan,architecture=model.architecture_identity(),
        corpus_sha256=file_sha256(data_path),base={p.name:file_sha256(p) for p in sorted(base.iterdir()) if p.is_file()},
        implementation={str(p.relative_to(source_root)):file_sha256(p) for p in implementation_files if p.is_file()},
        runtime=dict(torch=str(torch.__version__),transformers=transformers.__version__,
            transformers_init_sha256=file_sha256(Path(transformers.__file__)),cuda=torch.version.cuda,
            python=platform.python_version(),gpu=torch.cuda.get_device_name(),deterministic=True,tf32=False,
            mha_fastpath=False,cpu_threads=torch.get_num_threads(),python_hash_seed=os.environ.get('PYTHONHASHSEED'),
            cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    identity=json.loads(json.dumps(identity));store=CheckpointStore(run/'checkpoints')
    control=ControlChannel(run);control.install_signal_handlers();ledger=DiagnosticBudget(ledger_path,limit=300000)
    stage='pretrain';progress=Progress('SHINE_qwen3_pretrain')
    if args.resume:
        payload,manifest=store.load_latest(identity=identity)
        stage='qa' if payload['progress']['stage']=='SHINE_qwen3_qa' else 'pretrain'
        optimizer,scheduler=make_optimizer(stage)
        progress,cursor,life=restore_training_state(payload,model,optimizer,scheduler=scheduler,device='cuda')
        if cursor!={'next_ordinal':progress.micro_step} or life:raise ValueError('cursor/life mismatch')
        del payload
    else:
        if (store.root/'latest.json').exists():raise FileExistsError('existing run requires --resume')
        optimizer,scheduler=make_optimizer(stage)
    events=run/'events.jsonl'
    def log(value):
        with events.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
        print(json.dumps(value,allow_nan=False),flush=True)
    cached_owned=None
    def owned_bytes(*,refresh=False):
        nonlocal cached_owned
        if cached_owned is not None and not refresh:return cached_owned
        total=0
        for directory,_,names in os.walk(root):
            for name in names:
                path=Path(directory)/name
                if path.is_file():total+=path.stat().st_size
        cached_owned=total
        return total
    def guard(extra=0):
        # Only journal/log/ledger appends occur between saves. Reserve 32MiB
        # beyond the cached full inventory rather than stat thousands of NTFS
        # package files on every microbatch.
        if owned_bytes()+extra+32*(1<<20)>15*(1<<30):raise RuntimeError('approved 15GiB reproduction growth limit')
        if shutil.disk_usage(root).free<20*(1<<30)+extra:raise RuntimeError('reserved E: free space limit')
    def save(status):
        # Reserve enough for weights, moments and pending gradients, including
        # serialization overhead. No checkpoint is overwritten or auto-deleted.
        estimated=sum(p.numel()*p.element_size()+(0 if p.grad is None else p.grad.numel()*p.grad.element_size()) for p in parameters)
        estimated+=sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))
        estimated+=sum(b.numel()*b.element_size() for b in model.buffers())+16*(1<<20)
        guard(estimated)
        value=capture_training_state(model,optimizer,progress=progress,data_cursor={'next_ordinal':progress.micro_step},
            life_state={},scheduler=scheduler,device='cuda')
        d=store.begin(progress);store.save_rank(d,value)
        final=store.commit(d,world_size=1,identity=identity,progress=progress,status=status)
        owned_bytes(refresh=True)
        log(dict(event='saved',status=status,checkpoint=final.name,progress=asdict(progress)))
        return final
    if not args.resume:save('stopped')
    log(dict(event='ready',progress=asdict(progress),trainable_parameters=sum(p.numel() for p in parameters),
             architecture=model.architecture_identity(),owned_bytes=owned_bytes()))
    if args.action=='initialize':return
    if args.action=='evaluate':
        if not args.resume or not args.output:raise ValueError('evaluation requires pinned --resume and --output')
        output=Path(args.output).resolve()
        if not output.is_relative_to(root):raise ValueError('evaluation output must stay on E:')
        d,m=store.load_committed();before_sha=m['files'][0]['sha256'];rng=capture_rng('cuda');model.eval()
        natural=[x for x in corpus['qa_'+args.split] if x['kind']=='natural'][:args.contexts//2]
        synthetic=[x for x in corpus['qa_'+args.split] if x['kind']=='synthetic'][:args.contexts-args.contexts//2]
        rows=natural+synthetic
        if len(rows)!=args.contexts:raise ValueError('not enough evaluation contexts')
        jobs=[f'{i}/{j}/{v}' for i,row in enumerate(rows) for j,_ in enumerate(row['queries']) for v in ['memory','off','foreign','plaintext']]
        eid=dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),script_sha256=file_sha256(Path(__file__)),
            training_identity=identity,split=args.split,contexts=args.contexts,max_new_tokens=32)
        journal=EvaluationJournal(str(output)+'.journal.json',eid,jobs,resume=Path(str(output)+'.journal.json').exists())
        journal.install_signal_handlers();results=[];memories={};pause=False
        def encode(row,tag):
            if tag not in memories:
                ids=tokenizer.encode(row['source'],add_special_tokens=False);cost=len(ids)+model.memory_tokens
                key=ledger.reserve(run=str(output),kind='evaluation_source',student=cost)
                with torch.no_grad():memory=model.encode(torch.tensor([ids],device='cuda'))
                ledger.commit(key,student=cost);memories[tag]=memory
            return memories[tag]
        try:
            for i,row in enumerate(rows):
                other=rows[i^1]
                for j,question in enumerate(row['queries']):
                    for view in ['memory','off','foreign','plaintext']:
                        def compute():
                            messages=[dict(role='user',content=question['question'])];memory=None
                            if view=='memory':memory=encode(row,i)
                            elif view=='foreign':memory=encode(other,i^1)
                            elif view=='plaintext':messages.insert(0,dict(role='system',content='Answer only from this context, concisely.\n\n'+row['source']))
                            prompt=chat(tokenizer,messages,prompt=True)
                            key=ledger.reserve(run=str(output),kind='evaluation_query_'+view,student=len(prompt)+32)
                            result=model.generate(prompt,memory=memory,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id)
                            ledger.commit(key,student=result['input_tokens'])
                            text=tokenizer.decode(result['token_ids'],skip_special_tokens=True)
                            # Gold-prefix loss is counted separately and never used as EM.
                            gold=compile_messages(tokenizer,row['source'],messages+[dict(role='assistant',content=question['target'])])
                            key=ledger.reserve(run=str(output),kind='evaluation_gold_'+view,student=len(gold['input_ids']))
                            with torch.no_grad():nll=float(model.answer(torch.tensor([gold['input_ids']],device='cuda'),
                                torch.tensor([gold['labels']],device='cuda'),memory=memory))
                            ledger.commit(key,student=len(gold['input_ids']))
                            return dict(context=i,row_id=row['id'],kind=row['kind'],probe=j,view=view,
                                expected=question['target'],text=text,exact=normalized(text)==normalized(question['target']),
                                gold_nll=nll,source=row['source'] if row['kind']=='synthetic' else None,**result)
                        results.append(journal.evaluate(f'{i}/{j}/{view}',compute))
            journal.finish()
        except EvaluationPaused:pause=True
        finally:restore_rng(rng,'cuda')
        if file_sha256(d/m['files'][0]['file'])!=before_sha:raise RuntimeError('training payload changed during evaluation')
        if pause:
            log(dict(event='evaluation_paused',items=len(journal.results)));return
        summary={}
        for kind in ['natural','synthetic']:
            summary[kind]={view:dict(queries=sum(r['kind']==kind and r['view']==view for r in results),
                correct=sum(r['kind']==kind and r['view']==view and r['exact'] for r in results),
                mean_gold_nll=float(np.mean([r['gold_nll'] for r in results if r['kind']==kind and r['view']==view])))
                for view in ['memory','off','foreign','plaintext']}
        atomic_json(output,dict(identity=eid,summary=summary,records=results,training_payload_unchanged=True,
            diagnostic_only=True,M5_passed=False))
        log(dict(event='evaluation_finished',output=str(output),summary=summary));return
    if args.action!='train':raise ValueError('unknown action')
    existing=[json.loads(line) for line in events.read_text().splitlines()]
    micro_events=[x for x in existing if x['event']=='micro']
    if len(micro_events)!=progress.micro_step:raise ValueError('events/checkpoint progress differ; inspect instead of replaying')
    torch.cuda.reset_peak_memory_stats();stop_reason='micro_limit'
    while progress.micro_step<min(args.max_micro_steps,plan['total_steps']):
        request=control.poll()
        if request:
            final=save({'checkpoint':'checkpoint','pause':'paused','stop':'stopped'}[request.action]);control.acknowledge(request,final)
            if request.action!='checkpoint':return
        if args.pause_after is not None and progress.micro_step>=args.pause_after:
            save('paused');return
        if stage=='pretrain' and progress.micro_step==plan['pretrain_steps']:
            if progress.accumulation_index:raise RuntimeError('phase transition has pending gradients')
            save('checkpoint')
            optimizer,scheduler=make_optimizer('qa');stage='qa'
            progress=Progress('SHINE_qwen3_qa',progress.optimizer_step,progress.micro_step,0,progress.student_tokens,0)
            log(dict(event='phase_transition',semantics=plan['optimizer_transition'],progress=asdict(progress)))
            save('checkpoint')
        record=training_record(corpus,tokenizer,progress.micro_step,plan['pretrain_steps'])
        cost=token_cost(record,model.memory_tokens,checkpoint=plan['checkpoint'])
        # Keep all final evaluation costs within the same approved 300k ledger.
        if sum(e['student'] for e in ledger.read()['entries'])+cost>270000:
            stop_reason='reserved_final_evaluation_budget';break
        # Keep space for a complete interrupted gradient state before beginning work.
        guard(sum(p.numel()*p.element_size() for p in parameters)*4+32*(1<<20))
        key=ledger.reserve(run=str(run),kind='training_'+stage,student=cost)
        start=time.monotonic()
        memory=model.encode(torch.tensor([record['source_ids']],device='cuda'),checkpoint=plan['checkpoint'])
        loss=model.answer(torch.tensor([record['input_ids']],device='cuda'),torch.tensor([record['labels']],device='cuda'),
                           memory=memory,checkpoint=plan['checkpoint'])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss; reservation retained')
        (loss/plan['grad_accum']).backward();del memory
        accumulation=progress.accumulation_index+1;step=progress.optimizer_step
        grad_norm=None
        if accumulation==plan['grad_accum']:
            grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);accumulation=0;step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-start
        ledger.commit(key,student=cost)
        progress=Progress('SHINE_qwen3_'+stage,step,progress.micro_step+1,accumulation,progress.student_tokens+cost,0)
        log(dict(event='micro',row_id=record['row_id'],task=record['task'],loss=float(loss.detach()),
            student_input_tokens=cost,supervised_tokens=sum(t!=-100 for t in record['labels']),seconds=seconds,
            grad_norm=grad_norm,lr=optimizer.param_groups[0]['lr'],progress=asdict(progress),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        if torch.cuda.max_memory_reserved()>10*(1<<30):stop_reason='reserved_memory_limit';break
        if progress.micro_step%200==0 and progress.micro_step not in [plan['pretrain_steps'],plan['total_steps']]:save('checkpoint')
    final=save('stopped')
    atomic_json(run/f'summary-micro{progress.micro_step}.json',dict(reason=stop_reason,checkpoint=final.name,
        manifest_sha256=file_sha256(final/'manifest.json'),progress=asdict(progress),owned_bytes=owned_bytes(),
        diagnostic_only=True,formal_A_tokens_consumed=0))


if __name__=='__main__':main()
