#!/usr/bin/env python3
"""Small pinned SHINE U/P/C probe, writer timing and payload round trip."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer

from dsrsi.data.compiler import EpisodeCompiler,tokenizer_fingerprint
from dsrsi.data.shine import shine_episode
from dsrsi.evaluation.fixed_records import evidence_teacher_response
from dsrsi.evaluation.journal import EvaluationJournal,EvaluationPaused
from dsrsi.evaluation.m_only import normalized_answer
from dsrsi.model.cuda_execution import cuda_execution_identity,require_reference_kernels
from dsrsi.model.shine import ShineConfig,ShineQwen,GeneratedMemory
from dsrsi.training.checkpoint import CheckpointStore,restore_training_state,file_sha256
from dsrsi.training.device import resolve_device
from dsrsi.training.diagnostic_budget import DiagnosticBudget
from dsrsi.training.shine_pilot import encode_memories
from dsrsi.utils.atomic import atomic_json


def update_norm(a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    return float(((b.T@b)*(a@a.T).T).sum().clamp_min(0).sqrt())


def relative_update_change(positive,negative):
    numerator=denominator=0.
    by_id={x.record_id:x for x in negative}
    for memory in positive:
        for layer,(a,b) in memory.factors.items():
            other_a,other_b=by_id[memory.record_id].factors[layer]
            difference=update_norm(torch.cat((a,other_a)),torch.cat((b,-other_b),dim=1))
            numerator+=difference**2;denominator+=update_norm(a,b)**2
    return (numerator/max(denominator,1e-30))**.5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','run-dir','output','diagnostic-ledger'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--world',type=int,default=0)
    p.add_argument('--timing-repeats',type=int,default=5)
    args=p.parse_args()
    base,run,out,ledger_path=[Path(x).resolve() for x in (args.base,args.run_dir,args.output,args.diagnostic_ledger)]
    if any(not x.is_relative_to(Path('/mnt/e/ds-rsi')) for x in (base,run,out,ledger_path)):
        raise ValueError('all CUDA growth paths must remain on E:')
    if args.timing_repeats<1 or args.world<0 or out.exists() and not args.resume:raise ValueError('invalid/new evaluation required')
    device,_=resolve_device('cuda');torch.set_num_threads(2)
    store=CheckpointStore(run/'checkpoints');d,m=store.load_committed()
    if args.resume:
        prior=EvaluationJournal.read(str(out)+'.journal.json')
        d=store.root/prior['identity']['checkpoint'];m=json.loads((d/'manifest.json').read_text())
    for f in m['files']:
        if file_sha256(d/f['file'])!=f['sha256']:raise ValueError('payload checksum differs')
    c=m['identity']['train_config']
    if c['mode']!='generator':raise ValueError('shared writer generator required')
    code_root=Path(__file__).resolve().parents[1]
    for relative,expected in m['identity']['implementation'].items():
        if file_sha256(code_root/relative)!=expected:raise ValueError('training implementation differs')
    actual_base=dict(tokenizer=tokenizer_fingerprint(base),weights={n:file_sha256(base/n) for n in m['identity']['base']['weights']})
    if actual_base!=m['identity']['base']:raise ValueError('base/tokenizer differs')
    cfg=dict(m['identity']['architecture']['config']);cfg['layers']=tuple(cfg['layers'])
    torch.manual_seed(c['seed']);model=ShineQwen.from_local_checkpoint(base,device=device,config=ShineConfig(**cfg))
    require_reference_kernels(model)
    if cuda_execution_identity(device)!=m['identity']['runtime']:raise ValueError('runtime differs')
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=c['lr'],weight_decay=0.,foreach=False)
    payload=store.load_rank(d,m,rank=0);restore_training_state(payload,model,optimizer,device=device);model.eval()
    del optimizer,payload
    torch.cuda.reset_peak_memory_stats()
    compiler=EpisodeCompiler(AutoTokenizer.from_pretrained(base,local_files_only=True));ledger=DiagnosticBudget(ledger_path)
    identity=dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),script_sha256=file_sha256(Path(__file__)),
                  architecture=model.architecture_identity(),world=args.world,timing_repeats=args.timing_repeats,scope='one held-out zh world per U/P/C; no M5 claim')
    identity=json.loads(json.dumps(identity))
    timing_path=Path(str(out)+'.timing.json')
    with torch.no_grad():
        if timing_path.exists():
            timing=json.loads(timing_path.read_text())
            if timing['identity']!=identity:raise ValueError('timing identity differs')
        else:
            row=shine_episode(0,task='U',seed=c['seed']);episode=compiler.compile(row)
            seconds=[];memories=None
            for repeat in range(args.timing_repeats+1):
                cost=sum(len(s.input_ids)+model.memory_tokens for s in episode.source)
                charge=ledger.reserve(run=str(out),kind='writer_latency',student=cost)
                torch.cuda.synchronize();start=time.perf_counter()
                memories,tokens=encode_memories(model,episode)
                torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
                ledger.commit(charge,student=tokens)
            payload_path=Path(str(out)+'.memory.pt');assert not payload_path.exists()
            torch.cuda.synchronize();start=time.perf_counter()
            exported=dict(format='shine-generated-memory-v1',identity=identity,memories=[x.state() for x in memories],
                          source_sha256=hashlib.sha256(json.dumps([s.input_ids for s in episode.source]).encode()).hexdigest())
            with payload_path.open('xb') as f:torch.save(exported,f);f.flush();os.fsync(f.fileno())
            export_seconds=time.perf_counter()-start
            start=time.perf_counter();saved=torch.load(payload_path,map_location='cpu',weights_only=True)
            restored=[GeneratedMemory.from_state(x,device) for x in saved['memories']]
            torch.cuda.synchronize();load_seconds=time.perf_counter()-start
            response=episode.responses[0];checks=[]
            for active in (memories,restored):
                upper=len(response.prompt_ids)+16;charge=ledger.reserve(run=str(out),kind='payload_roundtrip_query',student=upper)
                answer=model.generate(response.prompt_ids,memories=active,life_id=episode.episode_id,eos_token_id=compiler.tokenizer.eos_token_id)
                ledger.commit(charge,student=answer['input_tokens']);checks.append(answer)
            assert checks[0]==checks[1],'payload reload changed canonical generation'
            timing=dict(identity=identity,scope='loaded model, pretokenized source; encoding + M2P, no optimizer/query/disk in writer timer',
                        first_write_seconds=seconds[0],warm_write_seconds=seconds[1:],warm_median_seconds=statistics.median(seconds[1:]),
                        source_input_tokens=sum(len(s.input_ids) for s in episode.source),extra_memory_tokens=model.memory_tokens,
                        memory_payload_bytes=sum(x.payload_bytes() for x in memories),serialized_bytes=payload_path.stat().st_size,
                        serialized_sha256=file_sha256(payload_path),export_cpu_and_fsync_seconds=export_seconds,reload_to_device_seconds=load_seconds,
                        roundtrip_generation_exact=True,roundtrip_answer_correct=normalized_answer(compiler.tokenizer.decode(checks[0]['token_ids'],skip_special_tokens=True))==normalized_answer(row['probes'][0]['target']),
                        adapter_to_base_frobenius={str(k):update_norm(*memories[0].factors[k])/float(model._sites[k].original.weight.float().norm()) for k in model.config.layers})
            atomic_json(timing_path,timing)
        jobs=[f'{task}/{variant}/{probe}/{view}' for task in ['U','P','C'] for variant in [0,1] for probe in ['p1','p2'] for view in ['memory','off','foreign','plaintext']]
        journal=EvaluationJournal(str(out)+'.journal.json',identity,jobs,resume=args.resume);journal.install_signal_handlers();records=[]
        for task in ['U','P','C']:
            for variant in [0,1]:
                row=shine_episode(args.world,variant=variant,task=task,split='dev',seed=c['seed']);episode=compiler.compile(row)
                opposite=shine_episode(args.world,variant=1-variant,task=task,split='dev',seed=c['seed']);other=compiler.compile(opposite)
                written=foreign=None
                for response,probe in zip(episode.responses,row['probes']):
                    for view in ['memory','off','foreign','plaintext']:
                        def compute():
                            nonlocal written,foreign
                            source_tokens=0
                            if view in {'memory','foreign'} and written is None:
                                upper=sum(len(s.input_ids)+model.memory_tokens for s in episode.source)
                                charge=ledger.reserve(run=str(out),kind='probe_source',student=upper)
                                written,n=encode_memories(model,episode);ledger.commit(charge,student=n);source_tokens+=n
                            if view=='foreign' and foreign is None:
                                upper=sum(len(s.input_ids)+model.memory_tokens for s in other.source)
                                charge=ledger.reserve(run=str(out),kind='probe_foreign_source',student=upper)
                                values,n=encode_memories(model,other);ledger.commit(charge,student=n);source_tokens+=n
                                by_id={x.record_id:x for x in values}
                                foreign=[GeneratedMemory(x.record_id,x.life_id,by_id[x.record_id].factors) for x in written]
                            active=written if view=='memory' else foreign if view=='foreign' else []
                            query=evidence_teacher_response(compiler,row,probe) if view=='plaintext' else response
                            student=view in {'memory','foreign'};upper=len(query.prompt_ids)+16
                            charge=ledger.reserve(run=str(out),kind='probe_query',student=upper if student else 0,teacher=0 if student else upper)
                            answer=model.generate(query.prompt_ids,memories=active,life_id=episode.episode_id,eos_token_id=compiler.tokenizer.eos_token_id)
                            ledger.commit(charge,student=answer['input_tokens'] if student else 0,teacher=0 if student else answer['input_tokens'])
                            text=compiler.tokenizer.decode(answer['token_ids'],skip_special_tokens=True)
                            other_target=next(p['target'] for p in opposite['probes'] if p['probe_id']==probe['probe_id'])
                            return dict(task=task,variant=variant,probe_id=probe['probe_id'],pair_kind=row['annotations']['pair_kind'],view=view,
                                selected_records=[x.record_id for x in active],source_input_tokens=source_tokens,expected=probe['target'],text=text,
                                content_swap_relative_update_change=relative_update_change(written,foreign) if view=='foreign' else None,
                                exact_match=normalized_answer(text)==normalized_answer(probe['target']),
                                follows_foreign=normalized_answer(text)==normalized_answer(other_target),**answer)
                        records.append(journal.evaluate(f"{task}/{variant}/{probe['probe_id']}/{view}",compute))
        journal.finish()
        summary={task:{view:dict(queries=sum(x['task']==task and x['view']==view for x in records),
                  correct=sum(x['task']==task and x['view']==view and x['exact_match'] for x in records)) for view in ['memory','off','foreign','plaintext']} for task in ['U','P','C']}
        atomic_json(out,dict(identity=identity,summary=summary,records=records,timing=timing,
                            diagnostic_only=True,M5_passed=False,training_state_modified=False,
                            inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            memory_statistics_reset_after_checkpoint_restore=True))
        print(json.dumps(dict(summary=summary,timing=timing),indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except EvaluationPaused as exc:print(str(exc),flush=True)
