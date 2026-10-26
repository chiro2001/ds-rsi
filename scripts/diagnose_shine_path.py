#!/usr/bin/env python3
"""Pin a SHINE checkpoint and compare parallel/cached gold-prefix logits."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dsrsi.data.compiler import EpisodeCompiler,tokenizer_fingerprint
from dsrsi.model.shine import ShineConfig, ShineQwen
from dsrsi.state.streaming_cache import StreamingCache
from dsrsi.training.checkpoint import CheckpointStore, file_sha256
from dsrsi.training.device import resolve_device
from dsrsi.training.diagnostic_budget import DiagnosticBudget
from dsrsi.training.shine_pilot import DirectFactors, encode_memories, fixture_rows
from dsrsi.utils.atomic import atomic_json


def representation_summary(values):
    """Descriptive sensitivity; no claim that distance measures retained facts."""
    x=torch.stack(values).double().flatten(1)
    energy=float(x.square().sum())
    centered=x-x.mean(0,keepdim=True)
    eigen=torch.linalg.eigvalsh(centered@centered.T).clamp_min(0)
    probability=eigen/eigen.sum().clamp_min(1e-30)
    return dict(contexts=len(values), norm_mean=float(x.norm(dim=1).mean()),
                centered_energy_fraction=float(centered.square().sum())/max(energy,1e-30),
                centered_effective_rank=float((-(probability*probability.clamp_min(1e-30).log()).sum()).exp()) if float(eigen.sum())>1e-20 else 0.,
                counterfactual_relative_distances=[float((x[i]-x[i+1]).norm()/x[i].norm().clamp_min(1e-30))
                                                   for i in range(0,len(values),2)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','run-dir','output','diagnostic-ledger'):p.add_argument('--'+name,required=True)
    p.add_argument('--contexts',type=int,default=4)
    p.add_argument('--probes',choices=['all','p1'],default='all')
    p.add_argument('--capture-encoding',action='store_true')
    args=p.parse_args()
    paths=[Path(v).resolve() for v in (args.base,args.run_dir,args.output,args.diagnostic_ledger)]
    if any(not v.is_relative_to(Path('/mnt/e/ds-rsi')) for v in paths):raise ValueError('CUDA paths must remain on E:')
    base,run,out,ledger_path=paths
    if out.exists() or args.contexts<1:raise ValueError('new output and positive context count required')
    device,_=resolve_device('cuda');torch.set_num_threads(2)
    store=CheckpointStore(run/'checkpoints');d,manifest=store.load_committed()
    config=manifest['identity']['train_config']
    if config['mode'] not in {'oracle','generator'}:raise ValueError('unknown SHINE mode')
    code_root=Path(__file__).resolve().parents[1]
    for relative,expected in manifest['identity']['implementation'].items():
        if file_sha256(code_root/relative)!=expected:raise ValueError('training implementation differs')
    base_id=dict(tokenizer=tokenizer_fingerprint(base),weights={name:file_sha256(base/name)
                  for name in manifest['identity']['base']['weights']})
    if base_id!=manifest['identity']['base']:raise ValueError('base/tokenizer identity differs')
    torch.manual_seed(config['seed'])
    payload=store.load_rank(d,manifest,rank=0)
    cfg=dict(manifest['identity']['architecture']['config']);cfg['layers']=tuple(cfg['layers'])
    model=ShineQwen.from_local_checkpoint(base,device=device,config=ShineConfig(**cfg))
    from dsrsi.model.cuda_execution import require_reference_kernels,cuda_execution_identity
    require_reference_kernels(model)
    if cuda_execution_identity(device)!=manifest['identity']['runtime']:
        raise ValueError('SHINE diagnostic runtime differs')
    if json.loads(json.dumps(model.architecture_identity()))!=manifest['identity']['architecture']:
        raise ValueError('architecture differs')
    rows=fixture_rows(worlds=config['worlds'],task=config['task'],seed=config['seed'])
    if args.contexts>len(rows) or args.capture_encoding and (config['mode']!='generator' or args.contexts%2):
        raise ValueError('encoding summary requires paired generator contexts within fixture')
    if config['mode']=='oracle':
        direct=DirectFactors(model,rows).to(device);direct.load_state_dict(payload['weights'],strict=True)
    else:
        direct=None
        params=dict(model.named_parameters())
        trainable={n:v for n,v in params.items() if v.requires_grad}
        if set(trainable)!=set(payload['weights']):raise ValueError('trainable layout differs')
        with torch.no_grad():
            for n,v in trainable.items():v.copy_(payload['weights'][n].to(device))
    model.eval()
    before={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad}
    del payload
    compiler=EpisodeCompiler(AutoTokenizer.from_pretrained(base,local_files_only=True))
    ledger=DiagnosticBudget(ledger_path)
    records=[];encoded={k:[] for k in ['raw','enriched','a_factors','b_factors']};update_pairs=[]
    snapshots={}
    hooks=[]
    if args.capture_encoding:
        for index,site in model._sites.items():
            def capture_raw(_module,_inputs,output,index=index):
                if model._sites[index].mode=='write':
                    snapshots[index]=output[0,-model.memory_tokens:].detach().cpu().float()
            hooks.append(model.base.layers[index].register_forward_hook(capture_raw))
        def capture_enriched(_module,inputs):snapshots['enriched']=inputs[0].detach().cpu().float()
        hooks.append(model.a_head.register_forward_pre_hook(capture_enriched))
    # Admit the complete diagnostic before loading any source/query into the model.
    upper=0
    for row in rows[:args.contexts]:
        episode=compiler.compile(row)
        if direct is None:upper+=sum(len(s.input_ids)+model.memory_tokens for s in episode.source)
        upper+=sum(2*len(r.input_ids) for r in episode.responses if args.probes=='all' or r.event_id=='p1')
    budget=ledger.read()
    if sum(e['student'] for e in budget['entries'])+upper>budget['student_limit']:
        raise ValueError('complete path diagnostic exceeds remaining budget')
    free_path=run/f'evaluation-micro{manifest["progress"]["micro_step"]}.json'
    free=json.loads(free_path.read_text()) if free_path.exists() else None
    if free and direct is None:
        h=hashlib.sha256()
        for name,tensor in before.items():
            h.update(name.encode());h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        reference=json.loads(Path(str(free_path)+'.journal.json').read_text())
        if reference['identity']['weights_sha256']!=h.hexdigest():raise ValueError('free-generation reference weights differ')
    free_index={(r['context_index'],r['probe_id']):r for r in free['records'] if r['view']=='memory'} if free else {}
    with torch.no_grad():
        for number,row in enumerate(rows[:args.contexts]):
            episode=compiler.compile(row)
            if direct is None:
                cost=sum(len(s.input_ids)+model.memory_tokens for s in episode.source)
                source_charge=ledger.reserve(run=str(out),kind='path_probe_source_encoding',student=cost)
                memories,actual=encode_memories(model,episode);ledger.commit(source_charge,student=actual)
                if args.capture_encoding:
                    encoded['raw'].append(torch.stack([snapshots[i] for i in model.config.layers]))
                    encoded['enriched'].append(snapshots['enriched'])
                    encoded['a_factors'].append(torch.cat([a.detach().cpu().flatten() for a,b in memories[0].factors.values()]))
                    encoded['b_factors'].append(torch.cat([b.detach().cpu().flatten() for a,b in memories[0].factors.values()]))
                    if number%2==0:prior=memories[0].state()
                    else:
                        from evaluate_shine_small import relative_update_change
                        from dsrsi.model.shine import GeneratedMemory
                        update_pairs.append(relative_update_change([GeneratedMemory.from_state(prior,'cpu')],
                            [GeneratedMemory.from_state(memories[0].state(),'cpu')]))
            else:memories=direct.memories(number,row)
            for response in episode.responses:
                if args.probes=='p1' and response.event_id!='p1':continue
                ids=torch.tensor([response.input_ids],device=device);labels=response.labels
                charge=ledger.reserve(run=str(run),kind='parallel_vs_cached_gold',student=2*len(response.input_ids))
                # Project only supervised positions, retaining both numerical paths.
                positions=[i for i in range(len(labels)-1) if labels[i+1]!=-100]
                hidden=model(ids,memories=memories,life_id=episode.episode_id)[0][0]
                parallel=model.head(hidden[positions]).float()
                cache=StreamingCache(model.base.config.layer_types,window=2048,rebase_interval=None)
                full_ok=cache_ok=disagreement=count=0;max_delta=0.;nll_full=nll_cache=0.;token_records=[]
                for t,token in enumerate(response.input_ids):
                    h,cache=model.cached(torch.tensor([[token]],device=device),cache,memories=memories,life_id=episode.episode_id)
                    if t+1<len(labels) and labels[t+1]!=-100:
                        gold=labels[t+1];cached=model.head(h[0,-1]).float();full=parallel[count]
                        full_ok+=int(full.argmax()==gold);cache_ok+=int(cached.argmax()==gold);count+=1
                        disagreement+=int(full.argmax()!=cached.argmax());max_delta=max(max_delta,float((full-cached).abs().max()))
                        nll_full+=float(-full.log_softmax(-1)[gold]);nll_cache+=float(-cached.log_softmax(-1)[gold])
                        token_records.append(dict(position=count-1,gold=gold,
                            gold_text=compiler.tokenizer.decode([gold]),parallel_top1=int(full.argmax()),cached_top1=int(cached.argmax()),
                            parallel_gold_nll=float(-full.log_softmax(-1)[gold]),cached_gold_nll=float(-cached.log_softmax(-1)[gold])))
                ledger.commit(charge,student=2*len(response.input_ids))
                records.append(dict(context=number,probe=response.event_id,tokens=count,parallel_correct=full_ok,cached_gold_correct=cache_ok,
                    top1_disagreements=disagreement,max_logit_delta=max_delta,parallel_nll=nll_full/count,cached_nll=nll_cache/count,
                    expected=row['probes'][episode.responses.index(response)]['target'],
                    free_generation=free_index.get((number,response.event_id)),token_records=token_records))
    for hook in hooks:hook.remove()
    assert all(torch.equal(v.detach().cpu(),before[n]) for n,v in model.named_parameters() if v.requires_grad)
    report=dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),
        script_sha256=file_sha256(Path(__file__)),scope='Gold-prefix numerical diagnostic, not free-generation capability',
        mode=config['mode'],contexts=args.contexts,probes=args.probes,student_tokens_upper_bound=upper,
        training_parameters_unchanged=True,
        free_reference_sha256=file_sha256(free_path) if free else None,
        encoding={k:representation_summary(v) for k,v in encoded.items()} if args.capture_encoding else None,
        counterfactual_relative_update_changes=update_pairs,
        records=records,total_tokens=sum(x['tokens'] for x in records),parallel_correct=sum(x['parallel_correct'] for x in records),
        cached_gold_correct=sum(x['cached_gold_correct'] for x in records),top1_disagreements=sum(x['top1_disagreements'] for x in records))
    atomic_json(out,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
