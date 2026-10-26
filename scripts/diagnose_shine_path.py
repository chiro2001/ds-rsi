#!/usr/bin/env python3
"""Pin a capacity checkpoint and compare parallel/cached gold-prefix logits."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dsrsi.data.compiler import EpisodeCompiler
from dsrsi.model.shine import ShineQwen
from dsrsi.state.streaming_cache import StreamingCache
from dsrsi.training.checkpoint import CheckpointStore, file_sha256
from dsrsi.training.device import resolve_device
from dsrsi.training.diagnostic_budget import DiagnosticBudget
from dsrsi.training.shine_pilot import DirectFactors, fixture_rows
from dsrsi.utils.atomic import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','run-dir','output','diagnostic-ledger'):p.add_argument('--'+name,required=True)
    p.add_argument('--contexts',type=int,default=4)
    args=p.parse_args()
    paths=[Path(v).resolve() for v in (args.base,args.run_dir,args.output,args.diagnostic_ledger)]
    if any(not v.is_relative_to(Path('/mnt/e/ds-rsi')) for v in paths):raise ValueError('CUDA paths must remain on E:')
    base,run,out,ledger_path=paths
    if out.exists() or args.contexts<1:raise ValueError('new output and positive context count required')
    device,_=resolve_device('cuda');torch.set_num_threads(2)
    store=CheckpointStore(run/'checkpoints');d,manifest=store.load_committed()
    config=manifest['identity']['train_config']
    if config['mode']!='oracle':raise ValueError('capacity diagnostic only')
    torch.manual_seed(config['seed'])
    payload=store.load_rank(d,manifest,rank=0)
    model=ShineQwen.from_local_checkpoint(base,device=device)
    if json.loads(json.dumps(model.architecture_identity()))!=manifest['identity']['architecture']:
        raise ValueError('architecture differs')
    rows=fixture_rows(worlds=config['worlds'],task=config['task'],seed=config['seed'])
    direct=DirectFactors(model,rows).to(device);direct.load_state_dict(payload['weights'],strict=True)
    compiler=EpisodeCompiler(AutoTokenizer.from_pretrained(base,local_files_only=True))
    ledger=DiagnosticBudget(ledger_path)
    records=[]
    with torch.no_grad():
        for number,row in enumerate(rows[:args.contexts]):
            episode=compiler.compile(row);memories=direct.memories(number,row)
            for response in episode.responses:
                ids=torch.tensor([response.input_ids],device=device);labels=response.labels
                charge=ledger.reserve(run=str(run),kind='parallel_vs_cached_gold',student=2*len(response.input_ids))
                # Project only supervised positions, retaining both numerical paths.
                positions=[i for i in range(len(labels)-1) if labels[i+1]!=-100]
                hidden=model(ids,memories=memories,life_id=episode.episode_id)[0][0]
                parallel=model.head(hidden[positions]).float()
                cache=StreamingCache(model.base.config.layer_types,window=2048,rebase_interval=None)
                full_ok=cache_ok=disagreement=count=0;max_delta=0.;nll_full=nll_cache=0.
                for t,token in enumerate(response.input_ids):
                    h,cache=model.cached(torch.tensor([[token]],device=device),cache,memories=memories,life_id=episode.episode_id)
                    if t+1<len(labels) and labels[t+1]!=-100:
                        gold=labels[t+1];cached=model.head(h[0,-1]).float();full=parallel[count]
                        full_ok+=int(full.argmax()==gold);cache_ok+=int(cached.argmax()==gold);count+=1
                        disagreement+=int(full.argmax()!=cached.argmax());max_delta=max(max_delta,float((full-cached).abs().max()))
                        nll_full+=float(-full.log_softmax(-1)[gold]);nll_cache+=float(-cached.log_softmax(-1)[gold])
                ledger.commit(charge,student=2*len(response.input_ids))
                records.append(dict(context=number,probe=response.event_id,tokens=count,parallel_correct=full_ok,cached_gold_correct=cache_ok,
                    top1_disagreements=disagreement,max_logit_delta=max_delta,parallel_nll=nll_full/count,cached_nll=nll_cache/count))
    report=dict(checkpoint=d.name,manifest_sha256=file_sha256(d/'manifest.json'),
        script_sha256=file_sha256(Path(__file__)),scope='Gold-prefix numerical diagnostic, not free-generation capability',
        records=records,total_tokens=sum(x['tokens'] for x in records),parallel_correct=sum(x['parallel_correct'] for x in records),
        cached_gold_correct=sum(x['cached_gold_correct'] for x in records),top1_disagreements=sum(x['top1_disagreements'] for x in records))
    atomic_json(out,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
