"""Revision-pinned small-corpus curriculum for the official Qwen3 SHINE pilot."""

import hashlib
import json
from pathlib import Path
import random
import re

from dsrsi.data.shine import shine_episode


DATA_VERSION='official-shine-small-corpus-recon-comp-mqa-v1'


def normalized(text):return re.sub(r'\s+',' ',text).strip().casefold()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def setup_tokenizer(base):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,padding_side='left')
    tokenizer.add_tokens(['<RECON>','<COMP>','<NOTHING>'])
    tokenizer.chat_template=(Path(__file__).parents[1]/'vendor/shine/chat_template.jinja').read_text()
    return tokenizer


def chat(tokenizer,messages,*,prompt=False):
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=prompt,enable_thinking=False)


def labels_for(tokenizer,ids):
    """Upstream mask_label semantics, including EOS and the following newline."""
    labels=[-100]*len(ids)
    start=tokenizer.convert_tokens_to_ids('<|im_start|>')
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    assistant=tokenizer.convert_tokens_to_ids('assistant')
    last=len(ids)
    for j in range(len(ids)-1,0,-1):
        if ids[j]==end:last=j
        elif ids[j]==assistant and ids[j-1]==start:
            labels[j+2:min(last+2,len(ids))]=ids[j+2:min(last+2,len(ids))]
    if not any(t!=-100 for t in labels):raise ValueError('no supervised response')
    return labels


def compile_messages(tokenizer,source,messages):
    ids=chat(tokenizer,messages)
    return dict(source_ids=tokenizer.encode(source,add_special_tokens=False),input_ids=ids,
                labels=labels_for(tokenizer,ids))


def token_cost(record,memory_tokens=176,*,checkpoint=False):
    n=len(record['source_ids'])+memory_tokens+len(record['input_ids'])
    # Conservative accounting: explicitly include rematerialized backbone inputs
    # if activation checkpointing is enabled, in addition to semantic forwards.
    return n*(2 if checkpoint else 1)


def short_text(tokenizer,text,limit):
    words=text.split()
    while words and len(tokenizer.encode(' '.join(words),add_special_tokens=False))>limit:
        words=words[:max(1,len(words)*limit//max(limit+1,len(tokenizer.encode(' '.join(words),add_special_tokens=False))))-1]
    return ' '.join(words)


def prepare_corpus(prefix,tokenizer):
    raw=json.loads(Path(prefix).read_text());texts=[];seen=set()
    for item in raw['pretrain']:
        text=short_text(tokenizer,item['text'],96)
        if len(text.split())<12 or normalized(text) in seen:continue
        seen.add(normalized(text));texts.append(dict(id=f"transmla-{item['upstream_index']}",source=text))
        if len(texts)==80:break
    if len(texts)!=80:raise ValueError('not enough distinct pretraining records')
    natural=[];seen=set()
    for number,item in enumerate(raw['mqa']):
        text=short_text(tokenizer,item['context'],192)
        if normalized(text) in seen:continue
        queries=[];conversation=item['conversations']
        for a,b in zip(conversation[::2],conversation[1::2]):
            if a['role']!='user' or b['role']!='assistant':raise ValueError('upstream conversation schema differs')
            if (normalized(b['content']) in normalized(text) and len(tokenizer.encode(b['content']))<=20
                    and len(tokenizer.encode(a['content']))<=48):
                queries.append(dict(question=a['content'],target=b['content']))
            if len(queries)==2:break
        if len(queries)!=2:continue
        seen.add(normalized(text));natural.append(dict(id=f'mqa-{number}',kind='natural',source=text,queries=queries))
        if len(natural)==24:break
    if len(natural)!=24:raise ValueError(f'not enough supported two-question passages: {len(natural)}')
    synthetic={}
    for split,count in [('train',8),('dev',4)]:
        synthetic[split]=[]
        for world in range(count):
            for variant in [0,1]:
                row=shine_episode(world,variant=variant,task='U',split=split,language='zh',seed=20261026)
                synthetic[split].append(dict(id=row['episode_id'],kind='synthetic',world=world,variant=variant,
                    source=row['source_events'][0]['content'],queries=[dict(question=p['query'],target=p['target']) for p in row['probes']]))
    train=natural[:16]+synthetic['train'];dev=natural[16:]+synthetic['dev']
    train_sources={normalized(x['source']) for x in texts[:64]+train}
    dev_sources={normalized(x['source']) for x in texts[64:]+dev}
    if train_sources & dev_sources:raise ValueError('exact source overlap across splits')
    return dict(version=DATA_VERSION,upstream={k:v for k,v in raw.items() if k not in ['pretrain','mqa']},
        pretrain_train=texts[:64]+[dict(id=x['id'],source=x['source']) for x in train],
        pretrain_dev=texts[64:],qa_train=train,qa_dev=dev,
        exact_source_overlap=0,selection='Pinned prefixes; 96-token pretraining texts, <=192-token MQA prefixes with two supported short spans; synthetic U fixtures',
        limitations=['Small selected corpus; not the paper benchmark or full 6B-token pretraining.',
                     'Exact source deduplication does not rule out semantic or upstream pretraining overlap.'])


def training_record(corpus,tokenizer,micro,pretrain_steps,seed=20261026):
    if micro<pretrain_steps:
        rows=corpus['pretrain_train'];row=rows[(micro*7)%len(rows)]
        rng=random.Random(seed+micro)
        completion=rng.random()<.5
        text=row['source'];source=text
        if completion:
            words=text.split();cut=max(1,min(len(words)-1,round(len(words)*(1-rng.uniform(.1,.3)))))
            source=' '.join(words[:cut])
        task='<COMP>' if completion else '<RECON>'
        messages=[dict(role='user',content=task),dict(role='assistant',content=text)]
        result=compile_messages(tokenizer,source,messages)
        result.update(stage='pretrain',task=task,row_id=row['id'])
    else:
        rows=corpus['qa_train'];row=rows[((micro-pretrain_steps)*7)%len(rows)]
        messages=[]
        for q in row['queries']:messages.extend([dict(role='user',content=q['question']),dict(role='assistant',content=q['target'])])
        result=compile_messages(tokenizer,row['source'],messages)
        result.update(stage='qa',task='MQA',row_id=row['id'])
    return result


def plan_steps(corpus,tokenizer,pretrain_budget=155000,qa_budget=80000,grad_accum=4):
    used=0;steps=0
    while True:
        costs=[token_cost(training_record(corpus,tokenizer,steps+i,10**9)) for i in range(grad_accum)]
        if used+sum(costs)>pretrain_budget:break
        used+=sum(costs);steps+=grad_accum
    pretrain_steps=steps;pretrain_tokens=used;used=0
    while True:
        costs=[token_cost(training_record(corpus,tokenizer,steps+i,pretrain_steps)) for i in range(grad_accum)]
        if used+sum(costs)>qa_budget:break
        used+=sum(costs);steps+=grad_accum
    return dict(pretrain_steps=pretrain_steps,total_steps=steps,pretrain_tokens=pretrain_tokens,
                qa_tokens=used,total_train_tokens=pretrain_tokens+used,grad_accum=grad_accum)
