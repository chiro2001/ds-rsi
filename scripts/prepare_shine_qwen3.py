#!/usr/bin/env python3
"""Stream revision-pinned Qwen3 files to E: without caching model weights locally."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.parse
import urllib.request


MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
PRETRAIN_REVISION = '92e313b9dd8eca6165f2ccd4d07e316712f26bd3'
MQA_REVISION = '6f07d80835a9c124ce5b56b197dd2aa137a188df'
RECEIVER = '''
import sys, pathlib, hashlib, os, json, shutil
p=pathlib.Path(sys.argv[1]); size=int(sys.argv[2]); expected=sys.argv[3]
assert str(p).startswith('/mnt/e/ds-rsi/reproduction/shine-qwen3-06b/')
assert shutil.disk_usage('/mnt/e/ds-rsi').free > 20*(1<<30)+size
p.parent.mkdir(parents=True,exist_ok=True)
assert not p.exists(), 'destination already exists'
tmp=p.with_name(p.name+'.part'); h=hashlib.sha256(); count=0
with tmp.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(1<<20)
  if not b: break
  f.write(b); h.update(b); count+=len(b)
 f.flush(); os.fsync(f.fileno())
assert count==size, (count,size)
assert expected=='-' or h.hexdigest()==expected, 'download checksum differs'
os.replace(tmp,p)
print(json.dumps(dict(file=str(p),bytes=count,sha256=h.hexdigest())),flush=True)
'''


def request(url):
    return urllib.request.urlopen(url,timeout=60)


def remote_bytes(host, path, data):
    command='python3 -c '+shlex.quote(RECEIVER)+' '+shlex.quote(str(path))+' '+str(len(data))+' '+hashlib.sha256(data).hexdigest()
    result=subprocess.run(['ssh',host,command],input=data,capture_output=True,check=True)
    return json.loads(result.stdout)


def download(host,path,url,size,sha):
    command='python3 -c '+shlex.quote(RECEIVER)+' '+shlex.quote(str(path))+' '+str(size)+' '+sha
    worker=subprocess.Popen(['ssh',host,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with request(url) as response:
            while chunk:=response.read(1<<20):worker.stdin.write(chunk)
        worker.stdin.close()
        output=worker.stdout.read();error=worker.stderr.read();status=worker.wait()
        if status:raise RuntimeError(error.decode())
        return json.loads(output)
    except BaseException:
        worker.stdin.close();worker.wait();raise


def json_prefix(url,count):
    decoder=json.JSONDecoder();buffer='';items=[]
    with request(url) as response:
        assert response.read(1)==b'['
        while len(items)<count:
            buffer=buffer.lstrip(' \r\n\t,')
            try:
                value,end=decoder.raw_decode(buffer)
                items.append(value);buffer=buffer[end:]
            except json.JSONDecodeError:
                chunk=response.read(65536)
                if not chunk:break
                buffer+=chunk.decode('utf-8')
    if len(items)!=count:raise ValueError('upstream JSON prefix shorter than requested')
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='wsl5080')
    p.add_argument('--root',default='/mnt/e/ds-rsi/reproduction/shine-qwen3-06b')
    p.add_argument('--part',choices=['model','data'],required=True)
    args=p.parse_args();root=Path(args.root)
    if str(root)!='/mnt/e/ds-rsi/reproduction/shine-qwen3-06b':raise ValueError('authorized root required')
    if args.part=='model':
        with request(f'https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/{MODEL_REVISION}?blobs=true') as f:meta=json.load(f)
        assert meta['sha']==MODEL_REVISION
        files=[]
        names={'LICENSE','config.json','generation_config.json','merges.txt','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json'}
        for record in meta['siblings']:
            name=record['rfilename']
            if name not in names:continue
            url=f'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{MODEL_REVISION}/{name}'
            lfs=record.get('lfs')
            if lfs:entry=download(args.host,root/'model'/name,url,lfs['size'],lfs['sha256'])
            else:
                with request(url) as f:data=f.read()
                entry=remote_bytes(args.host,root/'model'/name,data)
            files.append(entry);print(json.dumps(entry),flush=True)
        assert {Path(f['file']).name for f in files}==names
        manifest=dict(repository='Qwen/Qwen3-0.6B',revision=MODEL_REVISION,files=files)
        print(remote_bytes(args.host,root/'model/download-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode()))
    else:
        texts=[]
        for offset in range(0,512,100):
            query=urllib.parse.urlencode(dict(dataset='fxmeng/transmla_pretrain_6B_tokens',config='default',split='train',offset=offset,length=min(100,512-offset)))
            with request('https://datasets-server.huggingface.co/rows?'+query) as f:
                assert f.headers['x-revision']==PRETRAIN_REVISION
                rows=json.load(f)['rows']
            texts.extend(dict(upstream_index=r['row_idx'],**r['row']) for r in rows)
        url=f'https://huggingface.co/datasets/Yewei-Liu/ift_mqa_collection/resolve/{MQA_REVISION}/ift_pwc.json'
        mqa=json_prefix(url,512)
        value=dict(pretrain_repository='fxmeng/transmla_pretrain_6B_tokens',pretrain_revision=PRETRAIN_REVISION,
                   pretrain=texts,mqa_repository='Yewei-Liu/ift_mqa_collection',mqa_revision=MQA_REVISION,mqa=mqa)
        print(remote_bytes(args.host,root/'data/upstream-prefix.json',(json.dumps(value,ensure_ascii=False)+'\n').encode()),flush=True)


if __name__=='__main__':main()
