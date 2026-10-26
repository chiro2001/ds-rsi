#!/usr/bin/env python3
"""Observe controller/log boundaries without importing Torch or using the GPU."""

import json
import os
from pathlib import Path
import time


def main():
    root=Path('/mnt/e/ds-rsi/reproduction/shine-qwen3-06b')
    out=root/'control/timing-observation.json';temporary=out.with_suffix('.tmp')
    assert not out.exists()
    data=dict(poll_seconds=.5,labels={},started_unix=time.time())
    controller=json.loads((root/'control/serial-controller-submit.json').read_text())['pid']
    while True:
        state=json.loads((root/'control/experiment-status.json').read_text());label=state['phase']
        if 'pid' in state:
            if label not in data['labels']:
                data['labels'][label]=dict(launch_unix=state['updated_unix'],pid=state['pid'])
            item=data['labels'][label]
            path=root/'logs'/f'{label}.log'
            if path.exists():
                for line in path.read_text().splitlines():
                    if not line.startswith('{'):continue
                    event=json.loads(line)
                    if event.get('event')=='micro':
                        key='micro_'+str(event['progress']['micro_step'])+'_observed_unix'
                        if key not in item:item[key]=time.time()
                        if 'first_micro_observed_unix' not in item:
                            item['first_micro_observed_unix']=item[key]
                            item['launch_to_first_micro_upper_seconds']=item[key]-item['launch_unix']
                    if event.get('event')=='saved' and event.get('status')=='paused' and 'paused_saved_observed_unix' not in item:
                        item['paused_saved_observed_unix']=time.time()
        if label=='finished':data['finished_unix']=time.time()
        process=Path('/proc')/str(controller)/'cmdline'
        alive=process.exists() and bool(process.read_bytes())
        if not alive:data['controller_exit_observed_unix']=time.time()
        temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,out)
        if label=='finished' or not alive:return
        if time.time()-data['started_unix']>7200:raise RuntimeError('observer cap reached; inspect controller')
        time.sleep(.5)


if __name__=='__main__':main()
