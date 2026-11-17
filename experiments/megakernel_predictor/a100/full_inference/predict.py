"""ComposeDES selection using independent whole-layer stage profiles."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'fncs/orchestrator'))
from workflow_orchestrator import WorkflowController,COMPUTE_COMPLETED


def simulate(meta,cfg,p):
    compute_penalty,copy_penalty,issue,priority_weight=p
    deps=cfg['deps'];n=len(deps);kinds=meta['kinds'];work=meta['solo_ms']
    children=[[] for _ in deps]
    for v,parents in enumerate(deps):
        for u in parents:children[u].append(v)
    topo=[]
    while len(topo)<n:
        ready=[i for i,parents in enumerate(deps) if i not in topo and all(p in topo for p in parents)]
        if not ready:raise ValueError('cycle')
        topo.extend(ready)
    tails=[0.]*n
    for i in reversed(topo):tails[i]=work[i]+max((tails[j] for j in children[i]),default=0.)
    specs=[dict(name=str(i),host='gpu0',estimated_duration_ns=round(work[i]*1e6),
                children=[dict(child=str(j),size=0) for j in children[i]]) for i in topo]
    wf=WorkflowController(specs,'full-qwen-inference')
    ready={int(c.event['payload']['task_id']):c for c in wf.initial_commands()}
    active={};now=0.;last=-issue;ranks={i:k/(n-1) for k,i in enumerate(cfg['order'])}
    while ready or active:
        key=lambda i:(priority_weight*(1-ranks[i])+(1-priority_weight)*tails[i]/max(tails),-i)
        for i in sorted(ready,key=key,reverse=True):
            if now+1e-9<last+issue:break
            # The complete model has one sequential compute chain and one H2D engine.
            if any(kinds[j]==kinds[i] for j in active):continue
            active[i]=[work[i],ready.pop(i)];last=now
        if not active:
            if ready and last+issue>now:
                now=last+issue
                continue
            raise RuntimeError('deadlock')
        mixed=len({kinds[i] for i in active})==2
        rates={i:1/(1+(copy_penalty if kinds[i]=='io' else compute_penalty)*mixed) for i in active}
        dt=min(x[0]/rates[i] for i,x in active.items())
        if ready and last+issue>now+1e-9:dt=min(dt,last+issue-now)
        now+=dt
        for i in active:active[i][0]-=dt*rates[i]
        for i in [j for j,x in active.items() if x[0]<1e-8]:
            _,command=active.pop(i)
            event=dict(event_id=f'done/{i}',kind='compute.completed',correlation_id=command.event['correlation_id'],
                       payload=dict(task_id=str(i),attempt=command.event['payload']['attempt'],status='SUCCEEDED'))
            for c in wf.handle_event(COMPUTE_COMPLETED,event):ready[int(c.event['payload']['task_id'])]=c
    assert wf.done
    return now


def read(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    m=rows[0];times={label:[] for label in m['labels']}
    for x in rows[1:]:
        if x['type']=='ttft':times[x['label']].append(x)
    return m,times,rows[1:]


def freeze(paths,out):
    grid=list(itertools.product((0.,.025,.1,.3),(0.,.05,.2),(0.,.002),(0.,1.)))
    losses=[]
    for path in paths:
        meta,t,_=read(path);y=np.array([np.median([x['gpu_ms'] for x in t[c['label']]]) for c in meta['configs']])
        preds=np.array([[simulate(meta,c,p) for c in meta['configs']] for p in grid])
        residual=(preds-y)/y[0];residual-=np.median(residual,axis=1,keepdims=True)
        losses.append(np.mean(abs(residual),axis=1));print('calibrated',path,flush=True)
    loss=np.mean(losses,axis=0);weights=np.exp(-(loss-loss.min())/.005);weights/=weights.sum()
    out.write_text(json.dumps(dict(parameters=grid,weights=weights.tolist(),loss=loss.tolist(),
                    calibration=[str(p) for p in paths]),indent=2))


def select(meta,model):
    start=time.perf_counter()
    predictions=np.array([[simulate(meta,c,p) for c in meta['configs']] for p in model['parameters']])
    scores=np.array(model['weights'])@predictions
    return dict(label=meta['configs'][int(scores.argmin())]['label'],prediction_ms=scores.tolist(),
                selection_seconds=time.perf_counter()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('files',nargs='+',type=Path)
    a=p.parse_args();freeze(a.files,a.out)
