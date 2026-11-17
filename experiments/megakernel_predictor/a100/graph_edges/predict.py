"""ComposeDES DAG replay with a small shared GPU admission/slowdown model.

Fit only calibration graphs. Target input consists of graph structure and solo
node profiles; target end-to-end measurements are evaluation data, never fit data.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
from statistics import median
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'fncs'/'orchestrator'))
from workflow_orchestrator import WorkflowController,COMPUTE_COMPLETED
from experiment import topology


def simulate(meta,config,parameters):
    slots,interference,gap,copy_penalty,critical_weight=parameters
    work=meta['solo_ms'];n=len(work);c=meta['case'];edges=config['edges'];order=config['order']
    topo,_=topology(n,edges)
    children=[[] for _ in range(n)]
    for u,v in edges:children[u].append(v)
    tail=[0.]*n
    for i in reversed(topo):tail[i]=work[i]+max((tail[j] for j in children[i]),default=0.)
    specs=[dict(name=str(i),host='gpu0',estimated_duration_ns=round(work[i]*1e6),
                children=[dict(child=str(j),size=0) for j in children[i]]) for i in topo]
    wf=WorkflowController(specs,'cuda-edge-insertion')
    ready={int(x.event['payload']['task_id']):x for x in wf.initial_commands()}
    active={};now=0.;last_launch=-gap
    ranks={x:i/max(1,n-1) for i,x in enumerate(order)}
    claims={i:min(1.,c['blocks'][i//2]/(meta['sm_count']*4)) for i in range(1,n,2)}
    while ready or active:
        running_compute=[i for i in active if i%2]
        running_copy=[i for i in active if not i%2]
        priority=lambda i: (critical_weight*tail[i]/max(tail)-(1-critical_weight)*ranks[i],-i)
        for i in sorted(ready,key=priority,reverse=True):
            if now+1e-10<last_launch+gap:break
            fits=(not running_copy) if not i%2 else (
                len(running_compute)<slots and sum(claims[j] for j in running_compute)+claims[i]<=1.01)
            if fits:
                active[i]=[work[i],ready.pop(i)];last_launch=now
                (running_compute if i%2 else running_copy).append(i)
        if not active:raise RuntimeError('admission deadlock')
        nc=len(running_compute);ni=len(running_copy)
        rates={i:(1/(1+copy_penalty*nc)) if not i%2 else
               1/(1+interference*max(0,nc-1)+copy_penalty*ni) for i in active}
        dt=min(v[0]/rates[i] for i,v in active.items())
        # Only wake for a launch slot when at least one waiting node could fit.
        if ready and last_launch+gap>now+1e-10:
            dt=min(dt,last_launch+gap-now)
        now+=dt
        for i in active:active[i][0]-=dt*rates[i]
        done=[i for i,v in active.items() if v[0]<1e-9]
        for i in done:
            _,command=active.pop(i)
            event=dict(event_id=f'done/{i}',kind='compute.completed',
                       correlation_id=command.event['correlation_id'],
                       payload=dict(task_id=str(i),attempt=command.event['payload']['attempt'],status='SUCCEEDED'))
            for x in wf.handle_event(COMPUTE_COMPLETED,event):ready[int(x.event['payload']['task_id'])]=x
    assert wf.done
    return now


def read(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    m=rows[0];trials=[[] for _ in m['configs']]
    for x in rows[1:]:trials[x['index']].append(x['ms'])
    return m,np.array([median(x) for x in trials]),trials


def freeze(paths,out):
    grid=list(itertools.product((1,2,4), (0.,.3,1.), (0.,.002), (0.,.2), (0.,1.)))
    losses=np.zeros(len(grid));count=0
    for path in paths:
        meta,y,_=read(path)
        # All graph candidates have equal work; offset is removed to fit contrasts.
        for j,p in enumerate(grid):
            pred=np.array([simulate(meta,c,p) for c in meta['configs']])
            residual=(pred-y)/y[0];residual-=np.median(residual)
            losses[j]+=np.mean(np.abs(residual))
        count+=1
        print('calibrated',Path(path).name,flush=True)
    losses/=count
    weights=np.exp(-(losses-losses.min())/.01);weights/=weights.sum()
    model=dict(parameters=grid,weights=weights.tolist(),losses=losses.tolist(),
               calibration_files=[str(p) for p in paths],best=grid[int(losses.argmin())])
    out.write_text(json.dumps(model,indent=2));return model


def predict(meta,model):
    all_predictions=np.array([[simulate(meta,c,p) for c in meta['configs']] for p in model['parameters']])
    return np.array(model['weights'])@all_predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','predict'])
    p.add_argument('--model',type=Path,required=True);p.add_argument('files',type=Path,nargs='+')
    a=p.parse_args()
    if a.mode=='freeze':freeze(a.files,a.model)
    else:
        model=json.loads(a.model.read_text())
        for path in a.files:
            meta,y,_=read(path);pred=predict(meta,model);i=int(pred.argmin())
            output=dict(file=str(path),prediction_ms=pred.tolist(),selected=i,label=meta['configs'][i]['label'],
                        measured_ms=float(y[i]),baseline_ms=float(y[0]),best_ms=float(min(y)),
                        improvement=float(1-y[i]/y[0]),regret=float(y[i]/min(y)-1))
            path.with_suffix('.prediction.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))

if __name__=='__main__':main()
