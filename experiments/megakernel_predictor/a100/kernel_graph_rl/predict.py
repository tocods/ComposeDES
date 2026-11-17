"""ComposeDES kernel-level admission model and shared-parameter updating."""
import ctypes as C
import itertools,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'fncs/orchestrator'))
from workflow_orchestrator import WorkflowController,COMPUTE_COMPLETED

class Predictor:
    def __init__(self,meta):
        self.meta=meta;self.params=list(itertools.product((.25,.5,1.),(0.,.15,.5),(0.,.001)))
        self.weights=np.ones(len(self.params))/len(self.params);self.cache={};self.seconds=0.;self.calls=0
        self.lib=C.CDLL(str(Path(__file__).with_name('libadmission.so')))
        self.lib.admission.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_int),
            C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_double,C.c_double,C.c_double,C.POINTER(C.c_double)]
        self.lib.admission.restype=C.c_double
        self.kind=(C.c_int*len(meta['nodes']))(*(1 if n['type']==1 else 0 for n in meta['nodes']))
        self.duration=(C.c_double*len(meta['nodes']))(*meta['duration_ms'])
        self.demand=(C.c_double*len(meta['nodes']))(*(min(1.,n['blocks']/max(1,n['resident_blocks_per_sm']*108))
            if n['type']==0 else .01 for n in meta['nodes']))
    def predict(self,dag,trace=False):
        key=tuple(sorted(dag.extra));start=time.perf_counter()
        if key in self.cache and not trace:return self.cache[key]
        edges=dag.meta['edges']+list(map(list,dag.extra));ints=lambda x:(C.c_int*len(x))(*x)
        u=ints([x[0] for x in edges]);v=ints([x[1] for x in edges]);ends=(C.c_double*dag.n)() if trace else None
        predictions=[]
        for p in self.params:
            value=self.lib.admission(dag.n,len(edges),u,v,self.kind,self.duration,self.demand,*p,ends)
            if value<0:raise RuntimeError('admission deadlock')
            predictions.append(value)
        self.cache[key]=np.array(predictions);self.seconds+=time.perf_counter()-start;self.calls+=1
        if trace:self.validate_with_composedes(dag,list(ends))
        return self.cache[key]
    def score(self,dag):return float(self.weights@self.predict(dag))
    def update(self,dag,base,measured,baseline):
        residual=self.predict(dag)/self.predict(base)-measured/baseline
        # Observation changes shared parameter weights, not a common time offset.
        logs=np.log(np.maximum(self.weights,1e-100))*.995-.5*(residual/.01)**2
        logs-=logs.max();self.weights=np.exp(logs);self.weights/=self.weights.sum()
    def save(self,path):
        Path(path).write_text(json.dumps(dict(parameters=self.params,weights=self.weights.tolist(),simulation_seconds=self.seconds,calls=self.calls),indent=2))
    def load(self,path):
        x=json.loads(Path(path).read_text());assert x['parameters']==[list(p) for p in self.params];self.weights=np.array(x['weights'])
    def validate_with_composedes(self,dag,ends):
        specs=[dict(name=str(i),host='gpu0',estimated_duration_ns=max(1,round(self.meta['duration_ms'][i]*1e6)),
                    children=[dict(child=str(j),size=0) for j in dag.children[i]]) for i in dag.order]
        wf=WorkflowController(specs,'qwen-kernel-admission');ready={int(c.event['payload']['task_id']):c for c in wf.initial_commands()}
        rank={i:j for j,i in enumerate(dag.order)}
        for i in sorted(range(dag.n),key=lambda i:(ends[i],rank[i])):
            c=ready.pop(i)
            event=dict(event_id=f'done/{i}',kind='compute.completed',correlation_id=c.event['correlation_id'],
                       payload=dict(task_id=str(i),attempt=c.event['payload']['attempt'],status='SUCCEEDED'))
            for c in wf.handle_event(COMPUTE_COMPLETED,event):ready[int(c.event['payload']['task_id'])]=c
        assert wf.done
