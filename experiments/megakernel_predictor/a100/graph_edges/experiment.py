"""Edge insertion with fixed CUDA node creation order and independent validation."""
import argparse
import ctypes as C
import hashlib
import json
import random
from pathlib import Path
from statistics import median
import time

HERE = Path(__file__).resolve().parent


def topology(n, edges):
    parents = [set() for _ in range(n)]
    for u,v in edges:
        parents[v].add(u)
    order, levels = [], [0]*n
    while len(order)<n:
        ready=[i for i in range(n) if i not in order and parents[i].issubset(order)]
        if not ready:
            raise ValueError("cycle")
        for i in ready:
            levels[i]=max((levels[p]+1 for p in parents[i]),default=0)
        order.extend(ready)
    return order,levels


def adjustments(n, edges, target, shift):
    if not shift:
        return []
    _,level=topology(n,edges)
    reachable={target}
    while True:
        larger=reachable|{v for u,v in edges if u in reachable}
        if larger==reachable:
            break
        reachable=larger
    return [(u,target) for u in range(n)
            if u not in reachable and level[u]==level[target]+shift-1 and (u,target) not in edges]


def case(seed, n=65536, gates=32, shape="layered", rounds=32):
    rng=random.Random(seed)
    p1,p2=[],[]
    for g in range(gates):
        if shape=="layered":
            width=8
            group=g//width
            prev=list(range((group-1)*width,group*width)) if group else []
        elif shape=="fork":
            prev=[] if g<8 else [g-8]
        else:
            prev=list(range(g)) if g>=4 else []
        chosen=rng.sample(prev,min(2,len(prev)))
        p1.append(chosen[0] if chosen else -1)
        p2.append(chosen[1] if len(chosen)>1 else -1)
    edges=[]
    for g in range(gates):
        edges.append((2*g,2*g+1))
        edges.extend((2*p+1,2*g+1) for p in (p1[g],p2[g]) if p>=0)
    return dict(name=f"{shape}-s{seed}-n{n}-r{rounds}",seed=seed,n=n,gates=gates,
                p1=p1,p2=p2,rounds=[rounds*rng.choice([1,1,4]) for _ in range(gates)],
                blocks=[rng.choice([8,32,108,216]) for _ in range(gates)],edges=edges)


def candidates(c, count=80, seed=123):
    rng=random.Random(seed);n=2*c['gates'];base=list(map(tuple,c['edges']))
    default=list(range(n))
    result=[dict(label="original",edges=base,order=default),
            dict(label="original-AA",edges=base,order=default)]
    # Independent construction-order controls; all extra-edge candidates keep default order.
    result.extend([dict(label="copies-first",edges=base,order=list(range(0,n,2))+list(range(1,n,2))),
                   dict(label="compute-first",edges=base,order=list(range(1,n,2))+list(range(0,n,2)))])
    for window in (1,2,4,8):
        edges=base+[(2*(g-window)+1,2*g) for g in range(window,c['gates'])]
        result.append(dict(label=f"window-{window}",edges=edges,order=default))
    seen={tuple(sorted(x['edges'])) for x in result}
    for k in range(count):
        additions=(1,2,4,8)[k%4];edges=base.copy()
        for _ in range(additions):
            available=[pair for target in range(n) for shift in (1,2)
                       for pair in adjustments(n,edges,target,shift)]
            if not available:break
            edges.append(rng.choice(available))
        key=tuple(sorted(edges))
        if key not in seen:
            seen.add(key);result.append(dict(label=f"random-{k}-e{len(edges)-len(base)}",edges=edges,order=default))
    return result


class Runtime:
    def __init__(self,c):
        self.c=c;self.lib=C.CDLL(str(HERE/'libgraph_runtime.so'))
        ptr=C.POINTER(C.c_int)
        self.lib.data_create.argtypes=[C.c_int,C.c_int,ptr,ptr,ptr,ptr];self.lib.data_create.restype=C.c_void_p
        self.lib.graph_create.argtypes=[C.c_void_p,C.c_int,ptr,ptr,ptr,C.c_int];self.lib.graph_create.restype=C.c_void_p
        self.lib.graph_run.argtypes=[C.c_void_p,C.c_int];self.lib.graph_run.restype=C.c_double
        self.lib.graph_check.argtypes=[C.c_void_p];self.lib.graph_check.restype=C.c_double
        self.lib.graph_destroy.argtypes=[C.c_void_p];self.lib.data_destroy.argtypes=[C.c_void_p]
        arrays=[self.arr(c[k]) for k in ('p1','p2','rounds','blocks')]
        self.handle=self.lib.data_create(c['gates'],c['n'],*arrays)
        if not self.handle:raise RuntimeError('data allocation failed')
        self.graphs=[]

    @staticmethod
    def arr(x):return (C.c_int*len(x))(*x)

    def graph(self,config,only=-1):
        edges=config['edges']
        p=self.lib.graph_create(self.handle,len(edges),self.arr([u for u,v in edges]),
                               self.arr([v for u,v in edges]),self.arr(config['order']),only)
        if not p:raise RuntimeError('graph build failed')
        self.graphs.append(p);return p

    def run(self,p,repeat=8):
        ms=self.lib.graph_run(p,repeat)
        if ms<0:raise RuntimeError('execution failed')
        return ms

    def check(self,p):
        error=self.lib.graph_check(p)
        if not 0<=error<=1e-5:raise RuntimeError(f'numerical/dependency check failed {error}')
        return error

    def release(self,p):
        self.lib.graph_destroy(p);self.graphs.remove(p)

    def close(self):
        for p in self.graphs:self.lib.graph_destroy(p)
        self.graphs=[];self.lib.data_destroy(self.handle)


def measure(c,configs,path,session,repeat=16,check=True):
    runtime=Runtime(c);rng=random.Random(901+ord(session[0])+c['seed'])
    build=list(range(len(configs)));rng.shuffle(build)
    plans={i:runtime.graph(configs[i]) for i in build}
    errors={i:runtime.check(plans[i]) for i in build} if check else {}
    for _ in range(4):
        for i in build:runtime.run(plans[i],8)
    solo=[]
    # Solo node estimates are separate executions, no events are inserted into target graphs.
    for i in range(2*c['gates']):
        p=runtime.graph(configs[0],only=i)
        runtime.run(p,10)
        solo.append(median(runtime.run(p,16) for _ in range(7)))
    meta=dict(type='meta',case=c,configs=configs,solo_ms=solo,session=session,
              sm_count=runtime.lib.sm_count(),errors=errors,inner_replay=8)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as out:
        out.write(json.dumps(meta)+'\n');out.flush()
        for r in range(repeat):
            order=list(plans);rng.shuffle(order)
            for j,i in enumerate(order):
                ms=runtime.run(plans[i],8)
                out.write(json.dumps(dict(type='timing',repeat=r,position=j,index=i,ms=ms,
                                          wall_time=time.time()))+'\n')
            out.flush()
    runtime.close()
    return meta


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--session',default='A');parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--n',type=int,default=65536);parser.add_argument('--rounds',type=int,default=32)
    parser.add_argument('--gates',type=int,default=32);parser.add_argument('--shape',default='layered')
    parser.add_argument('--candidates',type=int,default=80);parser.add_argument('--repeat',type=int,default=16)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--skip-check',action='store_true')
    args=parser.parse_args()
    if args.manifest:
        data=json.loads(args.manifest.read_text());c,configs=data['case'],data['configs']
    else:
        c=case(args.seed,args.n,args.gates,args.shape,args.rounds);configs=candidates(c,args.candidates)
    meta=measure(c,configs,args.out,args.session,args.repeat,not args.skip_check)
    args.out.with_suffix('.manifest.json').write_text(json.dumps({k:meta[k] for k in ('case','configs')},indent=2))
    print(json.dumps(dict(case=c['name'],session=args.session,configs=len(configs),file=str(args.out))),flush=True)

if __name__=='__main__':main()
