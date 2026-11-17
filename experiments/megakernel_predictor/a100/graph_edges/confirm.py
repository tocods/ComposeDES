"""Independent-instance validation with longer replay batches after noisy A/A.

Configurations are frozen by calibration and process A, not reselected using B.
Each logical graph has three independently instantiated replicas. All are checked
from poisoned buffers. Timing batches contain 64 replays, with 32 untimed warmup
replays immediately beforehand. Report this separately from the original protocol.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from experiment import Runtime
from predict import read
from analyze import interval,hierarchical_interval


def one(root,index):
    summary=json.loads((root/'summary.json').read_text())['results'][index]
    meta,_,_=read(root/f'test-{index}-A.jsonl')
    requested={p['index'] for p in summary['policies'].values()}|{0,1,2,3,6,7}
    alias={};unique={}
    for i in sorted(requested):
        cfg=meta['configs'][i];key=(tuple(cfg['order']),tuple(sorted(map(tuple,cfg['edges']))))
        alias[i]=unique.setdefault(key,i)
    ids=sorted(set(alias.values()));rt=Runtime(meta['case']);rng=random.Random(9000+index)
    keys=[(i,k) for i in ids for k in range(3)];rng.shuffle(keys)
    plans={key:rt.graph(meta['configs'][key[0]]) for key in keys}
    errors={str(key):rt.check(p) for key,p in plans.items()}
    for _ in range(3):
        for p in plans.values():rt.run(p,64)
    result=dict(type='meta',case=meta['case'],ids=ids,aliases=alias,replicas=3,inner=64,
                untimed_warmup=32,repetitions=32,errors=errors,policies=summary['policies'])
    with (root/f'confirm-{index}.jsonl').open('w') as out:
        out.write(json.dumps(result)+'\n');out.flush()
        for r in range(32):
            rng.shuffle(keys)
            for pos,(i,k) in enumerate(keys):
                p=plans[(i,k)];rt.run(p,32);ms=rt.run(p,64)
                out.write(json.dumps(dict(type='timing',repeat=r,position=pos,index=i,replica=k,ms=ms))+'\n')
            out.flush()
    rt.close();print('confirmation complete',index,flush=True)


def summarize(root):
    results=[]
    for index in range(6):
        rows=[json.loads(x) for x in (root/f'confirm-{index}.jsonl').read_text().splitlines()];m=rows[0]
        timings={i:np.array([[next(x['ms'] for x in rows[1:] if x['index']==i and x['repeat']==r and x['replica']==k)
                             for k in range(3)] for r in range(32)]) for i in m['ids']}
        base=timings[0].mean(axis=1);policies={}
        for name,p in m['policies'].items():
            i=int(m['aliases'][str(p['index'])]);v=timings[i].mean(axis=1)
            policies[name]=dict(index=i,label=p['label'],ms=float(np.median(v)),
                 extra_edges=p['extra_edges'],same_graph_as_original=p['same_graph_as_original'],
                 **hierarchical_interval(timings[0],timings[i],seed=100+index))
        aa=interval(timings[0][:,0],timings[0][:,1],seed=200+index)
        # Genuine paired A/A: compare separately instantiated copies, not an alias of the same measurement.
        policies['AA']=dict(ms=float(np.median(timings[0][:,1])),**aa)
        results.append(dict(case=m['case'],policies=policies,replica_medians_ms={str(i):np.median(x,axis=0).tolist() for i,x in timings.items()}))
    (root/'confirmation.json').write_text(json.dumps(dict(protocol='three independent instances, 32 paired blocks, 64 replays per timing, 32 warmup',results=results),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:one(a.root,a.index)
    else:
        for i in range(6):subprocess.run([sys.executable,__file__,str(a.root),'--index',str(i)],check=True)
        summarize(a.root)
