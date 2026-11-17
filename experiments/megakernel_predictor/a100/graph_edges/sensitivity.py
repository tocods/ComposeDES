"""Keep graph instances fixed while varying replay count inside endpoint timing."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from experiment import Runtime
from predict import read
from analyze import hierarchical_interval


def aggregate(meta,records,ids):
    by_inner={}
    for n in (1,8,64):
        values={i:np.array([[next(x['ms'] for x in records if x['index']==i and x['inner']==n and x['repeat']==r and x['replica']==k)
                           for k in range(3)] for r in range(48)]) for i in ids}
        by_inner[n]={str(i):dict(label=meta['configs'][i]['label'],ms=float(np.median(v.mean(axis=1))),
                                extra_edges=len(meta['configs'][i]['edges'])-len(meta['case']['edges']),
                                **hierarchical_interval(values[0],v)) for i,v in values.items()}
    return dict(case=meta['case'],measurements=by_inner)


def summarize(root):
    results=[]
    for index in (0,1,2):
        rows=[json.loads(x) for x in (root/f'sensitivity-{index}.jsonl').read_text().splitlines()]
        meta,_,_=read(root/f'test-{index}-A.jsonl')
        results.append(aggregate(meta,rows[1:],rows[0]['indices']))
    (root/'sensitivity.json').write_text(json.dumps(results,indent=2))


def run(root):
    results=[]
    for index in (0,1,2):
        meta,_,_=read(root/f'test-{index}-A.jsonl')
        summary=json.loads((root/'summary.json').read_text())['results'][index]
        sparse=summary['policies']['sparse_edge_search_A']['index']
        ids=sorted({0,2,6,7,sparse});rng=random.Random(2500+index);rt=Runtime(meta['case'])
        keys=[(i,k) for i in ids for k in range(3)];rng.shuffle(keys)
        plans={key:rt.graph(meta['configs'][key[0]]) for key in keys}
        for p in plans.values():rt.check(p);rt.run(p,64)
        records=[]
        with (root/f'sensitivity-{index}.jsonl').open('w') as out:
            out.write(json.dumps(dict(type='meta',case=meta['case'],indices=ids,replicas=3,repeats=48,
                      warmup=32,inner_replays=[1,8,64]))+'\n')
            for rep in range(48):
                todo=[(i,k,n) for i,k in keys for n in (1,8,64)];rng.shuffle(todo)
                for pos,(i,k,n) in enumerate(todo):
                    p=plans[(i,k)];rt.run(p,32);ms=rt.run(p,n)
                    row=dict(type='timing',repeat=rep,position=pos,index=i,replica=k,inner=n,ms=ms)
                    records.append(row);out.write(json.dumps(row)+'\n')
                out.flush()
        rt.close()
        results.append(aggregate(meta,records,ids));print('sensitivity complete',index,flush=True)
    (root/'sensitivity.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--summarize',action='store_true')
    a=p.parse_args();(summarize if a.summarize else run)(a.root)
