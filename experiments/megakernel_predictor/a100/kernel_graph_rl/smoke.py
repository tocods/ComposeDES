import argparse,json,time
from pathlib import Path
from collections import Counter
import torch
from capture_model import Model,Request,Forward
from gpu import Runtime
from dag import DAG

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);m=Model('/root/autodl-tmp/models/Qwen3-8B');r=Request(m,512,1,101,4)
    f=Forward(r);g=f.capture();rt=Runtime(g);meta=rt.export(a.out/'graph.json');dag=DAG(meta)
    plan=rt.make();print('loaded',m.layers,m.params,'nodes',dag.n,'edges',len(meta['edges']),'types',Counter(n['type'] for n in meta['nodes']),'movable',len(dag.available()),'max level width',max(map(len,dag.buckets.values())),flush=True)
    print('original ms',[rt.run(plan) for _ in range(3)],flush=True)
    import random
    changed,history=dag.random(random.Random(13),8);new=rt.make(changed.extra)
    print('8-edge ms',[rt.run(new) for _ in range(3)],flush=True)
    if a.profile:
        meta['duration_ms'],meta['profiling_seconds']=rt.profile(dag.n);print('profile seconds',meta['profiling_seconds'],flush=True)
    meta.update(layers=m.layers,parameters=m.params,prefix=512,suffix=1,readback=False,fingerprint=dag.fingerprint(),sample_edges=changed.extra)
    (a.out/'graph.json').write_text(json.dumps(meta))
    rt.close(new);rt.close(plan);print('complete',flush=True)

if __name__=='__main__':main()
