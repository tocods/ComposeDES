"""Random auxiliary-edge discovery on an actual full-model CUDA kernel graph."""
import argparse,json,random,time,hashlib
from pathlib import Path
from statistics import median
import torch
from capture_model import Model,Request,Forward
from gpu import Runtime
from dag import DAG

def prepare(prefix,suffix,seed,out,readback=False,output_tokens=4):
    start=time.perf_counter();torch.set_num_threads(4)
    m=Model('/root/autodl-tmp/models/Qwen3-8B');r=Request(m,prefix,suffix,seed,output_tokens)
    source=Forward(r,parallel=True,readback=readback);rt=Runtime(source.capture());meta=rt.export(out/'graph.json');dag=DAG(meta)
    original=rt.make();rt.run(original,128)
    meta['duration_ms'],meta['profiling_seconds']=rt.profile(dag.n)
    rt.run(original,64)
    serial=Forward(r,parallel=False,readback=readback);serial_rt=Runtime(serial.capture());serial_plan=serial_rt.make()
    serial_meta=serial_rt.export(out/'serial-graph.json')
    # Compile both source graphs before warming / timing any candidate.
    rt.run(original,64);serial_rt.run(serial_plan,64)
    meta.update(layers=m.layers,parameters=m.params,prefix=prefix,suffix=suffix,seed=seed,readback=readback,
                fingerprint=dag.fingerprint(),performance_only=True,source_node_count=dag.n,
                source_edge_count=len(meta['edges']),movable_kernel_count=len(dag.available()),
                profiling_node_evaluations=sum(n['type'] in (0,1,2) for n in meta['nodes']),profiling_replays_per_node=6,
                setup_seconds=time.perf_counter()-start)
    (out/'graph.json').write_text(json.dumps(meta))
    return r,source,rt,original,serial,serial_rt,serial_plan,meta

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prefix',type=int,default=512)
    p.add_argument('--suffix',type=int,default=1);p.add_argument('--seed',type=int,default=701)
    p.add_argument('--samples',type=int,default=10000);p.add_argument('--readback',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    values=prepare(a.prefix,a.suffix,a.seed,a.out,a.readback);req,source,rt,base,serial,srt,sbase,meta=values
    rng=random.Random(a.seed);dag=DAG(meta);aa=rt.make()
    initial,history=dag.random(random.Random(13),8,enumerate_targets=True);extra=rt.make(initial.extra)
    handles={'original':(rt,base),'AA':(rt,aa),'random8':(rt,extra),'serial':(srt,sbase)}
    control=[]
    for rep in range(32):
        order=list(handles);rng.shuffle(order)
        for label in order:
            runtime,plan=handles[label];control.append(dict(repeat=rep,label=label,ms=runtime.run(plan,3)))
    (a.out/'initial-control.json').write_text(json.dumps(dict(rows=control,random8_edges=initial.extra),indent=2))
    print('control', {k:median(x['ms'] for x in control if x['label']==k) for k in handles},flush=True)
    rt.close(extra)
    seen=set();best=[];start=time.perf_counter()
    with (a.out/'random-search.jsonl').open('w') as log:
        for i in range(a.samples):
            while True:
                d,h=dag.random(rng,rng.choice((1,2,4,8,16,32)),enumerate_targets=True);key=tuple(sorted(d.extra))
                if key not in seen:seen.add(key);break
            plan=rt.make(d.extra);ms=rt.run(plan,3);rt.close(plan)
            row=dict(index=i,edges=d.extra,actions=h,ms=ms,replays=3,warmups=1)
            log.write(json.dumps(row)+'\n')
            best.append((ms,d.extra));best=sorted(best,key=lambda x:x[0])[:16]
            if i%32==0:
                log.write(json.dumps(dict(index=i,control=True,ms=rt.run(base,3)))+'\n');log.flush()
            if (i+1)%100==0:print('samples',i+1,'best',best[0][0],'seconds',time.perf_counter()-start,flush=True)
    (a.out/'shortlist.json').write_text(json.dumps(dict(best=best,seconds=time.perf_counter()-start,samples=a.samples,seed=a.seed),indent=2))
    plans={'original':base,'AA':aa};runtimes={k:rt for k in plans};plans['serial']=sbase;runtimes['serial']=srt
    for j,(score,edges) in enumerate(best):plans[f'best-{j}']=rt.make(edges);runtimes[f'best-{j}']=rt
    confirm=[]
    for rep in range(32):
        order=list(plans);rng.shuffle(order)
        for label in order:confirm.append(dict(repeat=rep,label=label,ms=runtimes[label].run(plans[label],3)))
    (a.out/'confirmation.json').write_text(json.dumps(confirm))
    for label,p in plans.items():runtimes[label].close(p)
    print('complete random probe',flush=True)

if __name__=='__main__':main()
