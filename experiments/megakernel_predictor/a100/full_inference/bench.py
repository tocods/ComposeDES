"""End-to-end full-model TTFT and autoregressive decode, with CPU-prefix reuse."""
import argparse
import gc
import json
from pathlib import Path
import random
from statistics import median
import time
import torch
from model import Model,Request
from runtime import Graph,capture


def graph_spec(layers):
    deps=[[]];kinds=['compute'];names=['embedding']
    copies=[];posts=[]
    for i in range(layers):
        load=len(deps);copies.append(load)
        deps.append([copies[-2]] if i else []);kinds.append('io');names.append(f'load-{i}')
        deps.extend([[posts[-1]] if i else [0],[load,load+1],[load+2]])
        kinds.extend(['compute']*3);names.extend([f'qkv-{i}',f'attn-{i}',f'post-{i}']);posts.append(load+3)
    deps.append([posts[-1]]);kinds.append('compute');names.append('output_head')
    return deps,kinds,names,copies,posts


def configs(layers,seed=96):
    base,kinds,names,copies,posts=graph_spec(layers);n=len(base)
    out=[dict(label='explicit-original',deps=base,order=list(range(n)))]
    out.append(dict(label='explicit-AA',deps=base,order=list(range(n))))
    out.append(dict(label='copies-first',deps=base,order=copies+[i for i in range(n) if i not in copies]))
    out.append(dict(label='compute-first',deps=base,order=[i for i in range(n) if i not in copies]+copies))
    for w in (1,2,4,8,16):
        deps=[x.copy() for x in base]
        for i in range(w,layers):deps[copies[i]].append(posts[i-w])
        out.append(dict(label=f'prefetch-{w}',deps=deps,order=list(range(n))))
    rng=random.Random(seed)
    for k in range(8):
        deps=[x.copy() for x in base]
        for i in rng.sample(range(1,layers),2 if k<4 else 4):
            deps[copies[i]].append(posts[max(0,i-rng.choice((1,2,4,8)))])
        out.append(dict(label=f'auxiliary-{k}',deps=deps,order=list(range(n))))
    return out,kinds,names


def ordinary(req,staged=False,resident=False):
    io=torch.cuda.Stream();start=torch.cuda.Event();ready=[torch.cuda.Event() for _ in req.host]
    def fn():
        if not resident:
            start.record()
            with torch.cuda.stream(io):
                io.wait_event(start)
                for i in range(req.model.layers):req.copy(i);ready[i].record()
        def wait(i):
            if not resident:torch.cuda.current_stream().wait_event(ready[i])
        if staged:
            req.embed()
            for i in range(req.model.layers):req.project(i);wait(i);req.attend(i);req.post(i)
            req.head()
        else:
            req.logits.copy_(req.model.dense_forward(req.ids_gpu[:,req.prefix:],req.arena,req.prefix,wait))
            req.next_token.copy_(req.logits.argmax(-1))
        if not resident:torch.cuda.current_stream().wait_event(ready[-1])
    g,s=capture(fn)
    return g,[s,io,start,*ready]


def build(req,smoke=False):
    cfg,kinds,names=configs(req.model.layers)
    if smoke:cfg=cfg[:1]+cfg[6:7]
    # Initialize valid inputs before warming/capturing isolated native fragments.
    req.reset(resident=True);req.forward();torch.cuda.synchronize()
    builders=[req.embed]
    for i in range(req.model.layers):
        builders.extend([lambda i=i:req.copy(i),lambda i=i:req.project(i),
                         lambda i=i:req.attend(i),lambda i=i:req.post(i)])
    builders.append(req.head)
    bodies=[];streams=[];solo=[];resources=[]
    for i,fn in enumerate(builders):
        g,s=capture(fn);bodies.append(g);streams.append(s)
        one=Graph([g],[[]],[0]);one.run()
        solo.append(median(one.run() for _ in range(5)))
        resources.append(Graph.pressure(g,torch.cuda.get_device_properties(0).multi_processor_count));one.close()
    plans={}
    for c in cfg:plans[c['label']]=Graph(bodies,c['deps'],c['order'])
    retained=[bodies,streams]
    for name,staged,resident in [('ordinary-full',False,False),('ordinary-staged',True,False),('resident-full',False,True)]:
        g,lifetime=ordinary(req,staged,resident);plans[name]=g;retained.append(lifetime)
    return plans,retained,dict(configs=cfg,kinds=kinds,names=names,solo_ms=solo,resource=resources)


def launch(plan):
    if isinstance(plan,Graph):plan.launch()
    else:plan.replay()


def ttft(req,plan,resident=False):
    req.reset(resident=resident)
    start,end=req.timers;before=time.perf_counter_ns()
    start.record();launch(plan);end.record()
    token=req.token();after=time.perf_counter_ns()
    return dict(ttft_ms=(after-before)/1e6,gpu_ms=start.elapsed_time(end),token=token)


def generate(req,plan,resident=False,check=False):
    first=ttft(req,plan,resident);tokens=[first['token']]
    errors=[req.check_logits(0)] if check else []
    intervals=[];start=time.perf_counter_ns()
    for step,g in enumerate(req.decode_graphs):
        before=time.perf_counter_ns();g.replay();tokens.append(req.token());after=time.perf_counter_ns()
        intervals.append((after-before)/1e6)
        if check:errors.append(req.check_logits(step+1))
    total=(time.perf_counter_ns()-start)/1e6
    assert tokens==req.reference_tokens,'autoregressive tokens differ from native complete model'
    return dict(**first,tokens=tokens,tpot_ms=median(intervals),decode_ms=total,
                total_ms=first['ttft_ms']+total,output_tokens_per_second=len(tokens)*1000/(first['ttft_ms']+total),
                errors=errors,per_token_ms=intervals)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prefix',type=int,default=2048);p.add_argument('--suffix',type=int,default=64)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeat',type=int,default=24);p.add_argument('--generate-repeat',type=int,default=4)
    p.add_argument('--session',default='A');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.cuda.reset_peak_memory_stats()
    began=time.time();m=Model(a.model);print('loaded complete model',m.layers,m.params,flush=True)
    req=Request(m,a.prefix,a.suffix,a.seed,a.tokens);print('native reference tokens',req.reference_tokens,flush=True)
    plans,lifetime,meta=build(req,a.smoke)
    req.reset(resident=True);req.forward();req.capture_decode()
    order=list(plans);rng=random.Random(f'{a.seed}/{a.session}');rng.shuffle(order)
    validation={}
    for label in order:
        print('checking',label,flush=True)
        result=generate(req,plans[label],label=='resident-full',check=True)
        validation[label]=dict(max_logit_relative_l2=max(x['relative_l2'] for x in result['errors']),
                               max_logit_absolute=max(x['max_absolute'] for x in result['errors']),tokens=result['tokens'])
        print('validated',label,'TTFT',result['ttft_ms'],flush=True)
    free,total=torch.cuda.mem_get_info()
    meta.update(type='meta',model='Qwen/Qwen3-8B',layers=m.layers,parameters=m.params,prefix=a.prefix,suffix=a.suffix,
        seed=a.seed,session=a.session,output_tokens=a.tokens,ids=req.ids,prompt_sha256=req.prompt_sha256,
        generated_text=m.tokenizer.decode(req.reference_tokens),h2d_bytes=req.bytes,validation=validation,
        torch_allocated=torch.cuda.memory_allocated(),torch_reserved=torch.cuda.memory_reserved(),
        driver_used_bytes=total-free,setup_seconds=time.time()-began,labels=list(plans),
        reference='native Transformers full model with FlashInfer FA2 attention')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w') as out:
        out.write(json.dumps(meta)+'\n');out.flush()
        for rep in range(a.repeat):
            rng.shuffle(order)
            for position,label in enumerate(order):
                row=ttft(req,plans[label],label=='resident-full')
                assert row['token']==req.reference_tokens[0]
                out.write(json.dumps(dict(type='ttft',repeat=rep,position=position,label=label,**row))+'\n')
            out.flush()
        for rep in range(a.generate_repeat):
            rng.shuffle(order)
            for position,label in enumerate(order):
                row=generate(req,plans[label],label=='resident-full')
                out.write(json.dumps(dict(type='generation',repeat=rep,position=position,label=label,**row))+'\n')
            out.flush()
    print('complete',a.out,flush=True)
    for plan in plans.values():
        if isinstance(plan,Graph):plan.close()

if __name__=='__main__':main()
