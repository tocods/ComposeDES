"""Matched full-graph evaluation budgets for random, GCN+DQN and ComposeDES."""
import argparse,json,random,time
from pathlib import Path
from statistics import median
import torch
from probe import prepare
from dag import DAG
from learn import Agent,state
from predict import Predictor

def target(dag,rng):
    for _ in range(10000):
        i=rng.randrange(dag.n)
        if dag.meta['nodes'][i]['type']==0 and (dag.options(i,1) or dag.options(i,2)):return i
    raise RuntimeError('no movable target')

def choices(dag,node,rng):
    out={0:dag}
    for a in (1,2):
        if dag.options(node,a):out[a]=dag.transition(node,a,rng)[0]
    return out

def rollout(base,method,rng,agent,predictor,steps=31):
    dag=base
    for _ in range(steps):
        node=target(dag,rng);opts=choices(dag,node,rng)
        if method=='dqn':a=agent.action(state(dag,node),list(opts),explore=False)
        elif method=='des':a=min(opts,key=lambda a:predictor.score(opts[a]))
        else:a=rng.choice(list(opts))
        dag=opts[a]
    return dag

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['random','dqn','des'],required=True)
    p.add_argument('--prefix',type=int,default=768);p.add_argument('--suffix',type=int,default=1)
    p.add_argument('--seed',type=int,default=901);p.add_argument('--budget',type=int,default=3200)
    p.add_argument('--frozen',type=Path);p.add_argument('--readback',action='store_true');a=p.parse_args()
    assert a.budget%32==0;a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    # Only model execution is inference-only; DQN training retains autograd.
    with torch.inference_mode():values=prepare(a.prefix,a.suffix,a.seed,a.out,a.readback)
    req,source,rt,baseplan,serial,srt,serialplan,meta=values
    rng=random.Random(a.seed+17);base=DAG(meta);agent=Agent(a.seed);predictor=Predictor(meta)
    if a.frozen:
        if a.method=='dqn':agent.load(a.frozen/'policy.pt')
        if a.method=='des':predictor.load(a.frozen/'predictor.json')
    zero=rollout(base,a.method,random.Random(a.seed+100000),agent,predictor) if a.frozen else base
    (a.out/'zero-shot.json').write_text(json.dumps(dict(edges=zero.extra,target_full_graph_timing_observations_used_for_choice=0,
        target_kernel_profiling_used=a.method=='des',common_preparation_warmup_graph_launches=324)))
    records=[];best=(float('inf'),[]);checkpoints={};start=time.perf_counter();losses=[]
    log=(a.out/'queries.jsonl').open('w')
    def evaluate(dag,label):
        nonlocal best
        plan=rt.make(dag.extra)
        ms=rt.run(plan,3);rt.close(plan)
        item=dict(query=len(records)+1,kind=label,ms=ms,edges=dag.extra,warmups=1,timed_replays=3)
        records.append(item);log.write(json.dumps(item)+'\n')
        if ms<best[0]:best=(ms,list(dag.extra))
        if len(records) in (8,32,128,512,1024,3200):checkpoints[len(records)]=dict(ms=best[0],edges=best[1])
        return ms
    for epoch in range(a.budget//32):
        baseline=evaluate(base,'baseline');dag=base;previous=baseline;node=target(dag,rng)
        for step in range(31):
            opts=choices(dag,node,rng)
            if a.method=='dqn':
                st=state(dag,node);action=agent.action(st,list(opts),explore=True)
            elif a.method=='des':
                action=rng.choice(list(opts)) if rng.random()<.2 else min(opts,key=lambda x:predictor.score(opts[x]))
            else:action=rng.choice(list(opts))
            new=opts[action];current=evaluate(new,'candidate')
            nxt=target(new,rng);valid=[0]+[j for j in (1,2) if new.options(nxt,j)]
            if a.method=='dqn' and not a.frozen:
                loss=agent.update(st,action,(previous-current)/baseline,state(new,nxt),valid,step==30)
                if loss is not None:losses.append(loss)
            if a.method=='des':predictor.update(new,base,current,baseline)
            dag=new;previous=current;node=nxt
        if epoch==9 and not a.frozen:agent.freeze_gcn()
        log.flush()
        if epoch%5==4:print(a.method,'epoch',epoch+1,'queries',len(records),'best',best[0],'seconds',time.perf_counter()-start,flush=True)
    log.close();assert len(records)==a.budget
    if a.method=='dqn' and not a.frozen:agent.save(a.out/'policy.pt')
    predictor.save(a.out/'predictor.json')
    selected=DAG(meta,best[1]);predictor.predict(selected,trace=True)
    policy=rollout(base,a.method,random.Random(a.seed+200000),agent,predictor)
    report=dict(method=a.method,budget=a.budget,epochs=a.budget//32,steps_per_epoch=31,seed=a.seed,
        source_fingerprint=meta['fingerprint'],prefix=a.prefix,suffix=a.suffix,readback=a.readback,
        query_graph_launches=a.budget*4,search_seconds=time.perf_counter()-start,
        common_preparation_warmup_graph_launches=324,
        profiling_seconds=meta['profiling_seconds'],profiling_node_evaluations=meta['profiling_node_evaluations'],
        setup_seconds=meta['setup_seconds'],
        profiling_node_replays=meta['profiling_node_evaluations']*6,
        best_edges=best[1],best_observed_ms=best[0],policy_edges=policy.extra,zero_shot_edges=zero.extra,
        checkpoints=checkpoints,losses=losses,numerical_correctness_validated=False,
        frozen_from=str(a.frozen) if a.frozen else None,composedes_dependency_replay_passed=True)
    (a.out/'selected.json').write_text(json.dumps(report,indent=2))
    # Fresh repetitions are never fed back into the search or policy weights.
    handles={'original':(rt,baseplan),'serial':(srt,serialplan),'best':(rt,rt.make(best[1])),
             'policy':(rt,rt.make(policy.extra)),'zero-shot':(rt,rt.make(zero.extra))}
    confirmation=[]
    for rep in range(32):
        order=list(handles);rng.shuffle(order)
        for label in order:
            runtime,plan=handles[label];confirmation.append(dict(repeat=rep,label=label,ms=runtime.run(plan,3)))
    (a.out/'confirmation.json').write_text(json.dumps(confirmation))
    for runtime,plan in handles.values():runtime.close(plan)
    print('complete',a.method,'confirmed',{k:median(x['ms'] for x in confirmation if x['label']==k) for k in handles},flush=True)

if __name__=='__main__':main()
