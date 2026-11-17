"""Paired held-out comparisons; selection on A, final reporting on process B."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from predict import read,predict


def interval(base,chosen,seed=42):
    base=np.asarray(base);chosen=np.asarray(chosen)
    assert base.shape==chosen.shape
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(base),(5000,len(base)))
    improvement=100*(1-np.median(chosen)/np.median(base))
    bootstrap=100*(1-np.median(chosen[idx],axis=1)/np.median(base[idx],axis=1))
    return dict(improvement_pct=float(improvement),ci95_pct=np.quantile(bootstrap,[.025,.975]).tolist())


def hierarchical_interval(base,chosen,seed=42):
    """Resample paired time blocks and independently instantiated graph replicas."""
    base=np.asarray(base);chosen=np.asarray(chosen)
    if np.array_equal(base,chosen):return dict(improvement_pct=0.,ci95_pct=[0.,0.])
    rng=np.random.default_rng(seed);b,n=base.shape
    blocks=rng.integers(0,b,(5000,b))
    br=rng.integers(0,n,(5000,n));cr=rng.integers(0,n,(5000,n))
    bsample=base[blocks[:,:,None],br[:,None,:]].mean(axis=2)
    csample=chosen[blocks[:,:,None],cr[:,None,:]].mean(axis=2)
    gain=100*(1-np.median(chosen.mean(axis=1))/np.median(base.mean(axis=1)))
    boot=100*(1-np.median(csample,axis=1)/np.median(bsample,axis=1))
    return dict(improvement_pct=float(gain),ci95_pct=np.quantile(boot,[.025,.975]).tolist())


def analyze(root):
    model=json.loads((root/'frozen-des.json').read_text())
    training=[read(root/f'cal-{s}-A.jsonl') for s in ('small','medium','heavy','wide')]
    fixed_scores=np.mean([np.log(y[:8]/y[0]) for _,y,_ in training],axis=0)
    fixed_scores[1]=np.inf # Do not treat an identical graph instance as an optimization.
    fixed=int(np.argmin(fixed_scores));result=[]
    for index in range(6):
        am,a,at=read(root/f'test-{index}-A.jsonl');bm,b,bt=read(root/f'test-{index}-B.jsonl')
        assert am['configs']==bm['configs']
        start=time.perf_counter();pred=predict(am,model);elapsed=time.perf_counter()-start
        labels=[c['label'] for c in am['configs']];rl=labels.index('GCN-DQN')
        eligible=[];seen=set()
        for i,cfg in enumerate(am['configs']):
            key=(tuple(cfg['order']),tuple(sorted(map(tuple,cfg['edges']))))
            if key not in seen:eligible.append(i);seen.add(key)
        extra_only=[i for i in eligible if i not in (2,3)]
        # Independent A measurements choose configurations; B is never used for policy selection.
        oracle=int(min(eligible,key=lambda i:a[i]));edges=int(min(extra_only,key=lambda i:a[i]))
        sparse=int(min([i for i in extra_only if len(am['configs'][i]['edges'])-len(am['case']['edges'])<=8],
                       key=lambda i:a[i]))
        model_pick=int(min(eligible,key=lambda i:pred[i]))
        shortlist=sorted(eligible,key=lambda i:pred[i])[:8]
        model_search=int(min(shortlist,key=lambda i:a[i]))
        random_ids=[i for i,l in enumerate(labels) if l.startswith('random-')]
        rng=np.random.default_rng(2026+index)
        search_winners=[min([0]+rng.choice(random_ids,7,replace=False).tolist(),key=lambda i:a[i]) for _ in range(200)]
        policies=dict(original=0,AA=1,fixed=fixed,GCN_DQN=rl,ComposeDES=model_pick,
                      ComposeDES_search8=model_search,search_all_A=oracle,edge_search_all_A=edges)
        policies['sparse_edge_search_A']=sparse
        measures={}
        for name,i in policies.items():
            measures[name]=dict(index=i,label=labels[i],ms=float(b[i]),extra_edges=len(am['configs'][i]['edges'])-len(am['case']['edges']),
                                same_graph_as_original=(am['configs'][i]['order']==am['configs'][0]['order'] and
                                  set(map(tuple,am['configs'][i]['edges']))==set(map(tuple,am['configs'][0]['edges']))),
                                **interval(bt[0],bt[i],seed=index+42))
        matched=[i for i,l in enumerate(labels) if l.startswith('matched-random-')]
        result.append(dict(case=am['case'],candidates=len(a),A_repetitions=len(at[0]),B_repetitions=len(bt[0]),
                           policies=measures,best_B_ms=float(min(b)),prediction_ms=pred.tolist(),
                           prediction_wall_seconds=elapsed,
                           random_search8_mean_ms=float(np.mean(b[search_winners])),
                           random_search8_mean_improvement_pct=float(np.mean(100*(1-b[search_winners]/b[0]))),
                           matched_random_fraction_faster_than_RL=float(np.mean(b[matched]<b[rl])) if matched else None,
                           correctness_max_error=max(bm['errors'].values()),
                           raw_ms_A=a.tolist(),raw_ms_B=b.tolist()))
        print(json.dumps(dict(case=index,policies=measures)),flush=True)
    output=dict(fixed_rule=training[0][0]['configs'][fixed]['label'],results=result,
                note='All choices use calibration, solo profiles, or process A; reported timings use independent process B.')
    (root/'summary.json').write_text(json.dumps(output,indent=2))
    lines=['| case | original ms | best selection from A | GCN-DQN | ComposeDES | model + 8 trials |',
           '|---|---:|---:|---:|---:|---:|']
    for i,row in enumerate(result):
        p=row['policies'];fmt=lambda n:f"{p[n]['improvement_pct']:.2f}% [{p[n]['ci95_pct'][0]:.2f}, {p[n]['ci95_pct'][1]:.2f}]"
        lines.append(f"| {i}: {row['case']['name']} | {p['original']['ms']:.4f} | {fmt('edge_search_all_A')} | {fmt('GCN_DQN')} | {fmt('ComposeDES')} | {fmt('ComposeDES_search8')} |")
    (root/'table.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
