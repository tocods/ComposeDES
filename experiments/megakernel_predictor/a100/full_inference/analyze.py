"""Evaluate frozen selection on full-model timings; no numerical-validity claim."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import numpy as np
from predict import read,select,simulate

def ranks(values):
    return np.array([sum(y<x for y in values)+(sum(y==x for y in values)-1)/2 for x in values])

def paired(rows,candidate,baseline='ordinary-full'):
    series={name:{x['repeat']:x['ttft_ms'] for x in rows
                  if x['type']=='ttft' and x['label']==name} for name in (candidate,baseline)}
    keys=sorted(set(series[candidate])&set(series[baseline]))
    values=[100*(series[baseline][k]-series[candidate][k])/series[baseline][k] for k in keys]
    rng=random.Random(1729)
    samples=sorted(median(rng.choices(values,k=len(values))) for _ in range(3000))
    return dict(paired_gain_pct=median(values),ci95_pct=[samples[74],samples[2924]],pairs=len(values))

def summarize(meta,rows):
    out={}
    for label in meta['labels']:
        ts=[x for x in rows if x['type']=='ttft' and x['label']==label]
        gs=[x for x in rows if x['type']=='generation' and x['label']==label]
        out[label]=dict(ttft_ms=median(x['ttft_ms'] for x in ts),gpu_ms=median(x['gpu_ms'] for x in ts),
            tpot_ms=median(x['tpot_ms'] for x in gs),total_ms=median(x['total_ms'] for x in gs),
            output_tokens_per_second=median(x['output_tokens_per_second'] for x in gs),
            **paired(rows,label))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();d=args.directory
    protocol=json.loads((d/'protocol.json').read_text());model=json.loads((d/'frozen.json').read_text())
    assert hashlib.sha256((d/'frozen.json').read_bytes()).hexdigest()==protocol['model_sha256']
    calibration=[]
    for path in sorted(d.glob('cal-*.jsonl')):
        m,_,rows=read(path);calibration.append(summarize(m,rows))
    candidates=[c for c in calibration[0] if c not in ('explicit-AA','resident-full','ordinary-staged','ordinary-full')]
    fixed=min(candidates,key=lambda c:sum(s[c]['ttft_ms']/s['explicit-original']['ttft_ms'] for s in calibration))
    output=dict(performance_only=protocol['performance_only'],numerical_correctness_validated=False,
                known_gpu_memory_integrity_failure=True,fixed_from_calibration=fixed,protocol=protocol,
                calibration=calibration,tests={},source_sha256={})
    for path in sorted(d.glob('test-*.jsonl')):
        m,_,rows=read(path);stats=summarize(m,rows)
        assert m['layers']==36 and m['parameters']==8190735360 and m['output_tokens']==32
        assert len(rows)==len(m['labels'])*(protocol['ttft_repeats']+protocol['generation_repeats'])
        assert all(len(x['tokens'])==32 for x in rows if x['type']=='generation')
        # Deliberately project metadata to model input fields. No target timing,
        # warmup outcome, generated token, or validation result enters select().
        prediction=select({k:m[k] for k in ('configs','kinds','solo_ms')},model)
        predicted=np.array(prediction['prediction_ms'])
        observed=np.array([stats[c['label']]['gpu_ms'] for c in m['configs']])
        order=ranks(predicted);truth=ranks(observed)
        prediction['spearman']=float(np.corrcoef(order,truth)[0,1]) if order.std()>0 else None
        prediction['relative_difference_mae_ms']=float(np.mean(abs((predicted-predicted[0])-(observed-observed[0]))))
        plain=[simulate(m,c,(0.,0.,0.,0.)) for c in m['configs']]
        prediction['no_interference_label']=m['configs'][int(np.argmin(plain))]['label']
        best=min(candidates,key=lambda c:stats[c]['ttft_ms'])
        t=dict(prefix=m['prefix'],suffix=m['suffix'],session=m['session'],seed=m['seed'],stats=stats,
            prediction=prediction,best_explicit_in_hindsight=best,
            memory_bytes={k:m[k] for k in ('torch_allocated','torch_reserved','driver_used_bytes')},
            h2d_bytes=m['h2d_bytes'],setup_seconds=m['setup_seconds'],
            independent_stage_sums_ms={kind:sum(v for k,v in zip(m['kinds'],m['solo_ms']) if k==kind)
                                       for kind in ('io','compute')},
            aa_vs_original=paired(rows,'explicit-AA','explicit-original'),
            token_sequence_matches=sum(x['reference_tokens_match'] for x in rows if x['type']=='generation'),
            generation_count=sum(x['type']=='generation' for x in rows))
        t['selected_regret_pct']=100*(stats[prediction['label']]['ttft_ms']/stats[best]['ttft_ms']-1)
        output['tests'][path.stem]=t
        output['source_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        print('analyzed',path.name,'selected',prediction['label'],'best',best,flush=True)
    for name,t in output['tests'].items():
        if t['session']=='B':
            train=output['tests'][name[:-1]+'A']
            chosen=min(candidates+['ordinary-full'],key=lambda c:train['stats'][c]['ttft_ms'])
            t['search_A_evaluate_B']=dict(label=chosen,**t['stats'][chosen])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n')
    print('saved',args.out,flush=True)

if __name__=='__main__':main()
