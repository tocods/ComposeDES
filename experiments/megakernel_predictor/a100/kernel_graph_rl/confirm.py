"""Rebuild the full model in a fresh process and compare all selected graphs."""
import argparse,json,random,time
from pathlib import Path
from statistics import median
import torch
from probe import prepare
from dag import DAG

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    choices={method:json.loads((a.case/method/'selected.json').read_text()) for method in ('random','dqn','des')}
    c=choices['random'];values=prepare(c['prefix'],c['suffix'],c['seed']+1000,a.out,c['readback'],output_tokens=32)
    req,source,rt,base,serial,srt,sbase,meta=values
    assert all(x['source_fingerprint']==meta['fingerprint'] for x in choices.values()),'source kernel DAG changed across processes'
    aa=rt.make();plans={'original':(rt,base),'AA':(rt,aa),'serial':(srt,sbase)};edges={}
    for method,selected in choices.items():
        for budget,value in selected['checkpoints'].items():
            label=f'{method}-{budget}';DAG(meta,value['edges']);plans[label]=(rt,rt.make(value['edges']));edges[label]=value['edges']
        for name,key in [('zero','zero_shot_edges'),('policy','policy_edges')]:
            label=f'{method}-{name}';plans[label]=(rt,rt.make(selected[key]));edges[label]=selected[key]
    rows=[];rng=random.Random(c['seed']+5000)
    for rep in range(32):
        labels=list(plans);rng.shuffle(labels)
        for label in labels:
            runtime,plan=plans[label];rows.append(dict(repeat=rep,label=label,ms=runtime.run(plan,3)))
    (a.out/'confirmation.json').write_text(json.dumps(dict(rows=rows,edges=edges,source_fingerprint=meta['fingerprint']),indent=2))
    # Also measure a complete 32-token generation. The selected full graph is
    # the first forward; all later forwards use the same complete decode path.
    req.reset(resident=True);source();req.capture_decode()
    final=['original','serial']+[f'{m}-{c["budget"]}' for m in choices]
    generations=[]
    for rep in range(8):
        rng.shuffle(final)
        for label in final:
            req.reset(resident=not c['readback']);runtime,plan=plans[label]
            start=time.perf_counter_ns();runtime.launch(plan);tokens=[req.token()];first=time.perf_counter_ns();intervals=[]
            for g in req.decode_graphs:
                before=time.perf_counter_ns();g.replay();tokens.append(req.token());intervals.append((time.perf_counter_ns()-before)/1e6)
            total=(time.perf_counter_ns()-start)/1e6
            generations.append(dict(repeat=rep,label=label,ttft_ms=(first-start)/1e6,tpot_ms=median(intervals),
                                    total_ms=total,output_tokens_per_second=32000/total,tokens=tokens))
    (a.out/'generation.json').write_text(json.dumps(generations))
    for runtime,plan in plans.values():runtime.close(plan)
    print('complete independent confirmation',a.case,flush=True)

if __name__=='__main__':main()
