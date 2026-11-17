import argparse,json,random,hashlib,math
from pathlib import Path
from statistics import median

def paired(rows,label):
    a={x['repeat']:x['ms'] for x in rows if x['label']=='original'}
    b={x['repeat']:x['ms'] for x in rows if x['label']==label}
    values=[100*(a[i]-b[i])/a[i] for i in sorted(a.keys()&b.keys())]
    rng=random.Random(1667);boots=sorted(median(rng.choices(values,k=len(values))) for _ in range(3000))
    return dict(ms=median(b.values()),gain_pct=median(values),ci95_pct=[boots[74],boots[2924]])

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.directory
    protocol=json.loads((root/'protocol.json').read_text());result=dict(protocol=protocol,performance_only=True,training={},heldout={})
    frozen=json.loads((root/'frozen.json').read_text())
    for recorded,digest in frozen['sha256'].items():
        parts=Path(recorded).parts;path=root.joinpath(*parts[parts.index('train'):])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'training state changed after freezing'
    result['frozen_training_sha256_verified']=True
    discovery=root/'discovery';rows=[json.loads(x) for x in (discovery/'random-search.jsonl').read_text().splitlines()]
    candidates=[x for x in rows if not x.get('control')];assert len(candidates)==10000
    assert len({tuple(sorted(map(tuple,x['edges']))) for x in candidates})==10000
    confirm=json.loads((discovery/'confirmation.json').read_text());meta=json.loads((discovery/'graph.json').read_text())
    assert meta['layers']==36 and meta['parameters']==8190735360
    result['discovery']=dict(samples=len(candidates),control_queries=len(rows)-len(candidates),
         nodes=meta['source_node_count'],edges=meta['source_edge_count'],kernels=sum(n['type']==0 for n in meta['nodes']),
         sampled_min_ms=min(x['ms'] for x in candidates),sampled_max_ms=max(x['ms'] for x in candidates),
         confirmation={label:paired(confirm,label) for label in sorted({x['label'] for x in confirm})})
    for method in protocol['methods']:
        d=root/'train'/method;selected=json.loads((d/'selected.json').read_text());queries=[json.loads(x) for x in (d/'queries.jsonl').read_text().splitlines()]
        assert len(queries)==3200 and sum(x['warmups']+x['timed_replays'] for x in queries)==12800
        assert all(math.isfinite(x['ms']) and x['ms']>0 for x in queries)
        if method=='dqn':assert len(selected['losses'])==3085 and all(math.isfinite(x) for x in selected['losses'])
        result['training'][method]=selected
    assert len({x['source_fingerprint'] for x in result['training'].values()})==1
    for spec in protocol['heldout']:
        name=spec['name'];d=root/name;independent=json.loads((d/'independent/confirmation.json').read_text());rows=independent['rows']
        meta=json.loads((d/'independent/graph.json').read_text())
        assert meta['layers']==36 and meta['parameters']==8190735360
        gen=json.loads((d/'independent/generation.json').read_text());assert all(len(x['tokens'])==32 for x in gen)
        case=dict(prefix=spec['prefix'],query_tokens=spec['query_tokens'],nodes=len(meta['nodes']),
                  kernels=sum(n['type']==0 for n in meta['nodes']),edges=len(meta['edges']),
                  timings={},generation={},methods={})
        for label in sorted({x['label'] for x in rows}):case['timings'][label]=paired(rows,label)
        for label in sorted({x['label'] for x in gen}):
            gs=[x for x in gen if x['label']==label]
            case['generation'][label]={k:median(x[k] for x in gs) for k in ('ttft_ms','tpot_ms','total_ms','output_tokens_per_second')}
        for method in protocol['methods']:
            selected=json.loads((d/method/'selected.json').read_text());queries=[json.loads(x) for x in (d/method/'queries.jsonl').read_text().splitlines()]
            assert len(queries)==512 and sum(x['warmups']+x['timed_replays'] for x in queries)==2048
            assert all(math.isfinite(x['ms']) and x['ms']>0 for x in queries)
            assert selected['source_fingerprint']==independent['source_fingerprint']
            case['methods'][method]=selected
        result['heldout'][name]=case
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('all query budgets, kernel DAG fingerprints, and 32-token generations verified',flush=True)

if __name__=='__main__':main()
