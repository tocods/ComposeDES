"""Check graph invariants and demonstrate that missing-copy dependencies are caught."""
from experiment import case,candidates,topology,adjustments,Runtime

c=case(5,n=256,gates=12,rounds=2)
for config in candidates(c,12):
    assert set(map(tuple,c['edges'])).issubset(set(map(tuple,config['edges'])))
    assert len(topology(24,config['edges'])[0])==24
for v in range(24):
    for shift in (1,2):
        for edge in adjustments(24,c['edges'],v,shift):
            _,a=topology(24,c['edges']);_,b=topology(24,c['edges']+[edge])
            assert b[v]==a[v]+shift
rt=Runtime(c)
valid=dict(edges=c['edges'],order=list(range(24)))
assert rt.check(rt.graph(valid))<=1e-5
# Force root computation to execute before its H2D transfer, after poisoning.
bad=dict(edges=[e for e in c['edges'] if tuple(e)!=(0,1)]+[(1,0)],order=list(range(24)))
assert rt.lib.graph_check(rt.graph(bad))==-2
rt.close()
print('DAG, level actions, numerical correctness, and broken-dependency rejection passed')
