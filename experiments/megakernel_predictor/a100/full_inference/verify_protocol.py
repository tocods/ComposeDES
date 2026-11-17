"""CPU-only checks of candidate DAGs and ComposeDES event progression."""
from bench import configs,graph_spec
from predict import simulate

def main():
    candidates,kinds,names=configs(36)
    original,*_=graph_spec(36)
    assert len(candidates)==17 and len(kinds)==146
    meta=dict(kinds=kinds,solo_ms=[.01]*len(kinds))
    base=simulate(meta,candidates[0],(0.,0.,0.,0.))
    for candidate in candidates:
        assert sorted(candidate['order'])==list(range(len(kinds)))
        for before,after in zip(original,candidate['deps']):assert set(before)<=set(after)
        seen=set()
        while len(seen)<len(kinds):
            ready={i for i,d in enumerate(candidate['deps']) if i not in seen and set(d)<=seen}
            assert ready,'cycle in candidate graph'
            seen.update(ready)
        # With no interference or issue overhead, adding constraints cannot help
        # these two serial chains executing on separate service resources.
        assert simulate(meta,candidate,(0.,0.,0.,0.))>=base-1e-9
        # Include idle gaps longer than individual work to exercise admission.
        assert simulate(meta,candidate,(.1,.2,.02,1.))>0
    print('17 complete-model DAGs: original dependencies preserved; acyclic; ComposeDES event checks passed')

if __name__=='__main__':main()
