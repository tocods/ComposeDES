"""Paper actions: retain level, move by one/two levels via safe predecessor buckets."""
from collections import deque
import hashlib,json,random

class DAG:
    def __init__(self,meta,extra=()):
        self.meta=meta;self.n=len(meta['nodes']);self.extra=list(map(tuple,extra))
        self.parents=[[] for _ in range(self.n)];self.children=[[] for _ in range(self.n)]
        for u,v in meta['edges']+list(map(list,self.extra)):
            self.parents[v].append(u);self.children[u].append(v)
        degree=list(map(len,self.parents));ready=deque(i for i,d in enumerate(degree) if d==0)
        self.order=[];self.levels=[0]*self.n
        while ready:
            u=ready.popleft();self.order.append(u)
            for v in self.children[u]:
                self.levels[v]=max(self.levels[v],self.levels[u]+1);degree[v]-=1
                if degree[v]==0:ready.append(v)
        if len(self.order)!=self.n:raise ValueError('cycle')
        self.buckets={}
        for i,l in enumerate(self.levels):self.buckets.setdefault(l,[]).append(i)
    def options(self,target,step):
        if step==0:return []
        candidates=self.buckets.get(self.levels[target]+step-1,[])
        # Only descendants at the selected level could invalidate an edge.
        limit=self.levels[target]+step-1;descendants={target};queue=[target]
        for u in queue:
            for v in self.children[u]:
                if v not in descendants and self.levels[v]<=limit:descendants.add(v);queue.append(v)
        return [u for u in candidates if u not in descendants]
    def available(self):
        return [i for i,node in enumerate(self.meta['nodes']) if node['type']==0 and (self.options(i,1) or self.options(i,2))]
    def transition(self,target,action,rng):
        if action==0:return self,None
        options=self.options(target,action)
        if not options:raise ValueError('invalid action')
        edge=(rng.choice(options),target)
        new=DAG(self.meta,self.extra+[edge]);assert new.levels[target]==self.levels[target]+action
        return new,edge
    def random(self,rng,steps,enumerate_targets=False):
        dag=self;history=[];kernels=[i for i,n in enumerate(self.meta['nodes']) if n['type']==0]
        for _ in range(steps):
            # Rejection sampling is uniform over currently movable kernels and
            # avoids enumerating every kernel's buckets for every action.
            if enumerate_targets:
                target=rng.choice(dag.available());actions=[a for a in (1,2) if dag.options(target,a)]
            else:
                for attempt in range(10000):
                    target=rng.choice(kernels);actions=[a for a in (1,2) if dag.options(target,a)]
                    if actions:break
                else:raise RuntimeError('no movable kernel found')
            action=rng.choice(actions);dag,edge=dag.transition(target,action,rng)
            history.append(dict(target=target,action=action,edge=edge))
        return dag,history
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(nodes=self.meta['nodes'],edges=self.meta['edges']),sort_keys=True).encode()).hexdigest()
