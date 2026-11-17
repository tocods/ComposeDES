import random
import numpy as np
import torch
from dag import DAG
from learn import Policy,state

def main():
    meta=dict(nodes=[dict(id=i,type=0,threads=128,blocks=i+1,copy_bytes=0) for i in range(5)],
              edges=[[0,2],[1,3],[2,4],[3,4]])
    dag=DAG(meta)
    for u in range(dag.n):
        for a in (1,2):
            for predecessor in dag.options(u,a):
                nxt=DAG(meta,[(predecessor,u)]);assert nxt.levels[u]==dag.levels[u]+a
    torch.manual_seed(11);net=Policy();target=4
    x=np.array([[n['threads'],n['blocks'],0,dag.levels[i],len(dag.parents[i]),len(dag.children[i])] for i,n in enumerate(meta['nodes'])],dtype=np.float32)
    x/=np.maximum(1,x.max(axis=0));adj=torch.eye(dag.n)
    for u,v in meta['edges']:adj[v,u]=1
    degree=adj.sum(1);adj=adj/torch.sqrt(degree[:,None]*degree[None,:])
    with torch.no_grad():
        expected=net.policy(torch.relu(net.gcn2(adj@torch.relu(net.gcn1(adj@torch.from_numpy(x))))))[target]
        actual=net([state(dag,target)])[0]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    print('safe level actions and exact local two-hop GCN equivalence passed')

if __name__=='__main__':main()
