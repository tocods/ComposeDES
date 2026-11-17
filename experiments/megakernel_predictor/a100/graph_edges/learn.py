"""Paper-inspired two-layer GCN and DQN; small synthetic graphs, CPU training.

This adapts the paper's three level actions, safe predecessor buckets, normalized
runtime-difference reward, replay buffer and target network. Network dimensions
are smaller; target selection is specified here because it is not fully specified
in the paper. This is not the authors' code or their SSTA benchmark.
"""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import random
from statistics import median
import time
import numpy as np
import torch
from torch import nn
from experiment import Runtime,case,topology,adjustments


def state(c,edges,target):
    n=2*c['gates'];_,levels=topology(n,edges)
    adj=np.eye(n,dtype=np.float32);fanin=np.zeros(n);fanout=np.zeros(n)
    for u,v in edges:adj[v,u]=1;fanin[v]+=1;fanout[u]+=1
    # Incoming-neighbor message passing, including self, row normalized.
    adj/=adj.sum(axis=1,keepdims=True)
    x=np.array([[256 if i%2 else 0,c['blocks'][i//2] if i%2 else 0,
                 c['n']*4 if not i%2 else 0,levels[i],fanin[i],fanout[i]] for i in range(n)],dtype=np.float32)
    x/=np.maximum(1,x.max(axis=0))
    return torch.tensor(x),torch.tensor(adj),target


class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(6,32);self.second=nn.Linear(32,32)
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))

    def forward(self,x,adj,target):
        h=torch.relu(self.first(adj@x));h=torch.relu(self.second(adj@h))
        return self.head(torch.cat([h[target],h.mean(0)],dim=-1))


def options(c,edges,target):
    return [None]+[adjustments(2*c['gates'],edges,target,s) for s in (1,2)]


def targets(c,seed,steps=8):
    rng=random.Random(seed)
    valid=[i for i in range(2*c['gates']) if any(options(c,c['edges'],i)[1:])]
    rng.shuffle(valid);return valid[:steps]


def train(out,epochs=100,seed=7):
    torch.set_num_threads(1);torch.manual_seed(seed);rng=random.Random(seed)
    cases=[case(31,1024,32,'layered',32),case(32,4096,32,'fork',256)]
    online=Policy();target=copy.deepcopy(online);opt=torch.optim.Adam(online.parameters(),lr=1e-4)
    memory=deque(maxlen=10000);steps=0;measurements=0;start=time.time();log=[]
    runtimes=[Runtime(c) for c in cases]
    def timing(rt,c,edges):
        nonlocal measurements
        cfg=dict(edges=edges,order=list(range(2*c['gates'])))
        p=rt.graph(cfg);rt.run(p,8);t=median(rt.run(p,8) for _ in range(3));rt.release(p)
        measurements+=1;return t
    for epoch in range(epochs):
        c=cases[epoch%len(cases)];rt=runtimes[epoch%len(cases)]
        edges=list(map(tuple,c['edges']));base=previous=timing(rt,c,edges)
        chosen=targets(c,seed+epoch)
        for j,node in enumerate(chosen):
            st=state(c,edges,node);opts=options(c,edges,node);valid=[0]+[a for a in (1,2) if opts[a]]
            epsilon=max(.05,.997**steps)
            if rng.random()<epsilon:action=rng.choice(valid)
            else:
                with torch.no_grad():q=online(*st)
                action=max(valid,key=lambda a:float(q[a]))
            new=edges.copy()
            if action:new.append(rng.choice(opts[action]))
            current=timing(rt,c,new) if action else previous
            reward=(previous-current)/base
            done=j==len(chosen)-1
            nxt=state(c,new,chosen[j+1] if not done else node)
            nxt_options=options(c,new,chosen[j+1] if not done else node)
            nxt_valid=[0]+[a for a in (1,2) if nxt_options[a]]
            memory.append((st,action,reward,nxt,nxt_valid,done));steps+=1
            if len(memory)>=16:
                batch=rng.sample(list(memory),16)
                predicted=torch.stack([online(*s)[a] for s,a,_,_,_,_ in batch])
                with torch.no_grad():
                    expected=torch.stack([torch.tensor(r) if d else r+.99*target(*n)[v].max()
                                          for _,_,r,n,v,d in batch])
                loss=nn.functional.smooth_l1_loss(predicted,expected);opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():
                    for p,t in zip(online.parameters(),target.parameters()):t.lerp_(p,.005)
            edges=new;previous=current
        if epoch==9:
            for layer in (online.first,online.second):
                for p in layer.parameters():p.requires_grad_(False)
        log.append(dict(epoch=epoch,case=c['name'],baseline_ms=base,final_ms=previous,
                        added=len(edges)-len(c['edges']),measurements=measurements,epsilon=epsilon))
        if epoch%10==9:print(json.dumps(log[-1]),flush=True)
    for rt in runtimes:rt.close()
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(online.state_dict(),out)
    out.with_suffix('.training.json').write_text(json.dumps(dict(seed=seed,epochs=epochs,cases=cases,
                  graph_evaluations=measurements,wall_seconds=time.time()-start,log=log),indent=2))


def propose(c,checkpoint,seed=19):
    torch.set_num_threads(1);net=Policy();net.load_state_dict(torch.load(checkpoint,weights_only=True));net.eval()
    edges=list(map(tuple,c['edges']));rng=random.Random(seed)
    for node in targets(c,seed):
        opts=options(c,edges,node);valid=[0]+[a for a in (1,2) if opts[a]]
        with torch.no_grad():q=net(*state(c,edges,node))
        action=max(valid,key=lambda a:float(q[a]))
        if action:edges.append(rng.choice(opts[action]))
    return dict(label='GCN-DQN',edges=edges,order=list(range(2*c['gates'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args();train(a.out,a.epochs,a.seed)
