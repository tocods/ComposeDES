"""GCN + DQN with the paper's published action space and hyperparameters.

Unspecified choices are explicit: 6->64->3072 GCN, incoming symmetric
normalization, uniform movable-kernel targets, 31 actions per epoch.
The two-hop target embedding is evaluated locally, exactly matching a full
two-layer GCN for that target, without materializing all node embeddings.
"""
import copy,random
from collections import deque
import numpy as np
import torch
from torch import nn

def state(dag,target):
    nodes=dag.meta['nodes'];features=np.array([[n['threads'],n['blocks'],n['copy_bytes'],dag.levels[i],
                      len(dag.parents[i]),len(dag.children[i])] for i,n in enumerate(nodes)],dtype=np.float32)
    features/=np.maximum(features.max(axis=0),1)
    incoming=dag.parents[target]+[target];degree=np.array([len(p)+1 for p in dag.parents],dtype=np.float32)
    aggregates=[];weights=[]
    for u in incoming:
        previous=dag.parents[u]+[u]
        norm=1/np.sqrt(degree[previous]*degree[u])
        aggregates.append((features[previous]*norm[:,None]).sum(axis=0))
        weights.append(1/np.sqrt(degree[u]*degree[target]))
    return np.array(aggregates,dtype=np.float32),np.array(weights,dtype=np.float32)

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.gcn1=nn.Linear(6,64);self.gcn2=nn.Linear(64,3072)
        self.policy=nn.Sequential(nn.Linear(3072,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,states):
        width=max(len(x[1]) for x in states);x=torch.zeros((len(states),width,6));w=torch.zeros((len(states),width))
        for i,(a,b) in enumerate(states):x[i,:len(b)]=torch.from_numpy(a);w[i,:len(b)]=torch.from_numpy(b)
        h=torch.relu(self.gcn1(x));h=(h*w.unsqueeze(-1)).sum(dim=1)
        return self.policy(torch.relu(self.gcn2(h)))

class Agent:
    def __init__(self,seed=907):
        torch.manual_seed(seed);self.rng=random.Random(seed);self.online=Policy();self.target=copy.deepcopy(self.online)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=.0001);self.memory=deque(maxlen=10000);self.steps=0
    def action(self,st,valid,explore=True):
        epsilon=max(.05,.997**self.steps) if explore else 0.
        if self.rng.random()<epsilon:return self.rng.choice(valid)
        with torch.no_grad():q=self.online([st])[0]
        return max(valid,key=lambda x:float(q[x]))
    def update(self,st,action,reward,nxt,valid,done):
        self.memory.append((st,action,reward,nxt,valid,done));self.steps+=1
        if len(self.memory)<16:return None
        batch=self.rng.sample(list(self.memory),16)
        q=self.online([x[0] for x in batch])[range(16),[x[1] for x in batch]]
        with torch.no_grad():
            qnext=self.target([x[3] for x in batch])
            y=torch.tensor([r if done else r+.99*float(qnext[i,valid].max())
                            for i,(_,_,r,_,valid,done) in enumerate(batch)])
        loss=nn.functional.smooth_l1_loss(q,y);self.optimizer.zero_grad();loss.backward();self.optimizer.step()
        with torch.no_grad():
            for p,t in zip(self.online.parameters(),self.target.parameters()):t.lerp_(p,.005)
        return float(loss)
    def freeze_gcn(self):
        for layer in (self.online.gcn1,self.online.gcn2):
            for p in layer.parameters():p.requires_grad_(False)
    def save(self,path):torch.save(dict(policy=self.online.state_dict(),steps=self.steps),path)
    def load(self,path):
        state=torch.load(path,weights_only=True);self.online.load_state_dict(state['policy']);self.steps=state['steps']
        self.target.load_state_dict(state['policy']);self.freeze_gcn()
