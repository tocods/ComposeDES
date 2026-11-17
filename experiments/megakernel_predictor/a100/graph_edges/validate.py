"""Freeze candidates before held-out GPU measurements and replay in two processes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from experiment import case,candidates,adjustments
from learn import propose


SHAPES=[(101,1024,32,'layered',32),(102,4096,32,'layered',256),
        (103,4096,32,'fork',1024),(104,16384,32,'random',128),
        (105,65536,32,'layered',32),(106,4096,48,'random',256)]


def prepare(root):
    checkpoints=[root/'gcn-dqn.pt',root/'frozen-des.json']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    for index,args in enumerate(SHAPES):
        c=case(*args);configs=candidates(c,80,seed=880+index)
        learned=propose(c,checkpoints[0]);configs.append(learned)
        # Random additions with exactly the learned policy's edge count.
        count=len(learned['edges'])-len(c['edges']);rng=random.Random(2026+index)
        if count:
            for k in range(24):
                edges=list(map(tuple,c['edges']))
                for _ in range(count):
                    options=[e for v in range(2*c['gates']) for s in (1,2)
                             for e in adjustments(2*c['gates'],edges,v,s)]
                    if not options:break
                    edges.append(rng.choice(options))
                configs.append(dict(label=f'matched-random-{k}',edges=edges,order=learned['order']))
        manifest=dict(case=c,configs=configs,frozen_time=time.time(),model_sha256=hashes)
        (root/f'test-{index}.manifest.json').write_text(json.dumps(manifest))
    (root/'frozen-validation.json').write_text(json.dumps(dict(shapes=SHAPES,model_sha256=hashes,
         sessions=['A','B'],repetitions=dict(A=24,B=48),inner_replays=8),indent=2))


def run(root):
    for session,repeats in [('A',24),('B',48)]:
        for index in range(len(SHAPES)):
            subprocess.run([sys.executable,'experiment.py','--manifest',str(root/f'test-{index}.manifest.json'),
                            '--out',str(root/f'test-{index}-{session}.jsonl'),'--session',session,
                            '--repeat',str(repeats)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('root',type=Path)
    a=p.parse_args();(prepare if a.mode=='prepare' else run)(a.root)
