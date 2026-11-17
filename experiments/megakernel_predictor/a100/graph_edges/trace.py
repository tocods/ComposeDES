"""Separate profiler execution; never supplies performance-selection inputs."""
import argparse
from pathlib import Path
import torch
from experiment import Runtime
from predict import read

p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--index',type=int,required=True)
a=p.parse_args();meta,_,_=read(a.input);rt=Runtime(meta['case'])
plans={i:rt.graph(meta['configs'][i]) for i in (0,a.index)}
for v in plans.values():rt.check(v);rt.run(v,16)
for repeat in range(4):
    for i in ((0,a.index) if repeat%2==0 else (a.index,0)):
        torch.cuda.nvtx.range_push(f'candidate-{i}/repeat-{repeat}')
        ms=rt.run(plans[i],1)
        torch.cuda.nvtx.range_pop()
        print(repeat,i,ms,flush=True)
rt.close()
