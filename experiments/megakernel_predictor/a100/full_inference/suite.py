"""Freeze full-model calibration before new shapes and prompts are evaluated."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CALIBRATION=[(2048,64,11),(4096,256,12)]
HELDOUT=[(4096,128),(8192,64),(8192,512)]


def run(model,out,prefix,suffix,seed,session,repeat=24):
    cmd=[sys.executable,str(HERE/'bench.py'),'--model',str(model),'--out',str(out),
         '--prefix',str(prefix),'--suffix',str(suffix),'--seed',str(seed),'--session',session,
         '--tokens','32','--repeat',str(repeat),'--generate-repeat','4']
    with out.with_suffix('.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('completed',out.name,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['calibrate','validate'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=='calibrate':
        for i,(prefix,suffix,seed) in enumerate(CALIBRATION):run(a.model,a.out/f'cal-{i}.jsonl',prefix,suffix,seed,'cal')
        subprocess.run([sys.executable,str(HERE/'predict.py'),'--out',str(a.out/'frozen.json'),
                        *[str(a.out/f'cal-{i}.jsonl') for i in range(len(CALIBRATION))]],check=True)
        (a.out/'protocol.json').write_text(json.dumps(dict(calibration=CALIBRATION,heldout=HELDOUT,
            sessions=['A','B'],output_tokens=32,ttft_repeats=24,generation_repeats=4,
            frozen_at=time.time(),model_sha256=hashlib.sha256((a.out/'frozen.json').read_bytes()).hexdigest()),indent=2))
    else:
        protocol=json.loads((a.out/'protocol.json').read_text())
        assert hashlib.sha256((a.out/'frozen.json').read_bytes()).hexdigest()==protocol['model_sha256']
        for session in ['A','B']:
            for i,(prefix,suffix) in enumerate(HELDOUT):
                run(a.model,a.out/f'test-{i}-{session}.jsonl',prefix,suffix,201+i+(100 if session=='B' else 0),session)

if __name__=='__main__':main()
