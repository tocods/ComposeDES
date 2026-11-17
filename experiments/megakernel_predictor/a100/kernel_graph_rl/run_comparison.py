"""Run the predeclared matched-budget protocol after random discovery finishes."""
import argparse,json,hashlib,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    if (log.parent/'selected.json').exists() and (log.parent/'confirmation.json').exists():
        print('resume completed',log.parent,flush=True);return
    if (log.parent/'generation.json').exists():
        print('resume completed',log.parent,flush=True);return
    with log.open('w') as f:subprocess.run([sys.executable,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('complete',log,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    assert (a.out/'discovery/confirmation.json').exists(),'finish the random-space experiment first'
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for method in protocol['methods']:
        out=a.out/'train'/method
        run([HERE/'search.py','--out',out,'--method',method,'--prefix',768,'--suffix',1,'--seed',901,'--budget',3200],out/'run.log')
    frozen={}
    for method in protocol['methods']:
        for file in (('policy.pt','predictor.json') if method=='dqn' else ('predictor.json',)):
            path=a.out/'train'/method/file;frozen[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.out/'frozen.json').write_text(json.dumps(dict(frozen_at=time.time(),sha256=frozen),indent=2))
    for case in protocol['heldout']:
        for method in protocol['methods']:
            out=a.out/case['name']/method
            run([HERE/'search.py','--out',out,'--method',method,'--prefix',case['prefix'],'--suffix',case['query_tokens'],
                 '--seed',case['seed'],'--budget',512,'--frozen',a.out/'train'/method],out/'run.log')
        run([HERE/'confirm.py','--case',a.out/case['name'],'--out',a.out/case['name']/'independent'],a.out/case['name']/'independent/run.log')
    print('complete all matched-budget experiments',flush=True)

if __name__=='__main__':main()
