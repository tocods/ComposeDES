"""Archive raw measurements, frozen policies and graph exports without weights."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    files=sorted(x for x in a.runs.rglob('*') if x.is_file() and x.suffix in ('.json','.jsonl','.log','.pt'))
    inventory=[]
    for path in files:
        inventory.append(dict(path=str(path.relative_to(a.runs)),bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    archive=a.out/'kernel-graph-raw.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in files:tar.add(path,arcname=str(path.relative_to(a.runs)))
    manifest=dict(archive=archive.name,bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=inventory,
                  excludes='Model weights, generated native binaries, credentials; no raw performance result is excluded.')
    (a.out/'artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()
