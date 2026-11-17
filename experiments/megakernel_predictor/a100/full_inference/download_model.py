"""Download public Qwen safetensors and tokenizer files from the official mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
a.directory.mkdir(parents=True,exist_ok=True)
base='https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/'
def fetch(name):
    dest=a.directory/name
    subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','5',
                    '--connect-timeout','20','--continue-at','-','--output',str(dest),base+name],check=True)
    print('downloaded',name,dest.stat().st_size,flush=True)
    return dest
for name in ('config.json','model.safetensors.index.json','tokenizer_config.json',
             'tokenizer.json','generation_config.json'):
    fetch(name)
index=json.loads((a.directory/'model.safetensors.index.json').read_text())
shards=sorted(set(index['weight_map'].values()))
assert all(re.fullmatch(r'model-\d+-of-\d+\.safetensors',s) for s in shards)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,shards))
checks={}
for file in sorted(a.directory.glob('*')):
    if file.name=='sha256.json' or not file.is_file():continue
    h=hashlib.sha256()
    with file.open('rb') as source:
        for chunk in iter(lambda:source.read(8<<20),b''):h.update(chunk)
    checks[file.name]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
(a.directory/'sha256.json').write_text(json.dumps(checks,indent=2))
print('complete',sum(x['bytes'] for x in checks.values()),flush=True)
