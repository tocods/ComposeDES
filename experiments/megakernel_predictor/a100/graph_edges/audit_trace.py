"""Read separate Nsight traces; these observations do not train the predictor."""
import argparse
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result=[]
    for s,e in sorted(intervals):
        if result and s<=result[-1][1]:result[-1][1]=max(e,result[-1][1])
        else:result.append([s,e])
    return result


def length(intervals):return sum(e-s for s,e in union(intervals))


p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=sqlite3.connect(a.sqlite);result=[]
for start,end,label in c.execute('select start,end,text from NVTX_EVENTS where text like "candidate-%"'):
    k=list(c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(start,end)))
    m=list(c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(start,end)))
    io=[(s,e) for s,e,b in m];span=max(e for s,e in k+io)-min(s for s,e in k+io)
    result.append(dict(label=label,kernels=len(k),copies=len(m),copy_bytes=sum(b for s,e,b in m),
                       gpu_span_us=span/1000,kernel_union_us=length(k)/1000,copy_union_us=length(io)/1000,
                       overlap_us=(length(k)+length(io)-length(k+io))/1000,
                       idle_between_activities_us=(span-length(k+io))/1000))
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
