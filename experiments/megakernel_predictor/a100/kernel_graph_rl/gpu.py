import ctypes as C
import json,time
from pathlib import Path
import torch

class Runtime:
    def __init__(self,graph):
        self.graph=graph;self.raw=graph.raw_cuda_graph();self.lib=C.CDLL(str(Path(__file__).with_name('libkernel_graph.so')))
        p=C.POINTER(C.c_int);l=self.lib
        l.kg_export.argtypes=[C.c_size_t,C.c_char_p]
        l.kg_create.argtypes=[C.c_size_t,C.c_int,p,p];l.kg_create.restype=C.c_void_p
        l.kg_time.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.POINTER(C.c_float)]
        l.kg_launch.argtypes=[C.c_void_p,C.c_size_t];l.plan_destroy.argtypes=[C.c_void_p]
        l.kg_profile.argtypes=[C.c_size_t,C.c_size_t,C.POINTER(C.c_float)]
    def export(self,path):
        assert self.lib.kg_export(self.raw,str(path).encode())==0
        return json.loads(Path(path).read_text())
    def make(self,edges=()):
        ints=lambda xs:(C.c_int*len(xs))(*xs)
        p=self.lib.kg_create(self.raw,len(edges),ints([e[0] for e in edges]),ints([e[1] for e in edges]))
        if not p:raise RuntimeError('CUDA graph construction failed')
        return p
    def run(self,plan,replays=3):
        result=C.c_float()
        if self.lib.kg_time(plan,torch.cuda.current_stream().cuda_stream,replays,C.byref(result)):raise RuntimeError('CUDA execution failed')
        return result.value
    def launch(self,plan):
        if self.lib.kg_launch(plan,torch.cuda.current_stream().cuda_stream):raise RuntimeError('launch failed')
    def close(self,plan):self.lib.plan_destroy(plan)
    def profile(self,n):
        start=time.perf_counter();values=(C.c_float*n)()
        if self.lib.kg_profile(self.raw,torch.cuda.current_stream().cuda_stream,values):raise RuntimeError('kernel profiling failed')
        return list(values),time.perf_counter()-start
