"""Explicit full-model graphs built from native PyTorch/FlashInfer fragments."""
import ctypes as C
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent


def capture(fn,pool=None):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn();stream.synchronize()
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream,pool=pool):fn()
    torch.cuda.current_stream().wait_stream(stream)
    # Retain the capture stream: its cuBLAS workspace belongs to this fragment.
    return graph,stream


class Graph:
    lib=None
    def __init__(self,bodies,deps,order):
        if Graph.lib is None:
            lib=C.CDLL(str(HERE/'libfull_graph.so'));p=C.POINTER(C.c_int)
            lib.full_create.argtypes=[C.c_int,C.POINTER(C.c_size_t),p,p,p];lib.full_create.restype=C.c_void_p
            lib.full_launch.argtypes=[C.c_void_p,C.c_size_t]
            lib.plan_run.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_float),C.c_void_p]
            lib.plan_destroy.argtypes=[C.c_void_p]
            lib.fragment_pressure.argtypes=[C.c_size_t,C.c_int];lib.fragment_pressure.restype=C.c_float
            Graph.lib=lib
        self.bodies=bodies;flat=[];offset=[0]
        for parents in deps:flat.extend(parents);offset.append(len(flat))
        ints=lambda x:(C.c_int*len(x))(*x)
        self.handle=self.lib.full_create(len(bodies),(C.c_size_t*len(bodies))(*(g.raw_cuda_graph() for g in bodies)),
                                         ints(order),ints(offset),ints(flat))
        if not self.handle:raise RuntimeError('full graph construction failed')

    def launch(self):
        if self.lib.full_launch(self.handle,torch.cuda.current_stream().cuda_stream):raise RuntimeError('graph launch failed')

    def run(self):
        out=C.c_float()
        if self.lib.plan_run(self.handle,torch.cuda.current_stream().cuda_stream,C.byref(out),None):raise RuntimeError('graph run failed')
        return out.value

    def close(self):
        if self.handle:self.lib.plan_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,'handle',None):self.close()

    @classmethod
    def pressure(cls,body,sms):return cls.lib.fragment_pressure(body.raw_cuda_graph(),sms)
