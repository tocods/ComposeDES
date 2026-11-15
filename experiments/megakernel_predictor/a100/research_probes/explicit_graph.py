"""Reuse captured kernel parameters in differently ordered explicit CUDA Graphs."""
from ctypes import CDLL, POINTER, c_float, c_int, c_size_t, c_void_p, byref
from pathlib import Path

import torch


def fragment(fn):
    fn()
    torch.cuda.current_stream().synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        fn()
    return graph


class Graph:
    _lib = None

    @classmethod
    def library(cls):
        if Graph._lib is None:
            lib = CDLL(str(Path(__file__).with_name("libexplicit_graph.so")))
            lib.plan_create.argtypes = [c_int, POINTER(c_size_t), POINTER(c_int),
                                        POINTER(c_int), POINTER(c_int), c_int]
            lib.plan_create.restype = c_void_p
            lib.plan_run.argtypes = [c_void_p, c_size_t, POINTER(c_float), POINTER(c_float)]
            lib.plan_destroy.argtypes = [c_void_p]
            Graph._lib = lib
        return Graph._lib

    def __init__(self, bodies, dependencies, order, trace=False):
        self.library()
        n = len(bodies)
        assert sorted(order) == list(range(n))
        assert all(p != i for i, parents in enumerate(dependencies) for p in parents)
        self.bodies = bodies
        offsets = [0]
        flattened = []
        for parents in dependencies:
            flattened.extend(parents)
            offsets.append(len(flattened))
        self.handle = self._lib.plan_create(
            n, (c_size_t * n)(*(f.raw_cuda_graph() for f in bodies)), (c_int * n)(*order),
            (c_int * len(offsets))(*offsets), (c_int * len(flattened))(*flattened), int(trace))
        if not self.handle:
            raise RuntimeError("Failed to construct the explicit CUDA graph")
        self.trace = (c_float * (2*n))() if trace else None
        self.time = c_float()

    def run(self):
        stream = torch.cuda.current_stream().cuda_stream
        if self._lib.plan_run(self.handle, stream, byref(self.time), self.trace):
            raise RuntimeError("CUDA graph run failed")
        return self.time.value

    def __del__(self):
        if getattr(self, "handle", None):
            self._lib.plan_destroy(self.handle)


def pressure(body, sms):
    """Query the static grid/occupancy proxy for a captured logical task."""
    query = Graph.library().fragment_pressure
    query.argtypes = [c_size_t, c_int]
    query.restype = c_float
    value = query(body.raw_cuda_graph(), sms)
    if value < 0:
        raise RuntimeError("Could not inspect captured launch resources")
    return value
