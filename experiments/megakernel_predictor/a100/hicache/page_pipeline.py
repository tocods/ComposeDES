"""Measure one dense GQA layer with real H2D cache promotion and CUDA Graph replay."""
import argparse
import itertools
import json
from pathlib import Path
import random

import flashinfer
import torch
from flashinfer.cascade import merge_states
from flashinfer.prefill import single_prefill_with_kv_cache as prefill


CANDIDATES = ("bulk_wait", "group_wait", "split_wait", "split_overlap")


class Layer:
    def __init__(self, prefix, suffix, resident, group, seed):
        self.prefix = prefix
        self.suffix = suffix
        self.resident = resident
        self.ranges = [
            (start, min(start + group, prefix))
            for start in range(resident, prefix, group)
        ]
        self.io = torch.cuda.Stream()
        self.compute = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in self.ranges]
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.host = [
            torch.randn((prefix, 8, 128), dtype=torch.bfloat16, generator=rng).pin_memory()
            for _ in range(2)
        ]
        self.device = [
            torch.empty((prefix + suffix, 8, 128), dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        self.q = torch.randn((suffix, 32, 128), dtype=torch.bfloat16, generator=rng).cuda()
        for dst, src in zip(self.device, self.host):
            dst[:prefix].copy_(src)
            dst[prefix:].copy_(torch.randn((suffix, 8, 128), dtype=torch.bfloat16, generator=rng))
        torch.cuda.synchronize()

    def load(self, start, end):
        for dst, src in zip(self.device, self.host):
            dst[start:end].copy_(src[start:end], non_blocking=True)

    def attention(self, start, end, causal=False, lse=True, query=None):
        k, v = self.device
        q = self.q if query is None else query
        return prefill(q, k[start:end], v[start:end], causal=causal,
                       backend="fa2", return_lse=lse)

    def combine(self, states):
        outputs, lse = zip(*states)
        return merge_states(torch.stack(outputs, dim=1), torch.stack(lse, dim=1))[0]

    def enqueue_load(self, candidate):
        if candidate == "bulk_wait":
            self.load(self.resident, self.prefix)
            self.ready[-1].record()
            return
        for (start, end), event in zip(self.ranges, self.ready):
            self.load(start, end)
            event.record()

    def compute_attention(self, candidate, query=None):
        if candidate != "split_overlap":
            self.compute.wait_event(self.ready[-1])
        if candidate in ("bulk_wait", "group_wait"):
            return self.attention(
                0, self.prefix + self.suffix, causal=True, lse=False, query=query)

        # All prefix keys are visible to every appended query; only the suffix
        # needs the triangular causal mask. Keep the merge order fixed.
        states = [self.attention(0, self.resident, query=query)] if self.resident else []
        states.append(self.attention(
            self.prefix, self.prefix + self.suffix, causal=True, query=query))
        for (start, end), event in zip(self.ranges, self.ready):
            if candidate == "split_overlap":
                self.compute.wait_event(event)
            states.append(self.attention(start, end, query=query))
        return self.combine(states)

    def run(self, candidate):
        self.io.wait_stream(self.compute)
        with torch.cuda.stream(self.io):
            self.enqueue_load(candidate)
        return self.compute_attention(candidate)

    def capture(self, fn):
        with torch.cuda.stream(self.compute):
            for _ in range(2):
                value = fn()
        self.compute.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.compute):
            value = fn()
        return graph, value

    def services(self, grouped_transfer):
        intervals = [(self.resident, self.prefix), *self.ranges]
        graph = {}
        graph["bulk_copy"] = self.capture(lambda: self.load(*intervals[0]))
        if grouped_transfer:
            graph["grouped_copy"] = self.capture(lambda: self.enqueue_load("group_wait"))
        for i, (start, end) in enumerate(self.ranges):
            graph[f"copy_{i}"] = self.capture(lambda: self.load(start, end))
            graph[f"partial_{i}"] = self.capture(lambda: self.attention(start, end))
        graph["whole_attention"] = self.capture(
            lambda: self.attention(0, self.prefix + self.suffix, causal=True, lse=False))
        graph["suffix"] = self.capture(
            lambda: self.attention(self.prefix, self.prefix + self.suffix, causal=True))
        names = ["suffix", *(f"partial_{i}" for i in range(len(self.ranges)))]
        if self.resident:
            graph["resident"] = self.capture(lambda: self.attention(0, self.resident))
            names.insert(0, "resident")
        for name in names:
            graph[name][0].replay()
        torch.cuda.synchronize()
        states = [graph[name][1] for name in names]
        graph["merge"] = self.capture(lambda: self.combine(states))
        return graph


def samples(graph, repetitions):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    results = []
    # Create events and warm the graph before recording samples.
    graph.replay()
    a.record()
    b.record()
    b.synchronize()
    for _ in range(repetitions):
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        results.append(1000 * a.elapsed_time(b))
    return results


def benchmark(prefix, suffix, resident, group, seed, repetitions, mode, grouped_transfer):
    layer = Layer(prefix, suffix, resident, group, seed)
    row = {
        "prefix": prefix, "suffix": suffix, "resident": resident, "group": group,
        "copied_bytes": (prefix - resident) * 2 * 8 * 128 * 2,
        "groups": len(layer.ranges), "seed": seed,
    }
    if mode == "calibration":
        row["service_us"] = {name: samples(graph, repetitions)
                             for name, (graph, _) in layer.services(grouped_transfer).items()}
        empty_graph, _ = layer.capture(lambda: layer.q[:1].clone())
        row["replay_floor_us"] = samples(empty_graph, repetitions)
        return row

    order = list(CANDIDATES)
    random.Random(seed).shuffle(order)
    captures = {name: layer.capture(lambda: layer.run(name)) for name in order}
    # Poison the promoted destination before each correctness replay.
    # A missing wait then exposes stale values instead of a previous replay's KV.
    outputs = {}
    for name, (graph, output) in captures.items():
        for kv in layer.device:
            kv[resident:prefix].fill_(float("nan"))
        graph.replay()
        outputs[name] = output.clone()
    ref = outputs["bulk_wait"]
    assert bool(ref.isfinite().all())
    row["max_abs_error"] = {}
    for name, output in outputs.items():
        error = (ref.float() - output.float()).abs().max().item()
        assert torch.allclose(ref, output, atol=0.003, rtol=0.02), (name, error)
        row["max_abs_error"][name] = error
    row["latency_us"] = {name: samples(graph, repetitions)
                         for name, (graph, _) in captures.items()}
    return row


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "calibration", "heldout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--sweep", choices=["initial", "transfer"], default="initial")
    args = parser.parse_args()
    if args.mode == "smoke":
        cases = [(8192, 128, 4096, 1024)]
    elif args.sweep == "initial":
        cases = itertools.product((8192,), (32, 128, 512, 2048), (0, 4096), (512, 1024, 2048))
    else:
        cases = itertools.product((12288,), (32, 128, 512, 2048), (0, 6144), (1024, 2048))
    seed = 173 if args.mode == "calibration" else 439
    if args.sweep == "transfer":
        seed += 867
    print(torch.cuda.get_device_name(0), torch.__version__, flashinfer.__version__, flush=True)
    with args.output.open("w", buffering=1) as out:
        for case in cases:
            row = benchmark(*case, seed, args.repetitions, args.mode, args.sweep == "transfer")
            print(json.dumps(row), file=out)
            print(args.mode, case, flush=True)


if __name__ == "__main__":
    main()
