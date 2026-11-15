"""Test page-group overlap against layerwise prefetch in a four-layer CUDA graph."""
import argparse
import itertools
import json
from pathlib import Path
import random

import torch

from page_pipeline import Layer, samples


LAYERS = 4
CANDIDATES = ("bulk_all", "group_all", "split_first", "split_all")
SUFFIXES = (128, 512, 2048)


def policy(candidate, i):
    if candidate == "split_all" or (candidate == "split_first" and i == 0):
        return "split_overlap"
    if candidate == "group_all":
        return "group_wait"
    return "bulk_wait"


def make_projection():
    rng = torch.Generator(device="cuda").manual_seed(615)
    return torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda", generator=rng) / 64


def bridge(query, weight):
    return (query.flatten(1) @ weight).reshape_as(query)


def build(layers, candidate, weight):
    first = layers[0]
    first.io.wait_stream(first.compute)
    # Queue layer N+1 H2D without waiting for layer N compute.
    with torch.cuda.stream(first.io):
        for i, layer in enumerate(layers):
            layer.enqueue_load(policy(candidate, i))

    outputs = []
    q = first.q
    for i, layer in enumerate(layers):
        if i > 0 and weight is not None:
            q = bridge(q, weight)
        q = layer.compute_attention(policy(candidate, i), q)
        outputs.append(q)
    return tuple(outputs)


def benchmark(suffix, cached, group, projection, repetitions):
    layers = [Layer(8192, suffix, cached, group, seed=1051 + i) for i in range(LAYERS)]
    first = layers[0]
    for layer in layers[1:]:
        layer.compute = first.compute
    weight = make_projection() if projection else None
    order = list(CANDIDATES)
    random.Random(suffix + cached + group + projection).shuffle(order)
    caps = {name: first.capture(lambda: build(layers, name, weight)) for name in order}
    outputs = {}
    for name, (graph, values) in caps.items():
        for layer in layers:
            for kv in layer.device:
                kv[cached:layer.prefix].fill_(float("nan"))
        graph.replay()
        outputs[name] = [v.clone() for v in values]
    reference = outputs["bulk_all"]
    errors = {}
    for name, measured in outputs.items():
        pairs = list(zip(reference, measured))
        errors[name] = max((ref.float() - out.float()).abs().max().item() for ref, out in pairs)
        assert all(torch.allclose(ref, out, atol=0.003, rtol=0.02)
                   for ref, out in pairs), (name, errors[name])
    return dict(prefix=8192, suffix=suffix, resident=cached, group=group, projection=projection,
                layers=LAYERS, max_abs_error=errors,
                latency_us={name: samples(graph, repetitions) for name, (graph, _) in caps.items()})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bridge", "measure"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=21)
    args = parser.parse_args()
    with args.output.open("w", buffering=1) as f:
        if args.mode == "bridge":
            weight = make_projection()
            for suffix in SUFFIXES:
                layer = Layer(8192, suffix, 4096, 2048, seed=673)
                graph, _ = layer.capture(lambda: bridge(layer.q, weight))
                row = dict(suffix=suffix, bridge_us=samples(graph, args.repetitions))
                print(json.dumps(row), file=f)
        else:
            for case in itertools.product(SUFFIXES, (0, 4096), (1024, 2048), (0, 1)):
                print(json.dumps(benchmark(*case, args.repetitions)), file=f)
                print(case, flush=True)


if __name__ == "__main__":
    main()
