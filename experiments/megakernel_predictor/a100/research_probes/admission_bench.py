"""Endpoint-only CUDA Graph replays for selective calibration and held-out slices."""

import json
from random import Random
from statistics import median

import torch

from explicit_graph import Graph, fragment, pressure
from joint_cases import downstream


ORDER_RULES = ("io", "compute", "critical", "alternating", "random-23", "random-41")
CAPS = ("dma", "2", "8", "24", "64")


def orders_for(deps, kinds, sizes, rules):
    paths = downstream(deps, sizes)
    result = {}
    for rule in rules:
        rng, sequence = Random(rule), []
        while len(sequence) < len(deps):
            choices = [i for i, parents in enumerate(deps)
                       if i not in sequence and all(p in sequence for p in parents)]
            if rule in ("io", "compute"):
                sequence.append(min(choices, key=lambda i: (kinds[i] != rule, i)))
            elif rule == "critical":
                sequence.append(max(choices, key=lambda i: (paths[i], -i)))
            elif rule == "alternating":
                latest = kinds[sequence[-1]] if sequence else "compute"
                sequence.append(min(choices, key=lambda i: (kinds[i] == latest, i)))
            elif rule == "short":
                sequence.append(min(choices, key=lambda i: (sizes[i], i)))
            else:
                sequence.append(rng.choice(choices))
        result[rule] = sequence
    return result


def run_case(file, *, name, cohort, session, deps, kinds, sizes, builders, layout,
             reset, validate, repeats, rules=ORDER_RULES, caps=CAPS):
    rng = Random(f"{name}/{session}")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    source_stream = torch.cuda.current_stream()
    capture_streams = {}

    def capture(i, fn):
        # A GEMM captured on a stream may embed a pointer to its cuBLAS
        # workspace. Give concurrently runnable compute stages disjoint
        # workspaces even when their fragments are later transplanted.
        stream = torch.cuda.Stream()
        capture_streams[i] = stream
        stream.wait_stream(source_stream)
        with torch.cuda.stream(stream):
            body = fragment(fn)
        source_stream.wait_stream(stream)
        return body

    # Kernel fragments and data addresses are shared by all candidates.
    compute = {i: capture(i,fn()) for i,fn in enumerate(builders) if kinds[i] == "compute"}
    io = {cap: {i: fragment(fn(cap)) for i,fn in enumerate(builders) if kinds[i] == "io"}
          for cap in caps}
    bodies = {cap: [ (compute if k == "compute" else io[cap])[i]
                     for i,k in enumerate(kinds)] for cap in caps}
    solo = {}
    cached = {}
    for cap, fragments in bodies.items():
        times, resource = [], []
        for stage in fragments:
            if stage not in cached:
                graph = Graph([stage], [[]], [0])
                graph.run()
                cached[stage] = [median(graph.run() for _ in range(7)), pressure(stage, sms)]
            t, p = cached[stage]
            times.append(t)
            resource.append(p)
        solo[cap] = {"ms":times, "peak_cta_fraction":resource}
    orders = orders_for(deps, kinds, sizes, rules)
    configs = {}
    for cap in caps:
        for tag, order in orders.items():
            key = cap, tuple(order)
            configs.setdefault(key, tag)
    keys = list(configs)
    rng.shuffle(keys)
    graph_execs = {}
    refs = ((cap, label, list(order)) for (cap, order), label in configs.items())
    file.write(json.dumps(dict(type="meta", name=name, cohort=cohort, layout=layout,
                               session=session, deps=deps, kinds=kinds, sizes=sizes, 
                               sms=sms, solo=solo, candidates=list(refs)))+"\n")
    for cap, order in keys:
        g = Graph(bodies[cap], deps, order)
        reset()
        g.run()
        validate()
        g.run()
        graph_execs[cap, order] = g
    for i in range(repeats):
        rng.shuffle(keys)
        for key in keys:
            cap, _ = key
            ms = graph_execs[key].run()
            file.write(json.dumps(dict(type="measure", name=name, session=session, cap=cap,
                                       rule=configs[key], repeat=i, ms=ms))+"\n")
