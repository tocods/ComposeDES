"""Test timing-event perturbation and collect isolated service on new shapes."""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch

from run_cta_curve import STAGES, request_ids
from run_readback import Model, Workload, write
from gpu_state import GPU


LEVELS = {"end": False, "boundary": "boundaries", "full": True}


def prepare(model, session, prefix, suffix):
    return Workload(model, dict(prefix=prefix, suffix=suffix, resident=0,
                                ids=request_ids(session, prefix, suffix, -1)))


def graph_pairs(model, args, output, gpu):
    work = prepare(model, args.session, 31232, 512)
    for block in range(2):
        reverse = bool(block) ^ (args.session == "B")
        names = [(level, q, tag) for level in LEVELS for q in (2, 8) for tag in ("a", "b")]
        if reverse:
            names.reverse()
        graphs = {}
        for level, q, tag in names:
            graphs[level, q, tag] = work.capture(q, "whole", timing=LEVELS[level])
        for g in graphs.values():
            work.trial(g, include_trace=False)
        comparisons = [("AB", level, 2) for level in LEVELS]
        comparisons += [("AA", level, q) for level in LEVELS for q in (2, 8)]
        comparisons += [("same", level, 2) for level in LEVELS]
        rng = random.Random(f"graphs/{args.session}/{block}")
        prior = None
        for rep in range(args.repeat):
            order = comparisons[:]
            rng.shuffle(order)
            for comparison, level, q in order:
                left = (q, "a")
                right = (8, "a") if comparison == "AB" else (q, "b") if comparison == "AA" else left
                arms = (0, 1) if (rep + reverse) % 2 == 0 else (1, 0)
                tokens = []
                for position, arm in enumerate(arms):
                    quota, tag = (left, right)[arm]
                    graph = graphs[level, quota, tag]
                    t0 = perf_counter_ns()
                    reading = work.trial(graph, include_trace=False)
                    t1 = perf_counter_ns()
                    name = f"{level}/{quota}/{tag}"
                    write(output, dict(reading, session=args.session, capture_block=block,
                                       capture_reverse=reverse, comparison=comparison, level=level,
                                       tested_quota=q, quota=quota, arm=arm, position=position,
                                       repeat=rep, prior=prior, compute_timers=len(graph[3]),
                                       h2d_timers=len(graph[4]) * (1 if level == "boundary" else 2),
                                       gpu=gpu.state, start_ns=t0, end_ns=t1))
                    prior = name
                    tokens.append(reading["token"])
                assert tokens[0] == tokens[1], tokens


def isolated(args, model, output):
    for prefix, suffixes in STAGES["heldout"].items():
        for suffix in suffixes:
            build = perf_counter_ns()
            work = prepare(model, args.session, prefix, suffix)
            graph = work.capture(2, "whole", mode="compute", timing=True)
            setup_us = (perf_counter_ns() - build) / 1000
            for repeat in range(args.repeat):
                start = perf_counter_ns()
                row = work.trial(graph)
                write(output, dict(row, prefix=prefix, suffix=suffix, repeat=repeat,
                                   setup_us=setup_us if repeat == 0 else 0,
                                   elapsed_us=(perf_counter_ns()-start)/1000))
            del graph, work
            torch.cuda.empty_cache()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("graph-pairs", "isolated"), required=True)
    p.add_argument("--session", choices=("A", "B"), default="A")
    p.add_argument("--repeat", type=int, default=8)
    args = p.parse_args()
    model = Model(args.model)
    with torch.cuda.stream(model.stream), args.output.open("w", buffering=1) as out, GPU() as gpu:
        if args.mode == "graph-pairs":
            graph_pairs(model, args, out, gpu)
        else:
            isolated(args, model, out)


if __name__ == "__main__":
    main()
