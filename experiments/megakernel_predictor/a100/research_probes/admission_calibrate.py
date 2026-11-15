"""Small calibration graphs that expose competing admission hypotheses."""

import argparse
from pathlib import Path
from random import Random

import torch

from admission_bench import run_case
from joint_cases import COMPUTE, DEPENDENCIES
from joint_probe import Fixture


def calibration_jobs():
    for op, small, large in (("attention", 128, 1536), ("gemm", 1024, 8192)):
        for size in (small, large):
            for layout in ("packed", "paged"):
                yield "chain", op, size, layout, 1
        for ratio in (1, 3):
            yield "pair", op, small, "packed", ratio
            yield "mix", op, large, "packed", ratio
            yield "producer-front", op, large, "paged", ratio
            yield "producer-back", op, large, "paged", ratio


def measure(job, session, repeats, output):
    graph, op, size, layout, ratio = job
    fixture = Fixture(op, size, layout)
    if graph == "chain":
        deps = DEPENDENCIES[graph]
        kinds = ["compute"] * COMPUTE + ["io"] * 6
        costs = [1] * COMPUTE
        rules = ("io", "compute", "critical", "alternating")
    elif graph == "pair":
        deps, kinds, costs = ((), ()), ["compute"] * 2, [1, ratio]
        rules = ("short", "compute", "critical", "alternating", "random-41")
    elif graph == "mix":
        deps, kinds, costs = ((), ()), ["compute", "io"], [ratio]
        rules = ("io", "compute")
    else:
        parent = 0 if graph == "producer-front" else 1
        deps = ((), (), (0, 1, 3), (parent,), ())
        kinds, costs = ["compute"] * 3 + ["io"] * 2, [ratio, 1, 1]
        rules = ("io", "compute", "critical", "alternating", "random-41")
    count = len(costs)
    builders = []
    for i, mult in enumerate(costs):
        consumed = [p-count+COMPUTE for p in deps[i] if p >= count]
        builders.append(lambda i=i, mult=mult, consumed=consumed:
                        lambda: fixture.compute(i, mult, consumed))
        fixture.compute(i, mult, consumed)
    for j in range(len(deps)-count):
        builders.append(lambda cap, j=j: lambda: fixture.copy(j, cap))
    reference = fixture.results[:count].clone()
    h = len(deps)-count

    def reset():
        if h:
            fixture.gpu[:h].fill_(float("nan"))
        fixture.results[:count].zero_()

    def validate():
        for i, mult in enumerate(costs):
            span = (size if op=="attention" else fixture.gpu.shape[2])*mult
            torch.testing.assert_close(fixture.results[i, :span], reference[i, :span], atol=0, rtol=0)
        if h:
            assert torch.equal(fixture.gpu[:h], fixture.copy_ref[:h])

    run_case(output, name="/".join(map(str, job)), cohort="chain" if graph=="chain" else graph,
             session=session, deps=deps, kinds=kinds,
             sizes=[(size/(1024 if op=="attention" else 8192))*x for x in costs]+[1]*h,
             builders=builders, layout=layout, reset=reset, validate=validate, repeats=repeats,
             rules=rules, caps=("dma",) if not h else ("dma","2","8","24","64"))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", default="A")
    p.add_argument("--repeat", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.manual_seed(177)
    jobs = list(calibration_jobs())
    if args.smoke:
        jobs = [j for j in jobs if j[1]=="attention" and j[-1]==3][:4]
    Random("calibration").shuffle(jobs)
    with torch.cuda.stream(torch.cuda.Stream()), args.output.open("w", buffering=1) as file:
        for job in jobs:
            print(job, flush=True)
            measure(job,args.session,args.repeat,file)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
