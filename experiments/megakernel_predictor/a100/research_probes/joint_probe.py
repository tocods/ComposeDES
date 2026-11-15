"""Joint creation-order/transfer benchmarks with endpoint-only timings."""
import argparse
import json
from pathlib import Path
from random import Random
from statistics import median

from flashinfer.prefill import single_prefill_with_kv_cache as attention
from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer
import torch

from explicit_graph import Graph, fragment
from joint_cases import CAPS, COMPUTE, COPIES, DEPENDENCIES, PAGE, TOKENS, WORK, create_orders, jobs


class Fixture:
    def __init__(self, kernel, shape, layout):
        shape = int(shape)
        dtype = torch.bfloat16
        self.layer = kernel
        pages = TOKENS // PAGE
        lanes = 1 if layout == "packed" else 4
        pool_shape = (COPIES, 2, pages * lanes, PAGE, 8, 128)
        self.host = torch.empty(pool_shape, dtype=dtype, pin_memory=True).normal_(std=0.15)
        self.pages = list(range(pages)) if lanes == 1 else [i*lanes + (i % lanes) for i in range(pages)]
        self.packed = layout == "packed"
        self.gpu = torch.empty((COPIES, 2, TOKENS, 8, 128), dtype=dtype, device="cuda")
        self.copy_index = torch.arange(TOKENS, dtype=torch.int32, device="cuda")
        source_indices = [page*PAGE + i for page in self.pages for i in range(PAGE)]
        self.source_index = torch.tensor(source_indices, dtype=torch.int32, device="cuda")
        self.copy_ref = self.host[:, :, self.pages].reshape(self.gpu.shape).cuda()
        self.gpu.copy_(self.copy_ref)
        self.resident = torch.randn_like(self.gpu[0])
        if kernel == "attention":
            self.query = torch.randn((COPIES+1, shape*3, 40, 128), dtype=dtype, device="cuda")
            d_out = (shape*3, 40, 128)
        else:
            self.weight = torch.randn((1024, shape), dtype=dtype, device="cuda")
            d_out = (TOKENS * 3, shape)
        self.results = torch.empty((COPIES+1, *d_out), dtype=dtype, device="cuda")

    def copy(self, i, cap):
        keys, values = self.gpu[i].unbind()
        host_k, host_v = self.host[i].flatten(1, 2).unbind()
        if cap == "dma":
            if self.packed:
                keys.copy_(host_k, non_blocking=True)
                values.copy_(host_v, non_blocking=True)
            else:
                for dest, source in ((keys, host_k), (values, host_v)):
                    for j, page in enumerate(self.pages):
                        dest[j*PAGE:(j+1)*PAGE].copy_(
                            source[page*PAGE:(page+1)*PAGE], non_blocking=True)
        else:
            transfer_hicache_one_layer(
                keys, values, self.copy_index, host_k, host_v, self.source_index,
                element_dim=1024, block_quota=int(cap), page_size=64)

    def compute(self, i, multiplier, copies):
        keys, values = self.gpu[copies[-1]-COMPUTE] if copies else self.resident
        out = self.results[i]
        if self.layer == "attention":
            rows = self.query.shape[1]//3 * multiplier
            out[:rows].copy_(attention(self.query[i, :rows], keys, values, causal=False, backend="fa2"))
        else:
            for j in range(multiplier):
                torch.mm(keys.view(TOKENS, -1), self.weight, out=out[j*TOKENS:(j+1)*TOKENS])


def measure_case(case, session, repeat, output, profile=False, matched=False):
    family, kernel, shape, layout = case
    deps = DEPENDENCIES[family]
    matches = WORK["gated"] if matched else WORK[family]
    rng = Random(f"{session}/{case}")
    fix = Fixture(kernel, shape, layout)
    fix.results.zero_()
    compute = [
        fragment(lambda: fix.compute(i, multiplier, [n for n in deps[i] if n >= COMPUTE]))
        for i, multiplier in enumerate(matches)
    ]
    reference = fix.results.clone()
    copy = {cap: [fragment(lambda: fix.copy(i, cap)) for i in range(COPIES)] for cap in CAPS}
    solo = [Graph([body], [[]], [0]) for body in compute]
    solo_copy = {cap: [Graph([body], [[]], [0]) for body in fragments]
                 for cap, fragments in copy.items()}

    def isolation(g):
        g.run()
        return median(g.run() for _ in range(7))

    isolated = [isolation(g) for g in solo]
    isolated_copy = {cap: [isolation(g) for g in gs] for cap, gs in solo_copy.items()}
    # Freeze critical-path construction using the same offline 8-CTA priors for all policies.
    # Coarse analytic weights make the actual order identical between measurement processes.
    order_prior = [weight * (shape/2048) for weight in matches] + [1.0] * COPIES
    orders = create_orders(family, order_prior)
    order_ids = dict(orders)
    cases = [(cap, name) for cap in CAPS for name in orders]
    if profile:
        cases = [(cap, name) for cap, name in cases
                 if cap in ("dma", "8") and name in ("io", "compute", "critical", "random-19")]
    rng.shuffle(cases)
    graphs = {}
    aliases = {}
    for cap, name in cases:
        key = cap, orders[name]
        if key in aliases:
            graphs[cap, name] = aliases[key]
            continue
        graph = Graph(compute + copy[cap], deps, orders[name])
        fix.gpu.fill_(float("nan"))
        fix.results.zero_()
        graph.run()
        assert torch.equal(fix.gpu, fix.copy_ref)
        torch.testing.assert_close(fix.results, reference, atol=0, rtol=0)
        graph.run()
        aliases[key] = graph
        graphs[cap, name] = graph
    label = dict(session=session, family=family, kernel=kernel, shape=shape, layout=layout)
    # A duplicate order is measured once per repeat and emitted under each policy name.
    # The evaluator deduplicates equivalent candidate graphs before counting a query budget.
    output.write(json.dumps(dict(**label, type="metadata", orders=order_ids, compute_work=matches,
                                isolated=isolated, copy=isolated_copy))+"\n")
    prior = None
    for index in range(repeat):
        rng.shuffle(cases)
        times = {}
        for cap, name in cases:
            key = cap, orders[name]
            if key not in times:
                if profile:
                    torch.cuda.nvtx.range_push(f"joint/{family}/{cap}/{name}/{index}")
                times[key] = graphs[cap, name].run()
                if profile:
                    torch.cuda.nvtx.range_pop()
            ms = times[key]
            output.write(json.dumps(dict(**label, type="measurement", cap=cap, order=name,
                                         repeat=index, prior=prior, ms=ms))+"\n")
            prior = cap, name


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--repeat", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--matched", action="store_true",
                   help="Use exactly the same seven compute sizes in all three dependency graphs")
    args = p.parse_args()
    seed = 41 if args.session == "A" else 67
    torch.manual_seed(seed)
    cases = list(jobs())
    if args.smoke:
        cases = [("gated", "attention", 128, "paged"), ("join", "gemm", 3072, "packed")]
    elif args.profile:
        cases = [(family, "attention", 1536, "packed") for family in ("gated", "join")]
    elif args.matched:
        cases = [(family, "attention", 1536, "packed") for family in DEPENDENCIES]
    else:
        Random(seed).shuffle(cases)
    with torch.cuda.stream(torch.cuda.Stream()), args.output.open("w", buffering=1) as output:
        for case in cases:
            print("Start", *case, flush=True)
            measure_case(case, args.session, args.repeat, output, args.profile, args.matched)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
