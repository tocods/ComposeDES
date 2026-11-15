"""Fixed-address native HiCache/FA2 and H2D/GEMM graph-order probes."""
import argparse
import json
from pathlib import Path
import random

from flashinfer.prefill import single_prefill_with_kv_cache as prefill
from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer
import torch

from explicit_graph import Graph, fragment


GROUPS = 6
TOKENS = 4096
LAYERS = {
    "attention": ((128, 512, 1536), ("2", "8", "24", "64")),
    "gemm": ((0, 1, 2), ("dma", "2", "8", "24", "64")),
}


def fixture(kind, workload, candidates):
    t = TOKENS
    dtype = torch.bfloat16
    shape = (2, GROUPS, t, 8, 128) if kind == "gemm" else (2, t, GROUPS, 8, 128)
    host = torch.empty(shape, dtype=dtype, pin_memory=True).normal_()
    src = [tuple(host[k, lid] if kind == "gemm" else host[k, :, lid] for k in (0, 1))
           for lid in range(GROUPS)]
    device = [(a.cuda().contiguous(), b.cuda().contiguous()) for a,b in src]
    resident = [torch.randn_like(device[0][0]) for _ in range(2)]
    index = torch.arange(t, device="cuda", dtype=torch.int32)
    live = [host, resident, index]
    if kind == "attention":
        q = torch.randn((workload, 40, 128), dtype=dtype, device="cuda")
        live.append(q)
        results = torch.empty((GROUPS + 1, *q.shape), device="cuda", dtype=dtype)

        def compute(lid):
            keys, values = resident if lid == 0 else device[lid - 1]
            results[lid].copy_(prefill(q, keys, values, causal=False, backend="fa2"))
    else:
        width = (1024, 3072, 8192)[workload]
        weight = torch.randn((1024, width), dtype=dtype, device="cuda")
        live.append(weight)
        results = torch.empty((GROUPS + 1, t, width), dtype=dtype, device="cuda")

        def compute(lid):
            keys = resident[0] if lid == 0 else device[lid-1][0]
            torch.mm(keys.view(t, 1024), weight, out=results[lid])

    def copy(lid, variant):
        a, b = src[lid]
        k, v = device[lid]
        if variant == "dma":
            k.copy_(a, non_blocking=True)
            v.copy_(b, non_blocking=True)
        else:
            transfer_hicache_one_layer(k, v, index, a, b, index, element_dim=1024,
                                      block_quota=int(variant), page_size=64)

    compute_graphs = [fragment(lambda: compute(i)) for i in range(GROUPS+1)]
    for lid in range(GROUPS+1):
        compute(lid)
    reference = results.clone()
    bodies = {cap: compute_graphs + [fragment(lambda: copy(i, cap)) for i in range(GROUPS)]
              for cap in candidates}
    return bodies, device, results, reference, live


def topology(n):
    # CPU->GPU copy i and attention i each maintain their own stream order.
    # Consumer i+1 also waits for true KV i. Consumer 0 is resident.
    ids_c = list(range(n+1))
    ids_h = list(range(n+1, 2*n+1))
    deps = [[]] + [[i, ids_h[i]] for i in range(n)]
    deps += [[]] + [[i] for i in ids_h[:-1]]
    staggered = [0]
    for h,c in zip(ids_h, ids_c[1:]):
        staggered.extend((h,c))
    lookahead = [0, *ids_h[:2]]
    for i,c in enumerate(ids_c[1:]):
        lookahead.append(c)
        if i + 2 < n:
            lookahead.append(ids_h[i+2])
    return deps, {"io-first": ids_h+ids_c, "interleaved": staggered, "lookahead-2": lookahead}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="A")
    p.add_argument("--repeat", type=int, default=14)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile", action="store_true", help="Mark a short endpoint-only run for Nsight Systems")
    args = p.parse_args()
    torch.manual_seed(91 if args.session == "A" else 222)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), args.output.open("w", buffering=1) as output:
        deps, orders = topology(GROUPS)
        for kind, (sizes, caps) in LAYERS.items():
            if args.smoke:
                sizes, caps = sizes[:1], caps[:2]
            if args.profile:
                sizes, caps = ((512,) if kind == "attention" else (2,)), ("8", "64")
            for size in sizes:
                print("Start", kind, size, flush=True)
                bodies, device, result, ref, live = fixture(kind, size, caps)
                rng = random.Random(f"{args.session}/{kind}/{size}")
                modes = (False,) if args.profile else (False, True)
                keys = [(cap, rule, trace) for cap in caps for rule in orders for trace in modes]
                rng.shuffle(keys)
                graphs = {(cap, rule, trace): Graph(bodies[cap], deps, orders[rule], trace)
                          for cap,rule,trace in keys}
                for graph in graphs.values():
                    for k,v in device:
                        k.fill_(float("nan"))
                        v.fill_(float("nan"))
                    graph.run()
                    torch.testing.assert_close(result, ref, atol=0, rtol=0)
                    graph.run()
                for repeat in range(args.repeat):
                    rng.shuffle(keys)
                    prior = None
                    for cap,rule,trace in keys:
                        graph = graphs[cap,rule,trace]
                        if args.profile:
                            torch.cuda.nvtx.range_push(f"probe/{kind}/{size}/{cap}/{rule}/{repeat}")
                        ms = graph.run()
                        if args.profile:
                            torch.cuda.nvtx.range_pop()
                        record = dict(session=args.session, pipeline=kind, shape=size, repeat=repeat,
                                      candidate=cap, construction=rule, events=trace, ms=ms,
                                      prior=prior, bytes=2*GROUPS*TOKENS*1024*2)
                        if trace:
                            record["nodes_ms"] = list(graph.trace)
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                        prior = (cap,rule,trace)
                del graphs, graph, bodies, device, result, ref, live
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
