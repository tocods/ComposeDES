"""Native SGLang HiSparse Resolve plus selected-KV FA2, with controlled cache misses."""
import argparse
import json
from pathlib import Path
import random

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha
import torch

from explicit_graph import Graph, fragment


TOPK = 1024
HISTORY = TOPK * 3
BLOCK = 16
HEADS, KV, DIM = 32, 8, 128
DTYPE = torch.bfloat16


def tensor(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device="cuda")


def distribution(batch, mode):
    if mode == "uniform":
        return [TOPK//4] * batch
    if mode == "concentrated":
        return [0] * (3*batch//4) + [TOPK] * (batch//4)
    if mode == "straggler":
        return [0] * (batch//4) + [TOPK//8] * (batch//2) + [3*TOPK//4] * (batch//4)
    raise ValueError(mode)


class SparseFixture:
    def __init__(self, count):
        self.n = count
        self.host = tuple(torch.empty((count*HISTORY, KV, DIM), dtype=DTYPE,
                                      pin_memory=True).normal_(std=.15) for _ in range(2))
        self.device = tuple(torch.empty((count*(TOPK+1), 1, KV, DIM), dtype=DTYPE,
                                        device="cuda") for _ in range(2))
        generator = random.Random(1369)
        chosen, remaining, logical = [], [], []
        for _ in range(count):
            blocks = generator.sample(range(HISTORY//BLOCK - 1), TOPK//BLOCK)
            chosen.append(blocks)
            tokens = [b*BLOCK + i for b in blocks for i in range(BLOCK)]
            logical.append(tokens)
            pool = sorted(set(range(HISTORY-1)) - set(tokens))
            generator.shuffle(pool)
            remaining.append(pool)
        self.logical = logical
        self.remaining = remaining
        self.topk = tensor(chosen)
        self.host_indices = torch.arange(count*HISTORY, device="cuda", dtype=torch.int64).view(count,HISTORY)
        self.slot_indices = torch.arange(count*(TOPK+1), device="cuda", dtype=torch.int32).view(count,TOPK+1)
        self.slot_tokens = torch.empty_like(self.slot_indices)
        self.lru = torch.empty((count,TOPK), dtype=torch.int16, device="cuda")
        self.initial_lru = torch.arange(TOPK, dtype=torch.int16, device="cuda").expand(count,-1).contiguous()
        self.output = torch.empty((count,TOPK), dtype=torch.int32, device="cuda")
        self.ids = torch.arange(count, dtype=torch.int32, device="cuda")
        self.lengths = torch.full((count,), HISTORY, dtype=torch.int32, device="cuda")
        self.query = torch.randn((count,HEADS,DIM), dtype=DTYPE, device="cuda")
        self.answer = torch.empty_like(self.query)
        self.workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        self.cached_indices = torch.tensor([i*HISTORY+t for i,l in enumerate(logical) for t in l])
        self.reference_kv = tuple(x[self.cached_indices].cuda() for x in self.host)

    def configure(self, mode):
        misses = distribution(self.n, mode)
        assert sum(misses) == self.n*TOPK//4
        slots = [self.logical[i][m:] + self.remaining[i][:m] + [HISTORY-1]
                 for i,m in enumerate(misses)]
        flat = torch.tensor([r*HISTORY+i for r, ts in enumerate(slots) for i in ts])
        self.initial_tokens = tensor(slots)
        self.initial_data = tuple(x[flat].cuda().reshape_as(dst)
                                  for x,dst in zip(self.host,self.device))
        self.reset()

    def reset(self):
        self.slot_tokens.copy_(self.initial_tokens)
        self.lru.copy_(self.initial_lru)
        for dst, source in zip(self.device,self.initial_data):
            dst.copy_(source)

    def resolve_fn(self, lo, hi):
        real = tensor([hi-lo])

        def run():
            load_blocks_to_device_buffer_mha(
                self.topk[lo:hi], self.slot_tokens, self.host_indices, self.slot_indices,
                *self.host, *self.device, self.output[lo:hi], self.ids[lo:hi],
                self.lengths[lo:hi], self.lru, item_size_bytes=KV*DIM*2,
                hot_buffer_size=TOPK, sparse_block_size=BLOCK, block_size=1024,
                num_real_reqs=real)
        return run

    def attention_fn(self, lo, hi):
        rows = hi-lo
        pointers = torch.arange(0,(rows+1)*TOPK,TOPK, dtype=torch.int32, device="cuda")
        indices = self.output[lo:hi].flatten()
        tails = torch.ones(rows, device="cuda", dtype=torch.int32)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", use_cuda_graph=True,
            paged_kv_indptr_buffer=pointers.clone(), paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=tails.clone(), use_tensor_cores=True, backend="fa2")
        wrapper.plan(pointers, indices.clone(), tails, HEADS, KV, DIM, 1,
                     q_data_type=DTYPE, kv_data_type=DTYPE)

        def run():
            wrapper.run(self.query[lo:hi], self.device, out=self.answer[lo:hi])
        return run

    def validate_slots(self):
        assert bool((self.output >= 0).all())
        indices = self.output.to(torch.long).flatten()
        actual_tokens = self.slot_tokens.flatten()[indices]
        expected = tensor(self.logical).flatten()
        assert torch.equal(actual_tokens, expected)
        for actual, expected in zip(self.device, self.reference_kv):
            assert torch.equal(actual[indices, 0], expected)


def parts(batch, mode):
    return {
        "batch": [(0,batch)],
        "halves": [(0,batch//2),(batch//2,batch)],
        "quarters": [(i,i+batch//4) for i in range(0,batch,batch//4)],
        "ready-first": [(0,3*batch//4),(3*batch//4,batch)] if mode == "concentrated"
                       else [(0,batch//4),(batch//4,batch)] if mode == "straggler"
                       else [(0,batch//2),(batch//2,batch)],
    }


def assemble(fix, groups):
    loaders = [fix.resolve_fn(lo,hi) for lo,hi in groups]
    attentions = [fix.attention_fn(lo,hi) for lo,hi in groups]
    bodies = [fragment(f) for f in loaders + attentions]
    n = len(groups)
    io = list(range(n))
    comp = list(range(n,2*n))
    deps = [[]] + [[i-1] for i in io[1:]]
    deps += [[0]] + [[n+i-1,i] for i in io[1:]]
    order1 = io+comp
    order2 = [i for both in zip(io,comp) for i in both]
    # Keep closures: their wrappers own the buffers pointed to by the graphs.
    return (Graph(bodies,deps,order1), Graph(bodies,deps,order2),
            Graph(bodies,deps,order2,trace=True), loaders+attentions)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="A")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--repeat", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.manual_seed(51 if args.session=="A" else 73)
    with torch.cuda.stream(torch.cuda.Stream()), args.output.open("w", buffering=1) as file:
        for count in (8,) if args.smoke else (8,32):
            fix = SparseFixture(count)
            for mode in ("uniform","concentrated","straggler"):
                print("Start", count, mode, flush=True)
                fix.configure(mode)
                loader = fix.resolve_fn(0,count)
                loader()
                fix.validate_slots()
                attend = fix.attention_fn(0,count)
                attend()
                reference = fix.answer.clone()
                measures = {}
                policies = parts(count,mode)
                for name, groups in policies.items():
                    fix.reset()
                    loader()
                    g1,g2,traced,live = assemble(fix,groups)
                    for graph in (g1,g2,traced):
                        fix.reset()
                        graph.run()
                        fix.validate_slots()
                        torch.testing.assert_close(fix.answer, reference, atol=.003, rtol=.018)
                    measures[name] = (g1,g2,traced,live)
                solo_resolve = Graph([fragment(loader)],[[]],[0])
                solo_compute = Graph([fragment(attend)],[[]],[0])
                choices = [(name, order) for name in measures for order in (0,1,2)]
                rng = random.Random(f"{args.session}/{mode}/{count}")
                for rep in range(args.repeat):
                    rng.shuffle(choices)
                    for name, order in choices:
                        fix.reset()
                        graph = measures[name][order]
                        ms = graph.run()
                        row = dict(session=args.session,batch=count,mode=mode,repeat=rep,
                                   plan=name,construction=("io-first","interleaved","trace")[order],
                                   group_sizes=[hi-lo for lo,hi in policies[name]],
                                   misses=count*TOPK//4, bytes=count*TOPK//4*KV*DIM*4,ms=ms)
                        if order==2:
                            row["nodes_ms"] = list(graph.trace)
                        file.write(json.dumps(row, separators=(",",":"))+"\n")
                    for name, g in (("resolve",solo_resolve),("attention",solo_compute)):
                        if name=="resolve":
                            fix.reset()
                        ms = g.run()
                        file.write(json.dumps(dict(session=args.session,batch=count,mode=mode,
                                                  repeat=rep,plan=name,construction="isolated",
                                                  ms=ms),separators=(",",":"))+"\n")
                del measures,graph,g1,g2,traced,live,solo_resolve,solo_compute,loader,attend
            del fix


if __name__ == "__main__":
    main()
