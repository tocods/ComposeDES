"""Held-out CUDA Graph slices of real Transformers Qwen3-8B decoder layers.

The stock decoder provides the numerical reference with real checkpoint weights.
The replay exposes native layer operations and SGLang HiCache copy stages with
stage boundaries matching projection, attention and post-attention. It is a
standalone slice, with neither an HTTP server nor continuous batching.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random

from flashinfer.prefill import single_prefill_with_kv_cache as prefill
from safetensors import safe_open
from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from admission_bench import run_case

LAYER_COUNT = 3
PAGE = 256
CASES = (
    (4096, 128, 0, "packed", 1, 0),
    (6144, 384, 0, "paged", 1, 0),
    (8192, 768, 2048, "packed", 1, 1),
    (12288, 288, 3072, "paged", 1, 0),
    (4096, 320, 0, "packed", 2, 1),
    (6144, 640, 2048, "paged", 2, 0),
    (8192, 192, 0, "paged", 2, 2),
    (12288, 512, 4096, "packed", 2, 2),
)


def flashinfer_reference(module, q, k, v, mask, **kwargs):
    output = prefill(q[0].transpose(0, 1).contiguous(),
                     k[0].transpose(0, 1).contiguous(), v[0].transpose(0, 1).contiguous(),
                     causal=True, backend="fa2", sm_scale=module.scaling)
    return output.unsqueeze(0), None


class QwenSlice:
    def __init__(self, model):
        self.config = AutoConfig.from_pretrained(model)
        ALL_ATTENTION_FUNCTIONS.register("admission_flashinfer", flashinfer_reference)
        self.config._attn_implementation = "admission_flashinfer"
        self.rope = Qwen3RotaryEmbedding(self.config).cuda()
        with torch.device("meta"):
            self.layers = [Qwen3DecoderLayer(self.config, i).eval() for i in range(LAYER_COUNT)]
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        requested = defaultdict(list)
        for i, layer in enumerate(self.layers):
            for name in layer.state_dict():
                key = f"model.layers.{i}.{name}"
                requested[index[key]].append((i,name,key))
        weights = [dict() for _ in self.layers]
        for file, entries in requested.items():
            with safe_open(model / file, framework="pt", device="cpu") as shard:
                for i,name,key in entries:
                    weights[i][name] = shard.get_tensor(key).cuda()
        for layer, weight in zip(self.layers, weights):
            layer.load_state_dict(weight, assign=True)
        self.emb_shard = model / index["model.embed_tokens.weight"]

    def inputs(self, n, seed):
        rng = Random(seed)
        ids = [rng.randrange(10, 50000) for _ in range(n)]
        with safe_open(self.emb_shard, framework="pt", device="cpu") as shard:
            # Only touch selected rows of the memory-mapped embedding tensor.
            lookup = shard.get_tensor("model.embed_tokens.weight")
            return lookup[ids].cuda().unsqueeze(0)


def pin_pages(tensor, layout):
    if layout == "packed":
        host = torch.empty_like(tensor, device="cpu", pin_memory=True)
        host.copy_(tensor)
        return host, list(range(tensor.shape[0]//PAGE))
    count = tensor.shape[0]//PAGE
    host = torch.empty((count*2*PAGE,*tensor.shape[1:]), dtype=tensor.dtype, pin_memory=True)
    for page in range(count):
        host[2*page*PAGE:(2*page+1)*PAGE].copy_(tensor[page*PAGE:(page+1)*PAGE])
    return host, list(range(0, count*2, 2))


class Request:
    def __init__(self, model, prefix, suffix, resident, layout, seed):
        self.model, self.prefix, self.resident = model, prefix, resident
        self.suffix, self.missing, self.layout = suffix, prefix-resident, layout
        self.x = model.inputs(prefix+suffix, seed)
        self.states = [self.x[:, prefix:].contiguous()] + [
            torch.empty_like(self.x[:,prefix:]) for _ in model.layers
        ]
        self.position = torch.arange(prefix+suffix, device="cuda").unsqueeze(0)
        self.rope = model.rope(self.x, self.position)
        self.suffix_rope = tuple(w[:,prefix:].contiguous() for w in self.rope)
        cache = DynamicCache(config=model.config)
        hx = self.x[:,:prefix].contiguous()
        for layer in model.layers:
            hx = layer(hx, past_key_values=cache, use_cache=True,
                       position_embeddings=tuple(w[:,:prefix] for w in self.rope))
        # Read back real prefix KV from the original framework's Cache object.
        data = [tuple(t[0].transpose(0,1).contiguous() for t in (x.keys,x.values))
                for x in cache.layers[:LAYER_COUNT]]
        self.prefix_kv = data
        self.arena, self.host, self.srcs = [], [], []
        self.q = [torch.empty((suffix,model.config.num_attention_heads,model.config.head_dim),
                              dtype=self.x.dtype, device="cuda") for _ in model.layers]
        self.a = [torch.empty_like(q) for q in self.q]
        for kv in data:
            kv_gpu = [torch.empty((prefix+suffix,*t.shape[1:]), device="cuda", dtype=t.dtype) for t in kv]
            for dst, src in zip(kv_gpu, kv):
                dst[:prefix].copy_(src)
            self.arena.append(kv_gpu)
            hosts, pages = zip(*(pin_pages(x[resident:prefix], layout) for x in kv))
            self.host.append(hosts)
            self.srcs.append(pages[0])
        self.src = torch.tensor([page*PAGE+t for page in self.srcs[0] for t in range(PAGE)],
                                device="cuda", dtype=torch.int32)
        self.dst = torch.arange(resident,prefix,device="cuda",dtype=torch.int32)
        hx = self.states[0]
        for layer in model.layers:
            hx = layer(hx, past_key_values=cache, use_cache=True,
                       position_embeddings=self.suffix_rope)
        self.reference = hx

    def projection(self, i):
        layer = self.model.layers[i]
        attn = layer.self_attn
        x = layer.input_layernorm(self.states[i])
        shaped = (1, self.suffix, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shaped)).transpose(1,2)
        k = attn.k_norm(attn.k_proj(x).view(shaped)).transpose(1,2)
        v = attn.v_proj(x).view(shaped)
        q,k = apply_rotary_pos_emb(q,k,*self.suffix_rope)
        self.q[i].copy_(q[0].transpose(0,1))
        self.arena[i][0][self.prefix:].copy_(k[0].transpose(0,1))
        self.arena[i][1][self.prefix:].copy_(v[0])

    def attention(self, i):
        k,v = self.arena[i]
        self.a[i].copy_(prefill(self.q[i], k, v, causal=True, backend="fa2"))

    def post(self, i):
        layer = self.model.layers[i]
        x = self.states[i] + layer.self_attn.o_proj(self.a[i].reshape(1,self.suffix,-1))
        self.states[i+1].copy_(x + layer.mlp(layer.post_attention_layernorm(x)))

    def copy(self, i, cap):
        k,v = self.arena[i]
        source_k, source_v = self.host[i]
        if cap != "dma":
            transfer_hicache_one_layer(k, v, self.dst, source_k, source_v, self.src,
                                       element_dim=k.shape[1]*k.shape[2], block_quota=int(cap), page_size=64)
        elif self.layout == "packed":
            k[self.resident:self.prefix].copy_(source_k, non_blocking=True)
            v[self.resident:self.prefix].copy_(source_v, non_blocking=True)
        else:
            for dst, src in ((k,source_k),(v,source_v)):
                for j, page in enumerate(self.srcs[i]):
                    lo = self.resident + j*PAGE
                    dst[lo:lo+PAGE].copy_(src[page*PAGE:(page+1)*PAGE], non_blocking=True)

    def reset(self):
        for pair in self.arena:
            for item in pair:
                item[self.resident:].fill_(float("nan"))
        for state in self.states[1:]:
            state.fill_(float("nan"))

    def seal(self):
        self.canonical = [x.clone() for x in self.states[1:]]
        reference = self.reference.float()
        rel = ((self.states[-1].float()-reference).norm()/reference.norm()).item()
        if rel > .008:
            raise AssertionError(f"Framework reference relative L2 error {rel:.5f}")
        return rel

    def validate(self):
        for actual, expected in zip(self.states[1:],self.canonical):
            torch.testing.assert_close(actual,expected, atol=0,rtol=0)
        for arena, kv in zip(self.arena,self.prefix_kv):
            for target, source in zip(arena,kv):
                assert torch.equal(target[:self.prefix],source)


def bench(model, case, session, repeat, output):
    prefix, suffix, resident, layout, branches, window = case
    requests = [
        Request(model,prefix,max(64,suffix//(i+1)),resident,layout,f"{case}/{i}")
        for i in range(branches)
    ]
    deps, kinds, sizes, builders = [], [], [], []
    last_read = None
    posts = []
    prior_post = [None]*branches
    for i in range(LAYER_COUNT):
        reads = []
        for j, req in enumerate(requests):
            parents = [] if last_read is None else [last_read]
            if window and i >= window:
                # A prefetch horizon of W layers for the framework slice.
                parents.append(posts[i-window][j])
            reads.append(len(deps))
            last_read = len(deps)
            deps.append(tuple(dict.fromkeys(parents)))
            kinds.append("io")
            sizes.append(req.missing/4096)
            builders.append(lambda cap, req=req,i=i: lambda: req.copy(i,cap))
        posts.append([])
        for j, req in enumerate(requests):
            proj = len(deps)
            deps.extend((() if prior_post[j] is None else (prior_post[j],),
                         (proj,reads[j]), (proj+1,)))
            kinds.extend(["compute"] * 3)
            rel = req.suffix/512
            sizes.extend([rel*.45,rel*prefix/8192,rel*.9])
            builders.extend(
                lambda fn=fn, i=i: lambda: fn(i) for fn in (req.projection,req.attention,req.post)
            )
            prior_post[j] = proj+2
            posts[-1].append(proj+2)
            # Seed stable live inputs for the standalone kernel fragments.
            req.projection(i)
            req.attention(i)
            req.post(i)

    def reset():
        for req in requests:
            req.reset()

    def validate():
        for req in requests:
            req.validate()

    native_relative_errors = [req.seal() for req in requests]
    print("Framework relative L2", native_relative_errors, flush=True)
    run_case(output, name="/".join(map(str,case)), cohort="qwen-framework", session=session,
             deps=deps,kinds=kinds,sizes=sizes,builders=builders,layout=layout,reset=reset,
             validate=validate,repeats=repeat)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    order = list(CASES[:1] if args.smoke else CASES)
    Random(args.session).shuffle(order)
    with torch.cuda.stream(torch.cuda.Stream()), args.output.open("w", buffering=1) as file:
        model = QwenSlice(args.model)
        for case in order:
            print("Start", *case, flush=True)
            bench(model,case,args.session,args.repeat,file)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
