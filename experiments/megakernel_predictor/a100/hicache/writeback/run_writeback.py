"""Replay fixed-budget, two-request Qwen workloads with native HiCache write-back."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch
from flashinfer.prefill import single_prefill_with_kv_cache as prefill
from sglang.kernels.ops.kvcache.hicache import (
    DEFAULT_BLOCK_QUOTA, _default_unroll, _jit_hicache_staged_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


SHAPES = (16384, 31744)
QUOTAS = (0, 2, 8, 24, 64, 108, 256)
WINDOWS = {"wide": 0, "tight": 39}
PAGE = 64
VICTIM = 2048
APPEND = 256
SHORT = 384
DECODE = 2
TOPICS = (
    "The geologists compared grain size and mineral diversity across sedimentary layers. ",
    "The tribunal reviewed testimony and reconciled it with the documentary record. ",
    "The control run isolates the effects of workload size and memory access patterns. ",
    "The paper evaluates latency, accuracy and throughput under independent workloads. ",
    "The library holds manuscript fragments that survived the fire in the adjacent district. ",
    "Measurements revealed variation across repeated trials and across calibration settings. ",
    "The instrument records fluorescence when illumination reaches the semiconductor. ",
    "The committee examined overlapping jurisdictions and proposed a revised procedure. ",
)


def requests(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    result = {}
    for split, count in (("calibration", 1), ("heldout", 4)):
        rows = []
        for length in SHAPES:
            for index in range(count):
                label = f"{split}-{length}-{index}"
                rng = random.Random(label)
                a = tokenizer.encode(label + " " + "".join(rng.choices(TOPICS, k=4300)))
                b = tokenizer.encode("New document " + "".join(rng.choices(TOPICS, k=100)))
                assert len(a) >= length and len(b) >= SHORT
                rows.append(dict(label=label, length=length, long=a[:length], short=b[:SHORT]))
        result[split] = rows
    result["victim"] = tokenizer.encode("Write-back document " + "".join(TOPICS * 100))[:VICTIM]
    assert len(result["victim"]) == VICTIM
    return result


def event():
    return torch.cuda.Event(enable_timing=True)


def elapsed(start, stop):
    return 1000 * start.elapsed_time(stop)


def stamp():
    point = event()
    point.record()
    return point


class Model:
    def __init__(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        self.core = self.model.model
        cfg = self.model.config
        self.layer_count = cfg.num_hidden_layers
        self.heads, self.kv_heads, self.dim = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
        assert (self.layer_count, self.heads, self.kv_heads, self.dim) == (40, 40, 8, 128)
        self.stream = torch.cuda.Stream()
        self.token_mailbox = torch.empty((2, 1), dtype=torch.long, pin_memory=True)
        torch.cuda.synchronize()

    def alloc(self, length):
        return [(torch.empty((length, self.kv_heads, self.dim), dtype=torch.bfloat16,
                             device="cuda"), torch.empty((length, self.kv_heads, self.dim),
                                                        dtype=torch.bfloat16, device="cuda"))
                for _ in self.core.layers]

    def forward(self, ids, kv, start, *, get_token=True, mark=None, attend=None,
                return_logits=False):
        count = ids.numel()
        end = start + count
        x = self.core.embed_tokens(ids.view(1, -1))
        pos = torch.arange(start, end, device="cuda").view(1, -1)
        cos, sin = self.core.rotary_emb(x, pos)
        for lid, (layer, (k, v)) in enumerate(zip(self.core.layers, kv)):
            if mark:
                mark(lid, "qkv")
            a = layer.self_attn
            z = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(z).view(1, count, self.heads, self.dim)).transpose(1, 2)
            keys = a.k_norm(a.k_proj(z).view(1, count, self.kv_heads, self.dim)).transpose(1, 2)
            q, keys = apply_rotary_pos_emb(q, keys, cos, sin)
            k[start:end].copy_(keys[0].transpose(0, 1))
            v[start:end].copy_(a.v_proj(z).view(count, self.kv_heads, self.dim))
            if mark:
                mark(lid, "attn")
            q = q[0].transpose(0, 1).contiguous()
            if attend is None:
                attention = prefill(q, k[:end], v[:end], causal=True, backend="fa2")
            else:
                attention = attend(lid, q, k[:end], v[:end])
            if mark:
                mark(lid, "post")
            x = x + a.o_proj(attention.reshape(1, count, self.heads * self.dim))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if not get_token:
            return None
        logits = self.model.lm_head(self.core.norm(x[:, -1:]))
        return logits if return_logits else logits.argmax(-1).reshape(1)

    def warm_prefix(self, ids, arena, length):
        for start in range(0, length, 1024):
            self.forward(ids[start:min(start + 1024, length)], arena, start, get_token=False)


class WriteBuffer:
    """Layer-first victim, page-first pinned destination and one fixed staging slab."""

    def __init__(self, model, ids):
        self.victim = model.alloc(VICTIM)
        model.warm_prefix(ids, self.victim, VICTIM)
        self.pristine = [(k[:SHORT + DECODE].clone(), v[:SHORT + DECODE].clone())
                         for k, v in self.victim]
        self.ptrs = tuple(torch.tensor([kv[i].data_ptr() for kv in self.victim],
                                       dtype=torch.uint64, device="cuda") for i in (0, 1))
        self.source_pages = torch.arange(0, VICTIM, PAGE, dtype=torch.int32, device="cuda")
        # Native copy routines need a CPU token-level destination mapping.
        order = list(range(VICTIM // PAGE))
        random.Random(7635).shuffle(order)
        self.destination = torch.tensor([p * PAGE + i for p in order for i in range(PAGE)],
                                        dtype=torch.int64)
        shape = (VICTIM, model.layer_count, model.kv_heads * model.dim)
        self.stage = tuple(torch.empty(shape, device="cuda", dtype=torch.bfloat16)
                           for _ in range(2))
        self.host = tuple(torch.empty(shape, device="cpu", dtype=torch.bfloat16, pin_memory=True)
                          for _ in range(2))
        self.writer = torch.cuda.Stream()
        element = 2 * model.kv_heads * model.dim
        self.native = _jit_hicache_staged_module(
            element_size=element, unroll=_default_unroll(element), block_quota=DEFAULT_BLOCK_QUOTA)
        self.args = (*self.host, self.destination, *self.stage, self.source_pages, *self.ptrs, PAGE)
        self.size = 2 * VICTIM * model.layer_count * element

    def reset(self):
        for (k, v), (a, b) in zip(self.victim, self.pristine):
            k[:a.shape[0]].copy_(a)
            v[:b.shape[0]].copy_(b)

    def schedule(self, ready, quota, detailed=False):
        with torch.cuda.stream(self.writer):
            self.writer.wait_event(ready)
            begin = stamp()
            packed = None
            if detailed:
                self.native.launch_all_lf_pf_controlled(*self.args, quota, 1, 0)
                packed = stamp()
                self.native.launch_all_lf_pf_controlled(*self.args, quota, 0, 1)
            elif quota:
                self.native.launch_all_lf_pf_controlled(*self.args, quota, 1, 1)
            else:
                self.native.launch_all_lf_pf_staged(*self.args)
            end = stamp()
        return begin, packed, end

    def check(self):
        for token in (0, 63, 64, VICTIM - 1):
            for layer in (0, len(self.victim) - 1):
                addr = self.destination[token].item()
                for i in (0, 1):
                    torch.testing.assert_close(
                        self.host[i][addr, layer], self.victim[layer][i][token].cpu().flatten(),
                        rtol=0, atol=0)


def episode(model, buffer, long_ids, short_ids, arena, window, quota, detailed=False,
            isolated=False, release_layer=None):
    prefix = long_ids.numel() - APPEND
    if release_layer is None:
        release_layer = WINDOWS[window]
    buffer.reset()
    model.stream.synchronize()
    start = stamp()
    observed = []
    transfer = None

    def mark(layer, phase):
        nonlocal transfer
        if detailed:
            observed.append((layer, phase, stamp()))
        if not isolated and layer == release_layer and phase == "qkv":
            ready = stamp()
            transfer = buffer.schedule(ready, quota, detailed)

    def publish(token, recipient):
        model.token_mailbox[recipient].copy_(token, non_blocking=True)
        return stamp()

    a = model.forward(long_ids[prefix:], arena, prefix, mark=mark)
    a_end = stamp()
    a_first = publish(a, 0)
    # Isolate the allocator's wait from the token egress copy. The latter
    # may also share the D2H engine, but it does not own these KV pages.
    a_end.synchronize()
    alloc_begin = perf_counter_ns()
    if transfer:
        transfer[-1].synchronize()
    blocked = (perf_counter_ns() - alloc_begin) / 1000 if transfer else 0.
    b = model.forward(short_ids, buffer.victim, 0)
    b_first = publish(b, 1)
    b_first.synchronize()
    a_token, b_token = (int(model.token_mailbox[i].item()) for i in (0, 1))
    a_last, b_last = a_first, b_first
    for i in range(DECODE):
        a = model.forward(a, arena, long_ids.numel() + i)
        a_last = publish(a, 0)
        a_last.synchronize()
        b = model.forward(b, buffer.victim, SHORT + i)
        b_last = publish(b, 1)
        b_last.synchronize()
    record = dict(a_ttft_us=elapsed(start, a_first), b_ttft_us=elapsed(start, b_first),
                  a_tpot_us=elapsed(a_first, a_last) / DECODE,
                  b_tpot_us=elapsed(b_first, b_last) / DECODE, alloc_block_us=blocked,
                  long_token=a_token, short_token=b_token, a_gpu_us=elapsed(start, a_end),
                  write_bytes=buffer.size if transfer else 0)
    if transfer:
        begin, packed, end = transfer
        record.update(write_start_us=elapsed(start, begin), write_end_us=elapsed(start, end))
        record["allocator_dependency_us"] = max(0., record["write_end_us"] - record["a_gpu_us"])
        if packed:
            record["pack_end_us"] = elapsed(start, packed)
    if detailed:
        record["points"] = [(lid, phase, elapsed(start, point)) for lid, phase, point in observed]
    return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("generate", "smoke", "calibration", "heldout"), required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompts-per-length", type=int)
    args = parser.parse_args()
    if args.mode == "generate":
        args.requests.write_text(json.dumps(requests(args.model), separators=(",", ":")) + "\n")
        return

    manifest = json.loads(args.requests.read_text())
    if args.prediction:
        frozen = json.loads(args.prediction.read_text())
        assert frozen["requests_sha256"] == sha256(args.requests.read_bytes()).hexdigest()
    else:
        frozen = None
    rows = manifest["calibration" if args.mode == "smoke" else args.mode]
    if args.mode == "smoke":
        rows = rows[:1]
    if args.prompts_per_length is not None:
        rows = [row for length in SHAPES
                for row in [row for row in rows if row["length"] == length][:args.prompts_per_length]]
    model = Model(args.model)
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(model.stream), args.output.open("w", buffering=1) as output:
        tensor = lambda ids: torch.tensor(ids, dtype=torch.long, device="cuda")
        buffer = WriteBuffer(model, tensor(manifest["victim"]))
        for quota in (0, 8, 108):
            start = stamp()
            buffer.schedule(start, quota, detailed=True)[-1].synchronize()
            buffer.check()
        for row in rows:
            print("Start", row["label"], flush=True)
            long_ids, short_ids = tensor(row["long"]), tensor(row["short"])
            arena = model.alloc(row["length"] + DECODE)
            model.warm_prefix(long_ids, arena, row["length"] - APPEND)
            ref = episode(model, buffer, long_ids, short_ids, arena, "wide", 0, isolated=True)
            if args.mode == "calibration":
                for repetition in range(args.repetitions):
                    record = episode(model, buffer, long_ids, short_ids, arena,
                                     "wide", 0, detailed=True, isolated=True)
                    output.write(json.dumps(dict(record, label=row["label"], length=row["length"],
                                                 quota=-1, window="isolated", repeat=repetition)) + "\n")
                buffer.reset()
                for quota in QUOTAS:
                    for repetition in range(args.repetitions):
                        model.stream.synchronize()
                        start = stamp()
                        begin, packed, end = buffer.schedule(start, quota, detailed=True)
                        end.synchronize()
                        output.write(json.dumps(dict(label=row["label"], length=row["length"],
                                                     quota=quota, window="write_only", repeat=repetition,
                                                     pack_us=elapsed(begin, packed),
                                                     dma_us=elapsed(packed, end))) + "\n")
            for repeat in range(args.repetitions):
                plan = frozen["choices"][str(row["length"])] if frozen else {}
                windows = plan or WINDOWS
                queues = [(window, quota) for window in windows for quota in
                          ((0, 8) if args.mode == "smoke" else QUOTAS)]
                if frozen:
                    queues += [(window, "predictor") for window in windows]
                random.Random(row["label"] + str(repeat)).shuffle(queues)
                for window, policy in queues:
                    torch.cuda.reset_peak_memory_stats()
                    quota = frozen["choices"][str(row["length"])][window]["quota"] if policy == "predictor" else policy
                    record = episode(model, buffer, long_ids, short_ids, arena, window, quota,
                                     detailed=args.mode == "calibration",
                                     release_layer=plan.get(window, {}).get("release_layer"))
                    assert (record["long_token"], record["short_token"]) == (
                        ref["long_token"], ref["short_token"]), record
                    output.write(json.dumps(dict(record, label=row["label"], length=row["length"],
                                                 quota=quota, policy=policy, window=window, repeat=repeat,
                                                 peak_bytes=torch.cuda.max_memory_allocated(),
                                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())) + "\n")
            del arena
            torch.cuda.empty_cache()
            print("Finished", row["label"], flush=True)


if __name__ == "__main__":
    main()
