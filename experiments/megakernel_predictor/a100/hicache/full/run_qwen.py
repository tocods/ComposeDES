"""Full Qwen3-8B host-prefix prefill over ordinary streams or MPK page tasks."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter_ns

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from pages import PageKernel, check, prefill
from selector import select

GROUP = 1024
CONFIGS = ("bulk", "dual", "mpk", "dual_all", "mpk_all")


def micros(start):
    return (perf_counter_ns() - start) / 1000


class Prefill:
    def __init__(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        self.core = self.model.model
        self.n = len(self.core.layers)
        self.compute = torch.cuda.Stream()
        self.io = torch.cuda.Stream()
        assert self.n == 36
        torch.cuda.synchronize()

    @contextmanager
    def request(self, row):
        self.prefix, self.suffix = row["prefix"], row["suffix"]
        self.resident = row.get("resident", 0)
        self.group = row.get("group", GROUP)
        if not 0 <= self.resident < self.prefix or self.group <= 0:
            raise ValueError("Expected a partial host prefix and a positive page group size")
        self.ranges = [(lo, min(lo + self.group, self.prefix))
                       for lo in range(self.resident, self.prefix, self.group)]
        self.token_ids = torch.tensor([row["ids"][self.prefix:]], device="cuda")
        self.positions = torch.arange(self.prefix, self.prefix + self.suffix, device="cuda")[None]
        prefix_ids = torch.tensor([row["ids"][:self.prefix]], device="cuda")
        saved = self.model(input_ids=prefix_ids, use_cache=True, logits_to_keep=1).past_key_values
        self.host = []
        for layer in saved.layers:
            cpu = [torch.empty((self.prefix, 8, 128), dtype=torch.bfloat16, pin_memory=True)
                   for _ in range(2)]
            for dst, src in zip(cpu, (layer.keys, layer.values)):
                dst.copy_(src[0].transpose(0, 1))
            self.host.append(cpu)
        self.device = [[torch.empty((self.prefix + self.suffix, 8, 128),
                                     dtype=torch.bfloat16, device="cuda") for _ in range(2)]
                       for _ in range(self.n)]
        if self.resident:
            for targets, sources in zip(self.device, self.host):
                for dst, src in zip(targets, sources):
                    dst[:self.resident].copy_(src[:self.resident], non_blocking=True)
        self.ready = torch.zeros(self.n, device="cuda", dtype=torch.int32)
        self.flags = tuple(self.ready.unbind())
        self.events = [[torch.cuda.Event(enable_timing=True) for _ in self.ranges]
                       for _ in range(self.n)]
        del saved
        torch.cuda.synchronize()
        with PageKernel(self.suffix, self.prefix, self.resident, self.group) as pages:
            self.pages = pages
            yield
        self.host.clear()
        self.device.clear()

    def copy(self, backend, record, mask, *, coalesce_whole=False):
        self.io.wait_stream(self.compute)
        with torch.cuda.stream(self.io):
            for lid, (targets, sources) in enumerate(zip(self.device, self.host)):
                if backend == "coalesced" or (coalesce_whole and not mask[lid]):
                    for dst, src in zip(targets, sources):
                        dst[self.resident:self.prefix].copy_(src[self.resident:self.prefix],
                                                            non_blocking=True)
                    self.events[lid][-1].record()
                    continue
                for i, (lo, hi) in enumerate(self.ranges):
                    for dst, src in zip(targets, sources):
                        dst[lo:hi].copy_(src[lo:hi], non_blocking=True)
                    if backend == "mpk" and mask[lid]:
                        self.pages.publish(self.flags[lid], i + 1)
                    if record or (backend == "dual" and mask[lid]) or i + 1 == len(self.ranges):
                        self.events[lid][i].record()

    def project(self, lid, layer, x, rope):
        a = layer.self_attn
        norm = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(norm).view(1, self.suffix, 32, 128)).transpose(1, 2)
        k = a.k_norm(a.k_proj(norm).view(1, self.suffix, 8, 128)).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        self.device[lid][0][self.prefix:].copy_(k[0].transpose(0, 1))
        self.device[lid][1][self.prefix:].copy_(a.v_proj(norm).view(self.suffix, 8, 128))
        return q[0].transpose(0, 1).contiguous()

    def post(self, layer, x, attn):
        x = x + layer.self_attn.o_proj(attn.reshape(1, self.suffix, 4096))
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def embedding(self):
        x = self.core.embed_tokens(self.token_ids)
        return x, self.core.rotary_emb(x, self.positions)

    def head(self, x):
        return self.model.lm_head(self.core.norm(x[:, -1:]))

    def run(self, backend, mask, *, wait_for_workers=True, defer_layers=(),
            merge_method="flashinfer", coalesce_whole=False):
        self.ready.zero_()
        self.copy(backend, False, mask, coalesce_whole=coalesce_whole)
        x, rope = self.embedding()
        for lid, layer in enumerate(self.core.layers):
            q = self.project(lid, layer, x, rope)
            k, v = self.device[lid]
            if mask[lid]:
                if lid in defer_layers:
                    self.compute.synchronize()
                attn = self.pages.attention(
                    q, k, v, self.flags[lid], self.events[lid], backend,
                    wait_for_workers=wait_for_workers, merge_method=merge_method)
            else:
                self.compute.wait_event(self.events[lid][-1])
                attn = prefill(q, k, v, causal=True, backend="fa2")
            x = self.post(layer, x, attn)
        return self.head(x)

    def calibrate(self, *, all_profiles=False):
        def time(fn, repetitions=3):
            latencies = []
            for _ in range(repetitions):
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                value = fn()
                b.record()
                b.synchronize()
                latencies.append(1000 * a.elapsed_time(b))
            return median(latencies), value

        data = {"layers": [], "arrival_us": {}, "adaptive_publish": True,
                "prefix": self.prefix, "suffix": self.suffix}
        if all_profiles:
            data.update(resident=self.resident, group=self.group, early_parts=self.pages.early)
        backends = ("dual", "mpk", "bulk") if all_profiles else ("dual", "mpk")
        for backend in backends:
            arrivals = []
            for _ in range(3):
                a = torch.cuda.Event(enable_timing=True)
                a.record()
                self.copy("coalesced" if backend == "bulk" else backend, True, [True] * self.n)
                self.io.synchronize()
                arrivals.append([[a.elapsed_time(e) * 1000 for e in
                                  (events[-1:] if backend == "bulk" else events)]
                                 for events in self.events])
            data["arrival_us"][backend] = [
                [median(x[lid][g] for x in arrivals) for g in range(len(arrivals[0][lid]))]
                for lid in range(self.n)]
        self.run("dual", [False] * self.n).argmax().item()
        self.run("mpk", [True] * self.n).argmax().item()
        if all_profiles:
            self.run("dual", [True] * self.n, merge_method="direct").argmax().item()

        data["embed_us"], (x, rope) = time(self.embedding)
        # H2D is complete here. Device tiles, their merge and normal model
        # operators are timed independently of page arrivals.
        for lid, layer in enumerate(self.core.layers):
            self.flags[lid].fill_(len(self.ranges))
            qkv, q = time(lambda: self.project(lid, layer, x, rope))
            k, v = self.device[lid]
            whole, attn = time(lambda: prefill(q, k, v, causal=True, backend="fa2"))
            parts = [time(lambda: self.pages.part(q, k, v, i))[0] for i in range(self.pages.count)]
            merge, old_output = time(self.pages.merge)
            direct = None
            if all_profiles:
                direct, direct_output = time(lambda: self.pages.merge("direct"))
                torch.testing.assert_close(old_output, direct_output, atol=0.008, rtol=0.02)
            def mpk_ready():
                stream = self.compute.cuda_stream
                p = self.pages
                check(p.library.pages_mpk(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                          p.output.data_ptr(), p.lse.data_ptr(),
                                          self.flags[lid].data_ptr(), stream))
                check(p.library.pages_wait())
            mpk, _ = time(mpk_ready)
            post, x = time(lambda: self.post(layer, x, attn))
            data["layers"].append(dict(qkv_us=qkv, post_us=post, whole_us=whole,
                                       parts_us=parts, merge_us=merge, mpk_us=mpk))
            if all_profiles:
                data["layers"][-1]["direct_merge_us"] = direct
        data["head_us"], _ = time(lambda: self.head(x))
        return data

    def measure_contention(self):
        # H2D writes layers 1–3 while attention reads layer 0. Capture both
        # branches so Python submission cannot delay one side of the pair.
        self.copy("coalesced", False, [False] * self.n)
        self.compute.wait_event(self.events[0][-1])
        x, rope = self.embedding()
        q = self.project(0, self.core.layers[0], x, rope)
        k, v = self.device[0]
        for i in range(self.pages.count):
            self.pages.part(q, k, v, i)
        self.pages.merge("direct")
        torch.cuda.synchronize()

        graphs = {}
        for kind in ("attention", "copy", "both"):
            start, attention_end, copy_end = [
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
            for event in (start, attention_end, copy_end):
                event.record()
            self.compute.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.compute):
                start.record()
                if kind != "attention":
                    self.io.wait_stream(self.compute)
                    with torch.cuda.stream(self.io):
                        for lid in range(1, 4):
                            for lo, hi in self.ranges:
                                for dst, src in zip(self.device[lid], self.host[lid]):
                                    dst[lo:hi].copy_(src[lo:hi], non_blocking=True)
                        copy_end.record()
                if kind != "copy":
                    for i in range(self.pages.count):
                        self.pages.part(q, k, v, i)
                    self.pages.merge("direct")
                    attention_end.record()
                if kind != "attention":
                    self.compute.wait_stream(self.io)
            graphs[kind] = (graph, start, attention_end, copy_end)

        result = {k: [] for k in graphs}
        for repetition in range(12):
            order = list(graphs)
            random.Random(repetition).shuffle(order)
            for kind in order:
                graph, start, attention_end, copy_end = graphs[kind]
                graph.replay()
                self.compute.synchronize()
                result[kind].append({
                    "attention_us": start.elapsed_time(attention_end) * 1000
                    if kind != "copy" else None,
                    "copy_us": start.elapsed_time(copy_end) * 1000
                    if kind != "attention" else None,
                })
        return result

    def measure(self, row, prediction, repetitions):
        expected = prediction["mask"]
        empty = [False] * self.n
        full = [True] * self.n
        paths = {"bulk": ("dual", empty), "dual": ("dual", expected), "mpk": ("mpk", expected),
                 "dual_all": ("dual", full), "mpk_all": ("mpk", full)}
        results = []
        ref = self.run("dual", empty)
        ref_token = ref.argmax().item()
        errors = {"bulk": 0.}
        for candidate, (backend, mask) in paths.items():
            if candidate == "bulk":
                continue
            for k, v in self.device:
                k[self.resident:self.prefix].fill_(float("nan"))
                v[self.resident:self.prefix].fill_(float("nan"))
            logits = self.run(backend, mask)
            assert logits.isfinite().all().item()
            errors[candidate] = (logits.float() - ref.float()).abs().max().item()
            assert errors[candidate] < 1.0, (candidate, errors[candidate])

        for repeat in range(repetitions):
            order = list(paths)
            random.Random(row["label"] + str(repeat)).shuffle(order)
            for candidate in order:
                backend, mask = paths[candidate]
                self.compute.synchronize()
                start = perf_counter_ns()
                selection = 0.
                if candidate in ("mpk", "dual"):
                    ts = perf_counter_ns()
                    mask, estimate, _ = select(prediction["profile"])
                    selection = micros(ts)
                    assert mask == expected
                logits = self.run(backend, mask)
                token = logits.argmax().item()
                ttft = micros(start)
                results.append(dict(candidate=candidate, repeat=repeat, label=row["label"],
                                    prefix=self.prefix, suffix=self.suffix, ttft_us=ttft,
                                    selector_us=selection, layers_split=sum(mask), first_token=token,
                                    reference_token=ref_token, max_logit_error=errors[candidate]))
        return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "calibration", "heldout"], required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    data = json.loads(args.requests.read_text())
    if args.mode == "smoke":
        cohort = data["calibration"][:1]
    else:
        cohort = data[args.mode]
    prediction = json.loads(args.prediction.read_text())["cases"] if args.prediction else {}
    model = Prefill(args.model)
    with torch.cuda.stream(model.compute), args.output.open("w", buffering=1) as out:
        for row in cohort:
            print("Starting", row["label"], flush=True)
            with model.request(row):
                if args.mode in ("calibration", "smoke"):
                    record = model.calibrate()
                    print(json.dumps(record), file=out)
                else:
                    case = prediction[f"{row['prefix']}/{row['suffix']}"]
                    for record in model.measure(row, case, args.repetitions):
                        print(json.dumps(record), file=out)
            print("Finished", row["label"], flush=True)


if __name__ == "__main__":
    main()
