"""Qwen3-14B host-prefix recovery using the native HiCache load kernel."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import torch
from flashinfer.prefill import single_prefill_with_kv_cache as prefill
from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer
from transformers import AutoTokenizer

from plans import CANDIDATES, PAGE, QUOTAS, case_key, ranges

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "writeback"))
sys.path.insert(0, str(BASE / "full"))
from run_writeback import Model, TOPICS
from merge_pages import merge_part_major


SHAPES = ((15872, 512), (31232, 512))
SPLITS = ("calibration", "heldout")


def dataset(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    data = {}
    for phase, count in (("calibration", 1), ("heldout", 3)):
        rows = []
        for prefix, suffix in SHAPES:
            for i in range(count):
                label = f"readback-{phase}-{prefix}-{suffix}-{i}"
                rng = random.Random(label)
                prompt = f"{label}. " + "".join(rng.choices(TOPICS, k=3900))
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert len(ids) >= prefix + suffix
                ids = ids[:prefix + suffix]
                for resident in (0, prefix // 2):
                    assert resident % PAGE == 0
                    rows.append(dict(label=f"{label}-{resident}", prefix=prefix, suffix=suffix,
                                     resident=resident, ids=ids))
        data[phase] = rows
    return data


def timer():
    return torch.cuda.Event(enable_timing=True, external=True)


def durations(start, events):
    return [start.elapsed_time(x) * 1000 for x in events]


class Workload:
    def __init__(self, model, row):
        self.model = model
        self.prefix, self.resident = row["prefix"], row["resident"]
        self.length = self.prefix + row["suffix"]
        self.missing = self.prefix - self.resident
        self.arena = model.alloc(self.length)
        ids = torch.tensor(row["ids"], device="cuda")
        self.suffix = ids[self.prefix:]
        model.warm_prefix(ids, self.arena, self.prefix)
        model.stream.synchronize()
        shape = (self.missing, model.layer_count, model.kv_heads, model.dim)
        host = [torch.empty(shape, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
        for layer, sources in enumerate(self.arena):
            for h, source in zip(host, sources):
                h[:, layer].copy_(source[self.resident:self.prefix])
        self.host = [tuple(h[:, i] for h in host) for i in range(model.layer_count)]
        self.bytes = sum(h.numel() * h.element_size() for h in host)
        self.dst = torch.arange(self.resident, self.prefix, dtype=torch.int32, device="cuda")
        self.src = torch.arange(self.missing, dtype=torch.int32, device="cuda")
        self.io = torch.cuda.Stream()
        self.shared_pool = torch.cuda.graph_pool_handle()
        self.mailbox = torch.empty_like(self.suffix[:1], device="cpu", pin_memory=True)
        self.logit_buffer = torch.empty((1, 1, model.model.config.vocab_size),
                                        dtype=torch.bfloat16, device="cuda")

    def copy(self, layer, lo, hi, quota):
        span = slice(lo - self.resident, hi - self.resident)
        k, v = self.arena[layer]
        k_cpu, v_cpu = self.host[layer]
        transfer_hicache_one_layer(k, v, self.dst[span], k_cpu, v_cpu, self.src[span],
                                  element_dim=self.model.kv_heads * self.model.dim,
                                  block_quota=quota, page_size=PAGE)

    def forward(self, layer_ranges, ready=None, record=None, *, after_attention=None,
                return_logits=False):
        def attend(layer, q, k, v):
            active = layer_ranges[layer]

            def evaluate(lo, hi, causal, name, index=None, return_lse=True):
                if ready is not None and index is not None:
                    if record:
                        record(layer, "wait:" + name)
                    self.model.stream.wait_event(ready[layer][index])
                if record:
                    record(layer, name)
                value = prefill(q, k[lo:hi], v[lo:hi], causal=causal, backend="fa2",
                                return_lse=return_lse)
                if record:
                    record(layer, name + ".end")
                return value

            if len(active) == 1:
                return evaluate(0, self.length, True, "whole", 0, False)
            states = []
            if self.resident:
                states.append(evaluate(0, self.resident, False, "resident"))
            states.append(evaluate(self.prefix, self.length, True, "suffix"))
            for index, (lo, hi) in enumerate(active):
                states.append(evaluate(lo, hi, False, f"part/{index}", index))
            if record:
                record(layer, "merge")
            result = merge_part_major(torch.stack([x for x, _ in states]),
                                      torch.stack([s for _, s in states]))
            if record:
                record(layer, "merge.end")
            return result

        def mark(layer, phase):
            if record:
                record(layer, "qkv.end" if phase == "attn" else phase)
            if phase == "post" and after_attention:
                after_attention(layer)

        return self.model.forward(self.suffix, self.arena, self.prefix, attend=attend,
                                  mark=mark if record or after_attention else None,
                                  return_logits=return_logits)

    def layout(self, kind, mask):
        return [ranges(self.prefix, self.resident, kind if cut else "whole") for cut in mask]

    def poison(self):
        for k, v in self.arena:
            k[self.resident:self.prefix].fill_(float("nan"))
            v[self.resident:self.prefix].fill_(float("nan"))
        self.model.stream.synchronize()

    def capture(self, quota, kind, *, mask=None, mode="joint", timing=False,
                window=None, logits=False):
        n = self.model.layer_count
        if window is not None and (mode != "joint" or not 1 <= window <= n):
            raise ValueError("Prefetch windows require a joint graph")
        plan = self.layout(kind, [True] * n if mask is None else mask)
        ready = [[torch.cuda.Event() for _ in layer] for layer in plan]
        start, finish = timer(), timer()
        fg, load = [], []

        def record(layer, phase):
            point = timer()
            point.record()
            fg.append((layer, phase, point))

        def submit_load(lid, trace=False):
            for j, (lo, hi) in enumerate(plan[lid]):
                if trace is True:
                    copy_start, copy_end = timer(), timer()
                    copy_start.record()
                self.copy(lid, lo, hi, quota)
                if trace is True:
                    copy_end.record()
                    load.append((lid, j, copy_start, copy_end))
                if trace == "boundaries" and (lid + 1) % 10 == 0 and j + 1 == len(plan[lid]):
                    point = timer()
                    point.record()
                    load.append((lid, j, point, point))
                ready[lid][j].record()

        def boundary(layer, stage):
            if (layer + 1) % 10 == 0 and stage == "post":
                record(layer, stage)

        record_compute = boundary if timing == "boundaries" else record if timing else None

        def advance(layer):
            next_layer = layer + window
            if next_layer < n:
                consumed = torch.cuda.Event()
                consumed.record()
                with torch.cuda.stream(self.io):
                    self.io.wait_event(consumed)
                    submit_load(next_layer, timing)

        # Initialize FlashInfer and HiCache JIT modules before graph capture.
        if mode != "compute":
            for lid in range(n):
                submit_load(lid)
        if mode != "copy":
            self.forward(plan)
        self.model.stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.shared_pool, stream=self.model.stream):
            start.record()
            if mode != "compute":
                self.io.wait_stream(self.model.stream)
                with torch.cuda.stream(self.io):
                    for lid in range(window or n):
                        submit_load(lid, timing)
            if mode != "copy":
                output = self.forward(plan, ready if mode == "joint" else None,
                                      record_compute,
                                      after_attention=advance if window else None,
                                      return_logits=logits)
                if logits:
                    self.logit_buffer.copy_(output)
                else:
                    self.mailbox.copy_(output, non_blocking=True)
            if mode != "compute":
                self.model.stream.wait_stream(self.io)
            finish.record()
        return graph, start, finish, fg, load

    def trial(self, captured, *, token=True, include_trace=True):
        graph, start, finish, fg, load = captured
        graph.replay()
        finish.synchronize()
        row = dict(ttft_us=durations(start, [finish])[0])
        if token:
            row["token"] = int(self.mailbox.item())
        if include_trace and fg:
            row["compute"] = [(i, p, t) for (i, p, _), t in
                              zip(fg, durations(start, [e for _, _, e in fg]))]
        if include_trace and load:
            row["loads"] = [(i, j, start.elapsed_time(a) * 1000, start.elapsed_time(b) * 1000)
                            for i, j, a, b in load]
        return row


def write(file, row):
    file.write(json.dumps(row, separators=(",", ":")) + "\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("generate", "smoke", *SPLITS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prediction", type=Path)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--only-quota", type=int, choices=QUOTAS)
    args = ap.parse_args()
    if args.mode == "generate":
        args.requests.write_text(json.dumps(dataset(args.model), separators=(",", ":")) + "\n")
        return
    all_rows = json.loads(args.requests.read_text())
    rows = all_rows["calibration" if args.mode == "smoke" else args.mode]
    frozen = None
    if args.prediction:
        frozen = json.loads(args.prediction.read_text())
        assert frozen["requests_sha256"] == sha256(args.requests.read_bytes()).hexdigest()
    if args.mode == "smoke":
        rows = rows[:1]
    model = Model(args.model)
    with torch.cuda.stream(model.stream), args.output.open("w", buffering=1) as out:
        for row in rows:
            key = case_key(row)
            print("Start", row["label"], key, flush=True)
            work = Workload(model, row)
            torch.cuda.reset_peak_memory_stats()
            solo = work.capture(2, "whole", mode="compute", timing=args.mode == "calibration")
            reference = work.trial(solo)["token"]
            if args.mode == "calibration":
                for kind in ("whole", "head", "groups"):
                    graph = solo if kind == "whole" else work.capture(2, kind, mode="compute", timing=True)
                    for _ in range(args.repeat):
                        write(out, dict(work.trial(graph), case=key, kind=kind, mode="compute"))
                    del graph
            del solo
            choices = CANDIDATES if args.mode != "smoke" else {
                "whole-2": (2, "whole"), "head-8": (8, "head"),
            }
            plans = {name: (quota, kind, None) for name, (quota, kind) in choices.items()
                     if args.only_quota is None or args.only_quota == quota}
            if frozen:
                for policy in ("coupled", "isolated"):
                    chosen = frozen["cases"][key][policy]
                    plans[policy] = (chosen["quota"], chosen["kind"], chosen["mask"])
            if args.mode == "calibration":
                for name, (quota, kind, _) in plans.items():
                    graph = work.capture(quota, kind, mode="copy", timing=True)
                    for _ in range(args.repeat):
                        write(out, dict(work.trial(graph, token=False), case=key, plan=name, mode="copy"))
                    del graph
            graphs = {name: work.capture(quota, kind, mask=mask,
                                        timing=args.mode == "calibration")
                      for name, (quota, kind, mask) in plans.items()}
            work.poison()
            checked = "whole-2" if args.only_quota is None else f"whole-{args.only_quota}"
            work.trial(graphs[checked])
            for layer in (0, model.layer_count - 1):
                for dst, source in zip(work.arena[layer], work.host[layer]):
                    torch.testing.assert_close(dst[work.resident:work.resident + 2].cpu(),
                                               source[:2], rtol=0, atol=0)
                    torch.testing.assert_close(dst[work.prefix - 2:work.prefix].cpu(),
                                               source[-2:], rtol=0, atol=0)
            for rep in range(args.repeat):
                order = list(graphs)
                random.Random(f'{row["label"]}/{rep}').shuffle(order)
                for name in order:
                    measured = work.trial(graphs[name])
                    write(out, dict(measured, case=key, plan=name, mode="joint", repeat=rep,
                                    label=row["label"], ref=reference, write_bytes=0,
                                    read_bytes=work.bytes, peak_bytes=torch.cuda.max_memory_allocated()))
            del graphs, work
            torch.cuda.empty_cache()
            print("Finished", row["label"], flush=True)


if __name__ == "__main__":
    main()
