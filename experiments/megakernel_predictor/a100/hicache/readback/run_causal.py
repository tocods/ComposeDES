"""Correctness, adjacent-pair timing, and attention/load traces."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch
from transformers import AutoTokenizer

from gpu_state import GPU
from plans import CANDIDATES, case_key
from run_readback import Model, TOPICS, Workload, write


CASES = {
    "32k-host": (31232, 512, 0),
    "16k-half": (15872, 512, 7936),
    "16k-host": (15872, 512, 0),
    "32k-half": (31232, 512, 15616),
    "short-suffix": (31232, 64, 0),
    "long-suffix": (31232, 1536, 0),
}
STOCK = "whole-2"
WINDOWS = (1, 2, 4)


def fresh_requests(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    data = {}
    for session in ("A", "B"):
        data[session] = {}
        for name, (prefix, suffix, resident) in CASES.items():
            label = f"hicache-causal-{session}-{name}"
            rng = random.Random(label)
            text = label + ". " + "".join(rng.choices(TOPICS, k=4250))
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) >= prefix + suffix
            data[session][name] = dict(ids=ids[:prefix + suffix], prefix=prefix, suffix=suffix,
                                       resident=resident, label=label)
    return data


def plan(name):
    if name == "A/A":
        name = STOCK
    if name == "last-8":
        return 8, "groups", [False] * 39 + [True], None
    if name == "first-1":
        return 1, "head", [True] + [False] * 39, None
    base, _, window = name.partition("@")
    quota, kind = CANDIDATES[base]
    return quota, kind, None, int(window) if window else None


def window_names():
    return [f"whole-{q}@{w}" for q in (2, 8) for w in WINDOWS] + [STOCK, "whole-8"]


def capture(work, name, **opts):
    if name.endswith("+trace"):
        name = name.removesuffix("+trace")
        opts["timing"] = True
    q, kind, mask, window = plan(name)
    return work.capture(q, kind, mask=mask, window=window, **opts)


def correctness(work, output, names, ids):
    resident_graph = capture(work, STOCK, mode="compute", logits=True)
    work.trial(resident_graph, token=False)
    ref = work.logit_buffer.detach().float().cpu().flatten()
    _, broken = verify(work, resident_graph)
    negative_detected = not bool(broken.isfinite().all()) or (ref - broken).abs().max().item() > 1.
    assert negative_detected
    del resident_graph
    margin = torch.topk(ref, 2).values.diff().abs().item()
    for name in names:
        graph = capture(work, name, logits=True)
        _, candidate = verify(work, graph)
        delta = (ref - candidate).abs()
        record = dict(**ids, plan=name, max_abs=delta.max().item(),
                      rmse=delta.square().mean().sqrt().item(),
                      top1=int(candidate.argmax()), ref_top1=int(ref.argmax()), ref_margin=margin,
                      finite=bool(candidate.isfinite().all()), negative_control=negative_detected)
        write(output, record)
        assert record["finite"] and record["max_abs"] <= 1. and record["rmse"] <= .20, record
        del graph


def verify(work, graph):
    work.poison()
    timing = work.trial(graph, token=False)
    return timing, work.logit_buffer.detach().float().cpu().flatten()


def pairwise(work, output, pairs, repeats, seed, gpu, ids):
    plans = sorted({name for pair in pairs for name in pair})
    graphs = {name: capture(work, name) for name in plans}
    rng = random.Random(seed)
    previous = None
    for name in plans:
        work.trial(graphs[name])
    reverse_first = {p: bool(rng.randrange(2)) for p in pairs}
    for rep in range(repeats):
        order = list(pairs)
        rng.shuffle(order)
        for standard, candidate in order:
            reverse = reverse_first[standard, candidate] ^ bool(rep % 2)
            pair = (candidate, standard) if reverse else (standard, candidate)
            for index, name in enumerate(pair):
                before = gpu.state
                t0 = perf_counter_ns()
                timing = work.trial(graphs[name], include_trace=False)
                write(output, dict(timing, **ids, pair=f"{standard}:{candidate}", repeat=rep,
                                   order=index, prior=previous, plan=name, start_ns=t0,
                                   end_ns=perf_counter_ns(), gpu_before=before, gpu_after=gpu.state))
                previous = name


def traces(work, output, repeats, ids, window=False):
    names = (STOCK, "whole-8", "whole-8@1", "whole-8@2", "whole-8@4") if window else (
        STOCK, "whole-8", "last-8", "whole-64")
    for name in names:
        for mode in ("joint",) if window else ("compute", "copy", "joint"):
            graph = capture(work, name, mode=mode, timing=True)
            work.trial(graph, token=mode != "copy")
            for repeat in range(repeats):
                write(output, dict(work.trial(graph, token=mode != "copy"), **ids,
                                   plan=name, mode=mode, repeat=repeat))
            del graph


def run_case(work, output, mode, repeats, gpu, ids, selected):
    name = ids["scenario"]
    if mode == "correctness":
        extra = ["last-8", "first-1"]
        if name in ("32k-host", "short-suffix", "long-suffix"):
            extra += window_names()
        correctness(work, output, list(dict.fromkeys([*CANDIDATES, *extra])), ids)
    elif mode in ("trace", "window-trace"):
        traces(work, output, repeats, ids, window=mode == "window-trace")
    elif mode == "window":
        pairwise(work, output, [(STOCK, "A/A"), *[(STOCK, p) for p in window_names() if p != STOCK]],
                 repeats, ids["label"], gpu, ids)
    elif mode == "instrumentation":
        variants = ("whole-8", "whole-8@2", "whole-8@4")
        pairs = [(p, p + "+trace") for p in variants]
        pairs.extend([("whole-8", "whole-8@2"), ("whole-8", "whole-8@4"),
                      ("whole-8+trace", "whole-8@2+trace"),
                      ("whole-8+trace", "whole-8@4+trace")])
        pairwise(work, output, pairs, repeats, ids["label"], gpu, ids)
    elif mode == "isolated":
        graph = capture(work, STOCK, mode="compute", timing=True)
        for _ in range(repeats):
            write(output, dict(work.trial(graph), mode="compute", **ids))
    else:
        if mode == "paired":
            pairs = [(STOCK, "A/A"), (STOCK, "whole-8"), ("whole-8", "last-8"),
                     (STOCK, "whole-64")] if name == "32k-host" else [
                         (STOCK, "A/A"), (STOCK, "whole-1"), ("whole-1", "first-1")]
        else:
            strategies = selected["cases"][name]["policies"]
            pairs = [(STOCK, "A/A"), *[(STOCK, p) for p in sorted(set(strategies.values()) - {STOCK})]]
            correctness(work, output, [STOCK, *[b for _, b in pairs]], ids)
        pairwise(work, output, pairs, repeats, ids["label"], gpu, ids)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("generate", "correctness", "paired", "trace",
                                     "window", "window-trace", "instrumentation", "isolated", "shape"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--session", choices=("A", "B"), default="A")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--repeat", type=int, default=12)
    p.add_argument("--selection", type=Path)
    args = p.parse_args()
    if args.mode == "generate":
        args.requests.write_text(json.dumps(fresh_requests(args.model), separators=(",", ":")) + "\n")
        return
    cohorts = json.loads(args.requests.read_text())[args.session]
    cases = {
        "trace": ["32k-host"], "window-trace": ["32k-host"],
        "instrumentation": ["32k-host"], "paired": ["32k-host", "16k-half"], "window": ["32k-host"],
        "correctness": list(CASES)[:4],
        "isolated": ["short-suffix", "long-suffix"], "shape": ["short-suffix", "long-suffix"],
    }
    selected = None
    if args.selection:
        selected = json.loads(args.selection.read_text())
        assert selected["requests_sha256"] == sha256(args.requests.read_bytes()).hexdigest()
    model = Model(args.model)
    with torch.cuda.stream(model.stream), args.output.open("w", buffering=1) as out, GPU() as gpu:
        for name in [args.case] if args.case else cases[args.mode]:
            row = cohorts[name]
            print("Start", args.mode, args.session, name, flush=True)
            work = Workload(model, row)
            ids = dict(session=args.session, scenario=name, label=row["label"], case=case_key(row),
                       read_bytes=work.bytes, kv_write_bytes=0)
            run_case(work, out, args.mode, args.repeat, gpu, ids, selected)
            del work
            torch.cuda.empty_cache()
            print("Finished", args.mode, args.session, name, flush=True)


if __name__ == "__main__":
    main()
