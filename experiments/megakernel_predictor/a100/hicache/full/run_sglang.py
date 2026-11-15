"""Benchmark host-only hits in the unmodified SGLang HiCache engine."""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter_ns, sleep

import sglang as sgl


def first_token(engine, ids, salt=None):
    return engine.generate(input_ids=ids, cache_salt=salt, sampling_params={
        "temperature": 0., "max_new_tokens": 1, "ignore_eos": True,
    })


def measure(engine, case, fillers, repetitions):
    ids = case["ids"]
    prefix = case["prefix"]
    # Complete prefix backup before filling practically all L1 token slots.
    print("Priming", case["label"], flush=True)
    for trial in range(-1, repetitions):
        # A trial's full request creates a cached suffix. A fresh namespace
        # makes the next trial reuse precisely the seeded prefix.
        salt = f"{case['label']}/{trial}"
        first_token(engine, ids[:prefix], salt)
        sleep(.18)
        for filler in fillers:
            first_token(engine, filler)
        sleep(.18)
        start = perf_counter_ns()
        out = first_token(engine, ids, salt)
        ttft_us = (perf_counter_ns() - start) / 1000
        meta = out.get("meta_info", {})
        tiers = meta.get("cached_tokens_details") or {}
        record = dict(candidate="sglang_hicache", label=case["label"], repeat=trial,
                      prefix=prefix, suffix=case["suffix"], ttft_us=ttft_us,
                      first_token=out["output_ids"][0], cached_tokens=meta.get("cached_tokens", 0),
                      cache_tiers=tiers)
        print(case["label"], trial, int(ttft_us), tiers, flush=True)
        if tiers.get("device") != 0 or tiers.get("host") != prefix or tiers.get("storage", 0):
            raise RuntimeError(f"expected a full host-only cache hit, observed {record}")
        if trial >= 0:
            yield record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text())
    cohort = [requests["calibration"][0]] if args.smoke else requests["heldout"]
    rng = random.Random(1246)
    fillers = [[rng.randrange(500, 140_000) for _ in range(8192)] for _ in range(2)]
    engine = sgl.Engine(
        model_path=args.model, dtype="bfloat16", tp_size=1, context_length=12288,
        max_running_requests=1, max_total_tokens=12288, chunked_prefill_size=8192,
        attention_backend="flashinfer", enable_hierarchical_cache=True,
        hicache_ratio=4.0, hicache_write_policy="write_through",
        hicache_mem_layout="page_first", hicache_io_backend="kernel",
        page_size=64, enable_cache_report=True, skip_tokenizer_init=True,
        cuda_graph_backend_prefill="breakable", cuda_graph_bs_prefill=[128, 512, 2048, 4096, 8192],
        random_seed=71, log_level="info", log_requests=True, log_requests_level=0,
        mem_fraction_static=0.8,
    )
    try:
        print("Warming native engine with an uncached short request", flush=True)
        first_token(engine, fillers[0][:128])
        print("Short request completed", flush=True)
        with args.output.open("w", buffering=1) as output:
            for case in cohort:
                for record in measure(engine, case, fillers, args.repetitions):
                    print(json.dumps(record), file=output)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
