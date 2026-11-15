"""Diagnostic on the SGLang scheduler's natural HiCache eviction path."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter_ns

import sglang as sgl


PAGE = 64
TOKENS = 34048
APPEND = 256


def sampling(count):
    return {"temperature": 0., "max_new_tokens": count, "ignore_eos": True}


async def measure(engine, ids, salt):
    started = perf_counter_ns()
    first = None
    n = 0
    meta = {}
    stream = await engine.async_generate(input_ids=ids, cache_salt=salt,
                                         sampling_params=sampling(3), stream=True)
    async for packet in stream:
        n = len(packet["output_ids"])
        if n and first is None:
            first = perf_counter_ns()
        meta = packet["meta_info"]
    done = perf_counter_ns()
    assert n == 3 and first is not None, (n, meta)
    return dict(ttft_us=(first - started) / 1000, tpot_us=(done - first) / 2000,
                cached_tokens=meta.get("cached_tokens"), tiers=meta.get("cached_tokens_details"))


async def mixed(engine, row, label):
    return await asyncio.gather(measure(engine, row["long"], label + "/a"),
                                measure(engine, row["short"], label + "/b"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quota", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=1)
    args = parser.parse_args()
    os.environ["COMPOSEDES_HICACHE_RELAYOUT_QUOTA"] = str(args.quota)
    data = json.loads(args.requests.read_text())
    engine = sgl.Engine(
        model_path=args.model, dtype="bfloat16", tp_size=1, context_length=32768,
        max_running_requests=4, max_total_tokens=TOKENS, chunked_prefill_size=1024,
        attention_backend="flashinfer", enable_hierarchical_cache=True,
        hicache_ratio=2., hicache_write_policy="write_back", hicache_mem_layout="page_first",
        hicache_io_backend="kernel", page_size=PAGE, enable_cache_report=True,
        skip_tokenizer_init=True, disable_cuda_graph=True, mem_fraction_static=.89,
        random_seed=71, log_level="info",
    )
    try:
        by_length = {}
        for row in data["heldout"]:
            by_length.setdefault(row["length"], []).append(row)
        with args.output.open("w", buffering=1) as output:
            for length, examples in sorted(by_length.items()):
                for row in examples[:args.prompts]:
                    label = row["label"]
                    if not engine.flush_cache().success:
                        raise RuntimeError("Could not clear the native cache")
                    pad_length = 31744 - length
                    engine.generate(input_ids=data["victim"], cache_salt=label + "/victim",
                                    sampling_params=sampling(1))
                    if pad_length:
                        engine.generate(input_ids=row["long"][:pad_length], cache_salt=label + "/pad",
                                        sampling_params=sampling(1))
                    engine.generate(input_ids=row["long"][:-APPEND], cache_salt=label + "/a",
                                    sampling_params=sampling(1))
                    print(f"MIXED_BEGIN {label}", flush=True)
                    a, b = engine.loop.run_until_complete(mixed(engine, row, label))
                    print(f"MIXED_END {label}", flush=True)
                    output.write(json.dumps(dict(label=label, length=length, quota=args.quota,
                                                 a=a, b=b, pool_tokens=TOKENS)) + "\n")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
