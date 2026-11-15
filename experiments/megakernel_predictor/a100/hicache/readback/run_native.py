"""Verify host-only prefix hits in the stock SGLang scheduler."""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter_ns, sleep

import sglang as sgl


def token(engine, ids, salt):
    return engine.generate(input_ids=ids, cache_salt=salt, sampling_params={
        "temperature": 0., "max_new_tokens": 1, "ignore_eos": True,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", type=int)
    args = parser.parse_args()
    data = json.loads(args.requests.read_text())
    cases = {}
    for row in data["heldout"]:
        if row["resident"] == 0 and args.prefix in (None, row["prefix"]):
            cases.setdefault(row["prefix"], row)
    engine = sgl.Engine(
        model_path=args.model, dtype="bfloat16", tp_size=1, context_length=32768,
        max_running_requests=1, max_total_tokens=32640, chunked_prefill_size=2048,
        attention_backend="flashinfer", enable_hierarchical_cache=True,
        hicache_ratio=4.0, hicache_write_policy="write_through",
        hicache_mem_layout="page_first", hicache_io_backend="kernel",
        page_size=64, enable_cache_report=True, skip_tokenizer_init=True,
        disable_cuda_graph=True, mem_fraction_static=.89, random_seed=71, log_level="info",
    )
    try:
        rng = random.Random(31213)
        fillers = [[rng.randrange(500, 140000) for _ in range(17664)] for _ in range(2)]
        with args.output.open("w", buffering=1) as output:
            for prefix, row in sorted(cases.items()):
                for _ in range(8):
                    if engine.flush_cache().success:
                        break
                    sleep(.5)
                else:
                    raise RuntimeError("Native cache did not become idle before priming")
                print("Prime", prefix, flush=True)
                salt = row["label"]
                token(engine, row["ids"][:prefix], salt)
                sleep(.6)
                for i, filler in enumerate(fillers):
                    token(engine, filler, f"{salt}/fill/{i}")
                sleep(.6)
                start = perf_counter_ns()
                result = token(engine, row["ids"], salt)
                latency = (perf_counter_ns() - start) / 1000
                meta = result["meta_info"]
                tiers = meta.get("cached_tokens_details") or {}
                raw = dict(prefix=prefix, suffix=row["suffix"], label=salt, wall_us=latency,
                           token=result["output_ids"][0], tiers=tiers)
                print(raw, flush=True)
                output.write(json.dumps(raw) + "\n")
                assert tiers.get("host") == prefix and tiers.get("device", 0) == 0, raw
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
