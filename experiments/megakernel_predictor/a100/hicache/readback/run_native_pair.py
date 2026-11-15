"""Within-engine A/B replay of the selected H2D CTA variant."""
import argparse
import json
import os
from pathlib import Path
import random
from time import perf_counter_ns, sleep

import sglang as sgl

from gpu_state import GPU
from run_native import token


def set_cap(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(str(value))
    pending.replace(path)


def native_engine(model, *, context_length=32768, total_tokens=32640, hicache_ratio=4.):
    return sgl.Engine(
        model_path=model, dtype="bfloat16", tp_size=1, context_length=context_length,
        max_running_requests=1, max_total_tokens=total_tokens, chunked_prefill_size=2048,
        attention_backend="flashinfer", enable_hierarchical_cache=True,
        hicache_ratio=hicache_ratio, hicache_write_policy="write_through",
        hicache_mem_layout="page_first", hicache_io_backend="kernel",
        page_size=64, enable_cache_report=True, skip_tokenizer_init=True,
        disable_cuda_graph=True, mem_fraction_static=.89, random_seed=71, log_level="info",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quota-file", required=True, type=Path)
    parser.add_argument("--session", choices=("A", "B"), required=True)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    row = json.loads(args.requests.read_text())[args.session]["32k-host"]
    quota_file = args.quota_file.resolve()
    set_cap(quota_file, 0)
    os.environ["COMPOSEDES_HICACHE_H2D_QUOTA_FILE"] = str(quota_file)
    engine = native_engine(args.model)
    prev = None
    try:
        rng = random.Random(args.session)
        fillers = [[rng.randrange(500, 140000) for _ in range(17664)] for _ in range(2)]
        ids = row["ids"]
        prefix = row["prefix"]
        for arm in range(2):
            token(engine, ids[:prefix], f"prime/{arm}")
            sleep(.35)
        with args.output.open("w", buffering=1) as output, GPU() as gpu:
            for repeat in range(-2, args.repeat):
                current = ids.copy()
                current[prefix] = 130000 + (repeat + 2)
                order = (0, 8) if (repeat % 2 == 0) ^ (args.session == "B") else (8, 0)
                matched = []
                for index, cap in enumerate(order):
                    set_cap(quota_file, 0)
                    for f, filler in enumerate(fillers):
                        token(engine, filler, f"filler/{f}")
                    sleep(.45)
                    set_cap(quota_file, cap)
                    before = gpu.state
                    started = perf_counter_ns()
                    response = token(engine, current, f"prime/{index}")
                    end = perf_counter_ns()
                    meta = response["meta_info"]
                    tiers = meta.get("cached_tokens_details") or {}
                    assert tiers.get("host") == prefix and tiers.get("device", 0) == 0, tiers
                    result = dict(scenario="32k-host-native", session=args.session, repeat=repeat,
                                  plan="whole-8" if cap else "whole-2", pair="whole-2:whole-8",
                                  order=index, prior=prev, ttft_us=(end - started) / 1000,
                                  token=response["output_ids"][0], tiers=tiers,
                                  read_bytes=prefix * 163840, kv_write_bytes=None,
                                  start_ns=started, end_ns=end, gpu_before=before,
                                  gpu_after=gpu.state)
                    prev = result["plan"]
                    matched.append(result)
                    print(result["plan"], repeat, round(result["ttft_us"] / 1000, 3),
                          tiers, flush=True)
                    sleep(.35)
                assert len({r["token"] for r in matched}) == 1, matched
                if repeat >= 0:
                    for record in matched:
                        output.write(json.dumps(record) + "\n")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
