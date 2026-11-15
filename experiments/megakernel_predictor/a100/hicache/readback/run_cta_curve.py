"""Native HiCache CTA pairs with fixed tokenizer inputs and verified cache hits."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from time import perf_counter_ns, sleep

from gpu_state import GPU
from run_native import token
from run_native_pair import native_engine, set_cap


STAGES = {
    "curve": {31232: (512, 1024, 1536)},
    "train": {15872: (512, 1024, 1536)},
    "heldout": {15872: (640, 1216), 31232: (768, 1280)},
}


def request_ids(session, prefix, length, repeat):
    history = random.Random(f"cta-history/{session}/{prefix}")
    suffix = random.Random(f"cta-suffix/{session}/{prefix}/{length}/{repeat}")
    lengths = sorted({n for stage in STAGES.values() for sizes in stage.values() for n in sizes})
    branch = 125000 + lengths.index(length) * 64 + repeat + 2
    return (history.choices(range(1000, 125000), k=prefix) + [branch] +
            suffix.choices(range(1000, 125000), k=length - 1))


def run(args):
    selection_digest = None
    cases = {}
    if args.stage == "heldout":
        if args.selection is None:
            raise ValueError("Freeze the policies before starting held-out measurement")
        frozen = args.selection.read_bytes()
        cases = json.loads(frozen)["cases"]
        assert cases.keys() == {
            f"{p}/{s}" for p, sizes in STAGES[args.stage].items() for s in sizes}
        selection_digest = sha256(frozen).hexdigest()
        print("SELECTION_SHA256", selection_digest, flush=True)
    quota_path = args.quota_file.resolve()
    set_cap(quota_path, 0)
    os.environ["COMPOSEDES_HICACHE_H2D_QUOTA_FILE"] = str(quota_path)
    engine = native_engine(args.model, context_length=32832, total_tokens=33280, hicache_ratio=6.)
    try:
        rng = random.Random(f"cta-fillers/{args.stage}/{args.session}")
        fillers = [rng.choices(range(1000, 125000), k=17792) for _ in range(2)]
        with args.output.open("w", buffering=1) as output, GPU() as gpu:
            for prefix, lengths in STAGES[args.stage].items():
                for _ in range(8):
                    if engine.flush_cache().success:
                        break
                    sleep(.5)
                else:
                    raise RuntimeError("HiCache did not become idle between prefixes")
                for arm in (0, 1):
                    token(engine, request_ids(args.session, prefix, lengths[0], 0)[:prefix],
                          f"{prefix}/prime/{arm}")
                    sleep(.45)
                previous = None
                for repeat in range(-2, args.repeat):
                    order = list(lengths)
                    rng.shuffle(order)
                    for length in order:
                        ids = request_ids(args.session, prefix, length, repeat)
                        checksum = sha256(json.dumps(ids).encode()).hexdigest()
                        caps = (0, 8) if (repeat % 2 == 0) ^ (args.session == "B") else (8, 0)
                        records = []
                        for arm, cap in enumerate(caps):
                            set_cap(quota_path, 0)
                            for f, filler in enumerate(fillers):
                                token(engine, filler, f"{prefix}/fill/{f}")
                            sleep(.35)
                            set_cap(quota_path, cap)
                            before = gpu.state
                            print(f"CTA_PAIR_BEGIN {prefix} {length} {repeat} {arm} {cap}", flush=True)
                            start = perf_counter_ns()
                            response = token(engine, ids, f"{prefix}/prime/{arm}")
                            end = perf_counter_ns()
                            print("CTA_PAIR_END", flush=True)
                            tiers = response["meta_info"].get("cached_tokens_details") or {}
                            assert tiers.get("host") == prefix and tiers.get("device", 0) == 0, tiers
                            name = "whole-8" if cap else "whole-2"
                            record = dict(scenario=f"{prefix}/{length}", session=args.session,
                                          prefix=prefix, suffix=length, stage=args.stage,
                                          repeat=repeat, order=arm, pair="whole-2:whole-8",
                                          prior=previous, plan=name, ttft_us=(end-start)/1000,
                                          read_bytes=prefix*163840, kv_write_bytes=None, tiers=tiers,
                                          token=response["output_ids"][0], input_sha256=checksum,
                                          start_ns=start, end_ns=end, gpu_before=before,
                                          gpu_after=gpu.state, selection_sha256=selection_digest)
                            if cases:
                                policies = cases[record["scenario"]]["policies"]
                                record["frozen_policies"] = sorted(k for k, v in policies.items() if v == name)
                            previous = name
                            records.append(record)
                            print(prefix, length, name, repeat, round(record["ttft_us"]/1000, 2),
                                  tiers, flush=True)
                            sleep(.35)
                        assert records[0]["token"] == records[1]["token"], records
                        if repeat >= 0:
                            for record in records:
                                output.write(json.dumps(record) + "\n")
    finally:
        engine.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quota-file", type=Path, required=True)
    p.add_argument("--session", choices=("A", "B"), required=True)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--selection", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
