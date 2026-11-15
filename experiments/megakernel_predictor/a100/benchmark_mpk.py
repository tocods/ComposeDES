"""Time executable MPK regions independently, then measure held-out requests."""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch

from mpk_regions import Model, PLANS, compile_runners
from workloads import OUTPUTS


def usec(start):
    return (perf_counter_ns() - start) / 1000


def calibrate(prefill, decode, prompt, repetitions):
    ids = torch.tensor(prompt["tokens"], device="cuda", dtype=torch.int64)
    whole = decode.plan("mpk_full")
    token = prefill.prefill(ids, prefill.plan("mpk_full"))
    pos = len(prompt["tokens"])
    # Build the same fixed KV history once for the independent stage timings.
    for pos in range(pos, pos + (OUTPUTS - 1) // 2):
        decode.setup(pos, torch.tensor([token], device="cuda"))
        decode.run(whole)
        token = decode.meta["output_tokens"][0, 0].item()
    pos += 1
    token = torch.tensor([token], device="cuda")
    result = []
    keys = list(PLANS)
    random.Random(prompt["context"] + 41).shuffle(keys)
    for _ in range(4):
        decode.setup(pos, token)
        decode.run(whole)
        torch.cuda.current_stream().synchronize()

    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for repeat in range(repetitions):
        candidate_order = keys[repeat % len(keys):] + keys[:repeat % len(keys)]
        for candidate in candidate_order:
            plan = decode.plan(candidate)
            start = perf_counter_ns()
            decode.setup(pos, token)
            decode.meta["output_tokens"][0, 0].item()
            result.append([candidate, repeat, "metadata_and_output", usec(start)])
            for stage, region in enumerate(plan):
                # Refresh inbound activation and KV, without the timers.
                decode.setup(pos, token)
                decode.run(plan[:stage])
                torch.cuda.synchronize()
                start = perf_counter_ns()
                events[0].record()
                region()
                events[1].record()
                region.wait()
                wall_us = usec(start)
                events[1].synchronize()
                gpu_us = events[0].elapsed_time(events[1]) * 1000
                result.append([candidate, repeat, f"region_{stage}", gpu_us])
                result.append([candidate, repeat, f"retire_{stage}", max(0, wall_us - gpu_us)])
    return result


def measure(prefill, decode, prompt, repetitions):
    ids = torch.tensor(prompt["tokens"], device="cuda", dtype=torch.int64)
    context = ids.numel()
    prefill_plan = prefill.plan("mpk_full")
    plans = {name: decode.plan(name) for name in PLANS}
    result = []
    for repeat in range(repetitions):
        # Every variant gets the same prefill path and its own decode history.
        order = list(plans)
        random.Random(100 * context + repeat).shuffle(order)
        for candidate in order:
            plan = plans[candidate]
            torch.cuda.synchronize()
            start = perf_counter_ns()
            first_token = prefill.prefill(ids, prefill_plan)
            ttft_us = usec(start)
            token = prefill.meta["output_tokens"][(context - 1) % prefill.n]
            out = [first_token]
            inter_token_us = []
            gpu_us = []
            start_gpu = torch.cuda.Event(enable_timing=True)
            end_gpu = torch.cuda.Event(enable_timing=True)
            # Materialize event handles before the service interval.
            start_gpu.record()
            end_gpu.record()
            end_gpu.synchronize()
            for pos in range(context, context + OUTPUTS - 1):
                start = perf_counter_ns()
                decode.setup(pos, token)
                start_gpu.record()
                decode.run(plan)
                end_gpu.record()
                token = decode.meta["output_tokens"][0]
                out.append(token[0].item())
                inter_token_us.append(usec(start))
                gpu_us.append(start_gpu.elapsed_time(end_gpu) * 1000)
            result.append({
                "candidate": candidate, "repeat": repeat, "context": context,
                "request": prompt["request"], "ttft_us": ttft_us,
                "token_us": inter_token_us, "gpu_us": gpu_us, "output": out,
            })
    assert all(row["output"] == result[0]["output"] for row in result), prompt["request"]
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--mode", choices=["calibration", "heldout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())[args.mode]
    model = Model(args.model)
    prefill, decode = compile_runners(model, args.build)
    # Pass an explicit stream pointer to the generated CUDA module and Torch.
    torch.cuda.synchronize()
    torch.cuda.set_stream(torch.cuda.Stream())
    with args.output.open("w", buffering=1) as out:
        for prompt in prompts:
            if args.mode == "calibration":
                for candidate, repeat, stage, us in calibrate(prefill, decode, prompt, args.repetitions):
                    row = dict(candidate=candidate, repeat=repeat, stage=stage, us=us,
                               context=prompt["context"], request=prompt["request"])
                    print(json.dumps(row), file=out)
            else:
                for row in measure(prefill, decode, prompt, args.repetitions):
                    print(json.dumps(row), file=out)
            print(f"Finished MPK {args.mode} {prompt['request']} length {prompt['context']}", flush=True)


if __name__ == "__main__":
    main()
