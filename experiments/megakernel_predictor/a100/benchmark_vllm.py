"""Measure batch-one, full-decode CUDA Graph using vLLM's synchronous engine."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from workloads import OUTPUTS


def run(engine, prompt, repeat, params, profile=False):
    import torch

    request = f"{prompt['request']}-{repeat}"
    start = perf_counter_ns()
    engine.add_request(request, {"prompt_token_ids": prompt["tokens"]}, params)
    last = start
    token_us = []
    tokens = []
    while engine.has_unfinished_requests():
        if profile:
            torch.cuda.nvtx.range_push("vllm_decode" if tokens else "vllm_prefill")
        try:
            finished = engine.step()
        finally:
            if profile:
                torch.cuda.nvtx.range_pop()
        if not finished:
            continue
        assert len(finished) == 1
        new = list(finished[0].outputs[0].token_ids)
        if len(new) > len(tokens):
            assert len(new) == len(tokens) + 1, (len(new), len(tokens))
            now = perf_counter_ns()
            token_us.append((now - last) / 1000)
            last = now
            tokens = new
    assert len(tokens) == OUTPUTS, (request, len(tokens))
    return dict(candidate="vllm_graph", context=prompt["context"],
                request=prompt["request"], repeat=repeat, ttft_us=token_us[0],
                token_us=token_us[1:], output=tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--mode", choices=["calibration", "heldout", "profile"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    data = json.loads(args.prompts.read_text())
    prompts = data[args.mode] if args.mode != "profile" else data["calibration"][:1]
    llm = LLM(
        model=args.model, dtype="bfloat16", max_model_len=1024,
        max_num_batched_tokens=1024, max_num_seqs=1,
        gpu_memory_utilization=0.78, enable_prefix_caching=False,
        skip_tokenizer_init=True, disable_log_stats=True,
        enforce_eager=False, compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    )
    engine = llm.llm_engine
    params = SamplingParams(
        temperature=0.0, ignore_eos=True, max_tokens=OUTPUTS, detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    warm = data["calibration"][0]
    for repeat in range(3):
        run(engine, warm, -repeat - 1, params)
    if args.mode == "profile":
        torch.cuda.cudart().cudaProfilerStart()

    with args.output.open("w", buffering=1) as out:
        for prompt in prompts:
            reference = None
            for repeat in range(args.repetitions):
                row = run(engine, prompt, repeat, params, args.mode == "profile")
                if reference is None:
                    reference = row["output"]
                assert row["output"] == reference, prompt["request"]
                print(json.dumps(row), file=out)
            print(f"Finished vLLM {args.mode} {prompt['request']} length {prompt['context']}",
                  flush=True)
    if args.mode == "profile":
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
