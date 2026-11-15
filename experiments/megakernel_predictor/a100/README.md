# Qwen3-8B MPK partitioning on one A100

The benchmark uses the complete BF16 Qwen3-8B on a single A100-PCIE-40GB.
All weights, prepared weight layouts, activations and KV pages reside on the GPU.
MPK uses Mirage commit `d2189e50063b54ed410cf7b8390b2c784114c7a2`.
The external baseline uses vLLM 0.33.0 with full decode CUDA Graph capture at
batch size one and prefix caching disabled.

## Decode plans

| Candidate | Compiled regions in a single decode step |
| --- | --- |
| `mpk_full` | Embedding through the greedy output head |
| `mpk_attn18_mlp18` | Embedding through the attention output projection and residual add of layer 18, followed by post-attention RMSNorm through the output head |
| `mpk_layer18_19` | Embedding through the MLP residual add of layer 18, followed by layer 19 through the output head |
| `vllm_graph` | vLLM's native decode graph and token return |

The layer numbers in the table are one-based. Each MPK region has its own
generated task graph and native launcher. All MPK decode candidates use
single-iteration `online_notoken` with the same MPK task implementations,
weight layouts and page size. The launch configuration uses 80 resident
workers and 48 scheduler warps for all regions, with the stock 96-way vocabulary
tile layout. The first 96-worker calibration showed a sporadic device-side
scheduler assertion, so its timings are excluded. The caller stream joins the
MPK execution streams and the host waits for both persistent kernels to retire
between launches. Prefill uses a single MPK plan with 8 query tokens per chunk
for all three MPK decode plans. vLLM uses its native prefill.

The compatibility patch also passes MPK's Ampere paged-attention registration
the 10-field parameter layout expected by the C++ task register. The Qwen3 KV
pool supplies a page stride of 512 rows for a page size of 256 tokens.

## Measurement and prediction

- The workload is batch one. Each request uses a frozen list of input IDs and
  returns 17 greedy token IDs with EOS disabled. The calibration split contains
  one request at each of 32, 96, 256 and 512 input tokens. The held-out split
  contains two different prompts at each length. The benchmark runs five
  repetitions per held-out request and candidate.
- MPK calibration records each region with CUDA events, after rebuilding any
  required prefix outside the interval. It measures the host-side retirement
  and token/metadata work separately. The baseline uses the median native vLLM decode step
  on calibration prompts as one indivisible service time.
- ComposeDES creates a workflow per candidate and context length. Its FNCS
  replay and longest-path routines must agree on the predicted completion time.
  The freeze step consumes calibration files only. Held-out TPOT measures the
  intervals between returned tokens. Held-out GPU spans are also recorded for
  MPK. Wall times exclude model startup and tokenization.
- There is no sweep over possible cut locations, batch sizes, or model
  architectures. This experiment tests the specific serial cuts in the table.

## A100 held-out results

| Candidate | Median TPOT (ms) | Change vs MPK full | Median MPK CUDA span (ms) |
| --- | ---: | ---: | ---: |
| vLLM CUDA Graph | 13.033 | -9.22% | — |
| MPK full | 14.357 | — | 14.203 |
| MPK between layers 18 and 19 | 14.520 | +1.13% | 14.358 |
| MPK between attention and MLP in layer 18 | 15.182 | +5.74% | 15.026 |

The frozen prediction ranks all four candidates correctly on 8/8 held-out
prompts and gets 48/48 pairwise orderings. The median absolute relative error
across the 32 per-prompt candidate medians is 0.117%, with a maximum of 0.434%.
The three MPK plans return identical greedy outputs for all prompts and
repetitions. vLLM matches all 17 tokens on 5/8 held-out prompts. The benchmark
uses native greedy outputs for each runtime.

For the 32-token calibration prompt, the attention/MLP split takes
`6.294 + 8.459 = 14.753 ms` in isolated CUDA spans, versus `13.958 ms`
for the monolithic region. Isolated host retirements sum to `0.099 ms` and
`0.041 ms`, respectively. The GPU spans account for most of the split penalty
under this scheduler and launch configuration.

A separate Nsight Systems run logged 16 calls to `cudaGraphLaunch_v10000`
for a vLLM request with 16 decode tokens. The trace was excluded from the
latency sample. The raw JSONL, frozen prediction, held-out score and Nsight
evidence are under `experiments/megakernel_predictor/runs/mpk-a100/regions/`.

## Reproduce

Prepare the A100 work directory with `model/` containing the Qwen3-8B Hugging
Face snapshot, `mirage/` at the pinned commit, and independent environments
`venv/` for Mirage and `vllm-venv/` for vLLM. Apply
`mpk-stream-join.patch` to `mirage/` before compiling. The MPK experiment used
Python 3.12, PyTorch 2.12.1+cu130, Transformers 4.57.1 and CUDA 13.0.

Run this directory's `run_gpu.sh calibration <work-directory>` on the A100.
Transfer `regions/calibration-mpk.jsonl` and
`regions/calibration-vllm.jsonl` to ComposeDES and freeze the predictions.

```sh
python experiments/megakernel_predictor/a100/predict_regions.py freeze \
    --mpk calibration-mpk.jsonl --vllm calibration-vllm.jsonl \
    --output prediction.json
```

Then run `run_gpu.sh heldout <work-directory>` on the A100. Transfer its
held-out JSONL files and score the frozen prediction.

```sh
python experiments/megakernel_predictor/a100/predict_regions.py score \
    --mpk heldout-mpk.jsonl --vllm heldout-vllm.jsonl \
    --prediction prediction.json --output results.json
```

Use `benchmark_vllm.py --mode profile` with Nsight Systems and
`--capture-range=cudaProfilerApi` to identify CUDA Graph replay within decode
ranges. Keep profile timings separate from the performance sample.
