# Native HiCache CPU-to-GPU reads and Qwen3-14B attention

## Setup

The matched runner uses a PCIe A100 40 GiB, the complete 40-layer
Qwen3-14B BF16 model, and the SGLang v0.5.23 page-first HiCache read
kernel. The kernel reads mapped host memory on the GPU. Its default
CUDA grid is limited to two CTAs. The variant changes the native
`block_quota` argument. Effective grid size also depends on the number
of rows. The flash-attention path uses FlashInfer FA2.

Each long request appends 512 tokens to a real 15,872- or 31,232-token
prefix. The old KV cache is either entirely in host RAM or half in HBM.
All plans within a case use identical tokens, resident locations,
mapped host page layout, and number of transferred KV bytes. The GPU
destination and graph memory pool are reused. The measured replay
contains CPU-to-GPU KV loads and full inference through first-token
delivery. Pinned host KV initialization takes place before timing.

Three load organizations use the same native kernel:

- Whole layer, with one ready event and the original full attention.
- A 1,024-token leading transfer followed by one transfer for the rest.
- Uniform groups of up to 4,096 tokens.

The grouped paths compute the resident KV and causal suffix before
waiting for each missing group. They merge the BF16 partials with
FP32 softmax weights derived from the base-2 LSE. Since the primitive
allocates its own outputs, stacking those outputs into the merge
input is included in the measured latency.

## Measurement and prediction

CUDA Graphs replay a load stream and a forward stream without CPU
interleaving differences between plans. The graph ends after the
first token reaches a pinned host mailbox. Standalone graphs measure
load and compute service. Additional calibration graphs mark the
actual overlap of the HiCache kernel with QKV projection, attention,
and subsequent compute. The model derives rates from those overlaps
and updates task progress at readiness or completion events.
ComposeDES replay and a longest-path check verify each frozen plan.

The offline selector evaluates five CTA limits and two partition
shapes, including mixed whole and split layers. It uses a small
margin when accepting a layer-level change. A separate frozen selector
uses isolated durations as an ablation. Both predictions are sealed
before the holdout is started. The holdout has three new prompts per
shape and eight randomized repetitions per candidate per prompt.
The calibration and holdout use disjoint token sequences.
The stock comparator within the matched runner is whole-layer
attention, original page-first reads, and the default limit of two
CTAs. A separate native SGLang check verifies real host-only cache
hits. Native timing includes the scheduler and the generation API.

## Results

All 1,632 replays returned the same first token as their fully resident
reference. The table reports median TTFT in milliseconds.

| Prefix + 512 new tokens | HBM resident | Stock 2 CTA | Fixed 1 CTA | ComposeDES with contention | ComposeDES isolated durations | Best swept fixed plan for the cell |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 15,872 | 0% | 231.17 | 243.60 | 230.15 | 230.93 | 223.28 (8 CTA, leading group) |
| 15,872 | 50% | 178.37 | 172.77 | 172.46 | 172.93 | 172.77 (1 CTA, whole) |
| 31,232 | 0% | 452.17 | 466.98 | 440.08 | 443.36 | 442.35 (8 CTA, whole) |
| 31,232 | 50% | 229.12 | 244.39 | 227.65 | 330.46 | 229.12 (2 CTA, whole) |

Fixed 1 CTA is the best one-size-fits-all plan on independent
calibration, selected by mean normalized median TTFT across the four
shapes. It stays unsplit. The contention-aware model chooses stock
for 15,872/0% and 31,232/50%, 1 CTA with a split at layer 1 for
15,872/50%, and 8 CTA with a split at layer 40 for 31,232/0%.
The chosen plans have 24 measurements per cell. P95 improves in
one of four cells relative to stock. Two policies with identical
device plans can still have different collection tails. Median
results are listed without assigning those tail differences to
a kernel change.

For the host-only 31,232-token prefix, isolated H2D is 416.7 ms at
2 CTA and 416.3 ms at 64 CTA. With the full forward active, the copy
stream completes at 417.3 and 418.4 ms in calibration, while TTFT is
421.9 and 570.0 ms. Held-out median TTFT is 452.2 and 586.0 ms. The
copy stream stays close to its observed bandwidth plateau while
the compute stream finishes much later. The plateau is specific to
the page layout, hardware, and copy implementation.

Across the 60 fixed shape/policy combinations, median absolute TTFT
prediction error is 19.18 ms with contention and 51.65 ms with
isolated durations. The model with isolated durations selects
4,096-token parts in all 40 layers and a 24-CTA copy cap for the
half-resident 31,232-token prefix. The measured median is 330.46 ms.
The model with contention keeps the stock configuration and measures
227.65 ms in its separate trial slot. This gives a coarse way to
reject strongly contending schedules. The prediction error remains
larger than the gap between several of the fastest candidates.

The paired results at the same CTA cap indicate that group attention
adds little over quota selection for these shapes. At 15,872/50%, the
layer-1 split at 1 CTA is 0.31 ms faster than 1 CTA with all layers
whole. At 31,232/0%, the layer-40 split at 8 CTA is 2.27 ms faster
than the all-whole 8-CTA path. At 15,872/0%, the all-layer leading
group gains 1.77 ms over all-whole at 8 CTA. Uniform 4,096-token
groups across every layer never win that sweep.

The stock SGLang engine separately reported zero GPU-resident tokens
and exact host cache hits of 15,872 and 31,232 tokens. Its API times
were 256.13 and 457.49 ms in the two checks, and both first tokens
matched the controlled replays. These two scheduler checks establish
cache placement and correctness. The controlled sweep provides the
latency distributions. The second native case was rerun in a fresh
engine after the first process briefly rejected a cache flush while
an asynchronous cache operation was pending.

## Reproduce

Use the original SGLang v0.5.23 Python environment and the complete
Qwen3-14B weights. Run the GPU steps sequentially.

```sh
python run_readback.py --mode generate --model /path/to/Qwen3-14B --requests requests.json
python run_readback.py --mode calibration --model /path/to/Qwen3-14B \
  --requests requests.json --output calibration-combined.jsonl
python predict_readback.py --requests requests.json \
  --calibration calibration-combined.jsonl --output prediction.json
python run_readback.py --mode heldout --model /path/to/Qwen3-14B \
  --requests requests.json --prediction prediction.json --repeat 8 --output heldout.jsonl
python score_readback.py --calibration calibration-combined.jsonl \
  --prediction prediction.json --heldout heldout.jsonl --output results.json
python run_native.py --model /path/to/Qwen3-14B --requests requests.json \
  --output native.jsonl
```

The recorded calibration was performed in two jobs, first the 2/8/24/64
CTA runs, then the 1 CTA run. The latter can be reproduced with
`--only-quota 1`. The combined telemetry, frozen prediction, holdout
trials and native check are under
`experiments/megakernel_predictor/runs/mpk-a100/hicache/readback/`.
