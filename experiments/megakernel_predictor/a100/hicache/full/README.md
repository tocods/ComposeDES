# Full Qwen3-8B host-prefix benchmark on A100

This uses the complete Qwen3-8B BF16 checkpoint on one A100-PCIE-40GB. All
36 layers, projections, RMSNorms, rotary position encoding, residuals, MLPs
and the output head execute for the requested suffix. Only the old prefix KV
is prepared in advance. The experimental driver allocates pinned host prefix
buffers and layer-local contiguous GPU destinations. This driver implements
the host-prefix restore required by the benchmark. The native SGLang baseline
runs separately with its own Unified Radix Cache and HiCache.

## MPK page readiness and the matched comparison

A CUDA stream issues the H2D for each pair of K/V pages. For a split layer
it publishes that group's completion through a GPU release store. MPK's
device scheduler reads the group's threshold with acquire semantics and
enqueues the matching query-tile tasks when the threshold is satisfied.
The suffix tasks have threshold zero. Each worker executes FlashInfer's
FA2 device routine with an explicit tile index and KV head. The tile has
64 grouped query rows, or 16 token positions for four query heads per KV head.
Workers and schedulers come from the pinned Mirage persistent runtime.
This integration retains 80 workers and 48 scheduler warps and waits for
the reusable queues to retire before advancing from the MPK attention call.

The same device routine also compiles into ordinary CUDA launches for the
dual-stream comparison. Dual stream waits on the corresponding CUDA event.
Both candidates use the same page-group size (1024 prefix tokens), input,
GPU destination and state merge. They use the same selected layers. Uncut
layers run the same whole FA2 attention and use one layer-completion event.
The final driver publishes group completion only for layers selected for MPK
splitting. An earlier run published on every layer, and its files are retained
under the run directory for the ablation.

## Workload and timing

- There are five independent calibration prompts and ten held-out prompts,
  two at each of five prefix/suffix shapes. Each held-out candidate runs
  five measured repetitions per prompt. All use exactly the same token IDs.
- Native SGLang 0.5.23 uses FlashInfer attention, a 64-token KV page,
  `page_first` + `kernel` HiCache I/O, write-through host caching and
  breakable prefill CUDA Graphs. The native baseline seeds only the intended
  prefix under a distinct cache salt per trial, then evicts device data with two
  unrelated requests. It accepts samples only when the cache report says the
  entire planned prefix was served by the host and zero tokens by the GPU.
- The experimental wall clock covers selection, submission, GPU prefix
  restore, suffix prefill and returning the argmax token from the GPU.
  The SGLang clock covers the synchronous local engine call for one generated
  token, including its request processing and IPC. Startup, prefix seeding,
  eviction and tokenization are outside both timing intervals.

## ComposeDES

Phase calibrations measure the prefix H2D timeline, each layer's QKV path,
whole attention, explicit page tile launches, an all-ready MPK dispatch,
state merge, the rest of the transformer layer and the output head. The
model partitions the independently measured all-ready MPK time across
groups using the matched tile timings. The model assumes the measured phase
costs are fixed during overlap.

For the final driver, the optimizer keeps the H2D queue's finish time and
the compute queue's finish time as two state variables. A path with later
completion on both queues can be pruned because every future dependency is
monotone in both clocks. `predict_full.py` validates the resulting masks and
completion times against ComposeDES workflow replay and the corresponding
longest-path certificate. Prediction inputs are frozen before the measured
requests. The online path reruns the same state search on the CPU once per
request, then submits a single plan for the complete forward. This experiment
uses calibrated service times by shape. It has no online re-estimation from
live DMA progress.

## A100 held-out medians

Times are in milliseconds.

| Prefix / new tokens | Native SGLang HiCache | Matched dual stream | MPK selected | MPK layers split |
| --- | ---: | ---: | ---: | --- |
| 4096 / 128 | 60.603 | 52.481 | 52.485 | none |
| 4096 / 512 | 73.553 | 69.274 | 68.963 | 1 |
| 8192 / 128 | 111.770 | 102.056 | 102.118 | none |
| 8192 / 512 | 116.606 | 102.991 | 111.051 | 34, 35, 36 |
| 8192 / 2048 | 243.876 | 267.567 | 268.504 | none |

The native baseline records an exact host-only hit in all 50 timed samples.
All six paths, including the forced-split controls, match the reference first
token in every sample. The MPK path measures 205 microseconds median and
431 microseconds p95 for online selection. Under the 8192/512 shape, the
predicted selected MPK plan is within 0.038 ms of the predicted no-split
path. The experiment observes an 8.060 ms loss relative to no-split. The
fixed-service event model misses this runtime cost. The native stack's TTFT
also includes different cache layout, transfer implementation and engine
overheads. The matched dual-stream comparison isolates the experimental
attention scheduler and the readiness signaling it requires.

## Launch-timing ablation

For the 8192/512 case the frozen plan selects layers 34–36. With per-layer
CUDA events enabled, the MPK path completes layer 33 and the final H2D later
than the matched dual path. A publication-only control finishes the last H2D
at 103.424 ms versus 103.319 ms for bulk wait. Removing the CPU-side MPK
worker join while retaining the GPU stream join leaves the large gap in place.
Delaying the first MPK launch until layer 34's query is ready brings the final
H2D completion from 107.732 ms to 103.494 ms in the same traced run.

A separate check disabled per-layer instrumentation. It used the two held-out
prompts and 15 repeats per candidate and prompt. The fixed plan skips online
search in this ablation.

| Candidate | Time to first token (ms) |
| --- | ---: |
| Whole attention | 105.208 |
| Matched dual stream | 104.960 |
| MPK with early submission | 110.742 |
| MPK submitted after the first selected query is ready | 105.625 |

These results show sensitivity to submission timing on this workload.
They leave CUDA's internal scheduling mechanism undetermined. The deferred
variant remains an experimental path in the trace harness. The main
benchmark keeps its original submission policy.

```sh
python trace_full.py --model /path/to/Qwen3-8B --requests requests.json \
  --prediction prediction.json --ablation defer --timer-only \
  --repetitions 15 --output defer-no-trace.jsonl
```

## Joint cut and submission selection

`run_timing.py` adds a per-layer choice between whole attention, early MPK
submission and MPK submission after the GPU makes its Q ready. A deferred
layer synchronizes the compute stream immediately after QKV. Prefix copying
continues on its own stream. The stage still uses the same device routine,
readiness counters and page-group size.

Seven new calibration prompts cover 4K, 8K and 12K prefixes. In addition to
the independent stage calibration, each prompt supplies paired one-cut
probes at layers 1, 9, 18, 27, 34 and 36. The probes measure Q readiness and
completion of the H2D stream for early and Q-ready submission. The replay
models the positive paired median offsets as effective delays on the compute
and H2D timelines and interpolates between measured layers. This surcharge
applies to an early **first** MPK submission. Q-ready submission adds a
separately measured event/synchronization gap. Fixed service times and the
first-submission approximation limit what this model can predict.

The search prunes dominated states separately before and after the first MPK
launch. The ComposeDES event replay and longest-path calculation verify the
result for each candidate under this model. The CPU reruns the search once
per timed request. The matched dual-stream control reruns the same MPK
selection and consumes the same cut mask. The independent dual selector
chooses its own cut mask. Both use the same CUDA tile and group settings.

Twenty-one new held-out prompts run five repetitions per candidate. All
735 measured trials match the reference first token. Times below are in
milliseconds. The last column is the early-minus-Q-ready difference with
a fixed split at layers 34–36. Only layer 34 defers in the Q-ready control.

| Prefix / new tokens | Joint MPK | Matched dual | Independent dual | Fixed-plan delay benefit |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 512 | 69.417 | 69.412 | 69.060 | 1.588 |
| 8192 / 128 | 103.424 | 103.342 | 103.867 | 4.553 |
| 8192 / 512 | 104.733 | 104.706 | 104.523 | 7.543 |
| 8192 / 1024 | 146.675 | 146.142 | 145.509 | 3.313 |
| 8192 / 2048 | 274.068 | 274.925 | 275.053 | 0.142 |
| 12288 / 512 | 154.413 | 154.118 | 155.056 | 8.643 |
| 12288 / 1024 | 170.792 | 169.580 | 169.550 | 4.606 |

The paired fixed-plan improvement is positive for 20/21 requests. Joint
selection takes 689 µs median and 1,239 µs p95. Independent dual selection
takes 218 µs median and 549 µs p95. The joint solver chooses whole attention
in the 8192/512 and 8192/2048 cases, so its results for those shapes include
the selector and whole attention. The original SGLang baseline belongs to
the earlier held-out experiment.

```sh
python timing_workloads.py --model /path/to/Qwen3-8B --output timing-requests.json
python run_timing.py --model /path/to/Qwen3-8B --requests timing-requests.json \
  --mode calibration --output timing-calibration.jsonl
python predict_timing.py --calibration timing-calibration.jsonl --output timing-prediction.json
python run_timing.py --model /path/to/Qwen3-8B --requests timing-requests.json \
  --mode heldout --prediction timing-prediction.json --output timing-heldout.jsonl
python score_timing.py --data timing-heldout.jsonl --prediction timing-prediction.json \
  --requests timing-requests.json --output timing-results.json
```

The joint timing artifacts are under the run directory's `timing/`
subdirectory.

## Mixed residency and page-group size

A100 PCIe 40 GB, Qwen3-8B BF16, all 36 layers, batch size one. The new
runner supports a GPU-resident leading prefix followed by a host-resident
prefix suffix. GPU-resident KV is staged before each request is timed.
Every policy transfers the same host KV bytes within a case. The whole
path uses one contiguous K copy and one contiguous V copy per layer.
Unsplit layers in a hybrid plan use the same whole-layer copies. The
grouped whole-attention control retains paged copies and waits for the
entire layer. Both split backends use the same FA2 tile and direct merge.
The MPK measurements defer the split-layer submission until Q is ready.
The TTFT here comes from the experimental full-model runner. The earlier
native SGLang measurements cover a separate host-only workload.

The direct merge reads FA2 states in their part-major layout, combines
base-2 LSE weights in FP32, and writes the BF16 attention output directly.
With nine states and 512 appended tokens, the avoided transposed vector
has 36 MiB. A standalone A100 CUDA Graph probe measured 184 µs for the
transpose plus FlashInfer merge and 40 µs for the direct kernel. The
full-model runner also measures both merges at the same partition.

The first sweep uses 8K+128, 8K+512, and 12K+512 token shapes, 0% and
50% GPU residency, and 512-, 1024-, and 2048-token groups. It calibrates
one prompt per shape and freezes the plans before measuring two distinct
held-out prompts per shape, with four repetitions for each candidate.
At 50% residency and 2048-token groups, whole-layer TTFT versus
all-layer dual-stream TTFT was 80.37 versus 79.00 ms for 8K+512 and
91.90 versus 90.71 ms for 12K+512. The smaller groups lost at these
shapes. The 36-layer solver can accumulate a large Pareto set and erase
the simulated overlap gain before the GPU runs.

### Bounded pruning and fresh validation

For fixed service times, each suffix schedule is monotone in the H2D
and compute completion clocks. A state with an earlier H2D clock and a
compute clock up to ε later can replay any suffix of the removed state
with at most ε added latency. This follows from the max-plus recurrence:
moving both input clocks later by ε delays any downstream time by at
most ε. Pruning with ε on the first 35 layers and exact dominance on the
last bounds prediction regret by 35ε for the 36-layer problem.

The follow-up fixes the group size at 2048 tokens, freezes ε=12 µs from
the original calibration, and uses three new prompts per shape with
five repetitions. Two earlier cases serve separately as diagnostics.
Values are medians of per-prompt medians in ms, including controller
time where relevant.

| Prefix + suffix | GPU residency | Whole | All layers, dual | Cached exact mask, dual | Exact online, dual | Bounded online, dual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K+128 | 0% | 101.785 | 103.131 | 101.813 | 101.854 | 102.090 |
| 8K+128 | 50% | 52.058 | 52.570 | 51.997 | 52.280 | 52.214 |
| 8K+512 | 0% | 102.681 | 103.740 | 102.556 | 102.807 | 102.808 |
| 8K+512 | 50% | 79.115 | 78.304 | 79.639 | 80.230 | 79.465 |
| 12K+512 | 0% | 150.144 | 151.518 | 149.559 | 149.808 | 149.783 |
| 12K+512 | 50% | 91.596 | 90.006 | 90.860 | 95.455 | 92.187 |

The online dual selector at 12K+512 with half residency dropped from
6.176 to 1.170 ms. Its modeled regret was 24.6 µs against a 420 µs
upper bound. Whole attention still beat the bounded online solver.
The same full model and group size changed winners as residency
changed. The CUDA-event dual path with all layers split won in both
50%-resident 512-token suffix cases on the new prompts. At 12K+512 with
half residency, using the legacy transpose and merge increased the
all-layer dual path from 90.006 to 92.683 ms.

All 900 measurements on the fresh prompts matched the whole-attention
first token. In the first sweep, 68 of 1,152 measurements differed,
all on one underlying 12K+512 prompt across placement and group sizes.
Replaying its 2048-token cases showed that both candidate tokens were
tied in the reference BF16 logits. The diagnostic includes legacy-merge
flips. These tests measure numerical equivalence at the checked
tolerances and first-token agreement, with ordinary BF16 rounding.

### Copy and attention interference

The contention probe captures the H2D and partial-attention streams in
a CUDA Graph, reads distinct KV buffers, and measures both engines from
a common device event. Across 18 cells, concurrent attention took
4.3% to 47.3% longer and the absolute H2D change stayed below 0.24%.
The current overlap correction interpolates partial-attention service
between the isolated and paired timings for the period with H2D
outstanding. It leaves whole attention, model projections, MLP, and
the H2D queue unchanged. ComposeDES replays the resulting job durations
and verifies the longest path. For the dual selector the median error
in the predicted TTFT gain dropped from 420 to 336 µs. The measured
winner agreed with the model in 8/18 cells before and after the
correction across four dual path choices. Full layer recovery with
different copy forms still calls for modeling those costs.

The local run directory is
`experiments/megakernel_predictor/runs/mpk-a100/hicache/full/granularity/`.
Use the separate CUDA Graph `contention.jsonl` for the interference
measurements. The old `contention` field in the original calibration file
includes CPU launch skew and served neither predictor.

```sh
python run_granularity.py --model /path/to/Qwen3-8B --mode generate --requests g-requests.json
python run_granularity.py --model /path/to/Qwen3-8B --mode calibration \
  --requests g-requests.json --output g-calibration.jsonl
python predict_granularity.py --calibration g-calibration.jsonl --requests g-requests.json \
  --output g-prediction.json
python run_granularity.py --model /path/to/Qwen3-8B --mode heldout \
  --requests g-requests.json --prediction g-prediction.json --output g-heldout.jsonl
python run_granularity.py --model /path/to/Qwen3-8B --mode contention \
  --requests g-requests.json --output g-contention.jsonl
python score_granularity.py --prediction g-prediction.json --heldout g-heldout.jsonl \
  --contention g-contention.jsonl --output g-results.json
python score_overlap.py --prediction g-prediction.json --results g-results.json \
  --contention g-contention.jsonl --output g-overlap.json
python run_granularity.py --model /path/to/Qwen3-8B --mode generate --validation \
  --requests validation-requests.json
python predict_trim.py --original-prediction g-prediction.json --requests validation-requests.json \
  --output validation-prediction.json
python run_granularity.py --model /path/to/Qwen3-8B --mode heldout --validation \
  --requests validation-requests.json --prediction validation-prediction.json \
  --repetitions 5 --output validation.jsonl
python score_trim.py --prediction validation-prediction.json --validation validation.jsonl \
  --output validation-results.json
```

## Reproduce

Use Mirage revision `d2189e50063b54ed410cf7b8390b2c784114c7a2`. Apply
the A100 experiment's `mpk-stream-join.patch`, then this directory's
`mpk-readiness.patch`. Build on SM80 with CUDA 13.0 and the FlashInfer
0.7.2.post1 installed include directory (including its bundled CCCL). The
experimental driver used PyTorch 2.14.0+cu130 and Transformers 5.20.2.
The separate SGLang environment used PyTorch 2.14.1+cu130.

```sh
bash build_pages.sh /path/to/mirage /path/to/flashinfer/data/include
python pages.py
python requests_data.py --model /path/to/Qwen3-8B --output requests.json
python run_qwen.py --model /path/to/Qwen3-8B --requests requests.json \
  --mode calibration --output calibration.jsonl
```

Freeze the predictor using a ComposeDES checkout with its FNCS submodule.
Then run the experimental and original SGLang GPU paths sequentially.

```sh
python predict_full.py --calibration calibration.jsonl --output prediction.json
python run_qwen.py --model /path/to/Qwen3-8B --requests requests.json \
  --mode heldout --prediction prediction.json --output heldout-qwen.jsonl
python run_sglang.py --model /path/to/Qwen3-8B --requests requests.json \
  --output heldout-sglang.jsonl
python score_full.py --qwen heldout-qwen.jsonl --sglang heldout-sglang.jsonl \
  --prediction prediction.json --output results.json
```

The executed measurements, calibrations, hashes, predictions and summaries
are in `experiments/megakernel_predictor/runs/mpk-a100/hicache/full/`.
