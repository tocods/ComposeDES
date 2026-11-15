# HiCache staged write-back CTA limits on A100

`hicache-cta.patch` applies to the installed SGLang v0.5.23 package root.
It limits the existing grid-stride relayout kernel and preserves the full
grid at a limit of zero. The native wrapper accepts
`COMPOSEDES_HICACHE_RELAYOUT_QUOTA`. The standalone harness can also time
relayout and D2H separately, using the same native kernel and host copy
routine. The limit is a count of launched CTAs. The CUDA driver determines
where those CTAs run.

## Controlled workload

The A100 PCIe 40 GiB run loads all 40 layers of Qwen3-14B BF16. Two
requests share one replay episode. Request A has a 16,384- or 31,744-token
prompt, including the final 256 tokens to prefill. Its resident prefix
is produced in 1,024-token chunks. Request B has a cold 384-token prompt
and must reuse the GPU KV allocation of a completed victim request. The
write stream moves the victim's 2,048 tokens across all 40 layers into
page-first pinned memory. With 8 KV heads and 128 elements per head, the
payload is exactly 320 MiB. Page size is 64 tokens. The permuted page
mapping, write payload, request content and allocations remain fixed
within every matched window and quota comparison.

The replay requests reuse immediately after A's GPU prefill completes.
The wide window releases the already-produced victim data at the start
of A's first layer; the tight window releases it before the final layer.
The intermediate drill chooses a release layer using roughly 40 ms of
remaining compute estimated from the original stage calibration.
The victim is locked through the native D2H completion. The host waits
on that event before passing the same KV buffers to B.

TTFT is measured from the common request epoch to the first token
arriving in a pinned mailbox. GPU first-token production and allocator
waiting are measured separately. This distinction matters when a large
write-back copy delays the scalar token copy. TPOT averages the next
two token intervals per request, with A and B interleaved. The measured
allocator wait is the host time spent waiting on the completion event.
The GPU timestamps also give the exposed dependency directly. The
controlled replay uses the native HiCache transfer implementation and
a fixed KV arena around the full Qwen forward. The native SGLang
scheduler diagnostic observes its own eviction path and is a separate
workload.

## Calibration and prediction

Independent prompts produce isolated QKV, attention, post-attention,
relayout and DMA timings. Concurrent probes give overlap intervals and
fit rates for attention and the other forward stages with each write
stage. For a fixed candidate the event simulation updates rates when
relayout turns into DMA or a compute stage completes. It represents the
write release, pack, DMA, allocation barrier and token egress as
dependencies. The model materializes those durations and replays them in
ComposeDES. Its simulated result agrees with the resulting DAG's longest
path. The decision minimizes B's predicted first-token time, with A as
a tie breaker. The predictor output and request manifest are hashed
before the held-out runs.

The primary comparison uses 4 new prompt pairs per context length and
25 interleaved repetitions per policy and window. The seven swept
launch grids are stock, 2, 8, 24, 64, 108 and 256 CTAs. Stock uses the
unlimited relayout launch. The fixed baseline is chosen on calibration.
The empirical oracle selects from the seven grids by held-out P99
TTFT for B. The report gives both linear interpolation and nearest-rank
P99 because rare stalls affect the interpolated estimate. The oracle and fixed-policy samples are measured
independently of the predictor's additional trial slot.

## A100 results

The held-out sweep completed 3,200 episodes. Each cell and policy has
100 measurements. Both first tokens match the no-write reference in
every episode. Peak allocated memory is 30.758 GiB at 16,384 tokens and
33.102 GiB at 31,744 tokens for every compared policy. The unlimited
stock launch won the calibration comparison across all grids. The best
enabled fixed limit on calibration was 256 CTAs. It also had the best
aggregate interpolated P99 across the held-out cells.

The table reports nearest-rank P99 for B in milliseconds. The oracle
uses the same percentile definition.

| Prompt tokens | Window | Stock | Fixed 256 | ComposeDES | Oracle |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16,384 | Wide | 158.17 | 157.42 | 157.54 (108) | 157.42 (256) |
| 16,384 | Tight | 233.80 | 180.77 | 180.77 (108) | 180.77 (256) |
| 31,744 | Wide | 187.32 | 183.28 | 181.98 (2) | 182.00 (2) |
| 31,744 | Tight | 205.05 | 205.11 | 205.74 (256) | 205.05 (stock) |

The median B TTFT is close among the leading policies. In the wide
window, several stock episodes have unusually long delays after the
write finished, with zero exposed write dependency. The nearest-rank
and interpolated P99 tell different stories for the stock path and
change the 16,384-token wide oracle. For 16,384 tight, stock and the
predicted policy give P95 B TTFT of 179.87 and 179.99 ms. The P99
difference is therefore sensitive to a small number of tail samples.

The intermediate drill adds 384 episodes. The actual window is about
41–43 ms. Two CTAs produce a median allocation wait of 22.36 ms at
16,384 tokens and 21.89 ms at 31,744 tokens. The model predicted 24.43
and 21.06 ms. Eight CTAs or higher and the stock path have no exposed
GPU write dependency in the observed median. In the wide window the
write fits even at two CTAs. In the tight window the independent D2H
stage alone is about 30 ms, exceeding the 3–5 ms available. Larger
relayout grids encounter this copy-engine floor.

At 31,744 tight, going from stock to eight CTAs changes A's interpolated
P99 TTFT from 132.62 to 105.17 ms. At the same time, B's P99 TTFT increases
from 205.05 to 212.87 ms, allocator blocking from 27.83 to 36.85 ms, and
A's P99 TPOT from 119.79 to 137.83 ms. A faster long-request token can
therefore coincide with slower page reuse and later tokens.

The separate native SGLang smoke confirms actual blocking write-back
when its scheduler evicts a dirty node. It moves 160 MiB in each
example. The two stock paths waited 15.50 and 15.65 ms, and the two
eight-CTA paths waited 19.65 and 60.75 ms. There is one measured mixed
episode per shape and quota, with cold-start and scheduler variation.
Use these logs to verify the eviction dependency. The reported P99
values above come from the controlled replay.

## Reproduce

Use the installed SGLang 0.5.23 package and the Qwen3-14B BF16 weight
directory. Keep the stock package intact and set `PYTHONPATH` to a
copy of the SGLang package with the patch applied. Run the GPU phases
serially. The results from this experiment are in
`experiments/megakernel_predictor/runs/mpk-a100/hicache/writeback/`.

```sh
mkdir -p patched
cp -a /path/to/site-packages/sglang patched/
patch -p1 -d patched < hicache-cta.patch
export PYTHONPATH="$PWD/patched"

python run_writeback.py --model /path/to/Qwen3-14B --mode generate \
  --requests requests.json
python run_writeback.py --model /path/to/Qwen3-14B --mode calibration \
  --requests requests.json --output calibration.jsonl --repetitions 5

python predict_writeback.py --requests requests.json \
  --calibration calibration.jsonl --output prediction.json
python run_writeback.py --model /path/to/Qwen3-14B --mode heldout \
  --requests requests.json --prediction prediction.json \
  --output heldout.jsonl --repetitions 25
python score_writeback.py --prediction prediction.json --heldout heldout.jsonl \
  --calibration calibration.jsonl --output results.json

python predict_writeback.py --requests requests.json \
  --calibration calibration.jsonl --output prediction-intermediate.json \
  --target-window-us 40000
python run_writeback.py --model /path/to/Qwen3-14B --mode heldout \
  --requests requests.json --prediction prediction-intermediate.json \
  --output intermediate.jsonl --prompts-per-length 2 --repetitions 12
python score_writeback.py --prediction prediction-intermediate.json \
  --heldout intermediate.jsonl --output results-intermediate.json \
  --prompts-per-length 2 --repetitions 12

patch -p1 -d patched < native-reclaim-trace.patch
python run_native.py --model /path/to/Qwen3-14B --requests requests.json \
  --quota 0 --output native-0.jsonl
```
