# HiCache page-group prefill pilot

The pilot runs on one A100-PCIE-40GB with PyTorch 2.14.0+cu130 and FlashInfer
0.7.2.post1 FA2. It uses the Qwen3-8B BF16 GQA geometry of 32 query heads,
8 KV heads and head dimension 128. Seeded synthetic Q/K/V tensors provide
controlled input. CUDA Graph replays issue actual pinned-host-to-device
copies into preallocated contiguous KV destinations. The measured interval
runs from replay admission to attention completion. End-to-end serving,
radix tree management and the MPK scheduler are outside this pilot.

## Single-layer schedules

| Name | Transfer and attention |
| --- | --- |
| `bulk_wait` | Bulk H2D of the missing prefix, followed by a single causal attention |
| `group_wait` | Grouped H2D, followed by that same causal attention |
| `split_wait` | Grouped H2D, followed by partial attention and state merge |
| `split_overlap` | The same grouped H2D and partial attention. Resident prefix and causal suffix can run immediately. Each missing prefix group waits on its own H2D event. |

The four plans use the same input, H2D byte count, KV destination and masks.
The suffix applies a causal mask. Cached prefix groups are visible to every
query in the suffix. Before each correctness replay, missing GPU rows are
overwritten with NaN to expose missing dependencies. Outputs are checked with
atol 0.003 and rtol 0.02. BF16 partials allow small reduction-order differences.

ComposeDES schedules a serial H2D queue and a serial compute queue with
explicit completion dependencies. It uses independent measurements of bulk
and per-group copies, partial attention, full attention and merge. The
`transfer` sweep also measures a complete grouped H2D burst without attention.
The predictor scales the isolated copy times to the measured burst total.
The isolated estimator remains available for the ablation. Both estimators
freeze before the held-out run.

## Four-layer schedules

One H2D stream queues layer-major copies ahead of forward execution. One
compute stream consumes each layer's attention output as the next query, with
either zero or one 4096×4096 BF16 projection between layers. The projection is
an independently timed source of intervening compute. The pilot compares all
bulk copies, all grouped copies, splitting the first layer, and splitting
every layer. All plans share the four layers' data and transfer byte count.
These are controlled attention chains with layerwise lookahead.

## Results

- The 8K single-layer sweep contains 24 configurations. Page-group overlap wins
  18. It wins all 12 with 512 or 2048 suffix tokens. Its best case is 1.953 ms
  vs 2.951 ms for bulk wait, a 33.8% decrease. The frozen predictor gets 19/24
  winners and 131/144 pairwise comparisons. Maximum absolute output difference
  across the single-layer sweep is 0.00048828125.
- The 8K four-layer sweep contains 24 configurations. The winners are all-layer
  splitting in 14, first-layer splitting in 4 and bulk wait in 6. The best
  grouped strategy reaches 8.754 ms vs 9.506 ms with bulk in one projected run
  with 2048 suffix tokens, a decrease of 7.9%. At a 1024-token page group in that
  regime, first-layer splitting reaches 8.759 ms vs 9.460 ms with bulk and
  9.627 ms with all-layer splitting. The frozen predictor gets 17/24 winners.
  Maximum absolute output difference across the four-layer sweep is
  0.0009765625.
- The 12K confirmation measures 16 new configurations. Calibrating independent
  grouped H2D improves median absolute percentage error from 1.85% to 0.72% and
  winner selection from 12/16 to 13/16. The three remaining wrong choices have
  1.02–13.31 µs of regret. Both estimators consume independent service times
  for the 12K shapes.

When the H2D arrival and compute chains partially overlap, the split wins
when the hidden service exceeds the extra split, reduction and grouped
transfer overhead. An optimization for MPK can express an attention tile in
terms of both query tiles and external KV page-group readiness, so a worker
can take a ready task while other page groups are in flight. The stock Ampere
path allocates Q shared memory proportional to the query-token count per CTA.
At a count of 512, BF16 Q alone needs 512 KiB for four query heads per KV head.
A new task therefore needs to tile the query length to cover the measured
winning prefill configurations on this GPU.

## Reproduce

Run the calibration on a CUDA device from this directory.

```sh
python page_pipeline.py --mode calibration --output calibration.jsonl
python layer_chain.py --mode bridge --output bridge.jsonl
python page_pipeline.py --mode calibration --sweep transfer --output transfer-calibration.jsonl
```

Use a ComposeDES checkout with its FNCS submodule initialized to freeze the
predictions.

```sh
python predict.py freeze --data calibration.jsonl --output prediction.json
python predict_chain.py freeze --data calibration.jsonl --bridge bridge.jsonl --output chain-prediction.json
python predict.py freeze --data transfer-calibration.jsonl --output transfer-grouped-prediction.json
python predict.py freeze --data transfer-calibration.jsonl --isolated-copy --output transfer-isolated-prediction.json
```

Then run the measured requests on the GPU.

```sh
python page_pipeline.py --mode heldout --repetitions 31 --output heldout.jsonl
python layer_chain.py --mode measure --repetitions 21 --output chain.jsonl
python page_pipeline.py --mode heldout --sweep transfer --repetitions 31 --output transfer-heldout.jsonl
```

Use `predict.py score` and `predict_chain.py score` with the corresponding
`--data`, `--prediction` and `--output` paths. The recorded pilot files are
under `experiments/megakernel_predictor/runs/mpk-a100/hicache/`.
