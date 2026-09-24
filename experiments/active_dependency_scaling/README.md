# Grok vertical × horizontal two-factor ablation

`run_scaling.py` extracts per-rank compute durations and exact acceleration
regions from the Grok-256 workflows and runs each selected rank as an independent
FNCS federate. It performs a 2 × 2 ablation:

- vertical optimization: aggregate 128 local compute events per rank into 16
  exact phase events;
- horizontal optimization: enable Active-dependency coordination across the
  rank-visible federates.

The rank-visible partition is fixed in all four cells. It exposes independent
timelines; the horizontal switch determines whether Active-dependency exploits
them. Cross-rank network edges are intentionally removed, so the experiment
measures a coordination upper bound.

```bash
python3 experiments/active_dependency_scaling/run_scaling.py \
  --ranks 8,32,64,128,256 --events-per-rank 128 --repeats 3
```

This is a coordination benchmark, not a replacement for the full Grok co-simulation.
It preserves Grok compute durations but excludes communication and backend execution.

The completed Grok-256 results are in `grok_rank_local/REPORT.md`,
`grok_rank_local/ablation_2x2.csv`, `grok_rank_local/results.csv`, and
`grok_rank_local/summary.json`.
