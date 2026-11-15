"""Summarize fixed-budget held-out recovery runs and empirical quota oracles."""
import argparse
from collections import defaultdict
from hashlib import sha256
from math import ceil
import json
from pathlib import Path
from statistics import mean, median


METRICS = (
    "a_ttft_us", "b_ttft_us", "a_tpot_us", "b_tpot_us", "alloc_block_us",
    "allocator_dependency_us", "a_gpu_us",
)


def percentile(values, probability):
    ordered = sorted(values)
    p = probability * (len(ordered) - 1)
    i = int(p)
    f = p - i
    return ordered[i] + f * (ordered[min(i + 1, len(ordered) - 1)] - ordered[i])


def score(rows, prediction, repetitions, prompts):
    cells = defaultdict(lambda: defaultdict(list))
    trials = defaultdict(list)
    for row in rows:
        cells[(row["length"], row["window"])][str(row["policy"])].append(row)
        trials[row["label"], row["window"], row["repeat"]].append(row)
    expected = {(int(length), window) for length, plan in prediction["choices"].items()
                for window in plan}
    assert cells.keys() == expected
    assert {r["write_bytes"] for r in rows} == {335544320}
    for matched in trials.values():
        assert len(matched) == 8
        assert len({(t["long_token"], t["short_token"]) for t in matched}) == 1
    fixed = str(prediction["fixed_quota_from_calibration"])
    summary, normalized = {}, defaultdict(list)
    for (length, window), cell in sorted(cells.items()):
        assert len(cell) == 8
        sample = {}
        for name, runs in sorted(cell.items()):
            assert len(runs) == prompts * repetitions
            counts = defaultdict(int)
            for row in runs:
                counts[row["label"]] += 1
            assert set(counts.values()) == {repetitions}
            assert len(counts) == prompts
            sample[name] = {
                metric: {"p50": median(values), "p95": percentile(values, .95),
                         "p99": percentile(values, .99),
                         "p99_nearest_rank": sorted(values)[ceil(.99 * len(values)) - 1]}
                for metric in METRICS
                for values in ([r[metric] for r in runs],)
            }
            sample[name]["budget"] = {
                "peak_allocated_bytes": max(r["peak_bytes"] for r in runs),
                "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in runs),
            }
            sample[name]["write_window"] = {
                "p50_us": median(r["a_gpu_us"] - r["write_start_us"] for r in runs),
                "p99_us": percentile([r["a_gpu_us"] - r["write_start_us"] for r in runs], .99),
            }
        quotas = [q for q in sample if q.isdigit()]
        for q in quotas:
            normalized[q].append(sample[q]["b_ttft_us"]["p99"] / sample["0"]["b_ttft_us"]["p99"])
        oracle = min(quotas, key=lambda q: sample[q]["b_ttft_us"]["p99"])
        nearest_oracle = min(quotas, key=lambda q: sample[q]["b_ttft_us"]["p99_nearest_rank"])
        median_oracle = min(quotas, key=lambda q: sample[q]["b_ttft_us"]["p50"])
        foreground_oracle = min(quotas, key=lambda q: sample[q]["a_ttft_us"]["p99"])
        prediction_q = prediction["choices"][str(length)][window]["quota"]
        summary[f"{length}/{window}"] = dict(
            length=length, window=window, samples_per_policy=prompts * repetitions,
            predicted_quota=prediction_q, oracle_p99_quota=int(oracle),
            oracle_p99_nearest_rank_quota=int(nearest_oracle),
            oracle_p50_quota=int(median_oracle), a_ttft_oracle=int(foreground_oracle),
            fixed_from_calibration=int(fixed), policies=sample,
            predictor_regret_p99_us=sample["predictor"]["b_ttft_us"]["p99"]
            - sample[oracle]["b_ttft_us"]["p99"],
            estimated_rank_regret_p50_us=sample[str(prediction_q)]["b_ttft_us"]["p50"]
            - sample[median_oracle]["b_ttft_us"]["p50"],
        )
    best_retrospective = min(normalized, key=lambda q: sum(normalized[q]) / len(normalized[q]))
    return dict(cases=summary, episodes=len(rows), tokens_checked=2 * len(rows),
                best_fixed_retrospective=int(best_retrospective), write_bytes=rows[0]["write_bytes"])


def best_enabled_cap(calibration):
    cells = defaultdict(list)
    for row in calibration:
        if row["window"] in ("wide", "tight"):
            cells[row["length"], row["window"], row["quota"]].append(row["b_ttft_us"])
    scores = defaultdict(list)
    for (length, window, quota), samples in cells.items():
        if quota > 0:
            scores[quota].append(median(samples) / median(cells[length, window, 0]))
    return min(scores, key=lambda q: mean(scores[q]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--prompts-per-length", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.heldout.read_text().splitlines()]
    frozen = json.loads(args.prediction.read_text())
    result = score(rows, frozen, args.repetitions,
                   args.prompts_per_length)
    if args.calibration:
        content = args.calibration.read_bytes()
        assert sha256(content).hexdigest() == frozen["calibration_sha256"]
        result["best_enabled_fixed_from_calibration"] = best_enabled_cap(
            json.loads(line) for line in content.splitlines())
    result["prediction_sha256"] = sha256(args.prediction.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
