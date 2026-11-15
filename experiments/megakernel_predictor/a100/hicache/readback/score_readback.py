"""Score native-load experiments and preserve source/target and token checks."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from plans import CANDIDATES


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summary(samples, repeats):
    labels = defaultdict(list)
    for row in samples:
        labels[row["label"]].append(row["ttft_us"])
    assert len(labels) == 3 and all(len(times) == repeats for times in labels.values())
    assert len({r["read_bytes"] for r in samples}) == 1
    assert all(r["write_bytes"] == 0 for r in samples)
    times = sorted(row["ttft_us"] for row in samples)
    return dict(p50_us=median(times), p95_us=times[ceil(.95 * len(times)) - 1],
                per_prompt_p50_us={name: median(values) for name, values in labels.items()},
                n=len(times), token_matches=sum(x["token"] == x["ref"] for x in samples),
                peak_bytes=max(x["peak_bytes"] for x in samples))


def score(rows, calibration, frozen, repeats):
    grouped = defaultdict(lambda: defaultdict(list))
    prior = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["case"]][row["plan"]].append(row)
    for row in calibration:
        if row["mode"] == "joint":
            prior[row["case"]][row["plan"]].append(row["ttft_us"])
    assert grouped.keys() == frozen["cases"].keys()
    fixed = min(CANDIDATES, key=lambda name: mean(
        median(v[name]) / median(v["whole-2"]) for v in prior.values()))
    results = {}
    errors = {"coupled": [], "isolated": []}
    for key, policies in grouped.items():
        assert policies.keys() == set(CANDIDATES) | {"coupled", "isolated"}
        assert len({r["read_bytes"] for runs in policies.values() for r in runs}) == 1
        aggregate = {name: summary(samples, repeats) for name, samples in policies.items()}
        lowest = min(CANDIDATES, key=lambda name: aggregate[name]["p50_us"])
        profiles = frozen["cases"][key]
        rows_all = [r for v in policies.values() for r in v]
        for mode in errors:
            predictions = profiles[mode]["candidates_us"]
            errors[mode].extend(abs(predictions[p] - aggregate[p]["p50_us"])
                                for p in CANDIDATES)
        results[key] = dict(stock="whole-2", fixed=fixed, oracle=lowest, metrics=aggregate,
                            choices={mode: profiles[mode] for mode in errors},
                            read_bytes=rows_all[0]["read_bytes"])
    return dict(cases=results, trials=len(rows), token_matches=sum(r["token"] == r["ref"] for r in rows),
                model_median_abs_error_us={k: median(x) for k, x in errors.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=8)
    args = p.parse_args()
    frozen = json.loads(args.prediction.read_text())
    assert frozen["calibration_sha256"] == sha256(args.calibration.read_bytes()).hexdigest()
    result = score(load(args.heldout), load(args.calibration), frozen, args.repeats)
    result["prediction_sha256"] = sha256(args.prediction.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
