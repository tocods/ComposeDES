"""Compare all candidates on the same held-out Qwen3-8B token sequences."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(rows):
    times = [row["ttft_us"] for row in rows]
    costs = [row.get("selector_us", 0.) for row in rows]
    return {
        "n": len(rows),
        "median_ms": median(times) / 1000,
        "p95_ms": quantiles(times, n=100, method="inclusive")[94] / 1000,
        "selector_median_us": median(costs),
        "selector_p95_us": quantiles(costs, n=100, method="inclusive")[94],
        "layers_split": rows[0].get("layers_split"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qwen, sglang = load(args.qwen), load(args.sglang)
    pred = json.loads(args.prediction.read_text())["cases"]
    assert len(qwen) == 250 and len(sglang) == 50
    assert {r["label"] for r in qwen} == {r["label"] for r in sglang}
    assert all(r["cache_tiers"] == {"device": 0, "host": r["prefix"]} for r in sglang)
    groups = defaultdict(lambda: defaultdict(list))
    by_request = defaultdict(lambda: defaultdict(list))
    references = {r["label"]: r["reference_token"] for r in qwen}
    for row in qwen + sglang:
        shape = f"{row['prefix']}/{row['suffix']}"
        row["token_match"] = row["first_token"] == references[row["label"]]
        groups[shape][row["candidate"]].append(row)
        by_request[row["label"]][row["candidate"]].append(row)
    cases = {}
    for shape, group in sorted(groups.items()):
        assert all(len(trials) == 10 for trials in group.values())
        cases[shape] = {
            "candidates": {name: summarize(rows) for name, rows in sorted(group.items())},
            "split_layers": [i + 1 for i, bit in enumerate(pred[shape]["mask"]) if bit],
            "predicted_ms": {k: v / 1000 for k, v in pred[shape]["scores_us"].items()},
            "token_matches": {name: sum(r["token_match"] for r in rows)
                              for name, rows in sorted(group.items())},
        }
    output = {
        "prediction_sha256": sha256(args.prediction.read_bytes()).hexdigest(),
        "cases": cases,
        "by_request": {label: {k: summarize(v) for k, v in sorted(group.items())}
                       for label, group in sorted(by_request.items())},
        "max_logit_error": max(r["max_logit_error"] for r in qwen),
        "first_token_matches": {
            name: sum(r["token_match"] for r in qwen + sglang if r["candidate"] == name)
            for name in sorted({r["candidate"] for r in qwen + sglang})
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
