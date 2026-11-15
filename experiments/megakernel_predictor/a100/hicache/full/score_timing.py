"""Verify and summarize the joint submission policy on the new holdout."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles


CANDIDATES = ("whole", "joint_mpk", "matched_dual", "dual_selected",
              "late_early", "late_q_ready", "late_dual")


def summarize(rows):
    ttft = [r["ttft_us"] for r in rows]
    selector = [r["selector_us"] for r in rows]
    return dict(
        ttft_ms=median(ttft) / 1000,
        p95_ttft_ms=quantiles(ttft, n=100, method="inclusive")[94] / 1000,
        selector_us=median(selector),
        p95_selector_us=quantiles(selector, n=100, method="inclusive")[94],
        matches=sum(r["reference_token"] == r["first_token"] for r in rows),
        n=len(rows),
    )


def score(args):
    requests = json.loads(args.requests.read_text())
    frozen = json.loads(args.prediction.read_text())["cases"]
    cases = {r["label"]: r for r in requests["heldout"]}
    old_hashes = {r["sha256"] for r in requests["calibration"]}
    assert len(cases) == len({r["sha256"] for r in cases.values()})
    assert not old_hashes.intersection(r["sha256"] for r in cases.values())
    assert old_hashes == {p["profile"]["sha256"] for p in frozen.values()}

    groups = defaultdict(lambda: defaultdict(list))
    by_label = defaultdict(lambda: defaultdict(list))
    every = defaultdict(list)
    keys = set()
    for raw in args.data.read_text().splitlines():
        r = json.loads(raw)
        label, candidate = r["label"], r["candidate"]
        key = label, candidate, r["repeat"]
        assert key not in keys
        keys.add(key)
        assert r["sha256"] == cases[label]["sha256"]
        shape = f"{r['prefix']}/{r['suffix']}"
        groups[shape][candidate].append(r)
        by_label[label][candidate].append(r)
        every[candidate].append(r)

    summary = {}
    request_row = {}
    for label, c in sorted(by_label.items()):
        assert set(c) == set(CANDIDATES) and all(len(rows) == args.repetitions for rows in c.values())
        timings = {name: summarize(rows) for name, rows in c.items()}
        request_row[label] = timings
    assert set(request_row) == set(cases)

    for shape, c in sorted(groups.items()):
        g = {name: summarize(rows) for name, rows in c.items()}
        plan = frozen[shape]["plan"]
        matches = [n for n, s in cases.items() if shape == f"{s['prefix']}/{s['suffix']}"]
        diffs = [request_row[n]["late_early"]["ttft_ms"] -
                 request_row[n]["late_q_ready"]["ttft_ms"] for n in matches]
        summary[shape] = dict(
            plans={name: [i + 1 for i, action in enumerate(plan) if action == name]
                   for name in ("early", "q_ready")},
            candidates=g,
            late_q_ready_gain_ms=g["late_early"]["ttft_ms"] - g["late_q_ready"]["ttft_ms"],
            paired_late_gain_ms=diffs,
            predicted_ms={k: v / 1000 for k, v in frozen[shape]["predicted_us"].items()},
        )
    out = dict(shapes=summary, requests=request_row,
               summary={k: summarize(v) for k, v in sorted(every.items())},
               prediction_sha256=sha256(args.prediction.read_bytes()).hexdigest(),
               requests_sha256=sha256(args.requests.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
