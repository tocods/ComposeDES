"""Audit hindsight fixed rules, same-shape counterexamples, and independent GPU traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import mean, median

from analyze_probes import summarize
from evaluate_joint import CAPS, CONFIGS, ORDERS, dedup, group, read


def best_constant(rows, keys, predicate):
    choices = []
    for key in keys:
        regrets = [
            100*(min(v for c,v in row["median"].items() if predicate(c,key))
                 / min(row["median"].values()) - 1)
            for row in rows
        ]
        choices.append(dict(key=key, mean_pct=mean(regrets), worst_pct=max(regrets)))
    return min(choices, key=lambda r: r["mean_pct"])


def matched(data, case, slower, faster):
    deltas = {}
    for session, table in data.items():
        first = table[case]["raw"][slower]
        second = table[case]["raw"][faster]
        assert first.keys() == second.keys()
        deltas.update({(session, i): first[i]-second[i] for i in first})
    return dict(case=case, faster=faster, slower=slower, **summarize(deltas))


def trace_stats(filepath):
    entries = defaultdict(list)
    with sqlite3.connect(filepath) as db:
        ranges = db.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'joint/%'").fetchall()
        for start, end, label in ranges:
            _, family, cap, order, _ = label.split("/")
            params = start, end
            computes = db.execute(
                """SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS s ON k.shortName = s.id
                WHERE k.start >= ? AND k.end <= ? AND s.value LIKE 'SinglePrefill%'
                ORDER BY k.start""", params,
            ).fetchall()
            if cap == "dma":
                transfers = db.execute(
                    """SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE start >= ? AND end <= ? AND copyKind = 1 AND bytes = 4194304
                    ORDER BY start""", params,
                ).fetchall()
                assert len(transfers) == 12
            else:
                transfers = db.execute(
                    """SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                    JOIN StringIds AS s ON k.shortName = s.id
                    WHERE k.start >= ? AND k.end <= ? AND s.value = 'hicache_transfer_per_layer'
                    ORDER BY k.start""", params,
                ).fetchall()
                assert len(transfers) == 6
            assert len(computes) == 7
            # Count each wall-time interval only once, even if multiple SM
            # transfer kernels run together.
            points = sorted({p for interval in computes+transfers for p in interval})
            overlap = concurrent_compute = concurrent_transfer = 0
            for left, right in zip(points, points[1:]):
                nc = sum(s <= left < e for s,e in computes)
                nt = sum(s <= left < e for s,e in transfers)
                duration = right-left
                if nc and nt:
                    overlap += duration
                if nc > 1:
                    concurrent_compute += duration
                if nt > 1:
                    concurrent_transfer += duration
            entries[family,cap,order].append(dict(
                overlap_ms=overlap/1e6,
                span_ms=(points[-1]-points[0])/1e6,
                copy_ms=sum(e-s for s,e in transfers)/1e6,
                compute_ms=sum(e-s for s,e in computes)/1e6,
                concurrent_compute_ms=concurrent_compute/1e6,
                concurrent_transfer_ms=concurrent_transfer/1e6,
            ))
    return [
        dict(family=f, cap=cap, order=o, n=len(rows),
             **{k: median(row[k] for row in rows) for k in rows[0]})
        for (f,cap,o),rows in sorted(entries.items())
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    d = args.directory
    data = {s: read(d/f"joint-{s}.jsonl", 16) for s in ("A", "B")}
    control = {}
    for s, cases in data.items():
        heldout = [row for case,row in cases.items() if group(case) != "train"]
        control[s] = dict(
            unique_replays=sum(len(dedup(row, CONFIGS)) for row in cases.values())*16,
            fixed=best_constant(heldout, CONFIGS, lambda c,k: c == k),
            fixed_order_oracle_cap=best_constant(heldout, ORDERS, lambda c,k: c[1] == k),
            fixed_cap_oracle_order=best_constant(heldout, CAPS, lambda c,k: c[0] == k),
        )
    attention = ("attention", 1536, "packed")
    witness = [
        matched(data, ("chain", *attention), ("dma", "io"), ("8", "io")),
        matched(data, ("gated", *attention), ("8", "io"), ("dma", "random-19")),
        matched(data, ("gated", *attention), ("dma", "io"), ("dma", "random-19")),
        matched(data, ("join", *attention), ("8", "io"), ("dma", "compute")),
    ]
    report = dict(controls=control, paired=witness, trace=trace_stats(d/"joint-node.sqlite"))
    if all((d/f"matched-{s}.jsonl").is_file() for s in ("A", "B")):
        cases = [(f, *attention) for f in ("chain", "gated", "join")]
        matched_data = {s: read(d/f"matched-{s}.jsonl", 16, cases) for s in ("A", "B")}
        assert all(c["compute_work"] == [3, 2, 1, 1, 1, 1, 1]
                   for data in matched_data.values() for c in data.values())
        report["matched_sizes"] = {
            s: dict(
                best_fixed=best_constant(data.values(), CONFIGS, lambda c,k: c == k),
                cases=[
                    dict(case=case, old_ms=data[case]["median"]["8", "io"],
                         best=best, best_ms=data[case]["median"][best])
                    for case in cases
                    for best in [min(data[case]["median"], key=data[case]["median"].get)]
                ],
            )
            for s,data in matched_data.items()
        }
    (d/"audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
