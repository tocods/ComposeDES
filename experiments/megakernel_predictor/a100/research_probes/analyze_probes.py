"""Summarize paired CUDA Graph measurements and sparse scheduling bounds."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from random import Random
import sqlite3
from statistics import median


ORDERS = ("io-first", "interleaved", "lookahead-2")


def read_rows(directory, name):
    files = sorted(directory.glob(f"{name}-[AB].jsonl"))
    if len(files) != 2:
        raise ValueError(f"Expected two fresh sessions for {name}, got {files}")
    for file in files:
        for line in file.read_text().splitlines():
            yield json.loads(line)


def round_tree(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {key: round_tree(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [round_tree(value) for value in obj]
    return obj


def summarize(values):
    """Within-session paired differences, and stratified percentile bootstrap."""
    by_session = defaultdict(list)
    for (session, _), value in sorted(values.items()):
        by_session[session].append(value)
    rng = Random(20261115)
    estimates = sorted(
        median([x for group in by_session.values() for x in rng.choices(group, k=len(group))])
        for _ in range(2000)
    )
    return dict(
        median_ms=median(values.values()),
        ci95_ms=[estimates[50], estimates[1949]],
        sessions_ms={key: median(group) for key, group in by_session.items()},
        n=len(values),
    )


def difference(a, b):
    assert a.keys() == b.keys()
    return {key: a[key] - b[key] for key in a}


def strong_sign(summary):
    """Require over 50 us and concordant session medians, as well as a nonzero CI."""
    for sign in (1, -1):
        if sign * summary["median_ms"] > .05 and all(
            sign * value > 0
            for value in [*summary["ci95_ms"], *summary["sessions_ms"].values()]
        ):
            return sign
    return 0


def overlap(nodes, first_stream_tasks):
    first = list(zip(nodes[:2*first_stream_tasks:2], nodes[1:2*first_stream_tasks:2]))
    second = list(zip(nodes[2*first_stream_tasks::2], nodes[2*first_stream_tasks+1::2]))
    return sum(max(0, min(end, other_end) - max(start, other_start))
               for start, end in first for other_start, other_end in second)


def analyze_order(directory):
    groups = defaultdict(dict)
    overlaps = defaultdict(list)
    intervals = defaultdict(list)
    for row in read_rows(directory, "order"):
        label = tuple(row[key] for key in ("pipeline", "shape", "events", "construction", "candidate"))
        rep = row["session"], row["repeat"]
        assert rep not in groups[label]
        groups[label][rep] = row["ms"]
        if label[2]:
            nodes = row["nodes_ms"]
            overlaps[label].append(overlap(nodes, 7))
            intervals[label].append([nodes[i+1]-nodes[i] for i in range(0, len(nodes), 2)])
    assert len(groups) == 162 and all(len(m) == 36 for m in groups.values())
    cases = []
    for pipeline, shape in sorted({key[:2] for key in groups}):
        caps = sorted({k[4] for k in groups if k[:2] == (pipeline, shape)})
        cases.append(dict(
            pipeline=pipeline, shape=shape, variants=[
                dict(events=events, order=order, candidate=cap,
                     median_ms=median(groups[key].values()),
                     overlap_ms=median(overlaps[key]) if events else None,
                     task_ms=[median(values) for values in zip(*intervals[key])] if events else None)
                for events in (False, True) for order in ORDERS for cap in caps
                for key in [(pipeline, shape, events, order, cap)]
            ], comparisons=[]
        ))
        for a, b in combinations(caps, 2):
            delta = dict()
            for events in (False, True):
                for order in ORDERS:
                    prefix = pipeline, shape, events, order
                    delta[events, order] = summarize(difference(groups[prefix + (a,)], groups[prefix + (b,)]))
            cases[-1]["comparisons"].append(dict(
                a=a, b=b,
                delta=[dict(events=events, order=order, **value) for (events, order), value in delta.items()],
                order_reversals=[
                    [o1, o2] for o1, o2 in combinations(ORDERS, 2)
                    if strong_sign(delta[False, o1]) * strong_sign(delta[False, o2]) == -1
                ],
                trace_reversals=[
                    order for order in ORDERS
                    if strong_sign(delta[False, order]) * strong_sign(delta[True, order]) == -1
                ],
            ))
    return cases


def analyze_sparse(directory):
    groups = defaultdict(dict)
    traces = defaultdict(dict)
    for row in read_rows(directory, "sparse"):
        label = tuple(row[key] for key in ("batch", "mode", "plan", "construction"))
        rep = row["session"], row["repeat"]
        assert rep not in groups[label]
        groups[label][rep] = row["ms"]
        if "nodes_ms" in row:
            assert row["misses"] == row["batch"] * 256
            assert row["bytes"] == row["misses"] * 4096
            traces[label][rep] = row["nodes_ms"], row["group_sizes"]
    assert len(groups) == 84 and all(len(m) == 48 for m in groups.values())
    cases = []
    for batch, mode in sorted({key[:2] for key in groups}):
        prefix = batch, mode
        resolve = groups[prefix + ("resolve", "isolated")]
        compute = groups[prefix + ("attention", "isolated")]
        lower = {r: max(compute[r], resolve[r] + compute[r]/batch) for r in resolve}
        base = groups[prefix + ("batch", "interleaved")]
        cases.append(dict(
            batch=batch, mode=mode, miss_mib=batch, variants=[],
            resolve_ms=median(resolve.values()), attention_ms=median(compute.values()),
            ideal_completion_ms=median(lower.values()),
            ideal_saving=summarize(difference(base, lower)),
        ))
        for plan in ("batch", "halves", "quarters", "ready-first"):
            predicted, trace_overlap, last_arrival = {}, [], []
            for rep, (nodes, sizes) in traces[prefix + (plan, "trace")].items():
                completions = nodes[1:2*len(sizes):2]
                last_arrival.append(completions[-1])
                trace_overlap.append(overlap(nodes, len(sizes)))
                finish = 0.0
                for arrival, n in zip(completions, sizes):
                    finish = max(finish, arrival) + compute[rep]*n/batch
                predicted[rep] = finish
            entry = dict(
                plan=plan, arrivals_bound_ms=median(predicted.values()),
                trace_last_arrival_ms=median(last_arrival), trace_overlap_ms=median(trace_overlap),
                clean=[],
            )
            for order in ORDERS[:2]:
                variant = groups[prefix + (plan, order)]
                comparator = groups[prefix + ("batch", order)]
                entry["clean"].append(dict(
                    order=order, median_ms=median(variant.values()),
                    delta=summarize(difference(variant, comparator)),
                ))
            cases[-1]["variants"].append(entry)
    return cases


def analyze_profile(directory):
    cases = defaultdict(list)
    with sqlite3.connect(directory / "order-node.sqlite") as connection:
        ranges = connection.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'probe/%'"
        ).fetchall()
        for start, end, label in ranges:
            _, pipeline, shape, candidate, order, _ = label.split("/")
            activities = connection.execute(
                """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                   JOIN StringIds AS s ON k.shortName = s.id
                   WHERE k.start >= ? AND k.end <= ? ORDER BY k.start""", (start, end),
            ).fetchall()
            copies = [(s,e) for s,e,name in activities if name == "hicache_transfer_per_layer"]
            compute = [(s,e) for s,e,name in activities
                       if name.startswith(("SinglePrefillWithKVCacheKernel", "ampere_bf16"))]
            assert len(copies) == 6 and len(compute) == 7
            tasks = copies + compute
            cases[pipeline, int(shape), candidate, order].append(dict(
                # Positive means the first compute kernel starts before the first readback.
                first_copy_minus_first_compute_ms=(copies[0][0] - compute[0][0])/1e6,
                copy_ms=sum(e-s for s,e in copies)/1e6,
                compute_ms=sum(e-s for s,e in compute)/1e6,
                overlap_ms=overlap([x for interval in tasks for x in interval], len(copies))/1e6,
                span_ms=(max(e for _,e in tasks) - min(s for s,_ in tasks))/1e6,
            ))
    assert len(cases) == 12 and all(len(rows) == 5 for rows in cases.values())
    return [
        dict(pipeline=pipeline, shape=shape, candidate=candidate, order=order, n=len(rows),
             **{key: median(row[key] for row in rows) for key in rows[0]})
        for (pipeline, shape, candidate, order), rows in sorted(cases.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyzers = dict(order=analyze_order, sparse=analyze_sparse, profile=analyze_profile)
    parser.add_argument("experiments", nargs="+", choices=analyzers)
    args = parser.parse_args()
    for name in args.experiments:
        result = round_tree(analyzers[name](args.directory))
        (args.directory / f"{name}-results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
