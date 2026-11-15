"""Freeze prefetch policies without observing end-to-end timings on new shapes."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from causal_analysis import Workflow, intervals, longest_path_certificate, nearest, paired, read, replay


STOCK = "whole-2"


def service(rows):
    profiles = [intervals(row) for row in rows]
    return {k: median(x[k][1] - x[k][0] for x in profiles) for k in profiles[0]}


def model(stages, copies, window, rates):
    ops = list(stages)
    last_op = {(layer, name): index for index, (layer, name) in enumerate(ops)}
    dependencies = []
    for layer, name in ops:
        dependencies.append(layer if name == "whole" else -1)
    prefetch = [last_op[lid - window, "whole"] if window and lid >= window else -1
                for lid in range(len(copies))]
    exec_times = list(stages.values())
    compute_left, copy_left = exec_times[0], copies[0]
    compute_wall = [0.] * len(ops)
    copy_wall = [0.] * len(copies)
    t = 0.
    i = j = 0
    while i < len(ops) or j < len(copies):
        a = i < len(ops) and dependencies[i] < j
        h = j < len(copies) and prefetch[j] < i
        assert a or h
        if a and h:
            phase = "attention" if ops[i][1] == "whole" else ops[i][1]
            ra, rh = rates.get(phase, 1.), rates.get("H2D", 1.)
        else:
            ra = rh = 1.
        step = min(compute_left / ra if a else float("inf"),
                   copy_left / rh if h else float("inf"))
        t += step
        if a:
            compute_wall[i] += step
            compute_left -= step * ra
            if compute_left <= 1e-8:
                i += 1
                if i < len(ops):
                    compute_left = exec_times[i]
        if h:
            copy_wall[j] += step
            copy_left -= step * rh
            if copy_left <= 1e-8:
                j += 1
                if j < len(copies):
                    copy_left = copies[j]

    # Materialize the coupled timing as an ordinary dependency graph.
    wf = Workflow()
    wf.add("ingress", 0.)
    i = j = 0
    while i < len(ops) or j < len(copies):
        advanced = False
        if i < len(ops) and dependencies[i] < j:
            parents = [f"compute/{i - 1}" if i else "ingress"]
            if dependencies[i] >= 0:
                parents.append(f"H2D/{dependencies[i]}")
            wf.add(f"compute/{i}", compute_wall[i], parents)
            i += 1
            advanced = True
        if j < len(copies) and prefetch[j] < i:
            parents = [f"H2D/{j - 1}" if j else "ingress"]
            if prefetch[j] >= 0:
                parents.append(f"compute/{prefetch[j]}")
            wf.add(f"H2D/{j}", copy_wall[j], parents)
            j += 1
            advanced = True
        assert advanced
    wf.add("token", 0., (f"compute/{i - 1}", f"H2D/{j - 1}"))
    graph = wf.serialize()
    ns = replay(graph)
    assert ns == longest_path_certificate(graph)["lower_ns"]
    assert abs(t - ns / 1000) < .2
    return ns / 1000


def freeze(old, trace, isolated, window, request_bytes):
    reference = old["cases"]["31232/512/0"]["rate_factors"]
    # Window A is the only end-to-end calibration for lookup and CTA-only.
    assert {r["session"] for r in window} == {"A"}
    results = paired(window)["32k-host"]
    gains = {key.split(":")[1]: row["median_gain_us"] for key, row in results.items()
             if key != f"{STOCK}:A/A"}
    gains[STOCK] = 0.
    lookup = max(gains, key=gains.get)
    cta_only = max((STOCK, "whole-8"), key=gains.get)
    uncertainty = results[f"{STOCK}:A/A"]["abs_delta_p95_us"]
    margin = uncertainty + 500.

    h2d = {}
    for name in (STOCK, "whole-8"):
        rows = [r["loads"] for r in trace if r["plan"] == name and r["mode"] == "copy"]
        h2d[name] = [median(r[i][3] - r[i][2] for r in rows) for i in range(len(rows[0]))]

    new = defaultdict(list)
    new["32k-host"] = [r for r in trace if r["plan"] == STOCK and r["mode"] == "compute"]
    for trace_row in isolated:
        new[trace_row["scenario"]].append(trace_row)
    plans = {}
    for name, records in new.items():
        nominal = service(records)
        estimates = {}
        for base, durations in h2d.items():
            for w in (None, 1, 2, 4):
                estimates[base if w is None else f"{base}@{w}"] = model(
                    nominal, durations, w, reference[base])
        raw = min(estimates, key=estimates.get)
        gain = estimates[STOCK] - estimates[raw]
        guarded = raw if gain > margin else STOCK
        plans[name] = dict(policies=dict(stock=STOCK, cta_only=cta_only, lookup=lookup,
                                         guarded=guarded, unguarded=raw),
                           estimates_us=estimates, raw_gain_us=gain)
    return dict(cases=plans, lookup_calibration_gain_us=gains, guard_us=margin,
                guard_aa_p95_us=uncertainty, requests_sha256=sha256(request_bytes).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    for name in ("prior", "trace", "isolated", "window-a", "requests", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(json.loads(args.prior.read_text()), list(read([args.trace])),
                    list(read([args.isolated])), list(read([args.window_a])),
                    args.requests.read_bytes())
    result["inputs_sha256"] = {key: sha256(getattr(args, key).read_bytes()).hexdigest()
                              for key in ("prior", "trace", "isolated", "window_a")}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
