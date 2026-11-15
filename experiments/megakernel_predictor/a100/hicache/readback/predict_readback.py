"""Fit co-run service rates then resolve CUDA-stream dependencies with ComposeDES."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

from plans import CANDIDATES, QUOTAS, case_key

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from workflow import Workflow, longest_path_certificate, replay


@dataclass(frozen=True)
class Job:
    name: str
    service: float
    phase: str
    wait_for: int = -1


def intervals(trace):
    marks = {(layer, name): stamp for layer, name, stamp in trace["compute"]}
    layers = 1 + max(layer for layer, _ in marks)
    output = {(-1, "embed"): (0., marks[0, "qkv"])}
    for (layer, name), start in marks.items():
        if name.endswith(".end") or name.startswith("wait:"):
            continue
        if name == "post":
            stop = marks.get((layer + 1, "qkv"), trace["ttft_us"])
        else:
            stop = marks[layer, name + ".end"]
        output[layer, name] = start, stop
    assert (layers - 1, "post") in output
    return output


def phase(name):
    if name in ("whole", "resident", "suffix") or name.startswith("part/"):
        return "attention"
    return name


def overlap(start, stop, spans):
    ends = spans[1]
    begin = bisect_right(ends, start)
    result = 0.
    for left, right in spans[0][begin:]:
        if left >= stop:
            break
        result += max(0., min(stop, right) - max(start, left))
    return result


def fraction(samples):
    # Trim timing outliers before projecting incremental work loss onto the
    # measured overlap. Intervals with almost no overlap carry little signal.
    slopes = [(delay / shared, shared) for delay, shared in samples if shared > 8.]
    if not slopes:
        return 1.
    center = median(x for x, _ in slopes)
    clipped = [(min(center + .35, max(center - .35, x)), w) for x, w in slopes]
    loss = sum(x*w*w for x, w in clipped) / (sum(w*w for _, w in clipped) + 2500.)
    return 1. - min(.8, max(0., loss))


def fit(rows):
    service = {"compute": {}, "copy": {}, "rates": {}, "isolated": {}}
    profiles = defaultdict(list)
    for row in rows:
        group = row.get("kind", row.get("plan"))
        profiles[row["mode"], group].append(row)

    for kind in ("whole", "head", "groups"):
        probes = [intervals(row) for row in profiles["compute", kind]]
        service["compute"][kind] = {
            key: median(max(.001, p[key][1] - p[key][0]) for p in probes) for key in probes[0]
        }
    for name, (_, kind) in CANDIDATES.items():
        probes = [row["loads"] for row in profiles["copy", name]]
        isolated_io = {
            (l, g): median(max(.001, p[i][3] - p[i][2]) for p in probes)
            for i, (l, g, _, _) in enumerate(probes[0])
        }
        service["copy"][name] = isolated_io
        compute_loss = defaultdict(list)
        copy_loss = []
        for joint in profiles["joint", name]:
            measured_compute = intervals(joint)
            activities = sorted(measured_compute.values())
            copy_spans = [(s, t) for _, _, s, t in joint["loads"]]
            copy_ends = [end for _, end in copy_spans]
            comp_ends = [end for _, end in activities]
            for key, (start, stop) in measured_compute.items():
                shared = overlap(start, stop, (copy_spans, copy_ends))
                base = service["compute"][kind][key]
                compute_loss[phase(key[1])].append((stop - start - base, shared))
            for layer, group, start, stop in joint["loads"]:
                shared = overlap(start, stop, (activities, comp_ends))
                copy_loss.append((stop - start - isolated_io[layer, group], shared))
        service["rates"][name] = {key: fraction(vals) for key, vals in compute_loss.items()}
        service["rates"][name]["H2D"] = fraction(copy_loss)
    return service


def jobs(service, q, kind, mask):
    # Same group and copy kernel as measured in the independent profiles;
    # unsplit layers use one load and one original attention call.
    compute = [Job("embed", service["compute"]["whole"][-1, "embed"], "embed")]
    copies = []
    families = []
    for layer, split in enumerate(mask):
        variant = kind if split else "whole"
        label = f"{variant}-{q}"
        sizes = service["copy"][label]
        load_at = len(copies)
        for (lid, group), time in sizes.items():
            if lid == layer:
                copies.append(Job(f"H2D/{lid}/{group}", time, "H2D"))
                families.append(label)
        stages = service["compute"][variant]
        for (lid, name), time in stages.items():
            if lid != layer:
                continue
            dependency = -1
            if name == "whole":
                dependency = load_at
            if name.startswith("part/"):
                dependency = load_at + int(name.split("/")[-1])
            compute.append(Job(f"layer/{lid}/{name}", time, phase(name), dependency))
    return compute, copies, families


def resolve(operations, rates):
    compute, copies, families = operations
    index_a = index_h = 0
    remaining_a, remaining_h = compute[0].service, copies[0].service
    wall_a, wall_h = [0.] * len(compute), [0.] * len(copies)
    now = 0.
    while index_a < len(compute) or index_h < len(copies):
        h_active = index_h < len(copies)
        a_active = index_a < len(compute) and compute[index_a].wait_for < index_h
        if not h_active and not a_active:
            raise ValueError("Unsatisfied attention readiness dependency")
        a_rate = h_rate = 1.
        if h_active and a_active:
            profile = rates.get(families[index_h], {})
            a_rate = profile.get(compute[index_a].phase, 1.)
            h_rate = profile.get("H2D", 1.)
        step = min(remaining_a / a_rate if a_active else float("inf"),
                   remaining_h / h_rate if h_active else float("inf"))
        now += step
        if a_active:
            wall_a[index_a] += step
            remaining_a -= step * a_rate
            if remaining_a <= 1e-8:
                index_a += 1
                if index_a < len(compute):
                    remaining_a = compute[index_a].service
        if h_active:
            wall_h[index_h] += step
            remaining_h -= step * h_rate
            if remaining_h <= 1e-8:
                index_h += 1
                if index_h < len(copies):
                    remaining_h = copies[index_h].service
    return now, wall_a, wall_h


def certificate(operations, result):
    total, wall_a, wall_h = result
    compute, copies, _ = operations
    graph = Workflow()
    graph.add("ingress", 0.)
    parent = "ingress"
    for job, duration in zip(copies, wall_h):
        graph.add(job.name, duration, (parent,), engine="native_hicache")
        parent = job.name
    parent = "ingress"
    for job, duration in zip(compute, wall_a):
        depends = [parent]
        if job.wait_for >= 0:
            depends.append(copies[job.wait_for].name)
        graph.add(job.name, duration, depends, engine="forward")
        parent = job.name
    graph.add("token", 0., (parent, copies[-1].name))
    tasks = graph.serialize()
    actual = replay(tasks)
    assert actual == longest_path_certificate(tasks)["lower_ns"]
    assert abs(actual / 1000 - total) < .35
    return len(tasks)


def select(service, coupled):
    rates = service["rates"] if coupled else {}
    n = 1 + max(l for l, _ in service["compute"]["whole"])
    all_layers = (True,) * n
    no_layers = (False,) * n
    candidates = {}

    def estimate(q, kind, mask):
        return resolve(jobs(service, q, kind, mask), rates)[0]

    gross = {(q, kind): estimate(q, kind, no_layers if kind == "whole" else all_layers)
             for q, kind in CANDIDATES.values()}
    for name, pair in CANDIDATES.items():
        candidates[name] = gross[pair]
    q0 = min(QUOTAS, key=lambda q: gross[q, "whole"])
    best = (gross[q0, "whole"], q0, "whole", no_layers)
    # Evaluate one-layer cuts and simple mixed masks. Carry forward only
    # changes which clear a small margin over timing quantization.
    for q in QUOTAS:
        for kind in ("head", "groups"):
            seeds = (no_layers, all_layers, (True,) + (False,) * (n - 1))
            initial = min(seeds, key=lambda mask: estimate(q, kind, mask))
            score = estimate(q, kind, initial)
            mask = list(initial)
            for _ in range(2):
                changed = False
                for layer in range(n):
                    mask[layer] = not mask[layer]
                    proposed = estimate(q, kind, mask)
                    if proposed + 75. < score:
                        score, changed = proposed, True
                    else:
                        mask[layer] = not mask[layer]
                if not changed:
                    break
            if score + 250. < best[0]:
                best = (score, q, kind, tuple(mask))
    score, q, kind, mask = best
    ops = jobs(service, q, kind, mask)
    nodes = certificate(ops, resolve(ops, rates))
    return dict(quota=q, kind=kind, mask=mask, ttft_us=score, graph_nodes=nodes,
                candidates_us=candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = json.loads(args.requests.read_text())
    data = defaultdict(list)
    for line in args.calibration.read_text().splitlines():
        row = json.loads(line)
        data[row["case"]].append(row)
    assert data.keys() == {case_key(r) for r in cohorts["heldout"]}
    cases = {}
    for key, rows in data.items():
        service = fit(rows)
        cases[key] = dict(coupled=select(service, True), isolated=select(service, False),
                          rate_factors=service["rates"],
                          copy_solo_us={p: sum(vals.values()) for p, vals in service["copy"].items()})
    output = dict(cases=cases, calibration_sha256=sha256(args.calibration.read_bytes()).hexdigest(),
                  requests_sha256=sha256(args.requests.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for name, case in cases.items():
        print(name, [(k, case[k]["quota"], case[k]["kind"], sum(case[k]["mask"]))
                     for k in ("coupled", "isolated")])


if __name__ == "__main__":
    main()
