"""Measure joint configuration selection on independent graph replays."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
from random import Random
from statistics import mean, median

import numpy as np

from admission_model import Candidate, PARAMETERS, dag_only, replay
from joint_cases import CAPS, COMPUTE, DEPENDENCIES, ORDERS, TOTAL, TRAIN_ORDERS, downstream, jobs


CONFIGS = tuple((cap, order) for cap in CAPS for order in ORDERS)
BUDGETS = (0, 1, 2, 4, 8, 16, "all")
PILOT_REPEATS = 24


def read(path, repeat, expected=None):
    cases, raw = {}, defaultdict(lambda: defaultdict(dict))
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            case = tuple(row[n] for n in ("family", "kernel", "shape", "layout"))
            if row["type"] == "metadata":
                assert case not in cases
                row["orders"] = {name: tuple(order) for name, order in row["orders"].items()}
                cases[case] = row
            else:
                key = row["cap"], row["order"]
                assert row["repeat"] not in raw[case][key]
                raw[case][key][row["repeat"]] = row["ms"]
    assert set(cases) == set(raw) == (set(jobs()) if expected is None else set(expected))
    for case in cases:
        assert set(raw[case]) == set(CONFIGS)
        assert all(len(times) == repeat for times in raw[case].values())
        cases[case]["raw"] = raw[case]
        cases[case]["median"] = {c: median(times.values()) for c, times in raw[case].items()}
    return cases


def group(case):
    family, _, shape, _ = case
    shape_seen = shape not in (512, 3072)
    if family == "chain":
        return "train" if shape_seen else "new-shape"
    if family == "gated":
        return "new-structure" if shape_seen else "new-structure-and-shape"
    return "new-structure-and-order"


def dedup(meta, pool):
    seen = set()
    unique = []
    for cap, name in pool:
        key = cap, meta["orders"][name]
        if key not in seen:
            unique.append((cap, name))
            seen.add(key)
    return unique


def candidate(meta, config):
    cap, label = config
    return Candidate(cap, meta["orders"][label], tuple(meta["isolated"] + meta["copy"][cap]))


def ratio_error(measured, selected):
    return 100*(measured[selected]/min(measured.values())-1)


def fixed_best(cases, rows):
    return min(rows, key=lambda option: mean(ratio_error(c["median"], option) for c in cases))


def good_ranking(values, targets):
    # Tie-aware training loss. Differences below 0.5% are down-weighted.
    loss, total = 0., 0.
    for i, j in combinations(values, 2):
        measured = targets[i] - targets[j]
        margin = min(1., abs(measured)/(.005*min(targets.values())))
        guessed = values[i] - values[j]
        if measured*guessed <= 0:
            loss += margin*(.5 if abs(guessed)<.00001 else 1)
        total += margin
    return loss/max(total, .001)


def order_features(meta, config):
    family, kernel, layout = (meta[n] for n in ("family", "kernel", "layout"))
    cap, label = config
    entry = candidate(meta, config)
    time = entry.work
    total_compute = sum(time[:COMPUTE])
    total_copy = sum(time[COMPUTE:])
    pressure = int(cap)/64 if cap != "dma" else 0.
    pure_dma = cap == "dma"
    ranks = {n: i/(TOTAL-1) for i,n in enumerate(entry.order)}
    criticality = downstream(DEPENDENCIES[family], time)
    top = max(criticality)
    copy_position = mean(ranks[n] for n in range(COMPUTE,TOTAL))
    relative_path = mean((1-ranks[n])*criticality[n]/top for n in range(TOTAL))
    compute_early = 1-min(ranks[n] for n in range(COMPUTE))
    cap_exposure = pressure * total_compute / (total_copy+total_compute)
    return np.array([
        1., total_compute, total_copy,
        dag_only(family, time, entry.order),
        pure_dma, pressure, cap_exposure,
        layout == "paged", kernel == "gemm",
        compute_early, copy_position, relative_path,
        pressure*compute_early, pressure*copy_position,
        pure_dma*(layout == "paged"),
        *(label == name for name in TRAIN_ORDERS),
    ], dtype=float)


def fit_regression(cases):
    features, values = [], []
    for row, pool in cases:
        design = np.stack([order_features(row,c) for c in pool])
        scores = np.array([row["median"][c] for c in pool])
        features.extend(design - design.mean(axis=0))
        values.extend(scores - scores.mean())
    x = np.stack(features)
    scale = np.maximum(.05, x.std(axis=0))
    x /= scale
    coeff = np.linalg.solve(x.T @ x + 2*np.eye(x.shape[1]), x.T @ np.array(values)) / scale
    return coeff


def fit_admission(cases):
    rank_loss = {}
    for params in PARAMETERS:
        losses = []
        for row, candidates in cases:
            predictions = {}
            cache = {}
            for config in candidates:
                entry = candidate(row, config)
                if entry not in cache:
                    cache[entry] = replay(DEPENDENCIES[row["family"]], entry, params)
                predictions[config] = cache[entry]
            losses.append(good_ranking(predictions, row["median"]))
        rank_loss[params] = mean(losses)
    chosen = min(rank_loss, key=rank_loss.get)
    # Candidate family contains GPU rate estimates, not node-level target fits.
    return chosen, rank_loss[chosen]


def scores_for(row, configs, coeff, params):
    scores = defaultdict(dict)
    for config in configs:
        entry = candidate(row, config)
        scores["compose-dag"][config] = dag_only(row["family"], entry.work, entry.order)
        scores["order-regression"][config] = float(order_features(row, config) @ coeff)
        scores["compose-admission"][config] = replay(DEPENDENCIES[row["family"]], entry, params)
    return dict(scores)


def closest(proposed, pool, orders):
    if proposed in pool:
        return proposed
    ref = orders[proposed[1]]
    ref_position = {n:i for i,n in enumerate(ref)}
    return min(pool, key=lambda c: (c[0] != proposed[0],
        sum(abs(i-ref_position[n]) for i,n in enumerate(orders[c[1]])), c))


def with_pilot(predicted, pilot):
    offset = median(pilot[c]-predicted[c] for c in pilot) if pilot else 0.
    return min(predicted, key=lambda c: pilot.get(c, predicted[c]+offset))


def describe(case, prior, truth, pool, fixed, layout_best, critical, coeff, params, evaluate_orders):
    measured = {c: truth["median"][c] for c in pool}
    order = lambda x: closest(x, pool, prior["orders"])
    fixed = order(fixed)
    choices = {
        "fixed-global": fixed,
        "fixed-layout": order(layout_best[case[-1]]),
        "critical-priority": order(critical[case[-1]]),
    }
    model_scores = scores_for(prior, pool, coeff, params)
    best = min(measured, key=measured.get)
    outcome = dict(case=case, pool=evaluate_orders, best=best,
                   best_ms=measured[best], choices={}, pilots={}, medians=[
                       dict(candidate=c, ms=measured[c], pilot_ms=prior["median"][c]) for c in pool
                   ])
    for method, selected in choices.items():
        outcome["choices"][method] = dict(candidate=selected, regret=ratio_error(measured, selected))
    for method, predicted in model_scores.items():
        selected = with_pilot(predicted, {})
        outcome["choices"][method] = dict(candidate=selected, regret=ratio_error(measured, selected))
    outcome["choices"]["direct-search"] = outcome["choices"]["fixed-global"]
    seeds = [(Random(f"{case}/{k}"), pool.copy()) for k in range(PILOT_REPEATS)]
    regrets = defaultdict(lambda: defaultdict(list))
    for rng, members in seeds:
        members.remove(fixed)
        rng.shuffle(members)
        members.insert(0, fixed)
        for budget in BUDGETS:
            n = len(members) if budget == "all" else budget
            pilot = {c: prior["median"][c] for c in members[:n]}
            if pilot:
                selected = min(pilot, key=pilot.get)
            else:
                selected = fixed
            regrets["direct-search"][budget].append(ratio_error(measured, selected))
            for method, prediction in model_scores.items():
                selected = with_pilot(prediction, pilot)
                regrets[method][budget].append(ratio_error(measured, selected))
    for method, budgets in regrets.items():
        outcome["pilots"][method] = {b: mean(values) for b,values in budgets.items()}
    return outcome


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--repeat", type=int, default=16)
    args = cli.parse_args()
    prior = read(args.directory/"joint-A.jsonl", args.repeat)
    truth = read(args.directory/"joint-B.jsonl", args.repeat)
    assert all(p["orders"] == truth[k]["orders"] for k,p in prior.items())
    training = [prior[c] for c in jobs() if group(c) == "train"]
    train_options = [(cap, tag) for cap, tag in CONFIGS if tag in TRAIN_ORDERS]
    train_sets = [(row, dedup(row, train_options)) for row in training]
    fixed = fixed_best(training, train_options)
    layouts = sorted({c[-1] for c in prior})
    by_layout = {layout: [row for row in training if row["layout"] == layout]
                 for layout in layouts}
    layout_best = {k: fixed_best(rows, train_options) for k,rows in by_layout.items()}
    critical = {k: fixed_best(rows, [(cap,"critical") for cap in CAPS])
                for k,rows in by_layout.items()}
    oracle_fixed = fixed_best(list(truth.values()), CONFIGS)
    coeff = fit_regression(train_sets)
    params, fit_loss = fit_admission(train_sets)
    report = dict(training_cases=[k for k in jobs() if group(k) == "train"],
                  training_endpoint_configs=sum(len(options) for _,options in train_sets),
                  fixed=fixed, fixed_by_layout=layout_best, critical=critical,
                  hindsight_fixed=oracle_fixed, admission=asdict(params),
                  training_pair_loss=fit_loss, readback_basis="session A isolated stages")
    detailed = defaultdict(list)
    for case in jobs():
        tag = group(case)
        if tag == "train":
            continue
        target = prior[case]
        pool = CONFIGS if case[0] == "join" else train_options
        actual = dedup(target, pool)
        params_for_case = [case, target, truth[case]]
        baselines = [fixed, layout_best, critical, coeff, params]
        detailed[tag].append(describe(*params_for_case, actual, *baselines, tag))
        if case[0] == "join":
            seen = {target["orders"][label] for label in TRAIN_ORDERS}
            novel = [c for c in actual if target["orders"][c[1]] not in seen]
            if novel:
                detailed["new-order-only"].append(describe(
                    *params_for_case, novel, *baselines, "new-order-only"))
    report["test"] = dict(detailed)
    report["summary"] = {
        tag: dict(n=len(rows), zero={method: {
            "mean_regret_pct":mean(r["choices"][method]["regret"] for r in rows),
            "worst_regret_pct":max(r["choices"][method]["regret"] for r in rows),
        } for method in rows[0]["choices"]}, budget={
            m:{b:mean(r["pilots"][m][b] for r in rows) for b in BUDGETS}
            for m in rows[0]["pilots"]})
        for tag,rows in detailed.items()
    }
    outfile = args.directory/"selection.json"
    outfile.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(params=report["admission"],best=fixed,
                          layout_best=layout_best,critical=critical,
                          hindsight_fixed=oracle_fixed,
                          summary=report["summary"]), indent=2))


if __name__ == "__main__":
    main()
