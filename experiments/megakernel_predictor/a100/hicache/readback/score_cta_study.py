"""Analyze CUDA timing events and freeze/evaluate load-aware native CTA rules."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean, median

from causal_analysis import nearest, paired, read
from select_window import model, service


BASELINE, ALTERNATIVE = "whole-2", "whole-8"
CONTRAST = f"{BASELINE}:{ALTERNATIVE}"


def summarize_graphs(rows):
    strata = defaultdict(list)
    synthetic = []
    counts = defaultdict(set)
    for r in rows:
        config = f'{r["level"]}/{r["comparison"]}/{r["tested_quota"]}'
        capture = "reverse" if r["capture_reverse"] else "forward"
        strata[config, capture].append(r)
        counts[r["level"]].add((r["compute_timers"], r["h2d_timers"]))
        synthetic.append(dict(
            scenario=config, session=f'{r["session"]}/{r["capture_block"]}',
            pair="left:right", plan=("left", "right")[r["arm"]], repeat=r["repeat"],
            order=r["position"], token=r["token"], ttft_us=r["ttft_us"],
            read_bytes=31232 * 163840, kv_write_bytes=0,
        ))
    stats = {key: value["left:right"] for key, value in paired(synthetic).items()}
    for (config, capture), items in strata.items():
        pairs = defaultdict(dict)
        for r in items:
            pairs[r["session"], r["capture_block"], r["repeat"]][r["arm"]] = r["ttft_us"]
        stats[config].setdefault("by_capture_order_us", {})[capture] = median(
            x[0] - x[1] for x in pairs.values())
    assert all(len(values) == 1 for values in counts.values()), counts
    return dict(stats=stats, marker_counts={k: list(v)[0] for k, v in counts.items()},
                rows=len(rows))


def gain_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"], row["session"], row["repeat"]].append(row)
    for (key, _, _), pair in grouped.items():
        assert len(pair) == 2
        assert len({r["input_sha256"] for r in pair}) == 1
        assert all(r["tiers"] == {"device": 0, "host": int(key.split("/")[0])}
                   for r in pair)
    return {key: x[CONTRAST] for key, x in paired(rows).items()}


def risk_rule(samples, features, depth, minimum=2):
    # The loss is paired TTFT regret normalized to stock, with equal weight
    # per shape. The two-point minimum leaf size limits the shape table.
    costs = {q: sum(max(0., gain if q == 2 else -gain) for _, _, gain in samples)
             for q in (2, 8)}
    q = min(costs, key=costs.get)
    best = costs[q], {"quota": q}
    if depth == 0:
        return best
    for feature in features:
        values = sorted({r[feature] for r in samples})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [r for r in samples if r[feature] <= threshold]
            right = [r for r in samples if r[feature] > threshold]
            if min(len(left), len(right)) < minimum:
                continue
            x, decision1 = risk_rule(left, features, depth-1, minimum)
            y, decision2 = risk_rule(right, features, depth-1, minimum)
            if x + y < best[0] - 1e-10:
                best = x+y, {"feature": ("added_tokens", "missing_bytes")[feature],
                             "threshold": threshold, "left": decision1, "right": decision2}
    return best


def use(rule, values):
    while "quota" not in rule:
        side = "left" if values[rule["feature"]] <= rule["threshold"] else "right"
        rule = rule[side]
    return f'whole-{rule["quota"]}'


def freeze(training, isolated, old, prior_selection, calibration):
    cases = gain_table(training)
    samples = []
    for key, r in cases.items():
        prefix, suffix = map(int, key.split("/"))
        samples.append((suffix, prefix * 163840, r["median_gain_us"] / r["baseline_p50_us"]))
    _, threshold = risk_rule(samples, (0,), 1)
    _, tree = risk_rule(samples, (0, 1), 2)
    byte_groups = sorted({row[1] for row in samples})
    assert len(byte_groups) == 2
    table = dict(feature="missing_bytes", threshold=sum(byte_groups)/2,
                 left=risk_rule([r for r in samples if r[1] == byte_groups[0]], (0,), 1, 1)[1],
                 right=risk_rule([r for r in samples if r[1] == byte_groups[1]], (0,), 1, 1)[1])
    plans = {}
    group = defaultdict(list)
    for row in isolated:
        group[row["prefix"], row["suffix"]].append(row)
    for (prefix, suffix), rows in sorted(group.items()):
        case = f"{prefix}/{suffix}"
        calibration_key = f"{prefix}/512/0"
        prior = old["cases"][calibration_key]["rate_factors"]
        nominal = service(rows)
        predictions = {}
        for q in (BASELINE, ALTERNATIVE):
            profiles = [r["loads"] for r in calibration if r["case"] == calibration_key and
                        r["mode"] == "copy" and r["plan"] == q]
            durations = [median(r[i][3] - r[i][2] for r in profiles) for i in range(40)]
            predictions[q] = model(nominal, durations, None, prior[q])
        expected = predictions[BASELINE] - predictions[ALTERNATIVE]
        values = dict(added_tokens=suffix, missing_bytes=prefix * 163840)
        plans[case] = dict(predicted_gain_us=expected, predictions_us=predictions, policies={
            "fixed-2": BASELINE, "fixed-8": ALTERNATIVE,
            "threshold": use(threshold, values), "tree": use(tree, values),
            "size-table": use(table, values),
            "composedes": ALTERNATIVE if expected > 0 else BASELINE,
            "guarded": ALTERNATIVE if expected > prior_selection["guard_us"] else BASELINE,
        })
    return dict(cases=plans, training_cases=sorted(cases), threshold=threshold, tree=tree,
                size_table=table,
                guard_us=prior_selection["guard_us"],
                costs=dict(native_training_pairs=len(training)//2,
                           native_training_api_us=sum(r["ttft_us"] for r in training),
                           new_isolated_runs=len(isolated),
                           new_isolated_setup_us=sum(r["setup_us"] for r in isolated),
                           new_isolated_run_us=sum(r["elapsed_us"] for r in isolated),
                           original_calibration_rows=len(calibration)))


def bootstrap_per_shape(rows):
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["scenario"], r["session"], r["repeat"]][r["plan"]] = r["ttft_us"]
    values = defaultdict(lambda: defaultdict(list))
    for (case, session, _), pair in sorted(pairs.items()):
        values[case][session].append(pair[BASELINE] - pair[ALTERNATIVE])
    rng = random.Random(37913)
    result = {}
    for case, group in sorted(values.items()):
        medians = []
        for _ in range(3500):
            all_draws = []
            for series in group.values():
                draws = []
                while len(draws) < len(series):
                    start = rng.randrange(len(series))
                    draws.extend(series[(start+i) % len(series)] for i in range(4))
                all_draws.extend(draws[:len(series)])
            medians.append(median(all_draws))
        result[case] = medians
    return result


def evaluate(rows, selection):
    measured = gain_table(rows)
    assert measured.keys() == selection["cases"].keys()
    for row in rows:
        declared = selection["cases"][row["scenario"]]["policies"]
        assert row["frozen_policies"] == sorted(k for k,v in declared.items() if v == row["plan"])
    by_case = {}
    portfolio = defaultdict(list)
    for case, m in measured.items():
        chosen = selection["cases"][case]
        actual = m["median_gain_us"]
        observed_max = max(0., actual)
        policies = {}
        for policy, quota in chosen["policies"].items():
            changed = quota == ALTERNATIVE
            gain = actual if changed else 0.
            interval = m["block_bootstrap_95_us"] if changed else [0., 0.]
            data = dict(quota=quota, gain_us=gain, interval_us=interval,
                        regret_us=observed_max-gain, regression=interval[1] < 0.,
                        relative_gain=gain/m["baseline_p50_us"])
            policies[policy] = data
            portfolio[policy].append(data)
        by_case[case] = dict(observed_8=m, policies=policies, predicted_gain_us=chosen["predicted_gain_us"])
    draw_by_case = bootstrap_per_shape(rows)
    summary = {}
    for policy, values in portfolio.items():
        chosen = [case for case, stats in by_case.items()
                  if stats["policies"][policy]["quota"] == ALTERNATIVE]
        averages = [sum(draw_by_case[case][i] for case in chosen) / len(by_case)
                    for i in range(3500)]
        summary[policy] = dict(
            mean_gain_us=mean(x["gain_us"] for x in values),
            mean_gain_95_us=[nearest(averages, .025), nearest(averages, .975)],
            mean_relative_gain=mean(x["relative_gain"] for x in values),
            mean_regret_us=mean(x["regret_us"] for x in values),
            regressions=sum(x["regression"] for x in values),
            switched_shapes=len(chosen),
        )
    return dict(cases=by_case, summary=summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("graphs", "freeze", "evaluate"), required=True)
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    for name in ("isolated", "historical-prediction", "prior-selection", "calibration", "selection"):
        ap.add_argument(f"--{name}", type=Path)
    args = ap.parse_args()
    rows = list(read(args.rows))
    if args.mode == "graphs":
        result = summarize_graphs(rows)
    elif args.mode == "freeze":
        old, prior = (json.loads(x.read_text()) for x in
                      (args.historical_prediction, args.prior_selection))
        result = freeze(rows, list(read([args.isolated])), old, prior,
                        list(read([args.calibration])))
        result["inputs_sha256"] = {x.name: sha256(x.read_bytes()).hexdigest() for x in (
            *args.rows, args.isolated, args.historical_prediction, args.prior_selection,
            args.calibration,
        )}
    else:
        assert {r["selection_sha256"] for r in rows} == {sha256(args.selection.read_bytes()).hexdigest()}
        result = evaluate(rows, json.loads(args.selection.read_text()))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
