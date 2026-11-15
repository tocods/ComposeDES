"""Paired uncertainty and event-graph counterfactuals for HiCache loads."""
import argparse
from collections import defaultdict
from math import ceil
import json
from pathlib import Path
import random
from statistics import median

from predict_readback import Workflow, intervals, longest_path_certificate, replay


def read(paths):
    for path in paths:
        for line in Path(path).read_text().splitlines():
            yield json.loads(line)


def nearest(values, p):
    sorted_values = sorted(values)
    return sorted_values[min(len(sorted_values) - 1, max(0, ceil(p * len(sorted_values)) - 1))]


def paired(rows):
    pairs = defaultdict(list)
    for row in rows:
        if "pair" in row:
            pairs[row["scenario"], row["session"], row["pair"], row["repeat"]].append(row)
    cohorts = defaultdict(lambda: defaultdict(list))
    for (case, session, pair, rep), observations in pairs.items():
        assert len(observations) == 2
        baseline, candidate = pair.split(":")
        times = {r["plan"]: r["ttft_us"] for r in observations}
        assert len({r["token"] for r in observations}) == 1, observations
        assert {r["order"] for r in observations} == {0, 1}
        assert {r["read_bytes"] for r in observations} == {observations[0]["read_bytes"]}
        assert all(not r["kv_write_bytes"] for r in observations)
        delta = times[baseline] - times[candidate]
        cohorts[case, pair][session].append(
            (rep, delta, times[baseline], times[candidate], observations[0]["plan"]))
    result = {}
    for (case, pair), sessions in sorted(cohorts.items()):
        blocks = [[r[1] for r in sorted(entries)] for entries in sessions.values()]
        samples = [d for series in blocks for d in series]
        rng = random.Random(case + pair)
        boot = []
        # Resample contiguous blocks within each independently started session.
        for _ in range(3500):
            values = []
            for series in blocks:
                draws = []
                length = len(series)
                while len(draws) < length:
                    start = rng.randrange(length)
                    draws.extend(series[(start + i) % length] for i in range(min(length, 4)))
                values.extend(draws[:length])
            boot.append(median(values))
        all_rows = [r for entries in sessions.values() for r in entries]
        normal, alternative = pair.split(":")
        result.setdefault(case, {})[pair] = dict(
            n=len(samples), median_gain_us=median(samples),
            block_bootstrap_95_us=[nearest(boot, .025), nearest(boot, .975)],
            abs_delta_p95_us=nearest([abs(x) for x in samples], .95),
            baseline_p50_us=median(r[2] for r in all_rows),
            candidate_p50_us=median(r[3] for r in all_rows),
            baseline_first_gain_us=median(r[1] for r in all_rows if r[4] == normal),
            candidate_first_gain_us=median(r[1] for r in all_rows if r[4] == alternative),
            per_session={k: dict(n=len(v), median_gain_us=median(x[1] for x in v))
                         for k, v in sessions.items()},
        )
    return result


def graph_replay(trace, *, compute=None, copy=None, merge_free=False, whole_last=None):
    obs = intervals(trace)
    measured = {(layer, name): stop - start for (layer, name), (start, stop) in obs.items()}
    stages = dict(compute if compute is not None else measured)
    if merge_free:
        for (lid, name) in stages:
            if name == "merge":
                stages[lid, name] = 0.
    if whole_last is not None:
        final_part = [(lid, name) for lid, name in stages if lid == 39 and
                      (name in ("resident", "suffix") or name.startswith("part/"))]
        total = sum(stages[k] for k in final_part)
        factor = min(1., whole_last / total)
        for key in final_part:
            stages[key] *= factor

    markers = {(layer, name): t for layer, name, t in trace["compute"]}
    wf = Workflow()
    wf.add("ingress", 0.)
    parent = "ingress"
    end = 0.
    for lid, group, start, stop in trace["loads"]:
        name = f"H2D/{lid}/{group}"
        # Include measured dispatch/timestamp gaps only in the measured replay.
        if copy is None:
            wait = max(0., start - end)
            wf.add(name + "/issue", wait, (parent,), engine="H2D")
            parent = name + "/issue"
        wf.add(name, copy[lid, group] if copy is not None else stop - start,
               (parent,), engine="H2D")
        end, parent = stop, name
    copy_end = parent
    parent = "ingress"
    end = 0.
    for (lid, stage), (start, stop) in obs.items():
        name = f"compute/{lid}/{stage}"
        needed = []
        if stage == "whole":
            needed.append(f"H2D/{lid}/0")
        if stage.startswith("part/"):
            needed.append(f"H2D/{lid}/{stage.split('/')[-1]}")
        if compute is None:
            observed_wait = markers.get((lid, "wait:" + stage), start)
            gap = max(0., observed_wait - end)
            wf.add(name + "/issue", gap, (parent,), engine="forward")
            parent = name + "/issue"
        wf.add(name, max(0., stages[lid, stage]), (parent, *needed), engine="forward")
        parent, end = name, stop
    wf.add("token", 0., (parent, copy_end))
    tasks = wf.serialize()
    observed = replay(tasks)
    cert = longest_path_certificate(tasks)
    assert cert["lower_ns"] == observed
    return observed / 1000


def trace_summary(rows):
    grouped = defaultdict(list)
    for item in rows:
        grouped[item["plan"], item["mode"]].append(item)
    results = {}
    plans = sorted({p for p, _ in grouped})
    for plan in plans:
        comp = [intervals(row) for row in grouped[plan, "compute"]]
        comp_times = {k: median(row[k][1] - row[k][0] for row in comp) for k in comp[0]}
        loads = [row["loads"] for row in grouped[plan, "copy"]]
        io_times = {(layer, group): median(row[i][3] - row[i][2] for row in loads)
                    for i, (layer, group, _, _) in enumerate(loads[0])}
        actual = grouped[plan, "joint"]
        records = []
        for row in actual:
            phase = intervals(row)
            marks = {(lid, n): t for lid, n, t in row["compute"]}
            waits = {f"{i}/{n[5:]}": max(0., marks[i, n[5:]] - t)
                     for (i, n), t in marks.items() if n.startswith("wait:")}
            base = graph_replay(row)
            data = dict(ttft_us=row["ttft_us"], measured_replay_us=base,
                        closure_error_us=base - row["ttft_us"],
                        last_kv_us=row["loads"][-1][3],
                        after_last_kv_us=row["ttft_us"] - row["loads"][-1][3],
                        no_contention_us=graph_replay(row, compute=comp_times, copy=io_times),
                        wait_by_attention_us=waits)
            if plan == "last-8":
                whole = [intervals(x)[39, "whole"] for x in grouped["whole-8", "compute"]]
                necessary = median(stop - start for start, stop in whole)
                data.update(merge_us=phase[39, "merge"][1] - phase[39, "merge"][0],
                            no_merge_us=graph_replay(row, merge_free=True),
                            no_split_overhead_us=graph_replay(
                                row, merge_free=True, whole_last=necessary))
            records.append(data)
        summary = {name: median(r[name] for r in records)
                   for name in records[0] if name != "wait_by_attention_us"}
        summary["wait_by_attention_us"] = {
            key: median(r["wait_by_attention_us"][key] for r in records)
            for key in records[0]["wait_by_attention_us"]
        }
        profiles = [intervals(row) for row in actual]
        summary["compute_by_layer_us"] = {
            f"{layer}/{name}": median(p[layer, name][1] - p[layer, name][0] for p in profiles)
            for layer, name in profiles[0]
        }
        summary["copy_ready_us"] = {
            f"{i}/{j}": median(r["loads"][index][3] for r in actual)
            for index, (i, j, _, _) in enumerate(actual[0]["loads"])
        }
        results[plan] = summary
    return results


def window_trace_summary(rows):
    runs = defaultdict(list)
    for row in rows:
        _, _, window = row["plan"].partition("@")
        window = int(window) if window else None
        post = {layer: t for layer, name, t in row["compute"] if name == "post"}
        end = 0.
        load_gaps = release_idle = service = 0.
        for layer, _, start, stop in row["loads"]:
            if end:
                load_gaps += max(0., start - end)
                if window and layer >= window:
                    release_idle += min(max(0., post[layer - window] - end),
                                        max(0., start - end))
            service += stop - start
            end = stop
        runs[row["plan"]].append(dict(ttft_us=row["ttft_us"], last_kv_us=end,
                                     copy_busy_us=service, copy_gaps_us=load_gaps,
                                     gated_wait_us=release_idle))
    return {name: {k: median(row[k] for row in values) for k in values[0]}
            for name, values in runs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", nargs="*", default=[])
    ap.add_argument("--trace", nargs="*", default=[])
    ap.add_argument("--verify", nargs="*", default=[])
    ap.add_argument("--window-trace", nargs="*", default=[])
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = {}
    if args.verify:
        rows = [r for r in read(args.verify) if "max_abs" in r]
        result["correctness"] = dict(rows=len(rows), finite=sum(r["finite"] for r in rows),
                                     max_abs=max(r["max_abs"] for r in rows),
                                     max_rmse=max(r["rmse"] for r in rows),
                                     token_matches=sum(r["top1"] == r["ref_top1"] for r in rows))
    if args.paired:
        result["paired"] = paired(read(args.paired))
    if args.trace:
        result["traces"] = trace_summary(list(read(args.trace)))
    if args.window_trace:
        result["window_traces"] = window_trace_summary(read(args.window_trace))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
