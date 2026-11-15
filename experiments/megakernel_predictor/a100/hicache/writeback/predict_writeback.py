"""Calibrate two coupled timelines and replay predicted write-back dependencies."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from workflow import Workflow, longest_path_certificate, replay


WINDOWS = {"wide": 0, "tight": 39}


def overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c))


def foreground(row):
    points = [(f"{layer}/{phase}", time) for layer, phase, time in row["points"]]
    times = [("embed", 0.), *points, ("done", row["a_gpu_us"])]
    return [(name, begin, end) for (name, begin), (_, end) in zip(times, times[1:])]


def fit_losses(samples):
    # Constrained least squares for the fraction of progress lost per
    # overlapping microsecond. Ridge keeps phases with little evidence stable.
    if not samples:
        return 0., 0.
    aa = sum(a * a for a, _, _ in samples) + 1000.
    bb = sum(b * b for _, b, _ in samples) + 1000.
    ab = sum(a * b for a, b, _ in samples)
    ay = sum(a * y for a, _, y in samples)
    by = sum(b * y for _, b, y in samples)
    clip = lambda x: min(0.9, max(0., x))
    candidates = [(edge, clip((by - ab * edge) / bb)) for edge in (0., 0.9)]
    candidates += [(clip((ay - ab * edge) / aa), edge) for edge in (0., 0.9)]
    determinant = aa * bb - ab * ab
    p = (bb * ay - ab * by) / determinant
    d = (aa * by - ab * ay) / determinant
    if 0 <= p <= 0.9 and 0 <= d <= 0.9:
        candidates.append((p, d))
    return min(candidates, key=lambda x: sum((a*x[0] + b*x[1] - y)**2
                                              for a, b, y in samples) +
               1000 * (x[0] ** 2 + x[1] ** 2))


def service(rows, solo, nominal):
    pack, dma = nominal
    compute = {}
    for phase in ("qkv", "attn", "post"):
        obs = []
        for row in rows:
            for name, start, end in foreground(row):
                if not name.endswith("/" + phase):
                    continue
                if name == f'{WINDOWS[row["window"]]}/qkv':
                    continue
                p = overlap(start, end, row["write_start_us"], row["pack_end_us"])
                d = overlap(start, end, row["pack_end_us"], row["write_end_us"])
                if p + d > 0.:
                    obs.append((p, d, end - start - solo[name]))
        compute[phase] = tuple(1. - loss for loss in fit_losses(obs))
    io = []
    for start, end, base in (("write_start_us", "pack_end_us", pack),
                             ("pack_end_us", "write_end_us", dma)):
        pairs = [(overlap(row[start], row[end], 0., row["a_gpu_us"]),
                  row[end] - row[start] - base) for row in rows]
        loss = sum(x*y for x, y in pairs) / (sum(x*x for x, _ in pairs) + 1000.)
        io.append(1. - min(.9, max(0., loss)))
    return compute, io


def materialize(solo, nominal, compute_rates, writer_rates, release):
    names = list(solo)
    timings = [0.] * len(names)
    copy = [0., 0.]
    idx = 0
    writer = -1
    a = solo[names[0]]
    w = 0.
    while idx < len(names) or writer in (0, 1):
        if idx < len(names) and names[idx] == f"{release}/qkv" and writer == -1:
            writer = 0
            w = nominal[0]
        has_a = idx < len(names)
        has_w = writer in (0, 1)
        phase = names[idx].rsplit("/", 1)[-1] if has_a else ""
        a_rate = compute_rates.get(phase, (1., 1.))[writer] if has_a and has_w else 1.
        w_rate = writer_rates[writer] if has_a and has_w else 1.
        step = min(a / a_rate if has_a else float("inf"),
                   w / w_rate if has_w else float("inf"))
        if has_a:
            timings[idx] += step
            a -= step * a_rate
            if a <= 1e-8:
                idx += 1
                if idx < len(names):
                    a = solo[names[idx]]
        if has_w:
            copy[writer] += step
            w -= step * w_rate
            if w <= 1e-8:
                writer += 1
                if writer == 1:
                    w = nominal[1]
    return dict(zip(names, timings)), copy


def graph_score(times, writes, layer, short_service):
    wf = Workflow()
    wf.add("arrival", 0.)
    prev = "arrival"
    a_end = sum(times.values())
    cursor = 0.
    release = 0.
    for name, duration in times.items():
        if name == f"{layer}/qkv":
            wf.add("relayout", writes[0], (prev,), resource="SM")
            wf.add("D2H", writes[1], ("relayout",), resource="copy")
            release = cursor
        wf.add(name, duration, (prev,), resource="SM")
        prev = name
        cursor += duration
    pack_end = release + writes[0]
    write_end = pack_end + writes[1]
    # Scalar first-token delivery can queue behind an already active D2H.
    serialized_delivery = pack_end <= a_end < write_end
    token_deps = (prev, "D2H") if serialized_delivery else (prev,)
    wf.add("long_first", 0., token_deps, resource="copy")
    wf.add("allocate", 0., (prev, "D2H"), resource="KV pages")
    wf.add("short_first", short_service, ("allocate", "long_first"), resource="SM")
    graph = wf.serialize()
    ns = replay(graph)
    assert ns == longest_path_certificate(graph)["lower_ns"]
    assert abs(ns / 1000 - (max(a_end, write_end) + short_service)) < .15
    return dict(a_ttft_us=write_end if serialized_delivery else a_end, a_gpu_us=a_end,
                b_ttft_us=ns / 1000, alloc_block_us=max(0., write_end - a_end),
                write_end_us=write_end)


def choose(rows, target_window_us=None):
    choices, details, fixed = {}, {}, {}
    for length in sorted({r["length"] for r in rows}):
        case = [row for row in rows if row["length"] == length]
        alone = [row for row in case if row["window"] == "isolated"]
        all_fg = [foreground(row) for row in alone]
        solo = {name: median(phases[i][2] - phases[i][1] for phases in all_fg)
                for i, (name, _, _) in enumerate(all_fg[0])}
        short = median(row["b_ttft_us"] - row["a_ttft_us"] for row in alone)
        if target_window_us is None:
            windows = WINDOWS
        else:
            remainder = sum(solo.values())
            available = {}
            for name, value in solo.items():
                if name.endswith("/qkv"):
                    available[int(name.split("/")[0])] = remainder
                remainder -= value
            layer = min(available, key=lambda n: abs(available[n] - target_window_us))
            windows = {"intermediate": layer}
        results = {key: {} for key in windows}
        details[length] = dict(solo_us=solo, short_us=short, quotas={})
        for quota in sorted({r["quota"] for r in case if r["quota"] >= 0}):
            io = [r for r in case if r["quota"] == quota and r["window"] == "write_only"]
            nominal = [median(r[name] for r in io) for name in ("pack_us", "dma_us")]
            drills = [r for r in case if r["quota"] == quota and r["window"] in WINDOWS]
            compute, copy = service(drills, solo, nominal)
            details[length]["quotas"][quota] = dict(nominal_us=nominal, compute_rates=compute,
                                                    writer_rates=copy)
            for window, layer in windows.items():
                times, writes = materialize(solo, nominal, compute, copy, layer)
                results[window][quota] = graph_score(times, writes, layer, short)
            for window in WINDOWS:
                actual = median(x["b_ttft_us"] for x in drills if x["window"] == window)
                base = median(x["b_ttft_us"] for x in case
                              if x["quota"] == 0 and x["window"] == window)
                fixed.setdefault(quota, []).append(actual / base)
        choices[length] = {}
        for window, candidates in results.items():
            winner = min(candidates, key=lambda q: (candidates[q]["b_ttft_us"],
                                                     candidates[q]["a_ttft_us"], q))
            selection = dict(quota=winner, estimates=candidates)
            if target_window_us is not None:
                selection["release_layer"] = windows[window]
            choices[length][window] = selection
    return choices, details, min(fixed, key=lambda q: sum(fixed[q]) / len(fixed[q]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-window-us", type=float)
    args = parser.parse_args()
    trials = [json.loads(line) for line in args.calibration.read_text().splitlines()]
    choices, profiles, fixed = choose(trials, args.target_window_us)
    args.output.write_text(json.dumps({
        "calibration_sha256": sha256(args.calibration.read_bytes()).hexdigest(),
        "requests_sha256": sha256(args.requests.read_bytes()).hexdigest(),
        "choices": choices, "profiles": profiles, "fixed_quota_from_calibration": fixed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
