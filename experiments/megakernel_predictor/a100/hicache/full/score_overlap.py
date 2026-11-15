"""Test whether paired-probe costs improve the prediction for dual-stream paths."""
import argparse
import json
from pathlib import Path
from statistics import median

from predict_full import longest_path_certificate, replay, tasks_for


def materialize_overlap(graph, inflation):
    """Resolve partial-attention service times while the fixed H2D queue is active."""
    dma_end = sum(t["estimated_duration_ns"] for t in graph if t.get("engine") == "H2D")
    release = {}
    for task in graph:
        name = task["name"]
        duration = task["estimated_duration_ns"]
        start = release.get(name, 0)
        if any(part in name for part in ("/resident/", "/partial/", "/merge")):
            # At reduced rate 1 / inflation for a wall interval a, the job
            # completes a / inflation units of its nominal GPU work.
            overlap = min(duration * inflation, max(0, dma_end - start))
            duration += round(overlap * (1 - 1 / inflation))
            task["estimated_duration_ns"] = duration
        for edge in task["children"]:
            child = edge["child"]
            release[child] = max(release.get(child, 0), start + duration)
    return graph


def compare(frozen, measured, probes):
    rows = {}
    for probe in probes:
        case = probe["case"]
        entry = frozen["cases"][case]
        profile = entry["profile"]
        n = len(profile["layers"])
        sample = probe["timings"]
        penalty = max(0, median(x["attention_us"] for x in sample["both"]) -
                      median(x["attention_us"] for x in sample["attention"]))
        service = median(sum(l["parts_us"]) + l["merge_us"] for l in profile["layers"])
        inflation = 1 + penalty / service
        paths = dict(bulk=[False] * n, dual_first=[True] + [False] * (n - 1),
                     dual_all=[True] * n, dual_select=entry["masks"]["dual"])
        fixed, coupled = {}, {}
        for key, mask in paths.items():
            tasks = tasks_for(profile, mask, "dual")
            fixed[key] = entry["scores_us"][key]
            duration = replay(materialize_overlap(tasks, inflation))
            assert duration == longest_path_certificate(tasks)["lower_ns"]
            coupled[key] = duration / 1000
        observations = measured["cases"][case]["ttft_us"]
        measured_gain = {k: observations["bulk"] - observations[k] for k in paths}
        errors = {}
        for name, estimate in (("fixed", fixed), ("overlap", coupled)):
            gain = {k: estimate["bulk"] - estimate[k] for k in paths}
            errors[name] = {
                "gains_us": gain,
                "absolute_error_us": {k: abs(gain[k] - measured_gain[k]) for k in paths if k != "bulk"},
                "choice": max(gain, key=gain.get),
            }
        rows[case] = dict(probe_penalty_us=penalty, measured_gains_us=measured_gain, **errors)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--contention", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(json.loads(args.prediction.read_text()), json.loads(args.results.read_text()),
                   [json.loads(line) for line in args.contention.read_text().splitlines()])
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
