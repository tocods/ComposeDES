"""ComposeDES schedule replay from independent single-layer services."""
import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflow import Workflow, longest_path_certificate, replay


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return "/".join(str(row[field]) for field in ("prefix", "suffix", "resident", "group"))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def graph(row, candidate, isolated_copy=False):
    service = {stage: median(us) for stage, us in row["service_us"].items()}
    floor = median(row["replay_floor_us"])
    wf = Workflow()
    wf.add("replay", floor)
    copy_nodes = ["bulk_copy"] if candidate == "bulk_wait" else [
        f"copy_{i}" for i in range(row["groups"])
    ]
    copy_us = {node: max(0., service[node] - floor) for node in copy_nodes}
    if not isolated_copy and "grouped_copy" in service and candidate != "bulk_wait":
        # Preserve individually measured transfer proportions and calibrate
        # the total time to one grouped DMA train, measured without attention.
        scale = max(0., service["grouped_copy"] - floor) / sum(copy_us.values())
        copy_us = {node: us * scale for node, us in copy_us.items()}
    previous = "replay"
    for node in copy_nodes:
        wf.add(node, copy_us[node], (previous,), engine="H2D")
        previous = node

    if candidate in ("bulk_wait", "group_wait"):
        wf.add("whole_attention", max(0., service["whole_attention"] - floor),
               (previous,), engine="SM")
        return wf.serialize()

    previous = copy_nodes[-1] if candidate == "split_wait" else "replay"
    for node in ("resident", "suffix"):
        if node in service:
            wf.add(node, max(0., service[node] - floor), (previous,), engine="SM")
            previous = node
    for i, copy in enumerate(copy_nodes):
        node = f"partial_{i}"
        wf.add(node, max(0., service[node] - floor), (previous, copy), engine="SM")
        previous = node
    wf.add("merge", max(0., service["merge"] - floor), (previous,), engine="SM")
    return wf.serialize()


def freeze(args):
    records = {}
    for row in read(args.data):
        predictions = {}
        for name in ("bulk_wait", "group_wait", "split_wait", "split_overlap"):
            tasks = graph(row, name, args.isolated_copy)
            ns = longest_path_certificate(tasks)["lower_ns"]
            assert ns == replay(tasks)
            predictions[name] = {"us": ns / 1000, "workflow": tasks}
        case = key(row)
        assert case not in records
        records[case] = predictions
    save(args.output, {"calibration_sha256": digest(args.data), "cases": records})


def score(args):
    predictions = json.loads(args.prediction.read_text())["cases"]
    results = []
    for row in read(args.data):
        case = key(row)
        measured = {name: median(vals) for name, vals in row["latency_us"].items()}
        predicted = {name: item["us"] for name, item in predictions[case].items()}
        ranking = sorted(measured, key=measured.get)
        expectation = sorted(predicted, key=predicted.get)
        expected_top = expectation[0]
        matches = sum(predicted[left] <= predicted[right]
                      for left, right in itertools.combinations(ranking, 2))
        delta = measured["split_overlap"] / measured["bulk_wait"] - 1
        overhead = measured["split_wait"] - measured["bulk_wait"]
        hidden = measured["split_wait"] - measured["split_overlap"]
        results.append({
            "case": case, "copied_bytes": row["copied_bytes"],
            "measured_us": measured, "predicted_us": predicted,
            "ranking": ranking, "predicted_ranking": expectation,
            "matched_pairs": matches, "winner_match": ranking[0] == expected_top,
            "winner_regret_us": measured[expected_top] - measured[ranking[0]],
            "overlap_vs_bulk_delta": delta, "split_overhead_us": overhead,
            "overlap_saved_us": hidden, "max_abs_error": row["max_abs_error"],
        })
    assert len(results) == len(predictions)
    save(args.output, {
        "prediction_sha256": digest(args.prediction), "rows": results,
        "winner_counts_measured": dict(Counter(row["ranking"][0] for row in results)),
        "winner_counts_predicted": dict(Counter(row["predicted_ranking"][0] for row in results)),
        "winner_matches": sum(row["winner_match"] for row in results),
        "matched_pairs": sum(row["matched_pairs"] for row in results),
        "total_pairs": len(results) * 6,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "score"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--isolated-copy", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        if args.prediction is None:
            parser.error("--prediction is required for score")
        score(args)


if __name__ == "__main__":
    main()
