"""Compose layerwise DMA and attention services into a four-layer event DAG."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
from statistics import median

from predict import Workflow, digest, graph, key, longest_path_certificate, read, replay, save


CANDIDATES = {
    "bulk_all": ("bulk_wait",) * 4,
    "group_all": ("group_wait",) * 4,
    "split_first": ("split_overlap", "bulk_wait", "bulk_wait", "bulk_wait"),
    "split_all": ("split_overlap",) * 4,
}


def chain(row, plans, projection_us):
    floor = median(row["replay_floor_us"])
    dest = Workflow()
    dest.add("replay", floor)
    dma_tail = compute_tail = "replay"
    for i, choice in enumerate(plans):
        tasks = graph(row, choice)
        if projection_us is not None and i > 0:
            name = f"layer{i}/projection"
            dest.add(name, max(0., projection_us - floor), (compute_tail,), engine="SM")
            compute_tail = name

        parents = defaultdict(list)
        for task in tasks:
            for child in task["children"]:
                parents[child["child"]].append(task["name"])
        alias = {}
        for task in tasks[1:]:
            source = task["name"]
            engine = task["engine"]
            name = f"layer{i}/{source}"
            upstream = [alias[parent] if parent != "replay"
                        else (dma_tail if engine == "H2D" else compute_tail)
                        for parent in parents[source]]
            if engine == "SM":
                # The first compute op may depend on only the final H2D event
                # in the layer graph. Carry forward the prior layer's output too.
                upstream.append(compute_tail)
                compute_tail = name
            else:
                dma_tail = name
            dest.add(name, task["estimated_duration_ns"] / 1000, upstream, engine=engine)
            alias[source] = name
    return dest.serialize()


def freeze(args):
    bridge = {row["suffix"]: median(row["bridge_us"]) for row in read(args.bridge)}
    result = {}
    for row in read(args.data):
        if row["suffix"] not in bridge or row["group"] == 512:
            continue
        for projection in (0, 1):
            case = f"{key(row)}/{projection}"
            durations = {}
            for name, plans in CANDIDATES.items():
                tasks = chain(row, plans, bridge[row["suffix"]] if projection else None)
                ns = longest_path_certificate(tasks)["lower_ns"]
                assert replay(tasks) == ns
                durations[name] = ns / 1000
            result[case] = durations
    save(args.output, {
        "calibration_sha256": digest(args.data), "bridge_sha256": digest(args.bridge),
        "cases": result,
    })


def score(args):
    predictions = json.loads(args.prediction.read_text())["cases"]
    results = []
    for row in read(args.data):
        case = f"{key(row)}/{row['projection']}"
        prediction = predictions[case]
        observed = {name: median(vals) for name, vals in row["latency_us"].items()}
        ranked = sorted(observed, key=observed.get)
        expected = sorted(prediction, key=prediction.get)
        pairs = sum(prediction[left] <= prediction[right]
                    for left, right in itertools.combinations(ranked, 2))
        results.append(dict(case=case, measured_us=observed, predicted_us=prediction,
                            ranking=ranked, predicted_ranking=expected, matched_pairs=pairs,
                            regret_us=observed[expected[0]] - observed[ranked[0]],
                            max_abs_error=row["max_abs_error"]))
    assert len(results) == len(predictions) and predictions.keys() == {r["case"] for r in results}
    save(args.output, {
        "prediction_sha256": digest(args.prediction), "rows": results,
        "winner_matches": sum(r["ranking"][0] == r["predicted_ranking"][0] for r in results),
        "matched_pairs": sum(r["matched_pairs"] for r in results),
        "total_pairs": len(results) * 6,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "score"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.bridge is None:
            parser.error("--bridge is required for freeze")
        freeze(args)
    else:
        if args.prediction is None:
            parser.error("--prediction is required for score")
        score(args)


if __name__ == "__main__":
    main()
