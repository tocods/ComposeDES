"""Freeze ComposeDES region predictions before reading held-out timings."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow import Workflow, longest_path_certificate, replay

MPK_STAGES = {
    "mpk_full": ["region_0", "retire_0"],
    "mpk_attn18_mlp18": ["region_0", "retire_0", "region_1", "retire_1"],
    "mpk_layer18_19": ["region_0", "retire_0", "region_1", "retire_1"],
}
CONTEXTS = (32, 96, 256, 512)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def freeze(args):
    services = defaultdict(list)
    for row in rows(args.mpk):
        assert row["request"].startswith("calibration-")
        services[row["context"], row["candidate"], row["stage"]].append(row["us"])
    for row in rows(args.vllm):
        assert row["request"].startswith("calibration-")
        services[row["context"], row["candidate"], "graph_step"].extend(row["token_us"])

    results = {}
    for n in CONTEXTS:
        estimates = {}
        for candidate, stages in (*MPK_STAGES.items(), ("vllm_graph", ["graph_step"])):
            wf = Workflow()
            names = stages if candidate == "vllm_graph" else ["metadata_and_output", *stages]
            for i, name in enumerate(names):
                us = statistics.median(services[n, candidate, name])
                wf.add(name, us, names[i - 1:i], stage=name)
            graph = wf.serialize()
            estimate = longest_path_certificate(graph)["lower_ns"]
            assert replay(graph) == estimate
            estimates[candidate] = {
                "host_us": estimate / 1000,
                "nodes_us": {node["name"]: node["estimated_duration_ns"] / 1000 for node in graph},
                "workflow": graph,
            }
        results[n] = {"candidates": estimates,
                      "ranking": sorted(estimates, key=lambda name: estimates[name]["host_us"])}
    json_save(args.output, {
        "calibration_sha256": {"mpk": digest(args.mpk), "vllm": digest(args.vllm)},
        "contexts": results,
        "unit": "microseconds",
    })


def medians(trials):
    n = len(trials)
    return {
        "tpot_us": statistics.median(x for row in trials for x in row["token_us"]),
        "tpot_trials_us": [statistics.median(row["token_us"]) for row in trials],
        "ttft_us": statistics.median(row["ttft_us"] for row in trials),
        "gpu_us": (statistics.median(x for row in trials for x in row["gpu_us"])
                   if "gpu_us" in trials[0] else None),
        "n": sum(len(row["token_us"]) for row in trials),
        "repetitions": n,
    }


def score(args):
    frozen = json.loads(args.prediction.read_text())
    requests = defaultdict(lambda: defaultdict(list))
    for row in rows(args.mpk) + rows(args.vllm):
        assert row["request"].startswith("heldout-")
        requests[row["request"]][row["candidate"]].append(row)

    summary = []
    pooled = defaultdict(list)
    for request, group in sorted(requests.items()):
        assert set(group) == {*MPK_STAGES, "vllm_graph"}
        n = group["mpk_full"][0]["context"]
        ref = group["mpk_full"][0]["output"]
        exact_mpk = all(row["output"] == ref for key in MPK_STAGES for row in group[key])
        assert exact_mpk
        v_tokens = group["vllm_graph"][0]["output"]
        prefix = next((i for i, (a, b) in enumerate(zip(ref, v_tokens)) if a != b), len(ref))

        prediction = frozen["contexts"][str(n)]
        stats = {name: medians(trials) for name, trials in group.items()}
        for name, trials in group.items():
            pooled[name].extend(trials)
            stats[name]["predicted_us"] = prediction["candidates"][name]["host_us"]
            stats[name]["relative_error"] = stats[name]["predicted_us"] / stats[name]["tpot_us"] - 1
        ranking = sorted(stats, key=lambda name: stats[name]["tpot_us"])
        mpk_gpu_rank = sorted(MPK_STAGES, key=lambda name: stats[name]["gpu_us"])
        predicted_mpk_gpu = sorted(
            MPK_STAGES,
            key=lambda name: sum(prediction["candidates"][name]["nodes_us"][stage]
                                 for stage in MPK_STAGES[name]),
        )
        expected = prediction["ranking"]
        expected_index = {key: i for i, key in enumerate(expected)}
        matched_pairs = sum(expected_index[a] < expected_index[b] for a, b in combinations(ranking, 2))
        summary.append({
            "request": request, "context": n, "ranks": ranking, "predicted_ranks": expected,
            "mpk_gpu_ranks": mpk_gpu_rank, "predicted_mpk_gpu_ranks": predicted_mpk_gpu,
            "mpk_gpu_winner_hit": mpk_gpu_rank[0] == predicted_mpk_gpu[0],
            "vllm_mpk_prefix_equal": prefix, "outputs": len(ref),
            "winner_hit": ranking[0] == expected[0], "full_rank_hit": ranking == expected,
            "mpk_winner_hit": [n for n in ranking if n in MPK_STAGES][0] ==
                              [n for n in expected if n in MPK_STAGES][0],
            "matched_pairs": matched_pairs, "total_pairs": len(list(combinations(ranking, 2))),
            "regret_us": stats[expected[0]]["tpot_us"] - stats[ranking[0]]["tpot_us"],
            "candidates": stats,
        })
    assert len(summary) == 8
    json_save(args.output, {
        "prediction_sha256": digest(args.prediction),
        "requests": summary,
        "pooled": {name: medians(trials) for name, trials in pooled.items()},
        "hits": {key: sum(q[key] for q in summary)
                 for key in ("winner_hit", "full_rank_hit", "mpk_winner_hit", "mpk_gpu_winner_hit", "matched_pairs",
                             "total_pairs")},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "score"])
    parser.add_argument("--mpk", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        if not args.prediction:
            parser.error("--prediction is required when scoring")
        score(args)


if __name__ == "__main__":
    main()
