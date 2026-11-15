"""Freeze paths and page sizes from independent calibration and replay their DAGs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from predict_full import longest_path_certificate, replay, tasks_for
from selector import copy_offsets, select


def key(row):
    return "/".join(str(row[name]) for name in ("prefix", "suffix", "resident", "group"))


def freeze(calibration, requests):
    cases = {}
    chosen = {}
    for line in calibration.read_text().splitlines():
        profile = json.loads(line)
        case = key(profile)
        profile["copy_offsets"] = {name: copy_offsets(times)
                                   for name, times in profile["arrival_us"].items()}
        legacy = dict(profile, layers=[dict(layer) for layer in profile["layers"]])
        for layer in profile["layers"]:
            layer["merge_us"] = layer["direct_merge_us"]
        n = len(profile["layers"])
        whole = [False] * n
        first = [True] + [False] * (n - 1)
        all_split = [True] * n
        masks = {}
        costs = {}
        for backend in ("dual", "mpk"):
            masks[backend], costs[backend], _ = select(profile, backend)
        configurations = {
            "bulk": (profile, whole, "dual"),
            "dual_first": (profile, first, "dual"),
            "dual_all": (profile, all_split, "dual"),
            "dual_legacy_all": (legacy, all_split, "dual"),
            "dual_select": (profile, masks["dual"], "dual"),
            "mpk_select": (profile, masks["mpk"], "mpk"),
            "mpk_matched": (profile, masks["dual"], "mpk"),
        }
        scores = {}
        for name, (stages, mask, backend) in configurations.items():
            graph = tasks_for(stages, mask, backend)
            completion = replay(graph)
            assert completion == longest_path_certificate(graph)["lower_ns"]
            scores[name] = completion / 1000
        for backend, cost in costs.items():
            assert abs(scores[backend + "_select"] - cost) < 0.5

        cases[case] = dict(profile=profile, masks=masks, scores_us=scores)
        shape = case.rsplit("/", 1)[0]
        group = profile["group"]
        for name in ("dual_select", "mpk_select"):
            score = scores[name] - scores["bulk"]
            choice = chosen.setdefault(shape, {}).get(name)
            if choice is None or (score, group) < (choice["margin_us"], choice["group"]):
                chosen[shape][name] = dict(group=group, margin_us=score)

    required = {key(row) for row in json.loads(requests.read_text())["heldout"]}
    assert cases.keys() == required, (cases.keys(), required)
    return dict(calibration_sha256=sha256(calibration.read_bytes()).hexdigest(),
                requests_sha256=sha256(requests.read_bytes()).hexdigest(),
                selected_group=chosen, cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frozen = freeze(args.calibration, args.requests)
    args.output.write_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
