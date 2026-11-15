"""Freeze a bounded-regret scheduler from the original calibration."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from predict_full import longest_path_certificate, replay, tasks_for
from predict_granularity import key
from selector import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-prediction", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    original = json.loads(args.original_prediction.read_text())
    data = json.loads(args.requests.read_text())
    cases = {}
    for case in sorted({key(req) for req in data["validation"]}):
        frozen = original["cases"][case]
        profile = frozen["profile"]
        trim_us = 12.
        mask, latency, _ = select(profile, "dual", trim_us=trim_us)
        graph = tasks_for(profile, mask, "dual")
        assert replay(graph) == longest_path_certificate(graph)["lower_ns"]
        assert abs(latency - replay(graph) / 1000) < 0.5
        regret = latency - frozen["scores_us"]["dual_select"]
        assert -0.5 <= regret <= (len(profile["layers"]) - 1) * trim_us + 0.5
        cases[case] = dict(frozen, trim_us=trim_us, trim_mask=mask,
                           trim_regret_us=regret, trim_prediction_us=latency)
    output = dict(cases=cases, requests_sha256=sha256(args.requests.read_bytes()).hexdigest(),
                  original_sha256=sha256(args.original_prediction.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
