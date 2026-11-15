"""Summarize the fresh scheduler validation and separately report diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def summarize(prediction, trials):
    cases = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_path = defaultdict(list)
    mismatches = []
    for row in trials:
        if not row["label"].startswith("diagnostic-"):
            cases[row["case"]][row["label"]][row["candidate"]].append(row)
            by_path[row["candidate"]].append(row)
        if row["first_token"] != row["reference_token"]:
            mismatches.append(row)
    assert len(cases) == 6
    summary = {}
    for case, prompts in sorted(cases.items()):
        assert len(prompts) == 3
        stats = {}
        for path in sorted(by_path):
            measures = [rows[path] for rows in prompts.values()]
            assert all(len(rows) == 5 for rows in measures)
            stats[path] = {
                column: median(median(r[column] for r in rows) for rows in measures)
                for column in ("ttft_us", "selector_us")
            }
        frozen = prediction["cases"][case]
        summary[case] = dict(paths=stats, exact_layers=sum(frozen["masks"]["dual"]),
                             trim_layers=sum(frozen["trim_mask"]),
                             trim_regret_us=frozen["trim_regret_us"])
    return dict(cases=summary, validation_trials=sum(map(len, by_path.values())),
                diagnostic_trials=len(trials) - sum(map(len, by_path.values())),
                validation_mismatches=sum(not r["label"].startswith("diagnostic-")
                                          for r in mismatches),
                mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.validation.read_text().splitlines()]
    report = summarize(json.loads(args.prediction.read_text()), rows)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
