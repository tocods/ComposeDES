"""Summarize held-out prefix recovery using one median per prompt and path."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def score(frozen, trials, contention):
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    selected_costs = defaultdict(list)
    for row in trials:
        cells[row["case"]][row["label"]][row["candidate"]].append(row["ttft_us"])
        if row["candidate"].endswith("_select"):
            selected_costs[row["candidate"]].append(row["selector_us"])
    assert cells.keys() == frozen["cases"].keys()
    paired_contention = {}
    for row in contention:
        t = row["timings"]
        paired_contention[row["case"]] = {
            engine: median(x[f"{engine}_us"] for x in t["both"]) /
            median(x[f"{engine}_us"] for x in t[engine])
            for engine in ("attention", "copy")
        }
    assert not contention or cells.keys() == paired_contention.keys()
    results = {}
    for case, prompts in cells.items():
        assert len(prompts) == 2
        per_prompt = {label: {name: median(values) for name, values in times.items()}
                      for label, times in prompts.items()}
        stats = {candidate: median(p[candidate] for p in per_prompt.values())
                 for candidate in next(iter(per_prompt.values()))}
        profile = frozen["cases"][case]
        scores = profile["scores_us"]
        comparable = {k: stats[k] for k in scores}
        measured_winner = min(comparable, key=comparable.get)
        predicted = min(scores, key=scores.get)
        results[case] = dict(
            ttft_us=stats, per_prompt_us=per_prompt, predicted_winner=predicted,
            measured_winner=measured_winner,
            prediction_regret_us=stats[predicted] - stats[measured_winner],
            selected_layers={name: sum(mask) for name, mask in profile["masks"].items()},
            contention_ratio=paired_contention.get(case),
        )
    chosen = {}
    for shape, policies in frozen["selected_group"].items():
        chosen[shape] = {}
        for name, info in policies.items():
            case = f'{shape}/{info["group"]}'
            stats = results[case]["ttft_us"]
            chosen[shape][name] = dict(
                group=info["group"], predicted_gain_us=-info["margin_us"],
                bulk_us=stats["bulk"], measured_us=stats[name],
                layers=results[case]["selected_layers"][name.split("_")[0]],
            )
    selector_stats = {}
    for name, samples in selected_costs.items():
        ordered = sorted(samples)
        selector_stats[name] = dict(median=median(ordered),
                                    p95=ordered[(95 * len(ordered) + 99) // 100 - 1])
    return dict(cases=results, chosen=chosen, selector_us=selector_stats,
                trials=len(trials), token_matches=sum(r["first_token"] == r["reference_token"]
                                                     for r in trials),
                max_logit_error=max(r["logit_error"] for r in trials))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--contention", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = score(json.loads(args.prediction.read_text()), read_jsonl(args.heldout),
                   read_jsonl(args.contention) if args.contention else [])
    args.output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
