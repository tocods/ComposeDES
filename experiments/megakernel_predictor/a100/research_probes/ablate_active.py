"""Supplementary controls for endpoint query selection, using the frozen parameters."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from active_evaluate import BUDGETS, Frozen, measure_outcome
from active_model import candidates_matrix, read_runs


def engine_pick(case, obs):
    for option in (("8","io"),("dma","io"),("2","io")):
        if option in case.labels:
            i = case.labels.index(option)
            if i not in obs:
                return i
    best = min(obs,key=obs.get) if obs else case.fixed
    prefer = ("io","compute","critical","alternating","random-23","random-41")
    available = set(range(len(case.labels)))-set(obs)
    return min(available,key=lambda i:(case.labels[i][0]!=case.labels[best][0],
                                       prefer.index(case.labels[i][1]),i))


def run(directory):
    a = read_runs(directory/"active-qwen-A.jsonl")
    b = read_runs(directory/"active-qwen-B.jsonl")
    frozen = json.loads((directory/"frozen.json").read_text())["model"]
    details = []
    control = []
    for name, case in a.items():
        truth = {"same-process":case.timings(1),"cross-process":b[name].timings(1)}
        engine_scores = {}
        for (cap,_), value in zip(case.labels,truth["cross-process"]):
            engine_scores[cap] = min(value,engine_scores.get(cap,float("inf")))
        winner, runner_up = sorted(engine_scores,key=engine_scores.get)[:2]
        relative = lambda ms: float(100*(ms/engine_scores[winner]-1))
        control.append(dict(case=name,best_cap=winner,second_cap=runner_up,
                            second_cap_gap_pct=relative(engine_scores[runner_up]),
                            same_cap_io_first_gap_pct=relative(truth["cross-process"][
                                case.labels.index((winner,"io"))])))
        pilot = case.timings(0)/case.scale
        prior = Frozen(candidates_matrix(case,True),frozen["concurrent-expanded"]["ensemble"])
        for method in ("engine-order-search","decision-frozen","static-top-score"):
            model = None if method=="engine-order-search" else prior
            observations = {}
            for budget in range(9):
                if budget in BUDGETS:
                    for target, timings in truth.items():
                        guess = model.scores(observations) if model else None
                        outcome = measure_outcome(case, observations, guess, timings)
                        details.append(dict(method=method,case=name,test=target,
                                            budget=budget,**outcome))
                if budget==8:
                    break
                if model is None:
                    point = engine_pick(case,observations)
                elif method=="decision-frozen":
                    point = model.pick(case,observations)
                elif not observations:
                    point = case.fixed
                else:
                    scores = model.scores({})
                    point = min(set(range(len(case.labels)))-set(observations),
                                key=lambda i:(scores[i],i))
                observations[point] = pilot[point]
    grouped = defaultdict(list)
    for d in details:
        grouped[d["method"],d["test"],d["budget"]].append(d["regret"])
    summary = [
        dict(method=m,test=t,budget=b,regret=mean(rows)) for (m,t,b),rows in sorted(grouped.items())
    ]
    (directory/"acquisition-controls.json").write_text(
        json.dumps(dict(summary=summary,details=details,engine_order_gaps=control),indent=2)+"\n"
    )
    for row in summary:
        if row["budget"] in (2,4,8):
            print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    run(args.directory)


if __name__=="__main__":
    main()
