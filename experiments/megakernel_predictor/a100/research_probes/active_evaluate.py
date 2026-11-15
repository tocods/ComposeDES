"""Freeze training, then evaluate shared target observations and adaptive queries."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np

from active_model import Ensemble, ScoreRegression, candidates_matrix, fit_models, read_runs


BUDGETS = (0, 1, 2, 4, 8)
PILOT_SEEDS = 12


class Frozen(Ensemble):
    def posterior(self, measured):
        return self.weights


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    directory = Path(__file__).parent
    sources = ("active_model.py","active_evaluate.py","admission_calibrate.py",
               "admission_framework.py","admission_bench.py","admission_model.py",
               "explicit_graph.cu","explicit_graph.py","joint_cases.py","joint_probe.py")
    return {name:digest(directory/name) for name in sources}


def freeze(directory):
    data = directory/"active-cal.jsonl"
    rows = read_runs(data)
    model = fit_models(list(rows.values()))
    model["sha256"] = dict(calibration=digest(data), code=provenance())
    model["budget_config"] = dict(budgets=BUDGETS, pilots=PILOT_SEEDS,
                                  available_rep_parity=0, scored_rep_parity=1)
    (directory/"frozen.json").write_text(json.dumps(model,indent=2)+"\n")
    print("Frozen", model["cases"], model["configs"], flush=True)


def independent_models(case, frozen):
    groups = frozen["model"]
    matrices = {x: candidates_matrix(case,x) for x in (False,True)}
    output = {
        k:ScoreRegression(case,v["coefficients"]) if k.startswith("regression") else
        Ensemble(matrices[k.startswith("concurrent")],v["ensemble"])
        for k,v in groups.items()
    }
    extended = output["concurrent-expanded"]
    output["concurrent-expanded-frozen"] = Frozen(extended.predictions,extended.weights)
    return output


def fixed_points(case, seed):
    rng = Random(f"{case.name}/{seed}/pilot")
    priority = {j:rng.random() for j in range(len(case.labels))}
    remaining = set(range(len(case.labels)))-{case.fixed}
    order = [case.fixed]
    while remaining:
        choice = max(remaining, key=lambda j: (
            min(case.distance[j,i] for i in order), priority[j]
        ))
        order.append(choice)
        remaining.remove(choice)
    return order


def local_pick(case, obs, seed):
    if not obs:
        return case.fixed
    best = min(obs,key=obs.get)
    available = set(range(len(case.labels)))-set(obs)
    # Alternate exploitation near the best and exploration of the candidate space.
    rng = Random(f"{case.name}/{seed}/local/{len(obs)}")
    priority = {j:rng.random() for j in sorted(available)}
    if len(obs)%2:
        return max(available,key=lambda j:(min(case.distance[j,i] for i in obs),priority[j]))
    return min(available,key=lambda j:(case.distance[best,j],priority[j]))


def measure_outcome(case, observed, scores, truth):
    if scores is None:
        chosen = min(observed,key=observed.get) if observed else case.fixed
    else:
        possibilities = scores.copy()
        for i,val in observed.items():
            possibilities[i] = val
        chosen = int(np.argmin(possibilities))
    regret = 100*(truth[chosen]/truth.min()-1)
    unseen_regret = None
    if scores is not None:
        remaining = sorted(set(range(len(truth)))-set(observed))
        best = remaining[np.argmin(scores[remaining])]
        unseen_regret = 100*(truth[best]/truth[remaining].min()-1)
    return dict(regret=regret,unseen_regret=unseen_regret,choice=case.labels[chosen])


def model_diagnostic(model, pops):
    weights = model.posterior(pops)
    w = np.bincount(model.predictions.argmin(axis=1),weights=weights)
    return dict(effective_parameters=float(1./(weights@weights)),
                supported_winners=int(np.count_nonzero(w>=.05)))


def assess(case, opposite, frozen):
    assert case.labels == opposite.labels and case.orders == opposite.orders
    assert case.deps == opposite.deps and case.kinds == opposite.kinds
    assert len(case.trials[0]) == len(opposite.trials[0]) == 16
    models = independent_models(case,frozen)
    pilots = case.timings(0)/case.scale
    truths = {"same-process":case.timings(1),"cross-process":opposite.timings(1)}
    runs, diagnostics = defaultdict(list), defaultdict(list)
    active_subset = ("concurrent-expanded","regression-expanded","original-expanded","direct-adaptive")
    for seed in range(PILOT_SEEDS):
        common = fixed_points(case,seed)
        for budget in BUDGETS:
            obs = {i:pilots[i] for i in common[:budget]}
            for label,m in list(models.items())+[("direct-shared",None)]:
                scores = m.scores(obs) if m else None
                for target,truth in truths.items():
                    runs["shared",target,label,budget].append(measure_outcome(case,obs,scores,truth))
                if label in ("concurrent-expanded","original-expanded"):
                    diagnostics[label,budget].append(model_diagnostic(m,obs))
        for tag in active_subset:
            model = models.get(tag)
            seen = {}
            for budget in range(max(BUDGETS)+1):
                if budget in BUDGETS:
                    scores = model.scores(seen) if model else None
                    for target,truth in truths.items():
                        runs["active",target,tag,budget].append(measure_outcome(case,seen,scores,truth))
                if budget<max(BUDGETS):
                    point = model.pick(case,seen) if model else local_pick(case,seen,seed)
                    assert point not in seen
                    seen[point] = pilots[point]
    summaries = []
    for (protocol,target,label,budget),items in runs.items():
        u = [x["unseen_regret"] for x in items if x["unseen_regret"] is not None]
        summaries.append(dict(protocol=protocol,test=target,method=label,budget=budget,
                              regret=mean(x["regret"] for x in items),
                              unseen_regret=mean(u) if u else None))
    diag = [dict(model=k,budget=b,**{d:mean(x[d] for x in values) for d in values[0]})
            for (k,b),values in diagnostics.items()]
    return dict(case=case.name,candidates=len(case.labels),metrics=summaries,diagnostics=diag,
                full_pilot={k:100*(p[pilots.argmin()]/p.min()-1) for k,p in truths.items()},
                fastest={k:dict(ms=float(p.min()),config=case.labels[p.argmin()]) for k,p in truths.items()})


def evaluate(directory):
    training = json.loads((directory/"frozen.json").read_text())
    if training["sha256"]["code"] != provenance():
        raise ValueError("Benchmark or predictor changes since the model was frozen")
    if training["sha256"]["calibration"] != digest(directory/"active-cal.jsonl"):
        raise ValueError("Calibration data changed since the model was frozen")
    a = read_runs(directory/"active-qwen-A.jsonl")
    b = read_runs(directory/"active-qwen-B.jsonl")
    assert len(a) == len(b) == 8 and set(a) == set(b)
    by_case = [assess(a[k],b[k],training) for k in sorted(a)]
    buckets = defaultdict(list)
    for c in by_case:
        for row in c["metrics"]:
            key = tuple(row[v] for v in ("protocol","test","method","budget"))
            buckets[key].append(row)
    summary = [
        dict(protocol=p,test=t,method=m,budget=i,regret=mean(x["regret"] for x in rows),
             unseen_regret=mean(x["unseen_regret"] for x in rows)
             if rows[0]["unseen_regret"] is not None else None)
        for (p,t,m,i),rows in sorted(buckets.items())
    ]
    spread = {t:mean(c["full_pilot"][t] for c in by_case) for t in by_case[0]["full_pilot"]}
    output = dict(summary=summary,all_candidate_pilot=spread,cases=by_case,
                  training=dict(cases=training["cases"],configs=training["configs"]))
    (directory/"active-results.json").write_text(json.dumps(output,indent=2)+"\n")
    for row in summary:
        if row["test"]=="cross-process" and row["budget"] in (0,4,8):
            print(row)
    print("Full pilot",spread,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("action",choices=("freeze","test"))
    args = p.parse_args()
    if args.action=="freeze":
        freeze(args.directory)
    else:
        evaluate(args.directory)


if __name__=="__main__":
    main()
