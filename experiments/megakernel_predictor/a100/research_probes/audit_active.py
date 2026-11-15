"""Paired workload contrasts for the pre-frozen active admission experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from active_model import read_runs
from joint_cases import downstream


COMPARISONS = (
    ("shared", "unseen_regret", "concurrent-expanded", "concurrent-expanded-frozen"),
    ("shared", "unseen_regret", "concurrent-expanded", "regression-expanded"),
    ("shared", "unseen_regret", "concurrent-expanded", "original-expanded"),
    ("shared", "unseen_regret", "concurrent-expanded", "concurrent-chain"),
    ("active", "regret", "concurrent-expanded", "regression-expanded"),
    ("active", "regret", "concurrent-expanded", "direct-adaptive"),
)


def contrasts(path):
    cases = json.loads(path.read_text())["cases"]
    blocks = [{tuple(r[k] for k in ("protocol","test","method","budget")):r
               for r in c["metrics"]} for c in cases]
    rng = np.random.default_rng(1995)
    samples = rng.integers(0,len(cases),size=(5000,len(cases)))
    out = []
    for protocol,metric,new,control in COMPARISONS:
        for target in ("same-process","cross-process"):
            for budget in (2,4,8):
                paired = np.array([row[protocol,target,new,budget][metric] -
                                    row[protocol,target,control,budget][metric] for row in blocks])
                ci = np.quantile(paired[samples].mean(axis=1),[.025,.975])
                out.append(dict(protocol=protocol,test=target,budget=budget,metric=metric,
                                new=new,control=control,delta_pp=float(paired.mean()),
                                ci95_pp=ci.tolist(),
                                strictly_better=int((paired < -1e-9).sum()),
                                strictly_worse=int((paired > 1e-9).sum())))
    return out


def stage_search(directory):
    """Additional direct-search control using the same isolated stage data."""
    a = read_runs(directory/"active-qwen-A.jsonl")
    b = read_runs(directory/"active-qwen-B.jsonl")
    budgets = (0, 1, 2, 4, 8)
    rows = []
    for name, case in a.items():
        priorities = ("io","compute","critical","alternating","random-23","random-41")
        prior = [max(downstream(case.deps,t)) for t in case.work]
        caps = sorted({cap for cap,_ in case.labels},
                      key=lambda cap:min(t for t,(c,_) in zip(prior,case.labels) if c==cap))
        preferences = {
            cap: sorted([i for i,(c,_) in enumerate(case.labels) if c==cap],
                        key=lambda i:priorities.index(case.labels[i][1]))
            for cap in caps
        }
        chosen = preferences[caps[0]][0]
        observations = {}
        measurements = case.timings(0)
        truths = {"same-process":case.timings(1),"cross-process":b[name].timings(1)}
        for budget in range(9):
            if observations:
                chosen = min(observations,key=observations.get)
            if budget in budgets:
                for scope, values in truths.items():
                    rows.append(dict(case=name,budget=budget,test=scope,
                                     regret=float(100*(values[chosen]/values.min()-1))))
            if budget == 8:
                break
            visited = {case.labels[i][0] for i in observations}
            unseen_caps = [cap for cap in caps if cap not in visited]
            if unseen_caps and budget%2 == 0:
                point = preferences[unseen_caps[0]][0]
            else:
                interesting = [case.labels[chosen][0], *caps]
                point = next(i for cap in interesting for i in preferences[cap] if i not in observations)
            observations[point] = measurements[point]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    rows = contrasts(args.directory/"active-results.json")
    (args.directory/"paired-audit.json").write_text(json.dumps(rows,indent=2)+"\n")
    audit = stage_search(args.directory)
    (args.directory/"stage-search-audit.json").write_text(json.dumps(audit,indent=2)+"\n")


if __name__ == "__main__":
    main()
