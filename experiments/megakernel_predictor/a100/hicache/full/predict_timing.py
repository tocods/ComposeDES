"""Build ComposeDES workflows for joint attention and submission decisions."""
import argparse
from bisect import bisect_right
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

from timing_selector import DUAL, EARLY, Q_READY, WHOLE, minimum_plan, prepare, simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from workflow import Workflow, longest_path_certificate, replay


def launch_lag(profile):
    probes = {int(k) - 1: v for k, v in profile["probes"].items()}
    position = sorted(probes)
    values = {
        i: tuple(max(0., median(trial[EARLY][k] - trial[Q_READY][k] for trial in probes[i]))
                 for k in ("dma_us", "query_us"))
        for i in position
    }
    interpolated = []
    for i in range(len(profile["layers"])):
        if i in values:
            interpolated.append(values[i])
            continue
        j = bisect_right(position, i)
        left, right = position[j - 1], position[j]
        weight = (i - left) / (right - left)
        interpolated.append(tuple(a + (b - a) * weight
                                  for a, b in zip(values[left], values[right])))
    return interpolated


def workflow(profile, plan):
    wf = Workflow()
    wf.add("request", 0.)
    wf.add("embedding", profile["embed_us"], ("request",))
    dma, compute = "request", "embedding"
    launched = False
    for i, (layer, action) in enumerate(zip(profile["layers"], plan)):
        prefix = f"layer/{i + 1}"
        mpk = action in (EARLY, Q_READY)
        if action == EARLY and not launched:
            dma_lag, q_lag = profile["early_lag_us"][i]
            dma_mark, q_mark = prefix + "/observed_dma_lag", prefix + "/observed_q_lag"
            wf.add(dma_mark, dma_lag, (dma,), engine="H2D")
            wf.add(q_mark, q_lag, (compute,), engine="SM")
            dma, compute = dma_mark, q_mark

        arrivals = []
        previous = 0.
        for j, offset in enumerate(profile["copies"]["mpk" if mpk else "dual"][i]):
            name = prefix + f"/copy/{j}"
            wf.add(name, offset - previous, (dma,), engine="H2D")
            dma = name
            previous = offset
            arrivals.append(name)

        q = prefix + "/qkv"
        wf.add(q, layer["qkv_us"], (compute,), engine="SM")
        attention = prefix + "/attention"
        if action == WHOLE:
            wf.add(attention, layer["whole_us"], (q, arrivals[-1]), engine="SM")
        else:
            if action == Q_READY:
                submit = prefix + "/submit_on_q_ready"
                wf.add(submit, profile["ready_submit_us"], (q,), engine="CPU")
                q = submit
            parts = layer["parts_us"]
            scale = layer["mpk_us"] / sum(parts) if mpk else 1.
            last = prefix + "/suffix"
            wf.add(last, parts[0] * scale, (q,), engine="SM")
            for j, (ready, part) in enumerate(zip(arrivals, parts[1:])):
                name = prefix + f"/partial/{j}"
                wf.add(name, part * scale, (last, ready), engine="SM")
                last = name
            wf.add(attention, layer["merge_us"], (last,), engine="SM")
        compute = prefix + "/after_attention"
        wf.add(compute, layer["post_us"], (attention,), engine="SM")
        launched |= mpk
    wf.add("head", profile["head_us"], (compute,), engine="SM")
    return wf.serialize()


def freeze(data, output):
    cases = {}
    for raw in data.read_text().splitlines():
        row = json.loads(raw)
        row["early_lag_us"] = launch_lag(row)
        profile = prepare({k: v for k, v in row.items() if k != "probes"})
        n = len(profile["layers"])
        joint, estimate = minimum_plan(profile)
        dual, dual_estimate = minimum_plan(profile, (WHOLE, DUAL))
        matched = tuple(WHOLE if action == WHOLE else DUAL for action in joint)
        late = (WHOLE,) * (n - 3)
        candidates = {
            "whole": (WHOLE,) * n, "joint_mpk": joint,
            "matched_dual": matched, "dual_selected": dual,
            "late_early": late + (EARLY,) * 3,
            "late_q_ready": late + (Q_READY, EARLY, EARLY),
            "late_dual": late + (DUAL,) * 3,
        }
        values = {}
        for name, plan in candidates.items():
            graph = workflow(profile, plan)
            ns = longest_path_certificate(graph)["lower_ns"]
            assert replay(graph) == ns
            values[name] = ns / 1000
            assert abs(values[name] - simulation(profile, plan)) < 0.5
        assert abs(values["joint_mpk"] - estimate) < 0.5
        assert abs(values["dual_selected"] - dual_estimate) < 0.5
        key = f"{profile['prefix']}/{profile['suffix']}"
        assert key not in cases
        cases[key] = dict(profile=profile, plan=joint, dual_plan=dual, predicted_us=values)

    output.write_text(json.dumps(dict(calibration_sha256=sha256(data.read_bytes()).hexdigest(),
                                      cases=cases), indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.calibration, args.output)


if __name__ == "__main__":
    main()
