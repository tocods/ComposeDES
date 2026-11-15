"""Build and verify the full 36-layer schedule before running held-out requests."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from selector import compact_profile, copy_offsets, planned_arrivals, select

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from workflow import Workflow, longest_path_certificate, replay


def tasks_for(row, mask, backend):
    wf = Workflow()
    wf.add("request", 0)
    wf.add("embedding", row["embed_us"], ("request",))
    edge = "request"
    previous_arrival = 0.
    arrivals = []
    for i, times in enumerate(planned_arrivals(row, mask, backend)):
        layer = []
        for j, t in enumerate(times):
            name = f"kv/{i}/{j}"
            wf.add(name, t - previous_arrival, (edge,), engine="H2D")
            edge = name
            layer.append(name)
            previous_arrival = t
        arrivals.append(layer)

    early_parts = row.get("early_parts", 1)
    edge = "embedding"
    for i, (layer, cut, ready) in enumerate(zip(row["layers"], mask, arrivals)):
        base = f"layer/{i}"
        q = base + "/qkv"
        wf.add(q, layer["qkv_us"], (edge,), engine="SM")
        if cut:
            parts = layer["parts_us"]
            scale = layer["mpk_us"] / sum(parts) if backend == "mpk" else 1.
            edge = q
            for j, duration in enumerate(parts[:early_parts]):
                name = base + f"/resident/{j}"
                wf.add(name, duration * scale, (edge,), engine="SM")
                edge = name
            for j, (node, duration) in enumerate(zip(ready, parts[early_parts:])):
                name = base + f"/partial/{j}"
                wf.add(name, duration * scale, (edge, node), engine="SM")
                edge = name
            attention = base + "/merge"
            wf.add(attention, layer["merge_us"], (edge,), engine="SM")
        else:
            attention = base + "/attention"
            wf.add(attention, layer["whole_us"], (q, ready[-1]), engine="SM")
        edge = base + "/mlp"
        wf.add(edge, layer["post_us"], (attention,), engine="SM")
    wf.add("head", row["head_us"], (edge,), engine="SM")
    return wf.serialize()


def freeze(data, output):
    cases = {}
    for key, row in compact_profile(json.loads(l) for l in data.read_text().splitlines()).items():
        if row.get("adaptive_publish"):
            row["copy_offsets"] = {backend: copy_offsets(times)
                                   for backend, times in row["arrival_us"].items()}
        mask, estimate, comparisons = select(row)
        bulk = [False] * len(mask)
        all_split = [True] * len(mask)
        scores = {}
        for name, selection, backend in (
            ("bulk", bulk, "dual"), ("dual", mask, "dual"), ("mpk", mask, "mpk"),
            ("dual_all", all_split, "dual"), ("mpk_all", all_split, "mpk"),
        ):
            graph = tasks_for(row, selection, backend)
            completion = longest_path_certificate(graph)["lower_ns"]
            assert completion == replay(graph)
            scores[name] = completion / 1000
        assert abs(scores["mpk"] - estimate) < 0.5
        cases[key] = dict(profile=row, mask=mask, scores_us=scores, comparisons_us=comparisons)
    output.write_text(json.dumps(dict(calibration_sha256=sha256(data.read_bytes()).hexdigest(),
                                      cases=cases), indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.calibration, args.output)


if __name__ == "__main__":
    main()
