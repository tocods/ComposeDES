"""Trace the A100 8K/512 counterexample with the original full-model runner."""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch

from run_qwen import Prefill


class Timeline:
    def __init__(self):
        self.origin_wall = perf_counter_ns()
        self.origin = self.event()
        self.stamps = {}

    @staticmethod
    def event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def mark(self, name):
        self.stamps[name] = ((perf_counter_ns() - self.origin_wall) / 1000, self.event())

    def results(self, copies):
        return {
            "stages": {name: {"cpu_us": host, "gpu_us": 1000 * self.origin.elapsed_time(device)}
                       for name, (host, device) in self.stamps.items()},
            "layer_h2d_complete_us": [1000 * self.origin.elapsed_time(layer[-1]) for layer in copies],
        }


class TracedPrefill(Prefill):
    timeline = None
    publication_mask = None
    defer_layer = None

    def copy(self, backend, record, mask):
        if self.timeline:
            self.timeline.mark("submit_h2d")
        if self.publication_mask is not None:
            backend, mask = "mpk", self.publication_mask
        super().copy(backend, record, mask)
        if self.timeline:
            self.timeline.mark("submitted_h2d")

    def project(self, lid, layer, x, rope):
        if self.timeline:
            self.layer_id = lid
            self.timeline.mark(f"layer/{lid + 1}/start")
        q = super().project(lid, layer, x, rope)
        if self.timeline:
            self.timeline.mark(f"layer/{lid + 1}/query")
        if self.defer_layer == lid:
            self.compute.synchronize()
        return q

    def post(self, layer, x, attn):
        if self.timeline:
            self.timeline.mark(f"layer/{self.layer_id + 1}/attention")
        x = super().post(layer, x, attn)
        if self.timeline:
            self.timeline.mark(f"layer/{self.layer_id + 1}/finish")
        return x

    def record(self, backend, mask, *, wait_for_workers=True, publication_mask=None,
               defer_layer=None, trace=True):
        self.compute.synchronize()
        start = perf_counter_ns()
        timeline = Timeline() if trace else None
        self.timeline = timeline
        self.publication_mask = publication_mask
        self.defer_layer = defer_layer
        try:
            logits = self.run(backend, mask, wait_for_workers=wait_for_workers)
            if timeline:
                timeline.mark("logits")
            token = logits.argmax().item()
            wall_us = (perf_counter_ns() - start) / 1000
            timing = timeline.results(self.events) if timeline else {}
            return dict(timing, first_token=token, wall_us=wall_us)
        finally:
            self.timeline = None
            self.publication_mask = None
            self.defer_layer = None


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timer-only", action="store_true")
    parser.add_argument("--ablation", choices=("publish", "wait", "defer"))
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text())["heldout"]
    mask = json.loads(args.prediction.read_text())["cases"]["8192/512"]["mask"]
    paths = {"bulk": dict(backend="dual", mask=[False] * len(mask)),
             "dual": dict(backend="dual", mask=mask), "mpk": dict(backend="mpk", mask=mask)}
    if args.ablation == "publish":
        paths["publish_only"] = {**paths["bulk"], "publication_mask": mask}
    if args.ablation == "wait":
        paths["mpk_stream_wait"] = {**paths["mpk"], "wait_for_workers": False}
    if args.ablation == "defer":
        paths["mpk_deferred_submit"] = {**paths["mpk"], "defer_layer": mask.index(True)}
    model = TracedPrefill(args.model)
    with torch.cuda.stream(model.compute), args.output.open("w", buffering=1) as out:
        for row in requests:
            if (row["prefix"], row["suffix"]) != (8192, 512):
                continue
            print("Tracing", row["label"], flush=True)
            with model.request(row):
                reference = model.run(**paths["bulk"])
                ref = reference.argmax().item()
                for path in paths.values():
                    settings = path.copy()
                    model.publication_mask = settings.pop("publication_mask", None)
                    model.defer_layer = settings.pop("defer_layer", None)
                    try:
                        logits = model.run(**settings)
                        torch.testing.assert_close(logits, reference, rtol=0.05, atol=0.5)
                        assert logits.argmax().item() == ref
                    finally:
                        model.publication_mask = None
                        model.defer_layer = None
                for repeat in range(args.repetitions):
                    order = list(paths)
                    random.Random(row["label"] + str(repeat)).shuffle(order)
                    for candidate in order:
                        record = model.record(**paths[candidate], trace=not args.timer_only)
                        assert record["first_token"] == ref
                        print(json.dumps(dict(record, label=row["label"], candidate=candidate,
                                              repeat=repeat)), file=out)


if __name__ == "__main__":
    main()
