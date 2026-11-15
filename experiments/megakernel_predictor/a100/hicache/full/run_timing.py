"""Calibrate submission lag and measure new full-model holdouts."""
import argparse
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter_ns

import torch

from run_qwen import Prefill, micros
from timing_selector import DUAL, EARLY, Q_READY, WHOLE, minimum_plan


PROBES = (0, 8, 17, 26, 33, 35)


def event():
    handle = torch.cuda.Event(enable_timing=True)
    handle.record()
    return handle


class SubmissionPrefill(Prefill):
    probe_layer = None
    q_event = None
    host_submit = None
    origin_wall = None

    def project(self, lid, layer, x, rope):
        q = super().project(lid, layer, x, rope)
        if lid == self.probe_layer:
            self.q_event = event()
            self.host_submit = micros(self.origin_wall)
        return q

    def probe(self, lid, ready):
        self.compute.synchronize()
        mask = tuple(i == lid for i in range(self.n))
        self.probe_layer = lid
        self.origin_wall = perf_counter_ns()
        origin = event()
        try:
            token = self.run("mpk", mask, defer_layers=(lid,) if ready else ()).argmax().item()
            return dict(query_us=origin.elapsed_time(self.q_event) * 1000,
                        dma_us=origin.elapsed_time(self.events[-1][-1]) * 1000,
                        host_submit_us=self.host_submit, first_token=token)
        finally:
            self.probe_layer = None

    def ready_submit(self):
        samples = []
        for _ in range(31):
            origin = event()
            self.compute.synchronize()
            stop = event()
            stop.synchronize()
            samples.append(origin.elapsed_time(stop) * 1000)
        return median(samples)

    def calibration(self):
        profile = super().calibrate()
        probes = {}
        for lid in PROBES:
            trials = []
            for repeat in range(5):
                pair = {name: self.probe(lid, name == Q_READY)
                        for name in ((EARLY, Q_READY) if repeat % 2 == 0 else (Q_READY, EARLY))}
                trials.append(pair)
            probes[str(lid + 1)] = trials
        return dict(profile, ready_submit_us=self.ready_submit(), probes=probes)

    def execute(self, plan):
        return self.run("dual" if DUAL in plan else "mpk",
                        tuple(x != WHOLE for x in plan),
                        defer_layers=frozenset(i for i, mode in enumerate(plan) if mode == Q_READY))

    def heldout(self, case, prediction, repetitions):
        profile = prediction["profile"]
        selected = tuple(prediction["plan"])
        dual_optimal = tuple(prediction["dual_plan"])
        same_mask = tuple(WHOLE if x == WHOLE else DUAL for x in selected)
        late_early = tuple(WHOLE if i < self.n - 3 else EARLY for i in range(self.n))
        late_ready = late_early[:-3] + (Q_READY, EARLY, EARLY)
        late_dual = late_early[:-3] + (DUAL,) * 3
        plans = {
            "whole": (WHOLE,) * self.n, "joint_mpk": selected, "matched_dual": same_mask,
            "dual_selected": dual_optimal, "late_early": late_early,
            "late_q_ready": late_ready, "late_dual": late_dual,
        }

        reference = self.execute(plans["whole"])
        token = reference.argmax().item()
        for candidate, plan in plans.items():
            for k, v in self.device:
                k[:self.prefix].fill_(float("nan"))
                v[:self.prefix].fill_(float("nan"))
            check = self.execute(plan)
            torch.testing.assert_close(check, reference, rtol=0.05, atol=0.5)

        for repeat in range(repetitions):
            order = list(plans)
            random.Random(f"{case['label']}/{repeat}").shuffle(order)
            for candidate in order:
                self.compute.synchronize()
                start = perf_counter_ns()
                selector_us = 0.
                plan = plans[candidate]
                if candidate in ("joint_mpk", "matched_dual", "dual_selected"):
                    selection_start = perf_counter_ns()
                    if candidate == "dual_selected":
                        plan, _ = minimum_plan(profile, (WHOLE, DUAL))
                    else:
                        plan, _ = minimum_plan(profile)
                        if candidate == "matched_dual":
                            plan = tuple(WHOLE if x == WHOLE else DUAL for x in plan)
                    selector_us = micros(selection_start)
                    assert plan == plans[candidate]
                first_token = self.execute(plan).argmax().item()
                total_us = micros(start)
                yield dict(candidate=candidate, label=case["label"], sha256=case["sha256"],
                           prefix=self.prefix, suffix=self.suffix, repeat=repeat, plan=plan,
                           ttft_us=total_us, selector_us=selector_us, first_token=first_token,
                           reference_token=token)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--mode", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "heldout" and args.prediction is None:
        parser.error("--prediction is required for a held-out run")
    cases = json.loads(args.requests.read_text())[args.mode]
    pred = json.loads(args.prediction.read_text())["cases"] if args.prediction else {}
    model = SubmissionPrefill(args.model)
    with torch.cuda.stream(model.compute), args.output.open("w", buffering=1) as out:
        for case in cases:
            print("Starting", case["label"], flush=True)
            with model.request(case):
                if args.mode == "calibration":
                    print(json.dumps(dict(model.calibration(), sha256=case["sha256"])), file=out)
                else:
                    key = f"{case['prefix']}/{case['suffix']}"
                    for row in model.heldout(case, pred[key], args.repetitions):
                        print(json.dumps(row), file=out)
            print("Finished", case["label"], flush=True)


if __name__ == "__main__":
    main()
