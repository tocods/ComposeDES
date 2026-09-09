#!/usr/bin/env python3
"""Compare baseline and backend-local-chain traces on a real GPU workflow."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_case_study


def read_trace(input_dir: Path):
    records = [
        json.loads(line)
        for line in (input_dir / "output-alpha4" / "control-plane.jsonl")
        .read_text()
        .splitlines()
    ]
    task_trace = {
        event["payload"]["task_id"]: {
            "start_time_ns": event["payload"].get("start_time_ns"),
            "finish_time_ns": event["payload"]["finish_time_ns"],
            "status": event["payload"]["status"],
        }
        for record in records
        if record["direction"] == "in" and record["topic"] == "compute/completed"
        for event in record["value"]["events"]
    }
    compute_dispatches = sum(
        len(record["value"]["events"])
        for record in records
        if record["direction"] == "out" and record["topic"] == "compute/dispatch"
    )
    network_trace = {
        event["payload"]["transfer_id"]: {
            "finish_time_ns": event["payload"]["finish_time_ns"],
            "status": event["payload"]["status"],
        }
        for record in records
        if record["direction"] == "in" and record["topic"] == "network/completed"
        for event in record["value"]["events"]
    }
    return task_trace, network_trace, compute_dispatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "case_study_results" / "exp1_gpu_spec" / "H100",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "case_study_results" / "backend_local_chain_validation.json",
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    previous = os.environ.pop("COSIM_BACKEND_LOCAL_CHAINS", None)
    try:
        baseline = run_case_study.run_simulation(args.input)
        baseline_trace, baseline_network_trace, baseline_dispatches = read_trace(args.input)
        os.environ["COSIM_BACKEND_LOCAL_CHAINS"] = "yes"
        local = run_case_study.run_simulation(args.input)
        local_trace, local_network_trace, local_dispatches = read_trace(args.input)
    finally:
        if previous is None:
            os.environ.pop("COSIM_BACKEND_LOCAL_CHAINS", None)
        else:
            os.environ["COSIM_BACKEND_LOCAL_CHAINS"] = previous

    if baseline is None or local is None:
        raise SystemExit("one of the co-simulation runs failed")
    if baseline["total_time"] != local["total_time"]:
        raise SystemExit(f"makespan changed: {baseline} / {local}")
    if baseline_trace != local_trace:
        raise SystemExit("per-task GPU start/finish trace changed")
    if baseline_network_trace != local_network_trace:
        raise SystemExit("network boundary completion trace changed")
    if local_dispatches >= baseline_dispatches:
        raise SystemExit("backend-local chains did not reduce dispatch events")

    result = {
        "input": str(args.input),
        "task_count": len(local_trace),
        "network_event_count": len(local_network_trace),
        "baseline_makespan_us": baseline["total_time"],
        "local_chain_makespan_us": local["total_time"],
        "baseline_dispatch_events": baseline_dispatches,
        "local_chain_dispatch_events": local_dispatches,
        "baseline_scheduler_rounds": baseline["scheduler_rounds"],
        "local_chain_scheduler_rounds": local["scheduler_rounds"],
        "per_task_start_finish_trace_equal": True,
        "network_completion_trace_equal": True,
        "makespan_equal": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
