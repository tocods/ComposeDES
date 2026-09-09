#!/usr/bin/env python3

import json
import sys
from pathlib import Path


output = Path(sys.argv[1])


def inspect(mode):
    records = [json.loads(line) for line in (output / mode / "control-plane.jsonl").read_text().splitlines()]
    outbound = [record for record in records if record["direction"] == "out"]
    compute_events = sum(
        len(record["value"]["events"])
        for record in outbound
        if record["topic"] == "compute/dispatch"
    )
    network_events = sum(
        len(record["value"]["events"])
        for record in outbound
        if record["topic"] == "network/dispatch"
    )
    end = next(
        record for record in outbound
        if record["topic"] == "control"
    )
    batches = [record["value"] for record in records if isinstance(record.get("value"), dict) and "events" in record["value"]]
    if any("microstep" not in batch for batch in batches):
        raise SystemExit(f"{mode} contains a batch without microstep")
    metrics = json.loads((output / mode / "coordination.json").read_text())
    compute_finishes = {
        event["payload"]["task_id"]: event["payload"]["finish_time_ns"]
        for record in records
        if record["direction"] == "in" and record["topic"] == "compute/completed"
        for event in record["value"]["events"]
    }
    terminal_summary = end["value"]["events"][0]["payload"]["summary"]
    return {
        "makespan_ns": end["logical_time_ns"],
        "compute_dispatch_events": compute_events,
        "network_dispatch_events": network_events,
        "dependency_updates": metrics["dependency_updates"],
        "scheduler_rounds": metrics["scheduler_rounds"],
        "compute_finishes_ns": compute_finishes,
        "backend_local_chains": terminal_summary["backend_local_chains"],
    }


baseline = inspect("baseline")
optimized = inspect("optimized")
flow_macro = inspect("flow-macro")
refinement = inspect("refinement")
local_chains = inspect("local-chains")
report = json.loads((output / "optimized" / "optimization.json").read_text())
refinement_report = json.loads((output / "refinement" / "optimization.json").read_text())
if baseline["makespan_ns"] != optimized["makespan_ns"]:
    raise SystemExit(f"exact contraction changed makespan: {baseline} != {optimized}")
if baseline["compute_dispatch_events"] != 3 or optimized["compute_dispatch_events"] != 2:
    raise SystemExit(f"unexpected event reduction: {baseline} / {optimized}")
if baseline["network_dispatch_events"] != 1 or optimized["network_dispatch_events"] != 1:
    raise SystemExit("network transfer was not preserved")
if report["original_task_count"] != 3 or report["optimized_task_count"] != 2:
    raise SystemExit(f"unexpected optimization report: {report}")
if not report["budget_satisfied"]:
    raise SystemExit("optimization certificate did not satisfy its error budget")
# GPUSim's CPU task length unit is microseconds in this fixture.
expected_macro_makespan_ns = 30000 + (120000 * 8 // 10) + 10000 + 10000
if flow_macro["makespan_ns"] != expected_macro_makespan_ns:
    raise SystemExit(
        f"flow macro differs from serialization + propagation theory: "
        f"{flow_macro['makespan_ns']} != {expected_macro_makespan_ns}"
    )
if refinement_report["refined_regions"] != ["critical"]:
    raise SystemExit(f"critical region was not selectively refined: {refinement_report}")
if refinement_report["retained_regions"] != ["side"]:
    raise SystemExit(f"non-critical region was not retained: {refinement_report}")
if refinement_report["final_certificate"]["gap_ns"] != 0:
    raise SystemExit(f"refinement did not close the certificate: {refinement_report}")
if local_chains["makespan_ns"] != baseline["makespan_ns"]:
    raise SystemExit(f"local chains changed makespan: {baseline} / {local_chains}")
if local_chains["compute_finishes_ns"] != baseline["compute_finishes_ns"]:
    raise SystemExit(
        f"local chains changed per-task finish times: "
        f"{baseline['compute_finishes_ns']} / {local_chains['compute_finishes_ns']}"
    )
if local_chains["compute_dispatch_events"] != 2:
    raise SystemExit(f"local chain did not reduce dispatches: {local_chains}")
if local_chains["backend_local_chains"] != {
    "enabled": True, "plan_count": 1, "dispatches": 1, "planned_tasks": 2
}:
    raise SystemExit(f"unexpected local plan summary: {local_chains}")

summary = {
    "baseline": baseline,
    "optimized": optimized,
    "flow_macro": flow_macro,
    "flow_macro_theory_ns": expected_macro_makespan_ns,
    "selective_refinement": {
        "makespan_ns": refinement["makespan_ns"],
        "refined_regions": refinement_report["refined_regions"],
        "retained_regions": refinement_report["retained_regions"],
        "final_gap_ns": refinement_report["final_certificate"]["gap_ns"],
    },
    "backend_local_chains": local_chains,
    "backend_local_chain_precision_preserved": True,
    "compute_event_reduction": 1,
    "exact_makespan_preserved": True,
    "certificate": report["final_certificate"],
}
(output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, sort_keys=True))
