#!/usr/bin/env python3
"""Check alpha.4 case-study results against control-plane/network invariants."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "case_study_results" / "all_data_alpha4.json"
SMALL = ROOT / "tests" / "control_plane_alpha4_small" / "output" / "summary.json"
OUTPUT = ROOT / "case_study_results" / "theory_validation_alpha4.json"


def indexed(entries):
    return {name: result for name, result in entries if result is not None}


data = json.loads(RESULTS.read_text())
experiments = data["results"]
required = {"gpu_spec", "parallelism", "topology", "bandwidth"}
if set(experiments) != required:
    raise SystemExit(f"incomplete experiment set: {sorted(experiments)}")

gpu = indexed(experiments["gpu_spec"])
if any(result["comm_time"] != 0 for result in gpu.values()):
    raise SystemExit("TP1 GPU sweep unexpectedly contains network activity")
if any(abs(result["total_time"] - result["compute_time"]) > 1e-6 for result in gpu.values()):
    raise SystemExit("TP1 GPU sweep changed time outside the compute backend")

parallel = indexed(experiments["parallelism"])
if parallel["TP4_PP1"]["network_events"] != 672:
    raise SystemExit("TP4 ring expansion did not produce 28 * 6 * 4 flows")
if parallel["TP1_PP4"]["network_events"] != 3:
    raise SystemExit("PP4 did not produce one flow for each of its three stage boundaries")

topology = indexed(experiments["topology"])
compute_delta_us = abs(
    topology["star"]["compute_time"] - topology["direct_p2p"]["compute_time"]
)
topology_ratio = topology["star"]["comm_time"] / topology["direct_p2p"]["comm_time"]
if compute_delta_us > 0.01 or abs(topology_ratio - 2.0) > 0.01:
    raise SystemExit("topology sweep violates compute invariance or hop-count scaling")

bandwidth = indexed(experiments["bandwidth"])
ordered_bps = [50, 100, 200, 400, 800, 1600, 6400]
totals = [bandwidth[f"{value}Gbps"]["total_time"] for value in ordered_bps]
if any(left <= right for left, right in zip(totals, totals[1:])):
    raise SystemExit("total time is not strictly decreasing with bandwidth")
compute_values = [bandwidth[f"{value}Gbps"]["compute_time"] for value in ordered_bps]
if max(compute_values) - min(compute_values) > 0.01:
    raise SystemExit("bandwidth sweep changed compute time")

comm_200 = bandwidth["200Gbps"]["comm_time"]
comm_400 = bandwidth["400Gbps"]["comm_time"]
fixed_propagation_us = 2 * comm_400 - comm_200
serialization_coefficient = (comm_400 - fixed_propagation_us) * 400
fit_errors = {}
for value in ordered_bps:
    observed = bandwidth[f"{value}Gbps"]["comm_time"]
    predicted = serialization_coefficient / value + fixed_propagation_us
    fit_errors[str(value)] = observed - predicted
if max(abs(value) for value in fit_errors.values()) > 1.1:
    raise SystemExit(f"bandwidth curve differs from A/B+C model: {fit_errors}")

small = json.loads(SMALL.read_text())
if not small["exact_makespan_preserved"]:
    raise SystemExit("exact contraction changed the small-case makespan")
if small["flow_macro"]["makespan_ns"] != small["flow_macro_theory_ns"]:
    raise SystemExit("small flow macro differs from serialization theory")
refinement = small["selective_refinement"]
if refinement["refined_regions"] != ["critical"] or refinement["retained_regions"] != ["side"]:
    raise SystemExit(f"selective refinement chose the wrong regions: {refinement}")
if refinement["final_gap_ns"] != 0:
    raise SystemExit(f"selective refinement did not close the certificate: {refinement}")

summary = {
    "control_plane": data["control_plane"],
    "checks_passed": 9,
    "tp4_ring_flows": parallel["TP4_PP1"]["network_events"],
    "topology_communication_ratio": topology_ratio,
    "bandwidth_compute_spread_us": max(compute_values) - min(compute_values),
    "fitted_fixed_propagation_us": fixed_propagation_us,
    "fitted_infinite_bandwidth_total_us": compute_values[0] + fixed_propagation_us,
    "bandwidth_fit_error_us": fit_errors,
    "small_flow_macro_matches_theory": True,
    "exact_contraction_preserves_makespan": True,
    "selective_refinement": refinement,
}
OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
