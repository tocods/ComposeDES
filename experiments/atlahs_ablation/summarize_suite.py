#!/usr/bin/env python3
"""Build the full HPC and LLM experiment tables from summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKLOADS = (
    ("LULESH-64", "HPC", ROOT / "lulesh64_frontier_partitioned" / "summary.json"),
    ("HPCG-8", "HPC", ROOT / "hpcg8_frontier_partitioned" / "summary.json"),
    ("ICON-8", "HPC", ROOT / "icon8_frontier_partitioned" / "summary.json"),
    ("HPCG-64", "HPC", ROOT / "hpcg64_frontier_partitioned" / "summary.json"),
    ("ICON-64", "HPC", ROOT / "icon64_frontier_partitioned" / "summary.json"),
    ("LAMMPS-64", "HPC", ROOT / "lammps64_frontier_partitioned" / "summary.json"),
    ("Grok-314B-256", "LLM training", ROOT / "grok256_frontier_partitioned" / "summary.json"),
)
LABELS = (
    ("vertical_off__horizontal_off", False, False),
    ("vertical_off__horizontal_on", False, True),
    ("vertical_on__horizontal_off", True, False),
    ("vertical_on__horizontal_on", True, True),
)


def main() -> int:
    loaded = []
    for name, category, path in WORKLOADS:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value["validation"]["network_completion_semantics_preserved"]:
            raise ValueError(f"{name}: network semantics differ across ablation cells")
        if not value["validation"]["simulated_makespan_within_10ppm"]:
            raise ValueError(f"{name}: makespan differs by more than 10 ppm")
        loaded.append((name, category, value))

    with (ROOT / "all_workloads.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "workload",
                "category",
                "ranks",
                "source_operations",
                "vertical_optimization",
                "horizontal_optimization",
                "compute_federates",
                "ranks_per_compute_federate",
                "safe_frontier_coordination",
                "wall_clock_seconds",
                "wall_clock_median_seconds",
                "speedup_vs_both_off",
                "compute_dispatch_events",
                "scheduler_rounds_median",
                "total_grants_median",
                "network_completion_count",
            ]
        )
        for name, category, value in loaded:
            dataset = value["dataset"]
            for label, vertical, horizontal in LABELS:
                cell = value["cells"][label]
                writer.writerow(
                    [
                        name,
                        category,
                        dataset["num_ranks"],
                        dataset["source_operations"],
                        vertical,
                        horizontal,
                        cell["compute_federates"],
                        cell.get(
                            "ranks_per_compute_federate",
                            dataset["num_ranks"] // cell["compute_federates"],
                        ),
                        cell.get("safe_frontier_coordination", horizontal),
                        json.dumps(cell["wall_clock_seconds"], separators=(",", ":")),
                        cell["wall_clock_median_seconds"],
                        cell["wall_clock_speedup_vs_all_off"],
                        cell["compute_dispatch_events"],
                        cell["scheduler_rounds_median"],
                        cell["total_grants_median"],
                        cell["network_completion_count"],
                    ]
                )

    with (ROOT / "workload_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "workload",
                "category",
                "ranks",
                "source_operations",
                "baseline_seconds",
                "horizontal_compute_federates",
                "horizontal_ranks_per_federate",
                "vertical_only_speedup",
                "horizontal_only_speedup",
                "both_speedup",
                "makespan_relative_span",
            ]
        )
        for name, category, value in loaded:
            dataset = value["dataset"]
            cells = value["cells"]
            writer.writerow(
                [
                    name,
                    category,
                    dataset["num_ranks"],
                    dataset["source_operations"],
                    cells["vertical_off__horizontal_off"]["wall_clock_median_seconds"],
                    cells["vertical_off__horizontal_on"]["compute_federates"],
                    cells["vertical_off__horizontal_on"].get(
                        "ranks_per_compute_federate",
                        dataset["num_ranks"]
                        // cells["vertical_off__horizontal_on"]["compute_federates"],
                    ),
                    cells["vertical_on__horizontal_off"]["wall_clock_speedup_vs_all_off"],
                    cells["vertical_off__horizontal_on"]["wall_clock_speedup_vs_all_off"],
                    cells["vertical_on__horizontal_on"]["wall_clock_speedup_vs_all_off"],
                    value["validation"]["simulated_makespan_relative_span"],
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
