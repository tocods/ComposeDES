#!/usr/bin/env python3
"""Run a vertical x horizontal FNCS ablation on Grok rank timelines.

This benchmark intentionally removes cross-rank communication.  It is an upper
bound for active-dependency coordination when rank timelines are separate
federates rather than hidden behind one central workflow orchestrator.

The vertical factor aggregates local events.  The horizontal factor enables
Active-dependency coordination over a fixed rank-visible federate partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BROKER = ROOT / "fncs" / "builds" / "local" / "bin" / "fncs_broker"
FNCS_LIBRARY = ROOT / "fncs" / "builds" / "local" / "lib" / "libfncs.so"
CLIENT_DIR = ROOT / "fncs" / "orchestrator"
MAX_TIME_NS = (1 << 63) - 1
ACTIVE_DEPENDENCIES = "__fncs/active_dependencies"
TASK_PATTERN = re.compile(r"^rank(?P<rank>\d+)_phase(?P<phase>\d+)_event(?P<event>\d+)$")
REGION_PATTERN = re.compile(r"^rank(?P<rank>\d+)-phase(?P<phase>\d+)$")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_rank_schedules(
    workflow: Path,
    rank_count: int,
    events_per_rank: int,
    aggregate_regions: bool = False,
) -> dict[str, list[int]]:
    data = json.loads(workflow.read_text())
    durations: dict[int, list[tuple[int, int, int]]] = {}
    if aggregate_regions:
        acceleration = data.get("acceleration") if isinstance(data, dict) else None
        regions = acceleration.get("regions", []) if acceleration else []
        for region in regions:
            match = REGION_PATTERN.match(str(region.get("id", "")))
            if not match or region.get("mode") != "exact":
                continue
            rank = int(match.group("rank"))
            if rank >= rank_count:
                continue
            bounds = region.get("duration_bounds_ns") or []
            if len(bounds) != 2 or int(bounds[0]) != int(bounds[1]) or int(bounds[0]) <= 0:
                raise ValueError(f"region {region.get('id')} is not exact")
            durations.setdefault(rank, []).append(
                (int(match.group("phase")), 0, int(bounds[0]))
            )
    else:
        tasks = data["tasks"] if isinstance(data, dict) else data
        for task in tasks:
            match = TASK_PATTERN.match(str(task.get("name", "")))
            if not match:
                continue
            rank = int(match.group("rank"))
            if rank >= rank_count:
                continue
            duration = int(task.get("simulated_duration_ns") or 0)
            if duration <= 0:
                raise ValueError(f"task {task.get('name')} has no positive duration")
            durations.setdefault(rank, []).append(
                (int(match.group("phase")), int(match.group("event")), duration)
            )

    missing = [rank for rank in range(rank_count) if rank not in durations]
    if missing:
        raise ValueError(f"workflow is missing requested ranks: {missing[:8]}")

    schedules: dict[str, list[int]] = {}
    for rank in range(rank_count):
        ordered = sorted(durations[rank])
        if events_per_rank:
            ordered = ordered[:events_per_rank]
        current = 0
        timeline = []
        for _, _, duration in ordered:
            current += duration
            timeline.append(current)
        if not timeline:
            raise ValueError(f"rank {rank} has an empty schedule")
        schedules[str(rank)] = timeline
    return schedules


def worker_main(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(CLIENT_DIR))
    from fncs_client import FncsClient

    schedules = json.loads(Path(args.schedules).read_text())
    targets = [int(value) for value in schedules[args.rank_key]]
    client = FncsClient(str(FNCS_LIBRARY))
    grants = 0
    current = 0
    started = time.perf_counter()
    try:
        client.initialize()
        if args.publish_empty_dependencies:
            client.publish_anon(ACTIVE_DEPENDENCIES, "epoch=1")
        for target in targets:
            if target <= current:
                raise RuntimeError(f"non-increasing target {target} after {current}")
            current = client.time_request(target)
            grants += 1
            if current != target:
                raise RuntimeError(f"unexpected grant {current}, requested {target}")
        current = client.time_request(MAX_TIME_NS)
        grants += 1
        if current != MAX_TIME_NS:
            raise RuntimeError(f"final grant {current} did not reach maximum time")
    finally:
        if client._lib.fncs_is_initialized():
            client.finalize()
    dump_json(
        Path(args.worker_result),
        {
            "rank_key": args.rank_key,
            "events": len(targets),
            "grants": grants,
            "last_event_ns": targets[-1],
            "final_grant_ns": current,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    return 0


def run_case(
    output: Path,
    schedules: dict[str, list[int]],
    event_mode: str,
    active: bool,
    repeat: int,
    port: int,
    timeout: int,
) -> dict[str, Any]:
    rank_count = len(schedules)
    mode = "active" if active else "global"
    result_dir = (
        output / f"ranks_{rank_count}" / event_mode / mode / f"repeat_{repeat}"
    )
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)
    schedules_path = result_dir / "schedules.json"
    dump_json(schedules_path, schedules)
    config = result_dir / "federate.zpl"
    config.write_text(
        "name = placeholder\ntime_delta = 1ns\nbroker = tcp://localhost:5570\n",
        encoding="utf-8",
    )
    metrics = result_dir / "coordination.json"
    broker_url = f"tcp://localhost:{port}"
    base_env = os.environ.copy()
    base_env.update(
        {
            "LD_LIBRARY_PATH": (
                f"{ROOT / 'fncs/builds/local/lib'}:{ROOT / 'deps/local/lib'}:"
                f"{base_env.get('LD_LIBRARY_PATH', '')}"
            ),
            "FNCS_LIBRARY": str(FNCS_LIBRARY),
            "FNCS_BROKER": broker_url,
            "FNCS_TIME_DELTA": "1ns",
            "FNCS_FATAL": "yes",
            "FNCS_CONFIG_FILE": str(config),
        }
    )
    broker_env = base_env.copy()
    broker_env.update(
        {
            "FNCS_ACTIVE_DEPENDENCY": "yes" if active else "no",
            "FNCS_COORDINATION_METRICS": str(metrics),
        }
    )
    processes: list[subprocess.Popen[str]] = []
    logs = []
    started = time.perf_counter()
    try:
        broker_log = (result_dir / "broker.log").open("w", encoding="utf-8")
        logs.append(broker_log)
        broker = subprocess.Popen(
            [str(BROKER), str(rank_count)],
            stdout=broker_log,
            stderr=subprocess.STDOUT,
            env=broker_env,
            text=True,
        )
        processes.append(broker)
        time.sleep(0.2)

        workers: list[tuple[str, subprocess.Popen[str]]] = []
        script = Path(__file__).resolve()
        for rank in range(rank_count):
            name = "orchestrator" if rank == 0 else f"rank-{rank:03d}"
            worker_env = base_env.copy()
            worker_env["FNCS_NAME"] = name
            worker_log = (result_dir / f"{name}.log").open("w", encoding="utf-8")
            logs.append(worker_log)
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--schedules",
                str(schedules_path),
                "--rank-key",
                str(rank),
                "--worker-result",
                str(result_dir / f"{name}.json"),
            ]
            if active and rank == 0:
                command.append("--publish-empty-dependencies")
            process = subprocess.Popen(
                command,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                env=worker_env,
                text=True,
            )
            processes.append(process)
            workers.append((name, process))

        deadline = time.monotonic() + timeout
        for name, process in workers:
            process.wait(timeout=max(1, deadline - time.monotonic()))
            if process.returncode != 0:
                log = (result_dir / f"{name}.log").read_text(errors="replace")
                raise RuntimeError(f"{name} exited {process.returncode}: {log[-2000:]}")
        broker.wait(timeout=max(1, deadline - time.monotonic()))
        if broker.returncode != 0:
            log = (result_dir / "broker.log").read_text(errors="replace")
            raise RuntimeError(f"broker exited {broker.returncode}: {log[-2000:]}")
    finally:
        wall_seconds = time.perf_counter() - started
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in logs:
            stream.close()

    coordination = json.loads(metrics.read_text())
    worker_results = [
        json.loads((result_dir / ("orchestrator.json" if rank == 0 else f"rank-{rank:03d}.json")).read_text())
        for rank in range(rank_count)
    ]
    expected_events = sum(len(values) for values in schedules.values())
    if sum(item["events"] for item in worker_results) != expected_events:
        raise RuntimeError("worker event count mismatch")
    if any(item["final_grant_ns"] != MAX_TIME_NS for item in worker_results):
        raise RuntimeError("not all workers reached the final grant")
    result = {
        "event_mode": event_mode,
        "mode": mode,
        "vertical_optimization": event_mode == "aggregated",
        "horizontal_optimization": active,
        "repeat": repeat,
        "rank_count": rank_count,
        "events_per_rank": len(next(iter(schedules.values()))),
        "total_events": expected_events,
        "wall_seconds": wall_seconds,
        "scheduler_rounds": coordination["scheduler_rounds"],
        "total_grants": coordination["total_grants"],
        "dependency_updates": coordination["dependency_updates"],
        "last_event_max_ns": max(item["last_event_ns"] for item in worker_results),
    }
    dump_json(result_dir / "result.json", result)
    return result


def summarize(output: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(
            (result["rank_count"], result["event_mode"], result["mode"]), []
        ).append(result)
    cells: dict[str, Any] = {}
    for (rank_count, event_mode, mode), samples in sorted(groups.items()):
        walls = [sample["wall_seconds"] for sample in samples]
        cells[f"ranks_{rank_count}__{event_mode}__{mode}"] = {
            "rank_count": rank_count,
            "event_mode": event_mode,
            "mode": mode,
            "vertical_optimization": event_mode == "aggregated",
            "horizontal_optimization": mode == "active",
            "samples": len(samples),
            "events_per_rank": samples[0]["events_per_rank"],
            "total_events": samples[0]["total_events"],
            "wall_seconds": walls,
            "wall_median_seconds": statistics.median(walls),
            "wall_mean_seconds": statistics.mean(walls),
            "wall_stdev_seconds": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "scheduler_rounds": sorted({sample["scheduler_rounds"] for sample in samples}),
            "total_grants": sorted({sample["total_grants"] for sample in samples}),
            "dependency_updates": sorted({sample["dependency_updates"] for sample in samples}),
        }
    for rank_count in sorted({result["rank_count"] for result in results}):
        baseline_global = cells[f"ranks_{rank_count}__baseline__global"]
        for event_mode in ("baseline", "aggregated"):
            global_cell = cells[f"ranks_{rank_count}__{event_mode}__global"]
            active_cell = cells[f"ranks_{rank_count}__{event_mode}__active"]
            active_cell["horizontal_speedup"] = (
                global_cell["wall_median_seconds"] / active_cell["wall_median_seconds"]
            )
            active_cell["round_reduction_fraction"] = 1 - (
                active_cell["scheduler_rounds"][0] / global_cell["scheduler_rounds"][0]
            )
        for event_mode in ("baseline", "aggregated"):
            for mode in ("global", "active"):
                cell = cells[f"ranks_{rank_count}__{event_mode}__{mode}"]
                cell["speedup_vs_both_off"] = (
                    baseline_global["wall_median_seconds"]
                    / cell["wall_median_seconds"]
                )
    summary = {
        "description": (
            "Grok rank-local compute schedules without cross-rank edges; "
            "two-factor vertical event aggregation x horizontal "
            "Active-dependency coordination ablation"
        ),
        "factors": {
            "vertical_optimization": (
                "aggregate 128 local compute events per rank into 16 exact phase events"
            ),
            "horizontal_optimization": (
                "enable Active-dependency coordination over the fixed rank-visible "
                "federate partition"
            ),
        },
        "fixed_conditions": {
            "federation_partition": "one FNCS federate per rank",
            "cross_rank_network": "excluded",
        },
        "cells": cells,
    }
    dump_json(output / "summary.json", summary)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "rank_count",
                "vertical_optimization",
                "horizontal_optimization",
                "event_mode",
                "coordination_mode",
                "events_per_rank",
                "total_events",
                "wall_median_seconds",
                "scheduler_rounds",
                "total_grants",
                "horizontal_speedup",
                "speedup_vs_both_off",
            ]
        )
        for rank_count in sorted({cell["rank_count"] for cell in cells.values()}):
            for event_mode, mode in (
                ("baseline", "global"),
                ("baseline", "active"),
                ("aggregated", "global"),
                ("aggregated", "active"),
            ):
                cell = cells[f"ranks_{rank_count}__{event_mode}__{mode}"]
                writer.writerow(
                    [
                        cell["rank_count"],
                        cell["vertical_optimization"],
                        cell["horizontal_optimization"],
                        cell["event_mode"],
                        cell["mode"],
                        cell["events_per_rank"],
                        cell["total_events"],
                        cell["wall_median_seconds"],
                        cell["scheduler_rounds"][0],
                        cell["total_grants"][0],
                        cell.get("horizontal_speedup", 1.0),
                        cell["speedup_vs_both_off"],
                    ]
                )
    primary_rank_count = max(cell["rank_count"] for cell in cells.values())
    with (output / "ablation_2x2.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "vertical_optimization",
                "horizontal_optimization",
                "compute_events",
                "scheduler_rounds",
                "total_grants",
                "wall_median_seconds",
                "speedup_vs_both_off",
            ]
        )
        for event_mode, mode in (
            ("baseline", "global"),
            ("baseline", "active"),
            ("aggregated", "global"),
            ("aggregated", "active"),
        ):
            cell = cells[f"ranks_{primary_rank_count}__{event_mode}__{mode}"]
            writer.writerow(
                [
                    cell["vertical_optimization"],
                    cell["horizontal_optimization"],
                    cell["total_events"],
                    cell["scheduler_rounds"][0],
                    cell["total_grants"][0],
                    cell["wall_median_seconds"],
                    cell["speedup_vs_both_off"],
                ]
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--schedules", help=argparse.SUPPRESS)
    parser.add_argument("--rank-key", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--publish-empty-dependencies", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--workflow",
        type=Path,
        default=ROOT / "experiments" / "atlahs_ablation" / "grok256" / "input" / "workflow-baseline.json",
    )
    parser.add_argument(
        "--accelerated-workflow",
        type=Path,
        default=ROOT / "experiments" / "atlahs_ablation" / "grok256" / "input" / "workflow-accelerated.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "active_dependency_scaling" / "grok_rank_local",
    )
    parser.add_argument("--ranks", default="8,32,64")
    parser.add_argument("--events-per-rank", type=int, default=128)
    parser.add_argument("--event-modes", default="baseline,aggregated")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker:
        return worker_main(args)
    if not BROKER.exists() or not FNCS_LIBRARY.exists():
        raise FileNotFoundError("build and install FNCS before running the benchmark")
    rank_counts = [int(value) for value in args.ranks.split(",") if value]
    event_modes = [value for value in args.event_modes.split(",") if value]
    if not rank_counts or min(rank_counts) < 2:
        raise ValueError("--ranks must contain values >= 2")
    if set(event_modes) != {"baseline", "aggregated"}:
        raise ValueError("--event-modes must contain baseline,aggregated")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    sequence = 0
    for rank_count in rank_counts:
        schedules_by_mode = {
            "baseline": load_rank_schedules(
                args.workflow.resolve(), rank_count, args.events_per_rank
            ),
            "aggregated": load_rank_schedules(
                args.accelerated_workflow.resolve(),
                rank_count,
                args.events_per_rank,
                aggregate_regions=True,
            ),
        }
        for repeat in range(1, args.repeats + 1):
            configurations = [
                (event_mode, active)
                for event_mode in event_modes
                for active in (False, True)
            ]
            offset = (repeat - 1) % len(configurations)
            ordered = configurations[offset:] + configurations[:offset]
            for event_mode, active in ordered:
                sequence += 1
                mode = "active" if active else "global"
                print(
                    f"running ranks={rank_count} events={event_mode} "
                    f"mode={mode} repeat={repeat}",
                    flush=True,
                )
                result = run_case(
                    output,
                    schedules_by_mode[event_mode],
                    event_mode,
                    active,
                    repeat,
                    5900 + sequence,
                    args.timeout,
                )
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps(summarize(output, results), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
