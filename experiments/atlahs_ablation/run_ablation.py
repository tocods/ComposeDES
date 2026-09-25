#!/usr/bin/env python3
"""Run a vertical x horizontal ComposeDES ablation on an ATLAHS GOAL trace.

The experiment varies two independent mechanisms:

* vertical optimization: exact contraction of closed, same-owner
  compute regions;
* horizontal optimization: long-lived compute Federates, each hosting one or
  more trace ranks, together with frontier-aware Active-dependency scheduling.

The GOAL trace is partitioned in operation order.  Compute durations and send
volumes come from the trace.  Every rank/phase contains several local compute
events so exact contraction has useful work to remove without crossing a
network boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOAL = Path(
    "/home/sdic/atlahs/data/hpc/lulesh/lulesh_64/lulesh_64.goal"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "atlahs_ablation" / "lulesh64"
BROKER = ROOT / "fncs" / "builds" / "local" / "bin" / "fncs_broker"
NS3 = ROOT / "ns3" / "build" / "src" / "fncs" / "examples" / "ns3-dev-fncs-example"
ORCHESTRATOR = ROOT / "fncs" / "orchestrator" / "workflow_orchestrator.py"
JAVA = Path(os.environ.get("JAVA_BIN", shutil.which("java") or "java"))

OP_RE = re.compile(
    r"^l\d+:\s+(calc|send|recv)\s+(\d+)b?"
    r"(?:\s+(?:from|to)\s+(\d+))?"
)
RANK_RE = re.compile(r"^rank\s+(\d+)\s+\{")
NETWORK_TASK_RE = re.compile(
    r"macro:rank(\d+)-phase(\d+)|rank(\d+)_phase(\d+)_event\d+"
)


def normalize_network_correlation(correlation_id: str) -> str:
    def replace(match: re.Match[str]) -> str:
        rank = match.group(1) or match.group(3)
        phase = match.group(2) or match.group(4)
        return f"rank{rank}_phase{phase}_boundary"

    return NETWORK_TASK_RE.sub(replace, correlation_id)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scan_goal_counts(goal: Path) -> tuple[int, list[int], Counter[str]]:
    num_ranks = 0
    counts: list[int] = []
    kinds: Counter[str] = Counter()
    current = -1
    with goal.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("num_ranks "):
                num_ranks = int(line.split()[1])
                counts = [0] * num_ranks
                continue
            rank_match = RANK_RE.match(line)
            if rank_match:
                current = int(rank_match.group(1))
                continue
            match = OP_RE.match(line)
            if match and current >= 0:
                counts[current] += 1
                kinds[match.group(1)] += 1
    if not num_ranks or len(counts) != num_ranks or any(count == 0 for count in counts):
        raise ValueError("GOAL trace has missing or empty ranks")
    return num_ranks, counts, kinds


def aggregate_goal(
    goal: Path, num_ranks: int, op_counts: list[int], phases: int
) -> tuple[list[list[int]], list[list[dict[int, int]]]]:
    calc_ns = [[0 for _ in range(phases)] for _ in range(num_ranks)]
    sends = [[defaultdict(int) for _ in range(phases)] for _ in range(num_ranks)]
    seen = [0] * num_ranks
    current = -1
    with goal.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            rank_match = RANK_RE.match(line)
            if rank_match:
                current = int(rank_match.group(1))
                continue
            match = OP_RE.match(line)
            if not match or current < 0:
                continue
            phase = min(phases - 1, seen[current] * phases // op_counts[current])
            seen[current] += 1
            kind, raw_value, raw_peer = match.groups()
            value = int(raw_value)
            if kind == "calc":
                calc_ns[current][phase] += value
            elif kind == "send" and raw_peer is not None:
                peer = int(raw_peer)
                if 0 <= peer < num_ranks and peer != current:
                    sends[current][phase][peer] += value
    return calc_ns, sends


def split_positive(total: int, pieces: int) -> list[int]:
    total = max(total, pieces)
    base, remainder = divmod(total, pieces)
    return [base + (1 if index < remainder else 0) for index in range(pieces)]


def make_task(name: str, host: str, duration_us: int) -> dict[str, Any]:
    duration_ns = duration_us * 1000
    return {
        "name": name,
        "period": "1",
        "host": host,
        "estimated_duration_ns": duration_ns,
        "simulated_duration_ns": duration_ns,
        "duration_bounds_ns": [duration_ns, duration_ns],
        "cpu_task": {"ram": "100", "pes_number": 1, "length": 10},
        "gpu_task": {
            "kernels": [{
                "block_num": 1,
                "thread_num": 1,
                "thread_length": duration_us,
                "hardware": "CPU",
                "requested_gddram_size": 0,
                "task_input_size": 0,
                "task_output_size": 0,
                "type": "浮点",
                "calcuType": 0,
            }],
            "requested_gddram_size": 0,
            "task_input_size": 0,
            "task_output_size": 0,
        },
        "ifManager": False,
        "ifInMaster": False,
        "deadline": "999999999",
        "appArgs": [],
        "middlewareArgs": [],
        "children": [],
    }


def prepare_workload(
    goal: Path,
    output: Path,
    phases: int,
    events_per_phase: int,
    bandwidth_gbps: float,
    delay_ns: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    num_ranks, op_counts, kinds = scan_goal_counts(goal)
    calc_ns, sends = aggregate_goal(goal, num_ranks, op_counts, phases)

    tasks: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}
    regions: list[dict[str, Any]] = []
    trace_send_bytes = 0
    modeled_send_bytes = 0
    modeled_network_edges = 0

    for rank in range(num_ranks):
        host = f"host{rank + 1}"
        for phase in range(phases):
            total_us = max(calc_ns[rank][phase] // 1000, events_per_phase)
            durations = split_positive(total_us, events_per_phase)
            members = []
            for event_index, duration_us in enumerate(durations):
                name = f"rank{rank}_phase{phase}_event{event_index}"
                members.append(name)
                task = make_task(name, host, duration_us)
                tasks.append(task)
                indexed[name] = task
            for left, right in zip(members, members[1:]):
                indexed[left]["children"].append({"child": right, "size": 0})
            duration_ns = sum(durations) * 1000
            regions.append({
                "id": f"rank{rank}-phase{phase}",
                "mode": "exact",
                "members": members,
                "duration_bounds_ns": [duration_ns, duration_ns],
                "closure": {
                    "causal": True,
                    "state": True,
                    "temporal": True,
                    "resource_non_interference": True,
                },
            })

    for rank in range(num_ranks):
        for phase in range(phases - 1):
            source = indexed[f"rank{rank}_phase{phase}_event{events_per_phase - 1}"]
            local_next = f"rank{rank}_phase{phase + 1}_event0"
            source["children"].append({"child": local_next, "size": 0})
            for peer, byte_count in sorted(sends[rank][phase].items()):
                trace_send_bytes += byte_count
                if byte_count <= 0:
                    continue
                destination = f"rank{peer}_phase{phase + 1}_event0"
                hops = max(1, abs(rank - peer))
                isolated_ns = math.ceil(
                    hops * (delay_ns + byte_count * 8 / bandwidth_gbps)
                )
                source["children"].append({
                    "child": destination,
                    "size": byte_count,
                    "src_host": f"host{rank + 1}",
                    "dst_host": f"host{peer + 1}",
                    "latency_bounds_ns": [hops * delay_ns, isolated_ns * num_ranks],
                })
                modeled_send_bytes += byte_count
                modeled_network_edges += 1

    trace_send_bytes += sum(
        sum(peer_bytes.values())
        for rank_phases in sends
        for peer_bytes in rank_phases[-1:]
    )

    hosts = []
    for rank in range(num_ranks):
        hosts.append({
            "name": f"host{rank + 1}",
            "video_card_infos": [{
                "gpu_infos": [{
                    "cores": 1,
                    "core_per_sm": 1,
                    "max_block_per_sm": 1,
                    "gddram": 80000,
                    "flops_per_core": 1,
                    "int_flops_per_core": 1,
                    "matrix_flops_per_core": 1,
                    "bw": 64,
                }],
                "name": "GPU",
                "pcie_bw": 64,
            }],
            "cpu_infos": [{"cores": 1, "mips": 1, "int_mips": 1, "matrix_mips": 1}],
            "ram": 1024,
            "ifMaster": rank == 0,
        })

    topology_lines = ["sim:", "  stop: 86400s", "", "nodes:"]
    for rank in range(num_ranks):
        topology_lines.extend([f"  - id: host{rank + 1}", "    type: host"])
    topology_lines.extend(["", "links:"])
    for rank in range(num_ranks - 1):
        topology_lines.extend([
            "  - type: p2p",
            f"    endpoints: [host{rank + 1}, host{rank + 2}]",
            f"    dataRate: {bandwidth_gbps}Gbps",
            f"    delay: {delay_ns / 1000:g}us",
        ])
    topology_lines.extend(["", "apps:"])
    for rank in range(num_ranks):
        topology_lines.extend([
            "  - type: fncs",
            f"    node: host{rank + 1}",
            f"    name: host{rank + 1}",
        ])

    network_model = {
        "abstraction": "store_and_forward_flow",
        "topology": "chain",
        "host_count": num_ranks,
        "bandwidth_gbps": bandwidth_gbps,
        "link_delay_ns": delay_ns,
        "switch_delay_ns": 0,
    }
    baseline = {"tasks": tasks, "collectives": [], "network_model": network_model}
    accelerated = json.loads(json.dumps(baseline))
    accelerated["acceleration"] = {
        "enabled": True,
        "error_budget_ns": 10**18,
        "regions": regions,
    }

    input_dir = output / "input"
    json_dump(input_dir / "hosts.json", hosts)
    json_dump(input_dir / "empty.json", [])
    json_dump(input_dir / "workflow-baseline.json", baseline)
    json_dump(input_dir / "workflow-accelerated.json", accelerated)
    (input_dir / "topology.yaml").write_text(
        "\n".join(topology_lines) + "\n", encoding="utf-8"
    )

    digest = hashlib.sha256()
    with goal.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "source": str(goal),
        "source_bytes": goal.stat().st_size,
        "source_sha256": digest.hexdigest(),
        "num_ranks": num_ranks,
        "source_operations": sum(op_counts),
        "source_operation_kinds": dict(kinds),
        "phases_per_rank": phases,
        "events_per_phase": events_per_phase,
        "baseline_compute_tasks": len(tasks),
        "accelerated_compute_tasks": len(regions),
        "exact_regions": len(regions),
        "modeled_network_edges": modeled_network_edges,
        "modeled_send_bytes": modeled_send_bytes,
        "trace_send_bytes": trace_send_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "link_delay_ns": delay_ns,
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    json_dump(output / "dataset.json", metadata)
    return metadata


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "<missing>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def write_orchestrator_zpl(path: Path, compute_workers: list[str]) -> None:
    lines = [
        "name = orchestrator",
        "time_delta = 1ns",
        "broker = tcp://localhost:5570",
        "values",
    ]
    for worker in compute_workers:
        lines.extend([
            f"    compute/completed/{worker}",
            f"        topic = {worker}/compute/completed",
            '        default = ""',
            "        type = string",
            "        list = false",
        ])
    lines.extend([
        "    network/completed",
        "        topic = ns3/network/completed",
        '        default = ""',
        "        type = string",
        "        list = false",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compute_worker_zpl(path: Path, worker: str, directed: bool) -> None:
    dispatch_topic = "orchestrator/compute/dispatch"
    if directed:
        dispatch_topic += f"/{worker}"
    path.write_text(
        "\n".join([
            f"name = {worker}",
            "time_delta = 1ns",
            "broker = tcp://localhost:5570",
            "values",
            "    compute/dispatch",
            f"        topic = {dispatch_topic}",
            '        default = ""',
            "        type = string",
            "        list = false",
            "    control",
            "        topic = orchestrator/control",
            '        default = ""',
            "        type = string",
            "        list = false",
        ]) + "\n",
        encoding="utf-8",
    )


def build_compute_partitions(
    all_hosts: list[dict[str, Any]],
    horizontal_optimization: bool,
    ranks_per_federate: int,
    partition_map_path: Path | None = None,
) -> tuple[list[str], dict[str, str], dict[str, list[dict[str, Any]]], str]:
    """Build stable long-lived Federates from rank or job-aware partitions."""
    host_names = [str(host["name"]) for host in all_hosts]
    if not horizontal_optimization:
        return ["gpusim"], {name: "gpusim" for name in host_names}, {
            "gpusim": [dict(host) for host in all_hosts]
        }, "centralized"
    if ranks_per_federate <= 0:
        raise ValueError("ranks_per_federate must be positive")

    if partition_map_path is None:
        raw_partition_by_host = {
            name: str(index // ranks_per_federate)
            for index, name in enumerate(host_names)
        }
        strategy = "contiguous_rank_groups"
    else:
        loaded = json.loads(partition_map_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("compute partition map must be a JSON object")
        missing = sorted(set(host_names) - set(loaded))
        extra = sorted(set(loaded) - set(host_names))
        if missing or extra:
            raise ValueError(
                f"compute partition map host mismatch: missing={missing}, extra={extra}"
            )
        raw_partition_by_host = {name: str(loaded[name]) for name in host_names}
        if any(not value for value in raw_partition_by_host.values()):
            raise ValueError("compute partition IDs must be non-empty")
        strategy = "explicit_dag_groups"

    partition_ids = list(dict.fromkeys(raw_partition_by_host.values()))
    worker_by_partition = {
        partition_id: f"gpusim-partition-{index:03d}"
        for index, partition_id in enumerate(partition_ids)
    }
    worker_by_host = {
        host: worker_by_partition[raw_partition_by_host[host]]
        for host in host_names
    }
    hosts_by_worker = {worker: [] for worker in worker_by_partition.values()}
    for host in all_hosts:
        hosts_by_worker[worker_by_host[str(host["name"])]].append(dict(host))
    return list(hosts_by_worker), worker_by_host, hosts_by_worker, strategy


def run_one(
    output: Path,
    acceleration: bool,
    horizontal_optimization: bool,
    repeat: int,
    port: int,
    timeout: int,
    quiet_worker_logs: bool = False,
    ranks_per_federate: int = 1,
    safe_frontiers: bool = True,
    partition_map_path: Path | None = None,
    frontier_update_quantum_ns: int = 0,
) -> dict[str, Any]:
    if frontier_update_quantum_ns < 0:
        raise ValueError("frontier_update_quantum_ns must be non-negative")
    label = (
        f"vertical_{'on' if acceleration else 'off'}__"
        f"horizontal_{'on' if horizontal_optimization else 'off'}"
    )
    result_dir = output / "runs" / label / f"repeat_{repeat}"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)
    input_dir = output / "input"
    dataset = json.loads((output / "dataset.json").read_text())
    host_count = int(dataset["num_ranks"])
    bandwidth_gbps = float(dataset["bandwidth_gbps"])
    delay_ns = int(dataset["link_delay_ns"])
    workflow = input_dir / (
        "workflow-accelerated.json" if acceleration else "workflow-baseline.json"
    )
    metrics = result_dir / "coordination.json"
    event_log = result_dir / "control-plane.jsonl"
    optimization = result_dir / "optimization.json"
    steady_start_marker = result_dir / "steady-start.marker"
    broker_url = f"tcp://localhost:{port}"
    run_id = f"atlahs-{output.name}-{label}-r{repeat}"

    all_hosts = json.loads((input_dir / "hosts.json").read_text())
    if len(all_hosts) != host_count:
        raise ValueError("hosts.json does not match dataset rank count")
    (
        compute_workers,
        compute_worker_by_host,
        hosts_by_worker,
        partition_strategy,
    ) = build_compute_partitions(
        all_hosts,
        horizontal_optimization,
        ranks_per_federate,
        partition_map_path,
    )

    config_dir = result_dir / "config"
    config_dir.mkdir()
    orchestrator_zpl = config_dir / "orchestrator.zpl"
    write_orchestrator_zpl(orchestrator_zpl, compute_workers)
    worker_specs: list[tuple[str, Path, Path]] = []
    for worker in compute_workers:
        worker_dir = result_dir / "workers" / worker
        worker_dir.mkdir(parents=True)
        worker_hosts = hosts_by_worker[worker]
        if horizontal_optimization:
            for host_index, host in enumerate(worker_hosts):
                host["ifMaster"] = host_index == 0
        hosts_path = config_dir / f"{worker}.hosts.json"
        zpl_path = config_dir / f"{worker}.zpl"
        json_dump(hosts_path, worker_hosts)
        write_compute_worker_zpl(zpl_path, worker, horizontal_optimization)
        worker_specs.append((worker, worker_dir, hosts_path))
    worker_map_path = config_dir / "compute-worker-map.json"
    json_dump(worker_map_path, compute_worker_by_host)

    base_env = os.environ.copy()
    base_env.update({
        "LD_LIBRARY_PATH": (
            f"{ROOT / 'fncs/builds/local/lib'}:{ROOT / 'deps/local/lib'}:"
            f"{base_env.get('LD_LIBRARY_PATH', '')}"
        ),
        "FNCS_LIBRARY": str(ROOT / "fncs" / "builds" / "local" / "lib" / "libfncs.so"),
        "FNCS_BROKER": broker_url,
        "FNCS_TIME_DELTA": "1ns",
        "FNCS_FATAL": "yes",
    })
    processes: list[subprocess.Popen[str]] = []
    streams = []

    def start(
        name: str,
        command: list[str],
        env: dict[str, str],
        discard_output: bool = False,
    ) -> subprocess.Popen[str]:
        if discard_output:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            processes.append(process)
            return process
        stream = (result_dir / f"{name}.log").open("w", encoding="utf-8")
        streams.append(stream)
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, text=True)
        processes.append(process)
        return process

    started = time.perf_counter()
    steady_started: float | None = None
    try:
        broker_env = base_env.copy()
        broker_env.update({
            "FNCS_ACTIVE_DEPENDENCY": "yes" if horizontal_optimization else "no",
            "FNCS_COORDINATION_METRICS": str(metrics),
        })
        broker = start(
            "broker", [str(BROKER), str(len(compute_workers) + 2)], broker_env
        )
        time.sleep(0.35)

        ns3_env = base_env.copy()
        ns3_env.update({
            "FNCS_CONFIG_FILE": str(ROOT / "ns3" / "fncs.worker.zpl"),
            "FNCS_NAME": "ns3",
            "COSIM_NS3_FLOW_MACRO": "yes",
            "COSIM_NS3_TOPOLOGY": "chain",
            "COSIM_NS3_HOST_COUNT": str(host_count),
            "COSIM_NS3_BANDWIDTH_GBPS": f"{bandwidth_gbps:g}",
            "COSIM_NS3_DELAY_NS": str(delay_ns),
            "COSIM_NS3_SWITCH_DELAY_NS": "0",
        })
        ns3 = start(
            "ns3",
            [str(NS3), f"--topo={input_dir / 'topology.yaml'}"],
            ns3_env,
            discard_output=quiet_worker_logs,
        )

        gpusim_processes: list[tuple[str, subprocess.Popen[str]]] = []
        for worker, worker_dir, hosts_path in worker_specs:
            worker_host_count = len(json.loads(hosts_path.read_text()))
            worker_heap_mb = max(256, worker_host_count * 8)
            gpusim_env = base_env.copy()
            gpusim_env.update({
                "FNCS_CONFIG_FILE": str(config_dir / f"{worker}.zpl"),
                "FNCS_NAME": worker,
                "COSIM_WORKER_IDLE_GRANT_NS": (
                    str((1 << 63) - 1)
                    if horizontal_optimization and safe_frontiers
                    else "1000000"
                ),
            })
            process = start(
                worker,
                [
                    str(JAVA), "-Xms16m", f"-Xmx{worker_heap_mb}m",
                    "--enable-native-access=ALL-UNNAMED",
                    f"-Djava.library.path={ROOT / 'GPUsim/lib'}",
                    "-cp", f"{ROOT / 'GPUsim/out/production/gpuworkflowsim'}:{ROOT / 'GPUsim/jars'}/*",
                    "backend.SimEngine", str(worker_dir), str(hosts_path),
                    str(input_dir / "empty.json"), str(input_dir / "empty.json"),
                    "-1", "false", "false", "0", "worker",
                ],
                gpusim_env,
                discard_output=quiet_worker_logs,
            )
            gpusim_processes.append((worker, process))

        orchestrator_env = base_env.copy()
        orchestrator_env.update({
            "FNCS_CONFIG_FILE": str(orchestrator_zpl),
            "FNCS_NAME": "orchestrator",
            "COSIM_RUN_ID": run_id,
        })
        orchestrator_command = [
            sys.executable, str(ORCHESTRATOR), str(workflow),
            "--event-log", str(event_log),
            "--optimization-report", str(optimization),
            "--steady-start-marker", str(steady_start_marker),
        ]
        if horizontal_optimization:
            orchestrator_command.extend([
                "--compute-worker-map", str(worker_map_path),
                "--active-dependency-coordination",
            ])
            if safe_frontiers:
                orchestrator_command.append("--safe-frontier-coordination")
            if frontier_update_quantum_ns:
                orchestrator_command.extend([
                    "--frontier-update-quantum-ns",
                    str(frontier_update_quantum_ns),
                ])
        orchestrator = start("orchestrator", orchestrator_command, orchestrator_env)

        deadline = time.monotonic() + timeout
        while orchestrator.poll() is None:
            if steady_started is None and steady_start_marker.exists():
                steady_started = time.perf_counter()
            for name, worker in [*gpusim_processes, ("ns3", ns3), ("broker", broker)]:
                return_code = worker.poll()
                if return_code is None:
                    continue
                terminal_published = event_log.exists() and (
                    '"kind":"control.end"' in tail(event_log, 2)
                )
                if not terminal_published:
                    raise RuntimeError(
                        f"{name} exited early ({return_code}) while orchestrator was running: "
                        f"{tail(result_dir / f'{name}.log')}"
                    )
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(orchestrator.args, timeout)
            time.sleep(0.05)
        if orchestrator.returncode != 0:
            raise RuntimeError(f"orchestrator exited {orchestrator.returncode}: {tail(result_dir / 'orchestrator.log')}")
        for name, worker in [*gpusim_processes, ("ns3", ns3)]:
            worker.wait(timeout=60)
            if worker.returncode != 0:
                raise RuntimeError(f"{name} exited {worker.returncode}: {tail(result_dir / f'{name}.log')}")
        broker.wait(timeout=20)
        if broker.returncode != 0:
            raise RuntimeError(f"broker exited {broker.returncode}: {tail(result_dir / 'broker.log')}")
    finally:
        wall_seconds = time.perf_counter() - started
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in streams:
            stream.close()

    if steady_started is None:
        raise RuntimeError("orchestrator did not publish a steady-start marker")
    startup_seconds = steady_started - started
    steady_wall_seconds = wall_seconds - startup_seconds

    records = [json.loads(line) for line in event_log.read_text().splitlines() if line.strip()]
    kinds: Counter[str] = Counter()
    topic_batches: Counter[str] = Counter()
    network_trace = []
    simulated_makespan_ns = None
    terminal_summary = None
    for record in records:
        topic_batches[f"{record['direction']}:{record['topic']}"] += 1
        batch = record.get("value")
        if not isinstance(batch, dict):
            continue
        for event in batch.get("events", []):
            kinds[event["kind"]] += 1
            if event["kind"] == "network.completed":
                network_trace.append((
                    event["correlation_id"],
                    int(event["payload"].get("finish_time_ns", record["logical_time_ns"])),
                    event["payload"].get("status"),
                ))
            if event["kind"] == "control.end":
                simulated_makespan_ns = int(record["logical_time_ns"])
                terminal_summary = event["payload"].get("summary")
    trace_json = json.dumps(sorted(network_trace), separators=(",", ":"))
    identity_json = json.dumps(
        sorted((correlation_id, status) for correlation_id, _, status in network_trace),
        separators=(",", ":"),
    )
    semantic_json = json.dumps(
        sorted(
            (normalize_network_correlation(correlation_id), status)
            for correlation_id, _, status in network_trace
        ),
        separators=(",", ":"),
    )
    coordination = json.loads(metrics.read_text())
    optimization_data = json.loads(optimization.read_text())
    result = {
        "label": label,
        "repeat": repeat,
        "vertical_optimization": acceleration,
        "horizontal_optimization": horizontal_optimization,
        "critical_path_event_acceleration": acceleration,
        "federate_partition": partition_strategy,
        "compute_federates": len(compute_workers),
        "ranks_per_compute_federate": (
            ranks_per_federate
            if horizontal_optimization and partition_map_path is None
            else host_count if not horizontal_optimization else None
        ),
        "active_dependency_coordination": horizontal_optimization,
        "safe_frontier_coordination": horizontal_optimization and safe_frontiers,
        "compute_federate_rank_counts": {
            worker: len(hosts_by_worker[worker]) for worker in compute_workers
        },
        "worker_logs_discarded": quiet_worker_logs,
        "wall_clock_seconds": wall_seconds,
        "startup_seconds": startup_seconds,
        "steady_wall_clock_seconds": steady_wall_seconds,
        "frontier_update_quantum_ns": frontier_update_quantum_ns,
        "simulated_makespan_ns": simulated_makespan_ns,
        "scheduler_rounds": coordination["scheduler_rounds"],
        "total_grants": coordination["total_grants"],
        "asynchronous_grants": coordination.get("asynchronous_grants", 0),
        "grants_by_simulator": coordination["grants_by_simulator"],
        "dependency_updates": coordination["dependency_updates"],
        "control_plane_records": len(records),
        "event_kinds": dict(sorted(kinds.items())),
        "topic_batches": dict(sorted(topic_batches.items())),
        "network_completion_count": len(network_trace),
        "network_trace_sha256": hashlib.sha256(trace_json.encode()).hexdigest(),
        "network_identity_sha256": hashlib.sha256(identity_json.encode()).hexdigest(),
        "network_semantic_sha256": hashlib.sha256(semantic_json.encode()).hexdigest(),
        "optimization": optimization_data,
        "terminal_summary": terminal_summary,
    }
    json_dump(result_dir / "result.json", result)
    return result


def summarize(output: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["label"]].append(result)
    cells = {}
    for label, samples in sorted(grouped.items()):
        walls = [sample["wall_clock_seconds"] for sample in samples]
        startups = [sample["startup_seconds"] for sample in samples]
        steady_walls = [sample["steady_wall_clock_seconds"] for sample in samples]
        rounds = [sample["scheduler_rounds"] for sample in samples]
        grants = [sample["total_grants"] for sample in samples]
        asynchronous_grants = [sample.get("asynchronous_grants", 0) for sample in samples]
        cells[label] = {
            "vertical_optimization": samples[0]["vertical_optimization"],
            "horizontal_optimization": samples[0]["horizontal_optimization"],
            "federate_partition": samples[0]["federate_partition"],
            "compute_federates": samples[0]["compute_federates"],
            "active_dependency_coordination": samples[0]["active_dependency_coordination"],
            "samples": len(samples),
            "wall_clock_seconds": walls,
            "startup_seconds": startups,
            "startup_median_seconds": statistics.median(startups),
            "steady_wall_clock_seconds": steady_walls,
            "steady_wall_clock_median_seconds": statistics.median(steady_walls),
            "steady_wall_clock_mean_seconds": statistics.mean(steady_walls),
            "steady_wall_clock_stdev_seconds": statistics.stdev(steady_walls) if len(steady_walls) > 1 else 0.0,
            "wall_clock_median_seconds": statistics.median(walls),
            "wall_clock_mean_seconds": statistics.mean(walls),
            "wall_clock_stdev_seconds": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "scheduler_rounds": rounds,
            "scheduler_rounds_median": statistics.median(rounds),
            "total_grants": grants,
            "total_grants_median": statistics.median(grants),
            "asynchronous_grants": asynchronous_grants,
            "asynchronous_grants_median": statistics.median(asynchronous_grants),
            "dependency_updates": sorted({sample["dependency_updates"] for sample in samples}),
            "control_plane_records": sorted({sample["control_plane_records"] for sample in samples}),
            "simulated_makespans_ns": sorted({sample["simulated_makespan_ns"] for sample in samples}),
            "network_trace_hashes": sorted({sample["network_trace_sha256"] for sample in samples}),
            "network_identity_hashes": sorted({sample["network_identity_sha256"] for sample in samples}),
            "network_semantic_hashes": sorted({sample["network_semantic_sha256"] for sample in samples}),
            "compute_dispatch_events": samples[0]["event_kinds"].get("compute.dispatch", 0),
            "compute_plan_dispatch_events": samples[0]["event_kinds"].get("compute.plan.dispatch", 0),
            "network_dispatch_events": samples[0]["event_kinds"].get("network.dispatch", 0),
            "network_completion_count": samples[0]["network_completion_count"],
        }

    baseline = cells["vertical_off__horizontal_off"]["wall_clock_median_seconds"]
    for cell in cells.values():
        cell["wall_clock_speedup_vs_all_off"] = baseline / cell["wall_clock_median_seconds"]

    makespans = {
        value
        for cell in cells.values()
        for value in cell["simulated_makespans_ns"]
    }
    network_hashes = {
        value
        for cell in cells.values()
        for value in cell["network_trace_hashes"]
    }
    network_identity_hashes = {
        value
        for cell in cells.values()
        for value in cell["network_identity_hashes"]
    }
    network_semantic_hashes = {
        value
        for cell in cells.values()
        for value in cell["network_semantic_hashes"]
    }
    makespan_span_ns = max(makespans) - min(makespans)
    makespan_relative_span = makespan_span_ns / max(makespans)
    summary = {
        "dataset": json.loads((output / "dataset.json").read_text()),
        "cells": cells,
        "validation": {
            "simulated_makespan_preserved": len(makespans) == 1,
            "simulated_makespan_within_1ppm": makespan_relative_span <= 1e-6,
            "simulated_makespan_within_10ppm": makespan_relative_span <= 1e-5,
            "simulated_makespan_span_ns": makespan_span_ns,
            "simulated_makespan_relative_span": makespan_relative_span,
            "network_completion_trace_preserved": len(network_hashes) == 1,
            "network_completion_identity_preserved": len(network_identity_hashes) == 1,
            "network_completion_semantics_preserved": len(network_semantic_hashes) == 1,
            "simulated_makespans_ns": sorted(makespans),
            "network_trace_hashes": sorted(network_hashes),
            "network_identity_hashes": sorted(network_identity_hashes),
            "network_semantic_hashes": sorted(network_semantic_hashes),
        },
    }
    json_dump(output / "summary.json", summary)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "vertical_optimization",
                "horizontal_optimization",
                "samples",
                "wall_clock_seconds",
                "startup_seconds",
                "steady_wall_clock_seconds",
                "wall_clock_median_seconds",
                "wall_clock_mean_seconds",
                "wall_clock_stdev_seconds",
                "speedup_vs_both_off",
                "compute_dispatch_events",
                "scheduler_rounds_median",
                "total_grants_median",
                "asynchronous_grants_median",
                "network_completion_count",
                "simulated_makespan_ns",
            ]
        )
        for label in (
            "vertical_off__horizontal_off",
            "vertical_off__horizontal_on",
            "vertical_on__horizontal_off",
            "vertical_on__horizontal_on",
        ):
            cell = cells[label]
            writer.writerow(
                [
                    cell["vertical_optimization"],
                    cell["horizontal_optimization"],
                    cell["samples"],
                    json.dumps(cell["wall_clock_seconds"], separators=(",", ":")),
                    json.dumps(cell["startup_seconds"], separators=(",", ":")),
                    json.dumps(cell["steady_wall_clock_seconds"], separators=(",", ":")),
                    cell["wall_clock_median_seconds"],
                    cell["wall_clock_mean_seconds"],
                    cell["wall_clock_stdev_seconds"],
                    cell["wall_clock_speedup_vs_all_off"],
                    cell["compute_dispatch_events"],
                    cell["scheduler_rounds_median"],
                    cell["total_grants_median"],
                    cell["asynchronous_grants_median"],
                    cell["network_completion_count"],
                    cell["simulated_makespans_ns"][0],
                ]
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--events-per-phase", type=int, default=8)
    parser.add_argument("--bandwidth-gbps", type=float, default=56.0)
    parser.add_argument("--delay-ns", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-start",
        type=int,
        default=1,
        help="first repeat to execute; prior result.json files are reused",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--ranks-per-federate",
        type=int,
        default=1,
        help="number of trace ranks hosted by each horizontal GPUSim Federate",
    )
    parser.add_argument(
        "--compute-partition-map",
        type=Path,
        help=(
            "JSON object mapping every host to a partition ID; use this to "
            "keep independent jobs or DAG components in separate Federates"
        ),
    )
    parser.add_argument(
        "--no-safe-frontiers",
        action="store_true",
        help="disable certified frontier requests in horizontal cells",
    )
    parser.add_argument(
        "--frontier-update-quantum-ns",
        type=int,
        default=0,
        help="round certified frontiers down before publishing (0 keeps exact updates)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument(
        "--quiet-worker-logs",
        action="store_true",
        help="discard verbose ns-3/GPUSim stdout during timing runs",
    )
    args = parser.parse_args()
    if args.repeat_start < 1 or args.repeat_start > args.repeats:
        parser.error("--repeat-start must be between 1 and --repeats")

    if not args.run_only:
        metadata = prepare_workload(
            args.goal.resolve(), args.output.resolve(), args.phases,
            args.events_per_phase, args.bandwidth_gbps, args.delay_ns,
        )
        print(json.dumps(metadata, sort_keys=True))
    if args.prepare_only:
        return 0

    for executable in (BROKER, NS3, JAVA):
        if not executable.exists():
            raise FileNotFoundError(executable)

    results = []
    configurations = [(False, False), (True, False), (False, True), (True, True)]
    for repeat in range(1, args.repeat_start):
        for acceleration, horizontal_optimization in configurations:
            label = (
                f"vertical_{'on' if acceleration else 'off'}__"
                f"horizontal_{'on' if horizontal_optimization else 'off'}"
            )
            result_path = (
                args.output.resolve() / "runs" / label /
                f"repeat_{repeat}" / "result.json"
            )
            if not result_path.exists():
                raise FileNotFoundError(
                    f"cannot resume: missing prior result {result_path}"
                )
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
    sequence = 0
    for repeat in range(args.repeat_start, args.repeats + 1):
        ordered = configurations[repeat - 1:] + configurations[:repeat - 1]
        for acceleration, horizontal_optimization in ordered:
            sequence += 1
            label = (
                f"vertical_{'on' if acceleration else 'off'}__"
                f"horizontal_{'on' if horizontal_optimization else 'off'}"
            )
            print(f"running {label} repeat={repeat}", flush=True)
            result = run_one(
                args.output.resolve(), acceleration, horizontal_optimization, repeat,
                5700 + sequence, args.timeout, args.quiet_worker_logs,
                args.ranks_per_federate, not args.no_safe_frontiers,
                args.compute_partition_map.resolve()
                if args.compute_partition_map else None,
                args.frontier_update_quantum_ns,
            )
            results.append(result)
            print(json.dumps({
                "label": label,
                "repeat": repeat,
                "wall_clock_seconds": result["wall_clock_seconds"],
                "scheduler_rounds": result["scheduler_rounds"],
                "simulated_makespan_ns": result["simulated_makespan_ns"],
            }, sort_keys=True), flush=True)

    summary = summarize(args.output.resolve(), results)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
