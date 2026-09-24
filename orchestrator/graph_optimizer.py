"""Closed-region contraction and critical-path-guided refinement."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Sequence, Tuple


class OptimizationError(ValueError):
    pass


_CLOSURE_FIELDS = (
    "causal",
    "state",
    "temporal",
    "resource_non_interference",
)


def _bounds(value: Any, label: str) -> Tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise OptimizationError(f"{label} must be [lower, upper]")
    lower, upper = value
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise OptimizationError(f"{label} values must be integers")
    if lower < 0 or upper < lower:
        raise OptimizationError(f"invalid {label}: {value!r}")
    return lower, upper


def task_duration_bounds(task: Dict[str, Any]) -> Tuple[int, int]:
    name = str(task.get("name") or "<unnamed>")
    if "duration_bounds_ns" in task:
        return _bounds(task["duration_bounds_ns"], f"task {name} duration_bounds_ns")
    duration = task.get("estimated_duration_ns", task.get("simulated_duration_ns"))
    if isinstance(duration, int) and duration >= 0:
        return duration, duration
    raise OptimizationError(
        f"task {name} needs duration_bounds_ns or estimated_duration_ns"
    )


def edge_duration_bounds(child: Dict[str, Any], src: str, dst: str) -> Tuple[int, int]:
    if "latency_bounds_ns" in child:
        return _bounds(child["latency_bounds_ns"], f"edge {src}->{dst} latency_bounds_ns")
    if int(child.get("size") or 0) == 0:
        return 0, 0
    raise OptimizationError(
        f"network edge {src}->{dst} needs latency_bounds_ns for certification"
    )


def _index_tasks(tasks: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for raw in tasks:
        task = copy.deepcopy(raw)
        name = str(task.get("name") or "")
        if not name or name in indexed:
            raise OptimizationError(f"invalid or duplicate task name: {name!r}")
        indexed[name] = task
    return indexed


def _topology(tasks: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[Tuple[str, Dict[str, Any]]]]]:
    incoming: Dict[str, int] = {name: 0 for name in tasks}
    outgoing: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for src, task in tasks.items():
        for child in task.get("children") or []:
            dst = str(child.get("child") or "")
            if dst not in tasks:
                raise OptimizationError(f"task {src} references missing child {dst}")
            outgoing[src].append((dst, child))
            incoming[dst] += 1
    ready = deque(sorted(name for name, count in incoming.items() if count == 0))
    order: List[str] = []
    while ready:
        src = ready.popleft()
        order.append(src)
        for dst, _ in outgoing[src]:
            incoming[dst] -= 1
            if incoming[dst] == 0:
                ready.append(dst)
    if len(order) != len(tasks):
        raise OptimizationError("workflow contains a cycle")
    return order, outgoing


def longest_path_certificate(tasks_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = _index_tasks(tasks_list)
    order, outgoing = _topology(tasks)
    lower_finish: Dict[str, int] = {}
    upper_finish: Dict[str, int] = {}
    upper_start: Dict[str, int] = {name: 0 for name in tasks}
    lower_start: Dict[str, int] = {name: 0 for name in tasks}

    for name in order:
        node_lower, node_upper = task_duration_bounds(tasks[name])
        lower_finish[name] = lower_start[name] + node_lower
        upper_finish[name] = upper_start[name] + node_upper
        for dst, child in outgoing[name]:
            edge_lower, edge_upper = edge_duration_bounds(child, name, dst)
            lower_start[dst] = max(lower_start[dst], lower_finish[name] + edge_lower)
            upper_start[dst] = max(upper_start[dst], upper_finish[name] + edge_upper)

    suffix_upper: Dict[str, int] = {}
    for name in reversed(order):
        node_upper = task_duration_bounds(tasks[name])[1]
        continuation = 0
        for dst, child in outgoing[name]:
            edge_upper = edge_duration_bounds(child, name, dst)[1]
            continuation = max(continuation, edge_upper + suffix_upper[dst])
        suffix_upper[name] = node_upper + continuation

    lower = max(lower_finish.values(), default=0)
    upper = max(upper_finish.values(), default=0)
    through_upper = {
        name: upper_start[name] + suffix_upper[name]
        for name in tasks
    }
    return {
        "lower_ns": lower,
        "upper_ns": upper,
        "gap_ns": upper - lower,
        "upper_path_through_ns": through_upper,
    }


def _validate_region(
    region: Dict[str, Any],
    tasks: Dict[str, Dict[str, Any]],
    collective_members: set[str],
    incoming_by_destination: Dict[str, List[str]],
) -> None:
    region_id = str(region.get("id") or "")
    members = region.get("members")
    if not region_id or not isinstance(members, list) or len(members) < 2:
        raise OptimizationError("each acceleration region needs an id and at least two members")
    if len(set(members)) != len(members) or any(member not in tasks for member in members):
        raise OptimizationError(f"region {region_id} has invalid members")
    closure = region.get("closure") or {}
    if any(closure.get(field) is not True for field in _CLOSURE_FIELDS):
        raise OptimizationError(f"region {region_id} does not satisfy all closure conditions")
    if collective_members.intersection(members):
        raise OptimizationError(f"region {region_id} crosses a collective boundary")

    member_set = set(members)
    host = tasks[members[0]].get("host")
    if not host or any(tasks[member].get("host") != host for member in members):
        raise OptimizationError(f"region {region_id} must stay on one host")

    internal_edges: List[Tuple[str, str]] = []
    incoming: List[Tuple[str, str]] = []
    outgoing: List[Tuple[str, str]] = []
    for src in members:
        task = tasks[src]
        for child in task.get("children") or []:
            dst = str(child.get("child") or "")
            if dst in member_set:
                if int(child.get("size") or 0) != 0:
                    raise OptimizationError(f"region {region_id} contains network communication")
                internal_edges.append((src, dst))
            else:
                outgoing.append((src, dst))
    for dst in members:
        incoming.extend(
            (src, dst)
            for src in incoming_by_destination.get(dst, [])
            if src not in member_set
        )
    expected = list(zip(members, members[1:]))
    if internal_edges != expected:
        raise OptimizationError(f"region {region_id} is not the declared linear chain")
    if any(dst != members[0] for _, dst in incoming):
        raise OptimizationError(f"region {region_id} has an external edge into its interior")
    if any(src != members[-1] for src, _ in outgoing):
        raise OptimizationError(f"region {region_id} has an external edge out of its interior")


def _merge_native_tasks(region: Dict[str, Any], tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    members = region["members"]
    first = tasks[members[0]]
    last = tasks[members[-1]]
    macro = copy.deepcopy(region.get("macro_task") or first)
    macro["name"] = f"macro:{region['id']}"
    macro["host"] = first["host"]
    macro["children"] = copy.deepcopy(last.get("children") or [])

    if "macro_task" not in region:
        gpu_modes = [bool((tasks[name].get("gpu_task") or {}).get("kernels")) for name in members]
        if any(gpu_modes) and not all(gpu_modes):
            raise OptimizationError(f"region {region['id']} mixes CPU and GPU tasks")
        if all(gpu_modes):
            kernels: List[Dict[str, Any]] = []
            for name in members:
                kernels.extend(copy.deepcopy(tasks[name]["gpu_task"]["kernels"]))
            macro["gpu_task"] = copy.deepcopy(first["gpu_task"])
            macro["gpu_task"]["kernels"] = kernels
        else:
            cpu_tasks = [tasks[name].get("cpu_task") or {} for name in members]
            if not all(isinstance(cpu.get("length"), int) for cpu in cpu_tasks):
                raise OptimizationError(f"region {region['id']} CPU tasks need integer lengths")
            reference = {key: value for key, value in cpu_tasks[0].items() if key != "length"}
            for cpu in cpu_tasks[1:]:
                if {key: value for key, value in cpu.items() if key != "length"} != reference:
                    raise OptimizationError(f"region {region['id']} CPU resource shapes differ")
            macro["cpu_task"] = copy.deepcopy(cpu_tasks[0])
            macro["cpu_task"]["length"] = sum(cpu["length"] for cpu in cpu_tasks)
            macro.setdefault("gpu_task", {"kernels": []})

    lower, upper = _bounds(region["duration_bounds_ns"], f"region {region['id']} duration_bounds_ns")
    estimate = region.get("estimated_duration_ns", (lower + upper) // 2)
    if not isinstance(estimate, int) or not lower <= estimate <= upper:
        raise OptimizationError(f"region {region['id']} has an invalid estimated_duration_ns")
    macro["duration_bounds_ns"] = [lower, upper]
    macro["estimated_duration_ns"] = estimate
    macro["simulated_duration_ns"] = estimate
    macro["_cosim_macro"] = {
        "region_id": region["id"],
        "members": list(members),
        "mode": region.get("mode", "approximate"),
    }
    return macro


def _contract(tasks_list: Sequence[Dict[str, Any]], region: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = _index_tasks(tasks_list)
    members = set(region["members"])
    entry = region["members"][0]
    macro = _merge_native_tasks(region, tasks)
    macro_name = macro["name"]
    result: List[Dict[str, Any]] = []
    for name, task in tasks.items():
        if name in members:
            continue
        rewritten = copy.deepcopy(task)
        for child in rewritten.get("children") or []:
            if child.get("child") == entry:
                child["child"] = macro_name
                child["dst_host"] = macro["host"]
        result.append(rewritten)
    result.append(macro)
    return result


def _contract_regions(
    tasks_list: Sequence[Dict[str, Any]],
    regions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Contract disjoint regions in one graph pass.

    Region validation guarantees that external edges enter only at a region's
    first member and leave only from its last member.  Rewriting every edge
    against the complete member-to-macro map is therefore equivalent to
    applying ``_contract`` repeatedly, while avoiding one full graph copy and
    index construction per region.
    """
    tasks = _index_tasks(tasks_list)
    member_to_macro: Dict[str, str] = {}
    macros: List[Dict[str, Any]] = []
    for region in regions:
        macro = _merge_native_tasks(region, tasks)
        macros.append(macro)
        for member in region["members"]:
            member_to_macro[member] = macro["name"]

    def rewrite_children(task: Dict[str, Any]) -> Dict[str, Any]:
        rewritten = copy.deepcopy(task)
        for child in rewritten.get("children") or []:
            destination = str(child.get("child") or "")
            macro_name = member_to_macro.get(destination)
            if macro_name is not None:
                child["child"] = macro_name
                child["dst_host"] = tasks[destination]["host"]
        return rewritten

    result = [
        rewrite_children(task)
        for name, task in tasks.items()
        if name not in member_to_macro
    ]
    result.extend(rewrite_children(macro) for macro in macros)
    return result


def optimize_workflow(
    tasks_list: Sequence[Dict[str, Any]],
    collectives: Sequence[Dict[str, Any]],
    acceleration: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not acceleration or not acceleration.get("enabled", True):
        return list(copy.deepcopy(tasks_list)), {"enabled": False}
    error_budget = acceleration.get("error_budget_ns", 0)
    if not isinstance(error_budget, int) or error_budget < 0:
        raise OptimizationError("error_budget_ns must be a non-negative integer")
    regions = copy.deepcopy(acceleration.get("regions") or [])
    if not isinstance(regions, list):
        raise OptimizationError("acceleration regions must be an array")

    original = _index_tasks(tasks_list)
    incoming_by_destination: Dict[str, List[str]] = defaultdict(list)
    for src, task in original.items():
        for child in task.get("children") or []:
            incoming_by_destination[str(child.get("child") or "")].append(src)
    collective_members = {
        str(participant.get(field) or "")
        for collective in collectives
        for participant in collective.get("participants") or []
        for field in ("src_task_id", "dst_task_id")
    }
    occupied: set[str] = set()
    by_id: Dict[str, Dict[str, Any]] = {}
    for region in regions:
        _validate_region(
            region, original, collective_members, incoming_by_destination
        )
        region_id = str(region["id"])
        if region_id in by_id or occupied.intersection(region["members"]):
            raise OptimizationError(f"region {region_id} is duplicate or overlaps another region")
        mode = str(region.get("mode") or "approximate")
        if mode not in {"exact", "approximate"}:
            raise OptimizationError(f"region {region_id} has invalid mode {mode!r}")
        region["mode"] = mode
        if "duration_bounds_ns" not in region:
            detailed = [task_duration_bounds(original[name]) for name in region["members"]]
            lower = sum(value[0] for value in detailed)
            upper = sum(value[1] for value in detailed)
            region["duration_bounds_ns"] = [lower, upper]
        lower, upper = _bounds(region["duration_bounds_ns"], f"region {region_id} duration_bounds_ns")
        if mode == "exact" and lower != upper:
            raise OptimizationError(f"exact region {region_id} must have equal duration bounds")
        if mode == "exact":
            detailed = [task_duration_bounds(original[name]) for name in region["members"]]
            if any(value[0] != value[1] for value in detailed):
                raise OptimizationError(f"exact region {region_id} contains uncertain tasks")
            detailed_duration = sum(value[0] for value in detailed)
            if lower != detailed_duration:
                raise OptimizationError(
                    f"exact region {region_id} duration {lower} does not match "
                    f"detailed duration {detailed_duration}"
                )
        cost = region.get("refine_cost", len(region["members"]) - 1)
        if not isinstance(cost, (int, float)) or cost <= 0:
            raise OptimizationError(f"region {region_id} has invalid refine_cost")
        region["refine_cost"] = float(cost)
        by_id[region_id] = region
        occupied.update(region["members"])

    retained = set(by_id)
    refined: List[str] = []

    def materialize() -> List[Dict[str, Any]]:
        regions_to_contract = [by_id[region_id] for region_id in sorted(retained)]
        return _contract_regions(tasks_list, regions_to_contract)

    graph = materialize()
    initial = longest_path_certificate(graph)
    certificate = initial
    iterations: List[Dict[str, Any]] = []
    while certificate["gap_ns"] > error_budget:
        candidates: List[Tuple[float, str, int]] = []
        for region_id in sorted(retained):
            region = by_id[region_id]
            if region["mode"] == "exact":
                continue
            macro_name = f"macro:{region_id}"
            possible_path = certificate["upper_path_through_ns"][macro_name]
            if possible_path < certificate["lower_ns"]:
                continue
            lower, upper = _bounds(region["duration_bounds_ns"], "duration_bounds_ns")
            benefit = upper - lower
            candidates.append((benefit / region["refine_cost"], region_id, possible_path))
        if not candidates:
            break
        _, selected, possible_path = max(candidates, key=lambda item: (item[0], item[1]))
        before = certificate
        retained.remove(selected)
        refined.append(selected)
        graph = materialize()
        certificate = longest_path_certificate(graph)
        iterations.append(
            {
                "region_id": selected,
                "possible_path_upper_ns": possible_path,
                "before_gap_ns": before["gap_ns"],
                "after_gap_ns": certificate["gap_ns"],
            }
        )

    report = {
        "enabled": True,
        "error_budget_ns": error_budget,
        "original_task_count": len(tasks_list),
        "optimized_task_count": len(graph),
        "initial_certificate": {
            key: initial[key] for key in ("lower_ns", "upper_ns", "gap_ns")
        },
        "final_certificate": {
            key: certificate[key] for key in ("lower_ns", "upper_ns", "gap_ns")
        },
        "budget_satisfied": certificate["gap_ns"] <= error_budget,
        "refined_regions": refined,
        "retained_regions": sorted(retained),
        "iterations": iterations,
    }
    return graph, report
