#!/usr/bin/env python3
"""FNCS federate that owns workflow DAG state and dispatches worker events."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, TextIO

from fncs_client import FncsClient
from graph_optimizer import OptimizationError, optimize_workflow


SCHEMA_VERSION = "2.0"
MAX_TIME_NS = (1 << 63) - 1

COMPUTE_DISPATCH = "compute/dispatch"
COMPUTE_COMPLETED = "compute/completed"
NETWORK_DISPATCH = "network/dispatch"
NETWORK_COMPLETED = "network/completed"
CONTROL = "control"
ACTIVE_DEPENDENCIES = "__fncs/active_dependencies"


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class Command:
    topic: str
    event: Dict[str, Any]


class WorkflowController:
    """Deterministic DAG and collective state machine, independent from FNCS."""

    SUPPORTED_COLLECTIVES = {"allreduce", "allgather", "reduce_scatter", "alltoall"}

    def __init__(
        self,
        task_specs: Iterable[Dict[str, Any]],
        run_id: str,
        max_compute_retries: int = 0,
        max_network_retries: int = 0,
        collective_specs: Iterable[Dict[str, Any]] | None = None,
        acceleration_report: Dict[str, Any] | None = None,
        backend_local_chains: bool = False,
        backend_local_chain_max_tasks: int = 32,
    ) -> None:
        self.run_id = run_id
        self.max_compute_retries = max(0, max_compute_retries)
        self.max_network_retries = max(0, max_network_retries)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.state: Dict[str, str] = {}
        self.parents: Dict[str, set[str]] = defaultdict(set)
        self.outgoing: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.graph_parents: Dict[str, set[str]] = defaultdict(set)
        self.graph_children: Dict[str, set[str]] = defaultdict(set)
        self.pending_inputs: Dict[str, int] = {}
        self.edge_state: Dict[str, str] = {}
        self.collectives: Dict[str, Dict[str, Any]] = {}
        self.collective_state: Dict[str, str] = {}
        self.collective_step: Dict[str, int] = {}
        self.collective_pending: Dict[str, set[str]] = defaultdict(set)
        self.collective_inputs: Dict[str, int] = defaultdict(int)
        self.task_attempt: Dict[str, int] = defaultdict(int)
        self.network_attempt: Dict[str, int] = defaultdict(int)
        self.transfer_context: Dict[str, Dict[str, Any]] = {}
        self.inflight_compute: set[str] = set()
        self.inflight_network: set[str] = set()
        self.seen_event_ids: set[str] = set()
        self.failed_reason: str | None = None
        self.acceleration_report = acceleration_report or {"enabled": False}
        self.backend_local_chains = backend_local_chains
        self.backend_local_chain_max_tasks = max(0, backend_local_chain_max_tasks)
        self.local_plans: Dict[str, List[str]] = {}
        self.local_plan_by_entry: Dict[str, str] = {}
        self.local_plan_by_task: Dict[str, str] = {}
        self.local_plan_dispatches = 0
        self._event_sequence = 0

        for raw_spec in task_specs:
            spec = copy.deepcopy(raw_spec)
            task_id = str(spec.get("name", ""))
            if not task_id or task_id in self.tasks:
                raise WorkflowError(f"invalid or duplicate task name: {task_id!r}")
            if not spec.get("host"):
                raise WorkflowError(f"task {task_id} is missing host placement")
            self.tasks[task_id] = spec
            self.state[task_id] = "BLOCKED"

        if not self.tasks:
            raise WorkflowError("workflow must contain at least one task")
        self._build_edges()
        self._build_collectives(collective_specs or [])
        self._validate_acyclic()
        if self.backend_local_chains:
            self._build_backend_local_chains()
        for task_id in self.tasks:
            self.pending_inputs[task_id] = (
                len(self.parents[task_id]) + self.collective_inputs[task_id]
            )
            if self.pending_inputs[task_id] == 0:
                self.state[task_id] = "READY"

    @classmethod
    def from_file(
        cls,
        path: str,
        run_id: str,
        max_compute_retries: int = 0,
        max_network_retries: int = 0,
        backend_local_chains: bool = False,
        backend_local_chain_max_tasks: int = 32,
    ) -> "WorkflowController":
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        collectives: Iterable[Dict[str, Any]] = []
        if isinstance(data, dict):
            tasks = data.get("tasks", [])
            collectives = data.get("collectives", [])
            acceleration = data.get("acceleration")
        else:
            tasks = data
            acceleration = None
        if not isinstance(tasks, list) or not isinstance(collectives, list):
            raise WorkflowError("workflow tasks and collectives must be arrays")
        try:
            tasks, acceleration_report = optimize_workflow(
                tasks, collectives, acceleration
            )
        except OptimizationError as error:
            raise WorkflowError(f"invalid acceleration configuration: {error}") from error
        return cls(
            tasks,
            run_id,
            max_compute_retries=max_compute_retries,
            max_network_retries=max_network_retries,
            collective_specs=collectives,
            acceleration_report=acceleration_report,
            backend_local_chains=backend_local_chains,
            backend_local_chain_max_tasks=backend_local_chain_max_tasks,
        )

    def _path_exists(self, source: str, destination: str) -> bool:
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == destination:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.graph_children[current] - visited)
        return False

    def _build_backend_local_chains(self) -> None:
        """Find exact local chains whose host resources cannot interleave."""
        collective_members = {
            participant[field]
            for collective in self.collectives.values()
            for participant in collective["participants"]
            for field in ("src_task_id", "dst_task_id")
        }
        tasks_by_host: Dict[str, List[str]] = defaultdict(list)
        for task_id, task in self.tasks.items():
            tasks_by_host[str(task["host"])].append(task_id)

        safe_hosts: set[str] = set()
        for host, host_tasks in tasks_by_host.items():
            if all(
                self._path_exists(left, right) or self._path_exists(right, left)
                for index, left in enumerate(host_tasks)
                for right in host_tasks[index + 1 :]
            ):
                safe_hosts.add(host)

        eligible_next: Dict[str, str] = {}
        eligible_previous: Dict[str, str] = {}
        for source, edges in self.outgoing.items():
            if len(edges) != 1 or source in collective_members:
                continue
            edge = edges[0]
            destination = edge["dst_task_id"]
            host = str(self.tasks[source]["host"])
            if (
                edge["bytes"] == 0
                and edge["src_host"] == edge["dst_host"] == host
                and host in safe_hosts
                and destination not in collective_members
                and self.parents[destination] == {source}
                and int(self.tasks[source].get("max_retries") or 0) == 0
                and int(self.tasks[destination].get("max_retries") or 0) == 0
            ):
                eligible_next[source] = destination
                eligible_previous[destination] = source

        sequence = 0
        for start in sorted(eligible_next):
            if start in eligible_previous:
                continue
            chain = [start]
            while chain[-1] in eligible_next:
                chain.append(eligible_next[chain[-1]])
            chunk_size = self.backend_local_chain_max_tasks or len(chain)
            for offset in range(0, len(chain), chunk_size):
                members = chain[offset : offset + chunk_size]
                if len(members) < 2:
                    continue
                sequence += 1
                plan_id = f"local-plan-{sequence:06d}"
                self.local_plans[plan_id] = members
                self.local_plan_by_entry[members[0]] = plan_id
                for task_id in members:
                    self.local_plan_by_task[task_id] = plan_id

    def _build_edges(self) -> None:
        for src_id, spec in self.tasks.items():
            for index, child in enumerate(spec.get("children") or []):
                dst_id = str(child.get("child", ""))
                if dst_id not in self.tasks:
                    raise WorkflowError(f"task {src_id} references missing child {dst_id}")
                if src_id in self.parents[dst_id]:
                    raise WorkflowError(f"duplicate edge {src_id} -> {dst_id}")
                edge_id = f"edge:{src_id}:{dst_id}:{index}"
                size = int(child.get("size") or 0)
                if size < 0:
                    raise WorkflowError(f"edge {edge_id} has negative size")
                edge = {
                    "edge_id": edge_id,
                    "src_task_id": src_id,
                    "dst_task_id": dst_id,
                    "src_host": child.get("src_host") or spec.get("host") or "",
                    "dst_host": child.get("dst_host") or self.tasks[dst_id].get("host") or "",
                    "bytes": size,
                }
                if size > 0 and (not edge["src_host"] or not edge["dst_host"]):
                    raise WorkflowError(f"network edge {edge_id} is missing a host")
                self.parents[dst_id].add(src_id)
                self.graph_parents[dst_id].add(src_id)
                self.graph_children[src_id].add(dst_id)
                self.outgoing[src_id].append(edge)
                self.edge_state[edge_id] = "WAIT_PRODUCER"

    def _build_collectives(self, specs: Iterable[Dict[str, Any]]) -> None:
        aliases = {
            "allreduce": "allreduce",
            "allgather": "allgather",
            "reducescatter": "reduce_scatter",
            "alltoall": "alltoall",
        }
        for raw_spec in specs:
            spec = copy.deepcopy(raw_spec)
            collective_id = str(spec.get("collective_id") or spec.get("id") or "")
            raw_type = str(spec.get("type") or "").lower().replace("_", "")
            collective_type = aliases.get(raw_type, raw_type)
            if not collective_id or collective_id in self.collectives:
                raise WorkflowError(f"invalid or duplicate collective id: {collective_id!r}")
            if collective_type not in self.SUPPORTED_COLLECTIVES:
                raise WorkflowError(f"unsupported collective type: {collective_type!r}")
            if str(spec.get("algorithm") or "ring").lower() != "ring":
                raise WorkflowError(f"collective {collective_id} only supports ring")

            participants = spec.get("participants") or []
            if not isinstance(participants, list) or len(participants) < 2:
                raise WorkflowError(f"collective {collective_id} needs at least two ranks")
            normalized: List[Dict[str, str]] = []
            source_ids: set[str] = set()
            target_ids: set[str] = set()
            for participant in participants:
                src_id = str(participant.get("src_task_id") or "")
                dst_id = str(participant.get("dst_task_id") or "")
                host = str(
                    participant.get("host")
                    or self.tasks.get(src_id, {}).get("host")
                    or ""
                )
                if src_id not in self.tasks or dst_id not in self.tasks or not host:
                    raise WorkflowError(
                        f"collective {collective_id} has invalid participant {participant!r}"
                    )
                if src_id in source_ids or dst_id in target_ids:
                    raise WorkflowError(
                        f"collective {collective_id} repeats a source or target task"
                    )
                source_ids.add(src_id)
                target_ids.add(dst_id)
                normalized.append(
                    {"src_task_id": src_id, "dst_task_id": dst_id, "host": host}
                )

            bytes_per_rank = int(spec.get("bytes_per_rank") or spec.get("bytes") or 0)
            if bytes_per_rank <= 0:
                raise WorkflowError(
                    f"collective {collective_id} must have positive bytes_per_rank"
                )
            spec.update(
                {
                    "collective_id": collective_id,
                    "type": collective_type,
                    "algorithm": "ring",
                    "bytes_per_rank": bytes_per_rank,
                    "participants": normalized,
                }
            )
            self.collectives[collective_id] = spec
            self.collective_state[collective_id] = "WAIT_SOURCES"
            self.collective_step[collective_id] = 0
            for dst_id in target_ids:
                self.collective_inputs[dst_id] += 1
                for src_id in source_ids:
                    self.graph_parents[dst_id].add(src_id)
                    self.graph_children[src_id].add(dst_id)

    def _validate_acyclic(self) -> None:
        degree = {task_id: len(self.graph_parents[task_id]) for task_id in self.tasks}
        ready = deque(task_id for task_id, count in degree.items() if count == 0)
        visited = 0
        while ready:
            src_id = ready.popleft()
            visited += 1
            for dst_id in self.graph_children[src_id]:
                degree[dst_id] -= 1
                if degree[dst_id] == 0:
                    ready.append(dst_id)
        if visited != len(self.tasks):
            raise WorkflowError("workflow contains a cycle")

    def _new_event(
        self, kind: str, correlation_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._event_sequence += 1
        return {
            "event_id": f"evt-{self._event_sequence:09d}",
            "kind": kind,
            "correlation_id": correlation_id,
            "payload": payload,
        }

    def initial_commands(self) -> List[Command]:
        return self._dispatch_ready_tasks()

    def _dispatch_ready_tasks(self) -> List[Command]:
        commands: List[Command] = []
        for task_id in sorted(self.tasks):
            if self.state[task_id] != "READY":
                continue
            plan_id = self.local_plan_by_entry.get(task_id)
            if plan_id:
                commands.append(self._dispatch_local_plan(plan_id))
                continue
            self.task_attempt[task_id] += 1
            attempt = self.task_attempt[task_id]
            task_spec = copy.deepcopy(self.tasks[task_id])
            task_spec["children"] = []
            payload = {
                "task_id": task_id,
                "attempt": attempt,
                "target_host": task_spec.get("host") or "",
                "task": task_spec,
            }
            correlation_id = f"task:{task_id}:attempt:{attempt}"
            commands.append(
                Command(
                    COMPUTE_DISPATCH,
                    self._new_event("compute.dispatch", correlation_id, payload),
                )
            )
            self.state[task_id] = "DISPATCHED"
            self.inflight_compute.add(task_id)
        return commands

    def _dispatch_local_plan(self, plan_id: str) -> Command:
        members = self.local_plans[plan_id]
        plan_tasks: List[Dict[str, Any]] = []
        for task_id in members:
            if self.state[task_id] not in {"READY", "BLOCKED"}:
                raise WorkflowError(
                    f"local plan {plan_id} member {task_id} is {self.state[task_id]}"
                )
            self.task_attempt[task_id] += 1
            attempt = self.task_attempt[task_id]
            task_spec = copy.deepcopy(self.tasks[task_id])
            task_spec["children"] = []
            plan_tasks.append(
                {
                    "task_id": task_id,
                    "attempt": attempt,
                    "target_host": task_spec["host"],
                    "correlation_id": f"task:{task_id}:attempt:{attempt}",
                    "task": task_spec,
                }
            )
            self.state[task_id] = "DISPATCHED"
            self.inflight_compute.add(task_id)
        self.local_plan_dispatches += 1
        return Command(
            COMPUTE_DISPATCH,
            self._new_event(
                "compute.plan.dispatch",
                f"plan:{plan_id}:attempt:1",
                {
                    "plan_id": plan_id,
                    "target_host": plan_tasks[0]["target_host"],
                    "tasks": plan_tasks,
                },
            ),
        )

    def handle_event(self, topic: str, event: Dict[str, Any]) -> List[Command]:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            raise WorkflowError("worker event is missing event_id")
        if event_id in self.seen_event_ids:
            return []
        self.seen_event_ids.add(event_id)
        kind = event.get("kind")
        if topic == COMPUTE_COMPLETED or kind == "compute.completed":
            commands = self._handle_compute_completed(event)
        elif topic == NETWORK_COMPLETED or kind == "network.completed":
            commands = self._handle_network_completed(event)
        else:
            raise WorkflowError(f"unsupported event topic={topic!r} kind={kind!r}")
        if not self.failed_reason:
            commands.extend(self._start_ready_collectives())
            commands.extend(self._dispatch_ready_tasks())
        return commands

    def _handle_compute_completed(self, event: Dict[str, Any]) -> List[Command]:
        payload = event.get("payload") or {}
        task_id = str(payload.get("task_id", ""))
        attempt = int(payload.get("attempt") or 0)
        status = str(payload.get("status", "SUCCEEDED")).upper()
        if task_id not in self.tasks:
            raise WorkflowError(f"completion for unknown task {task_id}")
        expected_correlation = f"task:{task_id}:attempt:{attempt}"
        if event.get("correlation_id") != expected_correlation:
            raise WorkflowError(
                f"completion correlation mismatch for task {task_id}: "
                f"{event.get('correlation_id')!r}"
            )
        if attempt != self.task_attempt[task_id]:
            return []
        if self.state[task_id] == "SUCCEEDED":
            return []
        if self.state[task_id] not in {"DISPATCHED", "RUNNING"}:
            raise WorkflowError(
                f"completion for task {task_id} in invalid state {self.state[task_id]}"
            )

        self.inflight_compute.discard(task_id)
        if status != "SUCCEEDED":
            retry_limit = int(
                self.tasks[task_id].get("max_retries", self.max_compute_retries)
            )
            if not self.failed_reason and self.task_attempt[task_id] <= retry_limit:
                self.state[task_id] = "READY"
            else:
                self.state[task_id] = status
                if not self.failed_reason:
                    self.failed_reason = f"task {task_id} completed with {status}"
            return []

        self.state[task_id] = "SUCCEEDED"
        if self.failed_reason:
            return []
        commands: List[Command] = []
        local_plan_id = self.local_plan_by_task.get(task_id)
        for edge in self.outgoing[task_id]:
            edge_id = edge["edge_id"]
            if (
                local_plan_id
                and self.local_plan_by_task.get(edge["dst_task_id"]) == local_plan_id
            ):
                self.edge_state[edge_id] = "LOCAL_COMPLETED"
                continue
            if edge["src_host"] == edge["dst_host"] or edge["bytes"] <= 0:
                self.edge_state[edge_id] = "LOCAL_READY"
                self._release_input(edge["dst_task_id"])
                continue
            self.edge_state[edge_id] = "NET_DISPATCHED"
            transfer = dict(edge)
            commands.append(self._dispatch_network(edge_id, transfer, "direct"))
        return commands

    def _dispatch_network(
        self,
        operation_id: str,
        payload: Dict[str, Any],
        operation_kind: str,
    ) -> Command:
        self.network_attempt[operation_id] += 1
        attempt = self.network_attempt[operation_id]
        transfer_id = f"transfer:{operation_id}:attempt:{attempt}"
        transfer = dict(payload)
        transfer["transfer_id"] = transfer_id
        transfer["attempt"] = attempt
        self.transfer_context[transfer_id] = {
            "operation_id": operation_id,
            "operation_kind": operation_kind,
            "payload": dict(payload),
        }
        self.inflight_network.add(transfer_id)
        return Command(
            NETWORK_DISPATCH,
            self._new_event("network.dispatch", transfer_id, transfer),
        )

    def _handle_network_completed(self, event: Dict[str, Any]) -> List[Command]:
        payload = event.get("payload") or {}
        transfer_id = str(payload.get("transfer_id", ""))
        status = str(payload.get("status", "SUCCEEDED")).upper()
        if event.get("correlation_id") != transfer_id:
            raise WorkflowError(
                f"completion correlation mismatch for transfer {transfer_id}: "
                f"{event.get('correlation_id')!r}"
            )
        context = self.transfer_context.pop(transfer_id, None)
        if context is None or transfer_id not in self.inflight_network:
            return []
        self.inflight_network.discard(transfer_id)
        operation_id = context["operation_id"]
        if status != "SUCCEEDED":
            if (
                not self.failed_reason
                and self.network_attempt[operation_id] <= self.max_network_retries
            ):
                return [
                    self._dispatch_network(
                        operation_id,
                        context["payload"],
                        context["operation_kind"],
                    )
                ]
            if not self.failed_reason:
                self.failed_reason = f"transfer {transfer_id} completed with {status}"
            return []

        if self.failed_reason:
            return []
        if context["operation_kind"] == "direct":
            if self.edge_state[operation_id] != "NET_COMPLETED":
                self.edge_state[operation_id] = "NET_COMPLETED"
                self._release_input(context["payload"]["dst_task_id"])
            return []

        collective_id = context["payload"]["collective_id"]
        self.collective_pending[collective_id].discard(operation_id)
        if self.collective_pending[collective_id]:
            return []
        self.collective_step[collective_id] += 1
        if self.collective_step[collective_id] < self._collective_step_count(
            collective_id
        ):
            return self._dispatch_collective_step(collective_id)

        self.collective_state[collective_id] = "COMPLETED"
        for participant in self.collectives[collective_id]["participants"]:
            self._release_input(participant["dst_task_id"])
        return []

    def _start_ready_collectives(self) -> List[Command]:
        commands: List[Command] = []
        for collective_id in sorted(self.collectives):
            if self.collective_state[collective_id] != "WAIT_SOURCES":
                continue
            sources = [
                participant["src_task_id"]
                for participant in self.collectives[collective_id]["participants"]
            ]
            if all(self.state[source] == "SUCCEEDED" for source in sources):
                self.collective_state[collective_id] = "RUNNING"
                commands.extend(self._dispatch_collective_step(collective_id))
        return commands

    def _collective_step_count(self, collective_id: str) -> int:
        collective = self.collectives[collective_id]
        rank_count = len(collective["participants"])
        if collective["type"] == "allreduce":
            return 2 * (rank_count - 1)
        return rank_count - 1

    def _dispatch_collective_step(self, collective_id: str) -> List[Command]:
        collective = self.collectives[collective_id]
        participants = collective["participants"]
        rank_count = len(participants)
        step = self.collective_step[collective_id]
        collective_type = collective["type"]
        if collective_type == "alltoall":
            peer_offset = step + 1
            bytes_per_flow = (
                collective["bytes_per_rank"] + rank_count - 1
            ) // rank_count
            phase = "alltoall"
        elif collective_type == "allgather":
            peer_offset = 1
            bytes_per_flow = collective["bytes_per_rank"]
            phase = "allgather"
        else:
            peer_offset = 1
            bytes_per_flow = (
                collective["bytes_per_rank"] + rank_count - 1
            ) // rank_count
            if collective_type == "allreduce":
                phase = "reduce_scatter" if step < rank_count - 1 else "allgather"
            else:
                phase = collective_type

        pending = self.collective_pending[collective_id]
        pending.clear()
        commands: List[Command] = []
        for rank, participant in enumerate(participants):
            peer = participants[(rank + peer_offset) % rank_count]
            operation_id = f"collective:{collective_id}:step:{step}:rank:{rank}"
            payload = {
                "collective_id": collective_id,
                "collective_type": collective_type,
                "algorithm": collective["algorithm"],
                "phase": phase,
                "step": step,
                "rank": rank,
                "src_task_id": participant["src_task_id"],
                "dst_task_id": peer["dst_task_id"],
                "src_host": participant["host"],
                "dst_host": peer["host"],
                "bytes": bytes_per_flow,
            }
            pending.add(operation_id)
            commands.append(
                self._dispatch_network(operation_id, payload, "collective")
            )
        return commands

    def _release_input(self, task_id: str) -> None:
        if self.pending_inputs[task_id] <= 0:
            raise WorkflowError(f"input count underflow for task {task_id}")
        self.pending_inputs[task_id] -= 1
        if self.pending_inputs[task_id] == 0:
            self.state[task_id] = "READY"

    @property
    def done(self) -> bool:
        return (
            not self.failed_reason
            and all(state == "SUCCEEDED" for state in self.state.values())
            and all(state == "COMPLETED" for state in self.collective_state.values())
            and not self.inflight_compute
            and not self.inflight_network
        )

    @property
    def terminal(self) -> bool:
        return self.done or (
            self.failed_reason is not None
            and not self.inflight_compute
            and not self.inflight_network
        )

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = defaultdict(int)
        for state in self.state.values():
            counts[state] += 1
        return {
            "run_id": self.run_id,
            "task_count": len(self.tasks),
            "states": dict(sorted(counts.items())),
            "inflight_compute": len(self.inflight_compute),
            "inflight_network": len(self.inflight_network),
            "collectives": dict(sorted(self.collective_state.items())),
            "compute_attempts": sum(self.task_attempt.values()),
            "network_attempts": sum(self.network_attempt.values()),
            "processed_event_ids": len(self.seen_event_ids),
            "failed_reason": self.failed_reason,
            "done": self.done,
            "acceleration": self.acceleration_report,
            "backend_local_chains": {
                "enabled": self.backend_local_chains,
                "plan_count": len(self.local_plans),
                "dispatches": self.local_plan_dispatches,
                "planned_tasks": sum(len(plan) for plan in self.local_plans.values()),
            },
        }

    def simulator_dependencies(
        self,
        orchestrator: str = "orchestrator",
        compute_worker: str = "gpusim",
        network_worker: str = "ns3",
    ) -> Dict[str, set[str]]:
        dependencies = {
            orchestrator: set(),
            compute_worker: set(),
            network_worker: set(),
        }
        if self.inflight_compute:
            dependencies[orchestrator].add(compute_worker)
        if self.inflight_network:
            dependencies[orchestrator].add(network_worker)
        # Either backend may receive a successor dispatch after any completion,
        # so idle workers must remain behind the DAG controller until terminal.
        if not self.terminal:
            dependencies[compute_worker].add(orchestrator)
            dependencies[network_worker].add(orchestrator)
        return dependencies


def encode_dependency_update(
    epoch: int, dependencies: Dict[str, set[str]]
) -> str:
    lines = [f"epoch={epoch}"]
    for consumer in sorted(dependencies):
        lines.append(f"{consumer}={','.join(sorted(dependencies[consumer]))}")
    return "\n".join(lines)


def make_batch(
    run_id: str,
    batch_sequence: int,
    time_ns: int,
    events: List[Dict[str, Any]],
    microstep: int = 0,
) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "batch_id": f"batch-{batch_sequence:09d}",
            "logical_time_ns": time_ns,
            "microstep": microstep,
            "events": events,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_batch(value: str, expected_run_id: str) -> Dict[str, Any]:
    batch = json.loads(value)
    if not isinstance(batch, dict):
        raise WorkflowError("batch must be a JSON object")
    if batch.get("schema_version") != SCHEMA_VERSION:
        raise WorkflowError(f"unsupported schema version {batch.get('schema_version')!r}")
    if batch.get("run_id") != expected_run_id:
        raise WorkflowError(
            f"run id mismatch: {batch.get('run_id')!r} != {expected_run_id!r}"
        )
    if not isinstance(batch.get("events"), list):
        raise WorkflowError("batch events must be an array")
    if not isinstance(batch.get("logical_time_ns"), int) or batch["logical_time_ns"] < 0:
        raise WorkflowError("batch logical_time_ns must be a non-negative integer")
    if not isinstance(batch.get("microstep", 0), int) or batch.get("microstep", 0) < 0:
        raise WorkflowError("batch microstep must be a non-negative integer")
    for event in batch["events"]:
        if not isinstance(event, dict):
            raise WorkflowError("batch event must be an object")
        for field in ("event_id", "kind", "correlation_id", "payload"):
            if field not in event:
                raise WorkflowError(f"batch event is missing {field}")
        if not isinstance(event["payload"], dict):
            raise WorkflowError("batch event payload must be an object")
    return batch


class EventLog:
    def __init__(self, path: str | None) -> None:
        self._stream: TextIO | None = None
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._stream = target.open("w", encoding="utf-8")

    def write(self, direction: str, topic: str, time_ns: int, value: Any) -> None:
        if not self._stream:
            return
        self._stream.write(
            json.dumps(
                {
                    "direction": direction,
                    "topic": topic,
                    "logical_time_ns": time_ns,
                    "value": value,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        if self._stream:
            self._stream.close()


def run(args: argparse.Namespace) -> int:
    if args.idle_grant_ns <= 0:
        raise WorkflowError("idle-grant-ns must be positive")
    run_id = args.run_id or f"run-{uuid.uuid4()}"
    controller = WorkflowController.from_file(
        args.workflow,
        run_id,
        max_compute_retries=args.max_compute_retries,
        max_network_retries=args.max_network_retries,
        backend_local_chains=args.backend_local_chains,
        backend_local_chain_max_tasks=args.backend_local_chain_max_tasks,
    )
    if args.backend_local_chains and args.max_compute_retries:
        raise WorkflowError("backend-local-chains requires max-compute-retries=0")
    if args.optimization_report:
        target = Path(args.optimization_report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(controller.acceleration_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    client = FncsClient(args.fncs_library)
    event_log = EventLog(args.event_log)
    batch_sequence = 0
    current_time = 0
    current_microstep = 0
    dependency_epoch = 0
    last_dependency_signature: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    def publish_commands(commands: List[Command]) -> None:
        nonlocal batch_sequence
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for command in commands:
            grouped[command.topic].append(command.event)
        for topic in sorted(grouped):
            batch_sequence += 1
            value = make_batch(
                run_id,
                batch_sequence,
                current_time,
                grouped[topic],
                current_microstep,
            )
            client.publish(topic, value)
            event_log.write("out", topic, current_time, json.loads(value))

    def publish_dependencies() -> None:
        nonlocal dependency_epoch, last_dependency_signature
        if not args.active_dependency_coordination:
            return
        dependencies = controller.simulator_dependencies(
            args.orchestrator_name,
            args.compute_worker_name,
            args.network_worker_name,
        )
        signature = tuple(
            (consumer, tuple(sorted(producers)))
            for consumer, producers in sorted(dependencies.items())
        )
        if signature == last_dependency_signature:
            return
        dependency_epoch += 1
        last_dependency_signature = signature
        value = encode_dependency_update(dependency_epoch, dependencies)
        client.publish_anon(ACTIVE_DEPENDENCIES, value)
        event_log.write(
            "control",
            ACTIVE_DEPENDENCIES,
            current_time,
            {"epoch": dependency_epoch, "value": value},
        )

    try:
        client.initialize()
        publish_commands(controller.initial_commands())
        publish_dependencies()
        while not controller.terminal:
            previous_time = current_time
            request_time = min(MAX_TIME_NS, current_time + args.idle_grant_ns)
            current_time = client.time_request(request_time)
            current_microstep = (
                current_microstep + 1 if current_time == previous_time else 0
            )
            commands: List[Command] = []
            incoming: List[tuple[int, int, str, Dict[str, Any]]] = []
            for topic in client.get_events():
                value = client.get_value(topic)
                if not value:
                    continue
                batch = parse_batch(value, run_id)
                event_log.write("in", topic, current_time, batch)
                for event in batch["events"]:
                    incoming.append(
                        (
                            batch["logical_time_ns"],
                            batch.get("microstep", 0),
                            topic,
                            event,
                        )
                    )
            for _, _, topic, event in sorted(
                incoming, key=lambda item: (item[0], item[1], item[2], item[3]["event_id"])
            ):
                commands.extend(controller.handle_event(topic, event))
            publish_commands(commands)
            publish_dependencies()

        batch_sequence += 1
        status = "completed" if controller.done else "failed"
        control_event = controller._new_event(
            "control.end",
            run_id,
            {"status": status, "summary": controller.summary()},
        )
        control_value = make_batch(
            run_id,
            batch_sequence,
            current_time,
            [control_event],
            current_microstep,
        )
        client.publish(CONTROL, control_value)
        event_log.write("out", CONTROL, current_time, json.loads(control_value))
        print(json.dumps(controller.summary(), ensure_ascii=False, sort_keys=True))
        client.time_request(MAX_TIME_NS)
        return 0 if controller.done else 2
    finally:
        if client._lib.fncs_is_initialized():
            client.finalize()
        event_log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", help="workflow jobs JSON file")
    parser.add_argument("--run-id", default=os.environ.get("COSIM_RUN_ID"))
    parser.add_argument("--event-log", help="JSONL control-plane event log")
    parser.add_argument("--optimization-report", help="write acceleration certificate JSON")
    parser.add_argument("--fncs-library", help="path to libfncs")
    parser.add_argument("--max-compute-retries", type=int, default=0)
    parser.add_argument("--max-network-retries", type=int, default=0)
    parser.add_argument(
        "--backend-local-chains",
        action="store_true",
        default=os.environ.get("COSIM_BACKEND_LOCAL_CHAINS", "").lower()
        in {"1", "true", "yes"},
        help="dispatch proven exact same-owner linear chains as backend-local plans",
    )
    parser.add_argument(
        "--backend-local-chain-max-tasks",
        type=int,
        default=int(os.environ.get("COSIM_BACKEND_LOCAL_CHAIN_MAX_TASKS", "32")),
        help="maximum tasks per local plan; 0 keeps the full proven chain",
    )
    parser.add_argument(
        "--active-dependency-coordination",
        action="store_true",
        default=os.environ.get("COSIM_ACTIVE_DEPENDENCIES", "").lower()
        in {"1", "true", "yes"},
    )
    parser.add_argument("--orchestrator-name", default="orchestrator")
    parser.add_argument("--compute-worker-name", default="gpusim")
    parser.add_argument("--network-worker-name", default="ns3")
    parser.add_argument(
        "--idle-grant-ns",
        type=int,
        default=int(os.environ.get("COSIM_IDLE_GRANT_NS", "1000000")),
        help="finite lookahead used while the orchestrator waits for worker events",
    )
    return parser


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    except (WorkflowError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"orchestrator error: {error}", file=sys.stderr)
        sys.exit(2)
