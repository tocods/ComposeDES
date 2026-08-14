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


SCHEMA_VERSION = "2.0"
MAX_TIME_NS = (1 << 63) - 1

COMPUTE_DISPATCH = "compute/dispatch"
COMPUTE_COMPLETED = "compute/completed"
NETWORK_DISPATCH = "network/dispatch"
NETWORK_COMPLETED = "network/completed"
CONTROL = "control"


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class Command:
    topic: str
    event: Dict[str, Any]


class WorkflowController:
    """Deterministic DAG state machine, independent from FNCS transport."""

    def __init__(self, task_specs: Iterable[Dict[str, Any]], run_id: str) -> None:
        self.run_id = run_id
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.state: Dict[str, str] = {}
        self.parents: Dict[str, set[str]] = defaultdict(set)
        self.outgoing: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pending_inputs: Dict[str, int] = {}
        self.edge_state: Dict[str, str] = {}
        self.transfer_to_edge: Dict[str, str] = {}
        self.task_attempt: Dict[str, int] = defaultdict(int)
        self.inflight_compute: set[str] = set()
        self.inflight_network: set[str] = set()
        self.failed_reason: str | None = None
        self._event_sequence = 0

        for raw_spec in task_specs:
            spec = copy.deepcopy(raw_spec)
            task_id = str(spec.get("name", ""))
            if not task_id or task_id in self.tasks:
                raise WorkflowError(f"invalid or duplicate task name: {task_id!r}")
            self.tasks[task_id] = spec
            self.state[task_id] = "BLOCKED"

        self._build_edges()
        self._validate_acyclic()
        for task_id in self.tasks:
            self.pending_inputs[task_id] = len(self.parents[task_id])
            if self.pending_inputs[task_id] == 0:
                self.state[task_id] = "READY"

    @classmethod
    def from_file(cls, path: str, run_id: str) -> "WorkflowController":
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, dict):
            data = data.get("tasks", [])
        if not isinstance(data, list):
            raise WorkflowError("workflow input must be a task array or an object with tasks")
        return cls(data, run_id)

    def _build_edges(self) -> None:
        for src_id, spec in self.tasks.items():
            for index, child in enumerate(spec.get("children") or []):
                dst_id = str(child.get("child", ""))
                if dst_id not in self.tasks:
                    raise WorkflowError(f"task {src_id} references missing child {dst_id}")
                if src_id in self.parents[dst_id]:
                    raise WorkflowError(f"duplicate edge {src_id} -> {dst_id}")
                edge_id = f"edge:{src_id}:{dst_id}:{index}"
                edge = {
                    "edge_id": edge_id,
                    "src_task_id": src_id,
                    "dst_task_id": dst_id,
                    "src_host": child.get("src_host") or spec.get("host") or "",
                    "dst_host": child.get("dst_host") or self.tasks[dst_id].get("host") or "",
                    "bytes": int(child.get("size") or 0),
                }
                self.parents[dst_id].add(src_id)
                self.outgoing[src_id].append(edge)
                self.edge_state[edge_id] = "WAIT_PRODUCER"

    def _validate_acyclic(self) -> None:
        degree = {task_id: len(self.parents[task_id]) for task_id in self.tasks}
        ready = deque(task_id for task_id, count in degree.items() if count == 0)
        visited = 0
        while ready:
            src_id = ready.popleft()
            visited += 1
            for edge in self.outgoing[src_id]:
                dst_id = edge["dst_task_id"]
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

    def handle_event(self, topic: str, event: Dict[str, Any]) -> List[Command]:
        kind = event.get("kind")
        if topic == COMPUTE_COMPLETED or kind == "compute.completed":
            commands = self._handle_compute_completed(event)
        elif topic == NETWORK_COMPLETED or kind == "network.completed":
            commands = self._handle_network_completed(event)
        else:
            raise WorkflowError(f"unsupported event topic={topic!r} kind={kind!r}")
        commands.extend(self._dispatch_ready_tasks())
        return commands

    def _handle_compute_completed(self, event: Dict[str, Any]) -> List[Command]:
        payload = event.get("payload") or {}
        task_id = str(payload.get("task_id", ""))
        attempt = int(payload.get("attempt") or 0)
        status = str(payload.get("status", "SUCCEEDED")).upper()
        if task_id not in self.tasks:
            raise WorkflowError(f"completion for unknown task {task_id}")
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
            self.state[task_id] = status
            self.failed_reason = f"task {task_id} completed with {status}"
            return []

        self.state[task_id] = "SUCCEEDED"
        commands: List[Command] = []
        for edge in self.outgoing[task_id]:
            edge_id = edge["edge_id"]
            if edge["src_host"] == edge["dst_host"] or edge["bytes"] <= 0:
                self.edge_state[edge_id] = "LOCAL_READY"
                self._release_input(edge["dst_task_id"])
                continue
            transfer_id = f"transfer:{edge_id}"
            self.transfer_to_edge[transfer_id] = edge_id
            self.edge_state[edge_id] = "NET_DISPATCHED"
            self.inflight_network.add(transfer_id)
            transfer = dict(edge)
            transfer["transfer_id"] = transfer_id
            commands.append(
                Command(
                    NETWORK_DISPATCH,
                    self._new_event("network.dispatch", transfer_id, transfer),
                )
            )
        return commands

    def _handle_network_completed(self, event: Dict[str, Any]) -> List[Command]:
        payload = event.get("payload") or {}
        transfer_id = str(payload.get("transfer_id", ""))
        status = str(payload.get("status", "SUCCEEDED")).upper()
        edge_id = self.transfer_to_edge.get(transfer_id)
        if edge_id is None:
            return []
        if self.edge_state[edge_id] == "NET_COMPLETED":
            return []
        if status != "SUCCEEDED":
            self.failed_reason = f"transfer {transfer_id} completed with {status}"
            return []

        self.inflight_network.discard(transfer_id)
        self.edge_state[edge_id] = "NET_COMPLETED"
        src_id, dst_id = self._edge_endpoints(edge_id)
        del src_id
        self._release_input(dst_id)
        return []

    def _edge_endpoints(self, edge_id: str) -> tuple[str, str]:
        for edges in self.outgoing.values():
            for edge in edges:
                if edge["edge_id"] == edge_id:
                    return edge["src_task_id"], edge["dst_task_id"]
        raise WorkflowError(f"unknown edge {edge_id}")

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
            "failed_reason": self.failed_reason,
            "done": self.done,
        }


def make_batch(run_id: str, batch_sequence: int, time_ns: int, events: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "batch_id": f"batch-{batch_sequence:09d}",
            "logical_time_ns": time_ns,
            "events": events,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_batch(value: str, expected_run_id: str) -> Dict[str, Any]:
    batch = json.loads(value)
    if batch.get("schema_version") != SCHEMA_VERSION:
        raise WorkflowError(f"unsupported schema version {batch.get('schema_version')!r}")
    if batch.get("run_id") != expected_run_id:
        raise WorkflowError(
            f"run id mismatch: {batch.get('run_id')!r} != {expected_run_id!r}"
        )
    if not isinstance(batch.get("events"), list):
        raise WorkflowError("batch events must be an array")
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
    run_id = args.run_id or f"run-{uuid.uuid4()}"
    controller = WorkflowController.from_file(args.workflow, run_id)
    client = FncsClient(args.fncs_library)
    event_log = EventLog(args.event_log)
    batch_sequence = 0
    current_time = 0

    def publish_commands(commands: List[Command]) -> None:
        nonlocal batch_sequence
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for command in commands:
            grouped[command.topic].append(command.event)
        for topic in sorted(grouped):
            batch_sequence += 1
            value = make_batch(run_id, batch_sequence, current_time, grouped[topic])
            client.publish(topic, value)
            event_log.write("out", topic, current_time, json.loads(value))

    try:
        client.initialize()
        publish_commands(controller.initial_commands())
        while not controller.done and not controller.failed_reason:
            current_time = client.time_request(MAX_TIME_NS)
            commands: List[Command] = []
            for topic in client.get_events():
                value = client.get_value(topic)
                if not value:
                    continue
                batch = parse_batch(value, run_id)
                event_log.write("in", topic, current_time, batch)
                for event in batch["events"]:
                    commands.extend(controller.handle_event(topic, event))
            publish_commands(commands)

        batch_sequence += 1
        status = "completed" if controller.done else "failed"
        control_event = controller._new_event(
            "control.end",
            run_id,
            {"status": status, "summary": controller.summary()},
        )
        control_value = make_batch(
            run_id, batch_sequence, current_time, [control_event]
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
    parser.add_argument("--fncs-library", help="path to libfncs")
    return parser


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    except (WorkflowError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"orchestrator error: {error}", file=sys.stderr)
        sys.exit(2)
