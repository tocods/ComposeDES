#!/usr/bin/env python3
import json
import sys
from pathlib import Path


case_dir = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in (case_dir / "output" / "control-plane.jsonl").read_text().splitlines()
]

network_dispatches = [
    record["value"]
    for record in records
    if record["direction"] == "out" and record["topic"] == "network/dispatch"
]
network_completions = [
    record["value"]
    for record in records
    if record["direction"] == "in" and record["topic"] == "network/completed"
]
compute_dispatches = [
    event
    for record in records
    if record["direction"] == "out" and record["topic"] == "compute/dispatch"
    for event in record["value"]["events"]
]

assert len(network_dispatches) == 2, len(network_dispatches)
assert [len(batch["events"]) for batch in network_dispatches] == [2, 2]
assert sum(len(batch["events"]) for batch in network_completions) == 4
assert len(compute_dispatches) == 4

pre_dispatch_times = {
    event["payload"]["task_id"]: record["logical_time_ns"]
    for record in records
    if record["direction"] == "out" and record["topic"] == "compute/dispatch"
    for event in record["value"]["events"]
    if event["payload"]["task_id"].startswith("pre")
}
post_dispatch_times = {
    event["payload"]["task_id"]: record["logical_time_ns"]
    for record in records
    if record["direction"] == "out" and record["topic"] == "compute/dispatch"
    for event in record["value"]["events"]
    if event["payload"]["task_id"].startswith("post")
}
last_network_time = max(
    event["payload"]["finish_time_ns"]
    for batch in network_completions
    for event in batch["events"]
)

assert set(pre_dispatch_times) == {"pre0", "pre1"}
assert set(post_dispatch_times) == {"post0", "post1"}
assert max(pre_dispatch_times.values()) == 0
assert min(post_dispatch_times.values()) >= last_network_time

print(
    "collective verification passed: "
    "2 ring steps, 4 ns-3 flows, post tasks released after the final barrier"
)
