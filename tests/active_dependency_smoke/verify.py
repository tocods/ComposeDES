#!/usr/bin/env python3

import json
import sys
from pathlib import Path


output = Path(sys.argv[1])
global_metrics = json.loads((output / "global" / "metrics.json").read_text())
active_metrics = json.loads((output / "active" / "metrics.json").read_text())

for mode in ("global", "active"):
    for name in ("orchestrator", "worker_a", "worker_b"):
        result = json.loads((output / mode / f"{name}.json").read_text())
        if result["requested"] != result["granted"]:
            raise SystemExit(f"{mode}/{name} changed the requested grant trace")

if active_metrics["scheduler_rounds"] >= global_metrics["scheduler_rounds"]:
    raise SystemExit(
        "active scheduling did not reduce rounds: "
        f"{active_metrics['scheduler_rounds']} >= {global_metrics['scheduler_rounds']}"
    )

summary = {
    "global_scheduler_rounds": global_metrics["scheduler_rounds"],
    "active_scheduler_rounds": active_metrics["scheduler_rounds"],
    "round_reduction": global_metrics["scheduler_rounds"]
    - active_metrics["scheduler_rounds"],
    "grant_trace_preserved": True,
}
message = json.loads((output / "message" / "consumer.json").read_text())
if message["grants"] != [10, 100] or message["events"] != [["wake", "at-10"]]:
    raise SystemExit(f"same-time wake-up failed: {message!r}")
summary["same_time_message_grant_ns"] = message["grants"][0]
(output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, sort_keys=True))
