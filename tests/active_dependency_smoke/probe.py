#!/usr/bin/env python3
"""Minimal FNCS federate used to compare global and active grants."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fncs" / "orchestrator"))

from fncs_client import FncsClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-dependencies", action="store_true")
    args = parser.parse_args()

    requested = [int(value) for value in args.schedule.split(",") if value]
    client = FncsClient(os.environ["FNCS_LIBRARY"])
    granted = []
    try:
        client.initialize()
        if args.publish_dependencies:
            client.publish_anon(
                "__fncs/active_dependencies",
                "epoch=1\norchestrator=\nworker_a=\nworker_b=",
            )
        for timestamp in requested:
            granted.append(client.time_request(timestamp))
    finally:
        if client._lib.fncs_is_initialized():
            client.finalize()

    Path(args.output).write_text(
        json.dumps(
            {
                "name": os.environ["FNCS_NAME"],
                "requested": requested,
                "granted": granted,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
