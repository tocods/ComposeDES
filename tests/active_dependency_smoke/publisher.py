#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fncs" / "orchestrator"))
from fncs_client import FncsClient


client = FncsClient(os.environ["FNCS_LIBRARY"])
grants = []
try:
    client.initialize()
    client.publish_anon(
        "__fncs/active_dependencies",
        "epoch=1\nconsumer=orchestrator\norchestrator=",
    )
    grants.append(client.time_request(10))
    client.publish("wake", "at-10")
    grants.append(client.time_request(100))
finally:
    if client._lib.fncs_is_initialized():
        client.finalize()
Path(sys.argv[1]).write_text(json.dumps({"grants": grants}) + "\n", encoding="utf-8")
