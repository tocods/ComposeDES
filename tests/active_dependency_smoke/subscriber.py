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
events = []
try:
    client.initialize()
    grants.append(client.time_request(100))
    events = [(key, client.get_value(key)) for key in client.get_events()]
    grants.append(client.time_request(100))
finally:
    if client._lib.fncs_is_initialized():
        client.finalize()
Path(sys.argv[1]).write_text(
    json.dumps({"grants": grants, "events": events}) + "\n", encoding="utf-8"
)
