"""Verify applied quota and cached prefix/uncached suffix in native logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def check(log, data):
    applied = {}
    window = None
    for line in log.read_text().splitlines():
        m = re.fullmatch(r"CTA_PAIR_BEGIN (\d+) (\d+) (-?\d+) (\d+) (\d+)", line)
        if m:
            assert window is None
            window = tuple(map(int, m.groups()))
            quotas, prefill = [], []
        elif line == "CTA_PAIR_END":
            assert window is not None
            prefix, suffix, rep, arm, cap = window
            assert quotas == [cap], (window, quotas)
            assert prefill == [(suffix, prefix)], (window, prefill)
            if rep >= 0:
                key = (prefix, suffix, rep, arm)
                assert key not in applied
                applied[key] = cap
            window = None
        elif window is not None:
            cap = re.search(r"HiCache experiment H2D quota = (\d+)", line)
            if cap:
                quotas.append(int(cap.group(1)))
            sizes = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
            if sizes:
                prefill.append(tuple(map(int, sizes.groups())))
    assert window is None
    observed = {(r["prefix"], r["suffix"], r["repeat"], r["order"]):
                0 if r["plan"] == "whole-2" else 8
                for r in map(json.loads, data.read_text().splitlines())}
    assert observed == applied
    return dict(requests=len(applied), quotas=dict(Counter(applied.values())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", type=Path, nargs="+")
    args = p.parse_args()
    result = {x.name: check(x, x.with_suffix(".jsonl")) for x in args.logs}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
