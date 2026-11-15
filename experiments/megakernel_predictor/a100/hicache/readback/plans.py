"""Comparable native copy and attention paths for the read-back sweep."""

QUOTAS = (1, 2, 8, 24, 64)
CANDIDATES = {
    **{f"whole-{q}": (q, "whole") for q in QUOTAS},
    **{f"head-{q}": (q, "head") for q in QUOTAS},
    **{f"groups-{q}": (q, "groups") for q in QUOTAS},
}
PAGE = 64
HEAD = 1024
GROUP = 4096


def case_key(row):
    return f"{row['prefix']}/{row['suffix']}/{row['resident']}"


def ranges(prefix, resident, kind):
    if kind == "whole":
        return [(resident, prefix)]
    if kind == "head":
        cut = resident + HEAD
        return [(resident, cut), (cut, prefix)] if cut < prefix else [(resident, prefix)]
    if kind == "groups":
        return [(lo, min(lo + GROUP, prefix)) for lo in range(resident, prefix, GROUP)]
    raise ValueError(kind)
