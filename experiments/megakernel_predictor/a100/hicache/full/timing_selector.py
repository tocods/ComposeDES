"""Choose whole attention, early MPK, or Q-ready MPK in the calibrated DAG."""
from selector import copy_offsets


WHOLE = "whole"
EARLY = "early"
Q_READY = "q_ready"
DUAL = "dual"


def prepare(profile):
    profile["copies"] = {k: copy_offsets(v) for k, v in profile["arrival_us"].items()}
    return profile


def transition(dma, compute, layer, offsets, action, launched, profile, index):
    requested = action != WHOLE
    charge = profile["early_lag_us"][index] if action == EARLY and not launched else (0., 0.)
    dma += charge[0]
    q = compute + charge[1] + layer["qkv_us"]
    copy_path = "mpk" if action in (EARLY, Q_READY) else "dual"
    arrival = offsets[copy_path][index]
    tail = dma + arrival[-1]
    if not requested:
        return tail, max(q, tail) + layer["whole_us"] + layer["post_us"]

    if action == Q_READY:
        q += profile["ready_submit_us"]
    parts = layer["parts_us"]
    scale = layer["mpk_us"] / sum(parts) if action in (EARLY, Q_READY) else 1.
    end = q + parts[0] * scale
    for at, us in zip(arrival, parts[1:]):
        end = max(end, dma + at) + us * scale
    return tail, end + layer["merge_us"] + layer["post_us"]


def simulation(profile, plan):
    dma, compute, launched = 0., profile["embed_us"], False
    for i, (layer, action) in enumerate(zip(profile["layers"], plan)):
        dma, compute = transition(dma, compute, layer, profile["copies"], action,
                                  launched, profile, i)
        launched |= action in (EARLY, Q_READY)
    return compute + profile["head_us"]


def minimum_plan(profile, allowed=(WHOLE, EARLY, Q_READY)):
    options = tuple(enumerate(allowed))
    frontier = {False: [(0., profile["embed_us"], 0, 0)], True: []}
    for i, layer in enumerate(profile["layers"]):
        next_paths = {False: [], True: []}
        for launched, sub in frontier.items():
            for dma, compute, packed, cuts in sub:
                for code, action in options:
                    end_dma, end_compute = transition(
                        dma, compute, layer, profile["copies"], action, launched, profile, i)
                    state = launched or action in (EARLY, Q_READY)
                    next_paths[state].append((end_dma, end_compute,
                                              packed | (code << (2 * i)), cuts + (code != 0)))
        for launched, paths in next_paths.items():
            earliest = float("inf")
            keep = []
            for state in sorted(paths, key=lambda p: (p[0], p[1], p[3])):
                if state[1] < earliest:
                    keep.append(state)
                    earliest = state[1]
            frontier[launched] = keep

    _, ready, packed, _ = min(frontier[False] + frontier[True],
                              key=lambda p: (p[1], p[3], p[0]))
    plan = tuple(allowed[(packed >> (2 * i)) & 3] for i in range(len(profile["layers"])))
    return plan, ready + profile["head_us"]
