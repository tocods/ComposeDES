"""Evaluate the two paths for each layer using independently calibrated timings."""
def copy_offsets(arrivals):
    previous = 0.
    offsets = []
    for layer in arrivals:
        offsets.append([x - previous for x in layer])
        previous = layer[-1]
    return offsets


def planned_arrivals(calibration, mask, backend):
    if not calibration.get("adaptive_publish"):
        return calibration["arrival_us"][backend]
    times = calibration["copy_offsets"]
    previous = 0.
    timeline = []
    for i, cut in enumerate(mask):
        copy_path = backend if cut else ("bulk" if "bulk" in times else "dual")
        offsets = times[copy_path][i]
        timeline.append([previous + duration for duration in offsets])
        previous = timeline[-1][-1]
    return timeline


def select_greedy(calibration, backend):
    # Greedily minimize the predicted completion of each model layer.
    # With conditional publication a split can delay later H2D work.
    arrival = calibration["arrival_us"][backend]
    offsets = calibration.get("copy_offsets")
    dma_cursor = 0.
    time = calibration["embed_us"]
    early_parts = calibration.get("early_parts", 1)
    mask = []
    comparisons = []
    for i, (layer, copies) in enumerate(zip(calibration["layers"], arrival)):
        q_ready = time + layer["qkv_us"]
        full_copy = copies[-1]
        if offsets:
            copies = [dma_cursor + d for d in offsets[backend][i]]
            full_copy = dma_cursor + offsets["dual"][i][-1]
        whole = max(q_ready, full_copy) + layer["whole_us"]
        if backend == "mpk":
            # Distribute the measured all-ready worker time in proportion
            # to the independently measured tiles, then add the shared merge.
            parts = layer["parts_us"]
            scale = layer["mpk_us"] / sum(parts)
        else:
            parts, scale = layer["parts_us"], 1.
        split = q_ready + sum(parts[:early_parts]) * scale
        for ready, cost in zip(copies, parts[early_parts:]):
            split = max(split, ready) + cost * scale
        split += layer["merge_us"]
        cut = split < whole
        mask.append(cut)
        dma_cursor = copies[-1] if cut else full_copy
        time = min(split, whole) + layer["post_us"]
        comparisons.append((q_ready, whole, split))
    return mask, time + calibration["head_us"], comparisons


def select(calibration, backend="mpk", *, trim_us=0.):
    if trim_us < 0:
        raise ValueError("The dominance tolerance must be nonnegative")
    if not calibration.get("adaptive_publish"):
        return select_greedy(calibration, backend)

    # Readiness publication consumes copy-stream time. A policy that reduces
    # compute completion but delays DMA may harm later layers. Keep both clocks.
    frontier = [(0., calibration["embed_us"], 0)]
    offsets = calibration["copy_offsets"]

    early_parts = calibration.get("early_parts", 1)
    for i, layer in enumerate(calibration["layers"]):
        missing = offsets[backend][i]
        bulk_tail = offsets.get("bulk", offsets["dual"])[i][-1]
        parts = layer["parts_us"]
        scale = layer["mpk_us"] / sum(parts) if backend == "mpk" else 1.
        early = sum(parts[:early_parts]) * scale
        jobs = tuple(zip(missing, (t * scale for t in parts[early_parts:])))
        paths = []
        for dma, compute, mask in frontier:
            q = compute + layer["qkv_us"]
            bulk_dma = dma + bulk_tail
            whole = max(q, bulk_dma) + layer["whole_us"]
            paths.append((bulk_dma, whole + layer["post_us"], mask))

            split = q + early
            for offset, duration in jobs:
                split = max(split, dma + offset) + duration
            split += layer["merge_us"] + layer["post_us"]
            paths.append((dma + missing[-1], split, mask | (1 << i)))

        # With fixed service times any future suffix is monotone in both
        # clocks. A retained state with an earlier DMA clock and up to
        # trim_us more compute delay can increase any suffix by at most
        # trim_us. Summing those losses bounds regret for the fixed profile.
        tolerance = trim_us if i + 1 < len(calibration["layers"]) else 0.
        frontier = []
        earliest = float("inf")
        for path in sorted(paths, key=lambda p: (p[0], p[1], p[2].bit_count())):
            if path[1] + tolerance < earliest:
                frontier.append(path)
                earliest = path[1]

    _, ready, bits = min(frontier, key=lambda p: (p[1], p[2].bit_count(), p[0]))
    mask = [bool(bits & (1 << i)) for i in range(len(calibration["layers"]))]
    return mask, ready + calibration["head_us"], []


def compact_profile(rows):
    return {f"{r['prefix']}/{r['suffix']}": r for r in rows}
