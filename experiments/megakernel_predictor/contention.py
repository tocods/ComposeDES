"""Fit active-window service slowdown from paired, timestamped H20 executions."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np


def spans(tiles):
    events = defaultdict(lambda: [0, 0])
    for (stage, _), (_, _, start, stop) in tiles.items():
        role = 0 if stage == 'gate' else 1
        events[start][role] += 1
        events[stop][role] -= 1
    edges = sorted(events)
    states = np.cumsum([events[t] for t in edges], axis=0)
    return np.asarray(edges), states


def integrate(edges, levels, start, stop):
    first = np.searchsorted(edges, start, side='right') - 1
    last = np.searchsorted(edges, stop, side='left')
    durations = np.minimum(edges[first + 1:last + 1], stop) - np.maximum(edges[first:last], start)
    return float(durations @ levels[first:last])


def fit(path, kinds=None, sm=78):
    frames = defaultdict(lambda: defaultdict(dict))
    with path.open() as file:
        for r in csv.DictReader(file):
            key = tuple(int(r[k]) for k in ('batch', 'layer', 'cohort', 'step', 'producers', 'trial'))
            tile = r['stage'], int(r['tile'])
            values = tuple(int(r[k]) for k in ('routes', 'weight_bytes', 'start_ns', 'end_ns'))
            frames[key][int(r['overlap'])][tile] = values
    features = defaultdict(list)
    diagnostics = defaultdict(list)
    scheduling = defaultdict(list)
    services = defaultdict(list)
    for key, variants in frames.items():
        serial, pipeline = variants[0], variants[1]
        if serial.keys() != pipeline.keys():
            raise ValueError(f'tile coverage differs for {key}')
        edges, activity = spans(pipeline)
        serial_edges, serial_activity = spans(serial)
        limits = np.array([key[-2], sm - key[-2]])
        scale = 1 / limits
        gate_ready = defaultdict(int)
        for (stage, tile), (_, _, start, stop) in pipeline.items():
            if stage == 'gate':
                expert = tile // 16
                gate_ready[expert] = max(gate_ready[expert], stop)
        for (stage, tile), (_, _, start, _) in pipeline.items():
            role = 0 if stage == 'gate' else 1
            previous = pipeline.get((stage, tile - limits[role]))
            tail = previous[3] if previous else 0
            if stage == 'gate' and previous:
                scheduling[key[-2], 'gate_gap_ns'].append(max(0, start - tail))
            if stage == 'down':
                ready = gate_ready[tile // 32]
                if ready >= tail:
                    scheduling[key[-2], 'ready_gap_ns'].append(max(0, start - ready))
                else:
                    scheduling[key[-2], 'down_gap_ns'].append(max(0, start - tail))
        length = edges[-1] - edges[0]
        both = (activity[:, 0] > 0) & (activity[:, 1] > 0)
        overlap = np.diff(edges) @ both[:-1]
        diagnostics[key[-2], 'forward', 'overlap_fraction'].append(float(overlap / length))
        diagnostics[key[-2], 'forward', 'pipeline_span_us'].append(float(length / 1000))
        for (stage, tile), (_, _, start, stop) in pipeline.items():
            role = 0 if stage == 'gate' else 1
            other = 1 - role
            base = serial[stage, tile]
            serial_ns = base[3] - base[2]
            if kinds is not None:
                services[key[-2], stage, kinds[key[1], stage], base[0]].append(serial_ns)
            coexist_ns = integrate(edges, (activity[:, other] > 0).astype(float), start, stop)
            # Opposite-role occupancy averaged during the measured tile. Include
            # self occupancy so an occupied but undersubscribed pair is distinct.
            pressure = integrate(edges, activity[:, other] * scale[other], start, stop) / (stop - start)
            self_pressure = integrate(edges, activity[:, role] * scale[role], start, stop) / (stop - start)
            serial_pressure = integrate(serial_edges, serial_activity[:, role] * scale[role], base[2], base[3]) / serial_ns
            features[stage].append((pressure, self_pressure - serial_pressure, (stop - start) / serial_ns))
            if coexist_ns / (stop - start) >= .75:
                diagnostics[key[-2], stage, 'mostly_overlap'].append((stop - start) / serial_ns)
            elif coexist_ns == 0:
                diagnostics[key[-2], stage, 'no_overlap'].append((stop - start) / serial_ns)
        for stage in ('gate', 'down'):
            for mode, data in (('serial', serial), ('pipeline', pipeline)):
                tiles = [r for (s, _), r in data.items() if s == stage]
                bw = sum(r[1] for r in tiles) / (max(r[3] for r in tiles) - min(r[2] for r in tiles))
                diagnostics[key[-2], stage, f'{mode}_logical_GBs'].append(bw)
    coeff = {}
    for stage, rows in features.items():
        x = np.asarray(rows, dtype=float)
        target = x[:, 2] - 1.
        low, high = np.quantile(target, [.02, .98])
        target = np.clip(target, low, high)
        fitted = np.linalg.lstsq(x[:, :2], target, rcond=None)[0]
        unconstrained = float(fitted[0])
        if fitted[0] < 0:
            fitted = [0., np.linalg.lstsq(x[:, 1:2], target, rcond=None)[0][0]]
        coeff[stage] = {'opposite': float(fitted[0]), 'unconstrained_opposite': unconstrained,
                        'self_delta': float(fitted[1]), 'tiles': len(x)}
    return dict(coefficients=coeff, pairs=len(frames),
                tile_ns={'/'.join(map(str, k)): median(v) for k, v in sorted(services.items())},
                scheduling={str(p): {name: median(scheduling[p, name])
                                     for name in ('gate_gap_ns', 'down_gap_ns', 'ready_gap_ns')}
                            for p in (26, 39, 52)},
                diagnostics={'/'.join(map(str, k)): {'median': median(v), 'n': len(v)}
                             for k, v in sorted(diagnostics.items())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--services', type=Path)
    args = parser.parse_args()
    kinds = None
    if args.services:
        with args.services.open() as file:
            kinds = {(int(r['layer']), r['stage']): int(r['type']) for r in csv.DictReader(file)}
    data = fit(args.trace, kinds)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
