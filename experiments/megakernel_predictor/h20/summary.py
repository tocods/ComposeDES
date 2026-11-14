#!/usr/bin/env python3
"""Summarize independent rounds and audit candidate ranking on a common workload."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import median


def rounds(folder, prefix=''):
    for path in sorted(folder.glob('[0-9]-*.csv')):
        round_id, label = path.stem.split('-', 1)
        with path.open() as file:
            data = [r for r in csv.DictReader(file) if int(r['step']) >= 4]
        if not data:
            raise ValueError(f'no decode samples in {path}')
        timing = 'wall_ms' if prefix else 'decode_ms'
        yield (prefix + label, int(round_id), median(float(r[timing]) for r in data),
               max(int(r['peak_bytes']) for r in data), len(data))


def summarize(folder, forecast_file, output, fp8=None):
    samples = {}
    for name, rnd, ms, peak, n in list(rounds(folder)) + (list(rounds(fp8, 'mono_')) if fp8 else []):
        samples.setdefault(name, []).append((rnd, ms, peak, n))
    actual = {name: median(ms for _, ms, _, _ in values) for name, values in samples.items()}
    forecasts = json.loads(forecast_file.read_text())['forecasts_ms']
    with output.open('w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['method', 'candidate', 'predicted_ms', 'measured_ms', 'pred_rank', 'actual_rank',
                         'min_round_ms', 'max_round_ms', 'rounds', 'peak_bytes', 'timed_forwards_per_round'])
        for policy, predictions in forecasts.items():
            names = predictions.keys() & samples.keys()
            if 'upstream' in names:
                names -= {'stock'}
            pred_order = sorted(names, key=predictions.get)
            actual_order = sorted(names, key=actual.get)
            for pred_rank, name in enumerate(pred_order, 1):
                scores = [ms for _, ms, _, _ in samples[name]]
                writer.writerow([policy, name, predictions[name], actual[name], pred_rank,
                                 actual_order.index(name) + 1, min(scores), max(scores), len(scores),
                                 max(peak for _, _, peak, _ in samples[name]), samples[name][0][3]])
            ready = [name for name in pred_order if name.startswith('ready_')]
            if ready:
                chosen = ready[0]
                true = min(ready, key=actual.get)
                swaps = sum(actual[a] > actual[b] for a, b in combinations(pred_order, 2))
                print(f'{policy}: predicted {chosen}, measured {true}, '
                      f'regret={(actual[chosen] / actual[true] - 1) * 100:.3f}%, '
                      f'pairwise inversions={swaps}/{len(names) * (len(names) - 1) // 2}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('forecast', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--fp8', type=Path)
    args = p.parse_args()
    summarize(args.folder, args.forecast, args.output, args.fp8)
