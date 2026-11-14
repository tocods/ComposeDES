#!/usr/bin/env python3
"""Extend the frozen candidate pool using independent FP8 service and training runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow import Workflow, longest_path_certificate


VARIANTS = ('flags', 'barrier', 'split_cached')


def component(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row['layer'], []).append(row['us'])
    return {layer: median(v) for layer, v in by_layer.items()}


def train_median(path):
    with path.open() as file:
        return median(float(row['wall_ms']) for row in csv.DictReader(file) if int(row['step']) >= 4)


def forecast(source, training, services, output):
    data = json.loads(source.read_text())
    estimates = {}
    floor = {}
    for variant in VARIANTS:
        stages = component(services / f'{variant}.jsonl')
        missing = set(range(40)) - stages.keys()
        if missing:
            raise ValueError(f'{variant}: missing layers {sorted(missing)}')
        wf = Workflow()
        previous = ()
        for layer in range(40):
            name = f'fp8/{layer}'
            wf.add(name, stages[layer], previous)
            previous = (name,)
        floor[variant] = longest_path_certificate(wf.serialize())['lower_ns'] / 1e6
    anchor = 'flags'
    common = train_median(training / f'0-{anchor}.csv') - floor[anchor]
    for variant in VARIANTS:
        estimates['mono_' + variant] = common + floor[variant]
    data['fp8_calibration'] = {'common_non_monomoe_ms': common, 'routed_kernel_ms': floor,
                               'method': '40 measured FP8 projection groups and one training runtime anchor'}
    for policy in data['forecasts_ms'].values():
        policy.update(estimates)
    output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('training', type=Path)
    p.add_argument('services', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    forecast(a.source, a.training, a.services, a.output)
