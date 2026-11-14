#!/usr/bin/env python3
"""Freeze B2 candidate rankings on fresh routing."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from service import Service, read_routes
from window import schedule
from workflow import Candidate, build, longest_path_certificate


def score(service, frames, candidate, policy, fit, witnesses=None):
    forwards = defaultdict(float)
    for frame in frames:
        tasks = build(service, frame, candidate, 78, 'dag')
        if candidate.mode == 8 and policy in ('gaps', 'tile'):
            if policy == 'tile':
                for task in tasks:
                    stage = task.get('stage')
                    if stage in ('gate', 'down'):
                        weight_type = service.weights[frame.layer, stage][1]
                        load = frame.loads[task['expert']]
                        key = f'{candidate.producers}/{stage}/{weight_type}/{load}'
                        task['estimated_duration_ns'] = round(fit['tile_ns'][key])
            apply_ready_costs(tasks, fit['scheduling'][str(candidate.producers)])
        if candidate.mode != 8 or policy == 'dag':
            duration = longest_path_certificate(tasks)['lower_ns']
        else:
            duration, overlap, durations = schedule(tasks, candidate.partition(78), fit['coefficients'])
            if witnesses is not None and not witnesses:
                for task in tasks:
                    task['estimated_duration_ns'] = round(durations[task['name']])
                check = longest_path_certificate(tasks)['lower_ns']
                if abs(check - duration) > 40:
                    raise ValueError('the ComposeDES certificate disagrees with the window simulation')
                witnesses.update(ns=duration, replayable_dag_ns=check, overlap_ns=overlap)
        forwards[frame.forward] += duration
    return median(forwards.values()) / 1e6


def apply_ready_costs(tasks, timing):
    by_name = {task['name']: task for task in tasks}
    for task in tuple(tasks):
        name = task['name']
        if name.startswith('ready/'):
            task['estimated_duration_ns'] += round(timing['ready_gap_ns'])
        if task.get('stage') not in ('gate', 'down'):
            continue
        for edge in task['children']:
            next_task = by_name[edge['child']]
            if next_task.get('stage') == task['stage']:
                gap = f"{task['stage']}_gap_ns"
                key = f'{name}/handoff'
                # Only the same CTA waits for this handoff. Expert readiness
                # starts from the completion of the projection itself.
                tasks.append({'name': key, 'host': 'gpu0', 'estimated_duration_ns': round(timing[gap]),
                              'children': [edge.copy()]})
                edge['child'] = key


POOL = {'tiled_whole': (7, 0), 'split_all': (9, 26),
        **{f'ready_{p}': (8, p) for p in (26, 39, 52)}}


def timed(path):
    with path.open() as f:
        return median(float(r['decode_ms']) for r in csv.DictReader(f) if int(r['step']) >= 4)


def predict(root, routes_name, output_name, calibration, upstream):
    fit = json.loads((root / calibration).read_text())
    service = Service(root / 'services-b2.csv')
    frames = list(read_routes(root / routes_name, 2))
    train = list(read_routes(root / 'routes-train-b2.csv', 2))
    anchor = Candidate(7, 0, 32, 64, 512, 512, 0)
    anchor_predict = score(service, train, anchor, 'dag', fit)
    residual = timed(root / 'train' / '0-tiled_whole.csv') - anchor_predict
    stock_label = 'upstream' if upstream else 'stock'
    stock = timed(root / 'train' / f'0-{stock_label}.csv')
    estimates, picks, witnesses = {}, {}, {}
    policies = ('dag', 'window', 'gaps', 'tile') if fit.get('tile_ns') else ('dag', 'window', 'gaps')
    for policy in policies:
        estimates[policy] = {}
        for name, (mode, p) in POOL.items():
            candidate = Candidate(mode, p, 32, 64, 512, 512, 0)
            witness = {}
            estimates[policy][name] = score(service, frames, candidate, policy, fit, witness) + residual
            if witness:
                witnesses[policy, name] = witness
        estimates[policy][stock_label] = stock
        picks[policy] = min((name for name in POOL if name.startswith('ready_')), key=estimates[policy].get)
    output = {'forecasts_ms': estimates, 'ready_picks': picks,
              'witnesses': {'/'.join(k): v for k, v in witnesses.items()},
              'train_non_routed_ms': residual, 'train_stock_ms': stock,
              'heldout_forwards': len({frame.forward for frame in frames})}
    (root / output_name).write_text(json.dumps(output, indent=2) + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--routes', default='routes-new-b2.csv')
    p.add_argument('--forecast', default='b2-forecast.json')
    p.add_argument('--calibration', default='contention.json')
    p.add_argument('--upstream', action='store_true')
    a = p.parse_args()
    predict(a.root, a.routes, a.forecast, a.calibration, a.upstream)
