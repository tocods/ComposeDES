#!/usr/bin/env python3
"""Rank fixed H20 candidate manifests using the ComposeDES critical path."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from service import Service, read_routes
from workflow import Candidate, build, contract_prologue, longest_path_certificate, replay, score_layer


def evaluate_forward(routes, service, candidate, sm, policy, uncertainty):
    times = [score_layer(service, r, candidate, sm, policy, uncertainty) for r in routes]
    return tuple(map(sum, zip(*times)))


def predict_capture(input_dir, output, sm, check_dag):
    # Take candidate definitions only. Previous scores and end-to-end outcomes
    # play no role in fitting or ranking.
    manifest = json.loads((input_dir / 'frozen.json').read_text())
    predictions = {}
    witnesses = {}
    for key, case in manifest.items():
        batch = int(key)
        service = Service(input_dir / f'services-b{batch}.csv')
        forwards = defaultdict(list)
        for route in read_routes(input_dir / f'routes-heldout-b{batch}.csv', batch):
            forwards[route.forward].append(route)
        for name, routes in forwards.items():
            if sorted(route.layer for route in routes) != list(range(40)):
                raise ValueError(f'batch {batch}, forward {name} lacks a complete layer sequence')
        candidates, distinct = {}, set()
        for name, config in sorted(case['candidates'].items()):
            candidate = Candidate(*config)
            # A full-grid split has no meaningful producer count.
            identity = candidate if candidate.mode != 9 or candidate.matched else Candidate(
                9, 0, candidate.gate_rows, candidate.down_rows, candidate.gate_threads,
                candidate.down_threads, 0)
            if identity in distinct:
                continue
            candidates[name] = identity
            distinct.add(identity)
        first_forward = next(iter(forwards.values()))
        policies = {}
        for policy in ('additive', 'dag'):
            scores = {}
            for name, candidate in candidates.items():
                totals = [evaluate_forward(routes, service, candidate, sm, policy, True)
                          for routes in forwards.values()]
                scores[name] = {
                    key: median(col) / 1e6 for key, col in zip(('point_ms', 'lower_ms', 'upper_ms'), zip(*totals))
                }
            policies[policy] = scores
        predictions[key] = {'capture_forwards': len(forwards), 'service_samples': dict(service.sample_counts),
                            'policies': policies}
        if check_dag:
            candidate = candidates['ready_26']
            example = build(service, first_forward[0], candidate, sm, 'dag')
            estimate = longest_path_certificate(example)['lower_ns']
            makespan = replay(example)
            contracted, report = contract_prologue(example)
            compact = replay(contracted)
            if makespan != estimate or compact != estimate or not report['budget_satisfied']:
                raise ValueError(f'ComposeDES replay, critical path, and contraction differ in batch {batch}')
            witnesses[key] = {'layer': first_forward[0].layer, 'nodes': len(example),
                              'edges': sum(len(task['children']) for task in example),
                              'replay_ns': makespan, 'critical_path_ns': estimate,
                              'contracted_nodes': len(contracted), 'contracted_replay_ns': compact}
            (output.parent / f'example-b{batch}.json').write_text(json.dumps({'tasks': example}, indent=2) + '\n')
        print(f'B{batch} scored {len(candidates)} candidates using {len(forwards)} forwards', flush=True)
    output.write_text(json.dumps({'prediction': predictions, 'witness': witnesses}, indent=2) + '\n')


def compare(prediction_path, measurements, output_path):
    plan = json.loads(prediction_path.read_text())
    samples = defaultdict(list)
    with measurements.open(newline='') as source:
        for row in csv.DictReader(source):
            samples[row['batch'], row['candidate']].append(float(row['decode_ms']))
    actual = {key: median(values) for key, values in samples.items()}
    if any(len(values) != 3 for values in samples.values()):
        raise ValueError('expected three complete measurement rounds per candidate')
    with output_path.open('w', newline='') as target:
        out = csv.writer(target, lineterminator='\n')
        out.writerow(('batch', 'pool', 'policy', 'selected', 'winner', 'regret_pct', 'separated_by_bounds',
                       'selected_prediction_ms', 'selected_measured_ms', 'stock_ms'))
        for batch, case in plan['prediction'].items():
            for policy, scores in case['policies'].items():
                pools = {'tiled': sorted(scores), 'ready': sorted(n for n in scores if n.startswith('ready_'))}
                for pool, names in pools.items():
                    pick = min(names, key=lambda n: scores[n]['point_ms'])
                    best = min(names, key=lambda n: actual[batch, n])
                    regret = 100 * (actual[batch, pick] / actual[batch, best] - 1)
                    margin_clear = all(scores[pick]['upper_ms'] < scores[n]['lower_ms'] for n in names if n != pick)
                    out.writerow((batch, pool, policy, pick, best, round(regret, 6), int(margin_clear),
                                  round(scores[pick]['point_ms'], 6), actual[batch, pick], actual.get((batch, 'stock'), '')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest='operation', required=True)
    pred = tasks.add_parser('predict')
    pred.add_argument('--inputs', type=Path, required=True)
    pred.add_argument('--output', type=Path, required=True)
    pred.add_argument('--sm', type=int, default=78)
    pred.add_argument('--check-dag', action='store_true')
    check = tasks.add_parser('compare')
    check.add_argument('--prediction', type=Path, required=True)
    check.add_argument('--measurements', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.operation == 'predict':
        predict_capture(args.inputs, args.output, args.sm, args.check_dag)
    else:
        compare(args.prediction, args.measurements, args.output)


if __name__ == '__main__':
    main()
