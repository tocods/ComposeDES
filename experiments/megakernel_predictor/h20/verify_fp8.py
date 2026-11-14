#!/usr/bin/env python3
"""Check output agreement and per-round FP8 latency on the fixed input trace."""
import argparse
from itertools import chain, combinations
import json
from pathlib import Path
from statistics import median

from summary import rounds


def verify(folder):
    data = list(rounds(folder, 'mono_'))
    for name in sorted({r[0] for r in data}):
        values = [ms for candidate, _, ms, _, _ in data if candidate == name]
        print(name, f'{median(values):.3f} ms', f'rounds {[round(ms, 3) for ms in values]}')
    by_variant = {}
    for path in sorted(folder.glob('[0-9]-*.out.json')):
        rnd, variant = path.name.removesuffix('.out.json').split('-', 1)
        outputs = list(chain.from_iterable(chain.from_iterable(json.loads(path.read_text()))))
        if len(outputs) != 384:
            raise ValueError(f'{path}: expected 384 decoded tokens')
        by_variant.setdefault(variant, {})[rnd] = outputs
    for variant, outputs in by_variant.items():
        comparisons = [sum(x != y for x, y in zip(a, b, strict=True))
                       for a, b in combinations(outputs.values(), 2)]
        print(variant, 'within-variant max disagreements', max(comparisons, default=0))
    for a, b in combinations(sorted(by_variant), 2):
        matches = [sum(x == y for x, y in zip(by_variant[a][rnd], by_variant[b][rnd], strict=True))
                   for rnd in sorted(by_variant[a].keys() & by_variant[b].keys())]
        print(a, b, 'roundwise agreements out of 384', matches)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    verify(p.parse_args().folder)
