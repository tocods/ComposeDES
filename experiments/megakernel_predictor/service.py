"""Fit H20 projection service from isolated timings and ordered routing counts."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECTIONS = {'gate': 512, 'down': 2048}


@dataclass(frozen=True)
class Route:
    layer: int
    cohort: int
    step: int
    loads: tuple[int, ...]

    @property
    def forward(self):
        return self.cohort, self.step


def read_routes(path, batch):
    with path.open(newline='') as source:
        for row in csv.DictReader(source):
            values = tuple(map(int, row['loads'].split(':')))
            if sum(values) != 8 * batch or min(values) <= 0 or max(values) > batch:
                raise ValueError(f'{path}: invalid top-8 routing histogram')
            yield Route(int(row['layer']), int(row['cohort']), int(row['step']), values)


def work(loads, stage, blocks, rows):
    columns = PROJECTIONS[stage]
    if blocks <= 0 or rows <= 0 or columns % rows:
        raise ValueError(f'invalid tile configuration for {stage}')
    tiles = columns // rows
    cta = [0] * blocks
    for tile in range(len(loads) * tiles):
        cta[tile % blocks] += loads[tile // tiles]
    return tiles, cta


def features(loads, stage, blocks, rows, weight_bytes):
    tiles, ctas = work(loads, stage, blocks, rows)
    waves = (len(loads) * tiles + blocks - 1) // blocks
    return [1., len(loads) * weight_bytes / 2**20, max(ctas), waves]


def fit_relative(x, y):
    weighted = x / y[:, None]
    best = None
    for mask in range(1, 1 << x.shape[1]):
        cols = [i for i in range(x.shape[1]) if mask & (1 << i)]
        values = np.linalg.lstsq(weighted[:, cols], np.ones(len(y)), rcond=None)[0]
        if np.any(values < 0):
            continue
        err = np.mean((weighted[:, cols] @ values - 1) ** 2)
        if best is None or err < best[0]:
            coeff = np.zeros(x.shape[1])
            coeff[cols] = values
            best = err, coeff
    return best[1]


class Service:
    def __init__(self, path: Path):
        observations, overhead = defaultdict(list), defaultdict(list)
        self.weights = {}
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                stage = row['stage']
                blocks, threads, rows, kind = (int(row[key]) for key in ('blocks', 'threads', 'rows', 'type'))
                layer = int(row['layer'])
                loads = tuple(map(int, row['loads'].split(':')))
                weight_bytes = int(row['expert_bytes']) * (2 if stage == 'gate' else 1)
                self.weights[layer, stage] = weight_bytes, kind
                plain = float(row['plain_us'])
                terms = features(loads, stage, blocks, rows, weight_bytes)
                target = max(.1, float(row['us']) - plain)
                observations[stage, blocks, threads, rows, kind].append((terms, target))
                overhead[layer].append(tuple(float(row[key]) for key in
                    ('plain_us', 'coop_us', 'barrier_us', 'setup_us', 'reduce_us')))
        self.overhead = {layer: np.median(rows, axis=0) for layer, rows in overhead.items()}
        self.models = {}
        self.sample_counts = Counter()
        for key, rows in observations.items():
            x, y = map(np.asarray, zip(*rows))
            coeff = fit_relative(x, y)
            relative_error = np.abs(x @ coeff - y) / np.maximum(.1, x @ coeff)
            radius = min(.95, float(np.quantile(relative_error, .90)))
            self.models[key] = coeff, radius
            self.sample_counts[key[0]] += len(y)

    @lru_cache(maxsize=4096)
    def stage(self, layer, loads, stage, blocks, threads, rows):
        weight_bytes, kind = self.weights[layer, stage]
        coeff, radius = self.models[stage, blocks, threads, rows, kind]
        terms = features(loads, stage, blocks, rows, weight_bytes)
        return max(.1, float(terms @ coeff)), radius
