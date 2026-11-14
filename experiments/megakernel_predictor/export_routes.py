#!/usr/bin/env python3
"""Export ordered expert loads from the resident-Qwen capture harness."""
import argparse
import csv
from pathlib import Path
import struct


HEADER = struct.Struct('<4i')
HIDDEN = 2048
EXPERTS = 256
TOP_K = 8


def integer_bytes(count):
    return 4 * count


def export(source, target):
    writer = csv.writer(target, lineterminator='\n')
    writer.writerow(('layer', 'cohort', 'step', 'loads'))
    while header := source.read(HEADER.size):
        if len(header) != HEADER.size:
            raise ValueError('incomplete capture header')
        layer, batch, cohort, step = HEADER.unpack(header)
        if not 0 <= layer < 40 or not 1 <= batch <= 8:
            raise ValueError('invalid capture header')
        input_bytes = integer_bytes(batch * (HIDDEN + EXPERTS))
        ids_bytes = integer_bytes(batch * TOP_K)
        output_bytes = integer_bytes(batch * HIDDEN)
        payload = source.read(input_bytes + ids_bytes + output_bytes)
        if len(payload) != input_bytes + ids_bytes + output_bytes:
            raise ValueError('incomplete capture payload')
        ids = struct.unpack_from(f'<{batch * TOP_K}i', payload, input_bytes)
        load = {}
        for token in range(batch):
            selected = ids[token * TOP_K:(token + 1) * TOP_K]
            if len(set(selected)) != TOP_K or min(selected) < 0 or max(selected) >= EXPERTS:
                raise ValueError('invalid token expert IDs')
            for expert in selected:
                load[expert] = load.get(expert, 0) + 1
        writer.writerow((layer, cohort, step, ':'.join(map(str, load.values()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    with args.capture.open('rb') as source, args.csv.open('w', newline='') as target:
        export(source, target)
