#!/usr/bin/env python3
"""Strict coverage and timing analysis for Complete FP8 runtime-input-transform A/B."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FAMILIES = ('one_window', 'runtime_k')
DTYPES = ('fp8_e4m3', 'fp8_e5m2')
TRANSFORMS = ('identity', 'absolute', 'square')


def analyze(path: Path):
    lines = path.read_text().splitlines()
    if '# fp8_complete_runtime_input_transform_regression_complete' not in lines:
        raise ValueError('Incomplete input-transform benchmark: completion marker missing')
    rows = list(csv.DictReader(line for line in lines if line and not line.startswith('#')))
    if len(rows) != 72:
        raise ValueError(f'Expected exactly 72 timed combinations, found {len(rows)}')
    plans = defaultdict(dict)
    for row in rows:
        key = (row['family'], row['outer'], row['R'], row['K'], row['input_dtype'])
        if key[0] not in FAMILIES or key[-1] not in DTYPES:
            raise ValueError(f'Unexpected family/input: {key}')
        transform = row['transform']
        if transform not in TRANSFORMS:
            raise ValueError(f'Unexpected transform: {transform}')
        variant = (transform, row['version'], row['output_dtype'])
        if variant in plans[key]:
            raise ValueError(f'Duplicate variant {key}: {variant}')
        latency = float(row['median_ms'])
        if latency <= 0:
            raise ValueError(f'Invalid latency: {row}')
        plans[key][variant] = latency
    if len(plans) != 8:
        raise ValueError(f'Expected eight geometry/dtype cells, found {len(plans)}')
    ratios = defaultdict(list)
    for key, candidates in sorted(plans.items()):
        dtype = key[-1]
        expected = {(t, 'compile_time', dtype) for t in TRANSFORMS}
        expected.update((t, 'runtime', dtype) for t in TRANSFORMS)
        expected.update((t, 'runtime', 'fp32') for t in TRANSFORMS)
        if set(candidates) != expected:
            raise ValueError(f'Incomplete candidates at {key}: missing {expected-set(candidates)} extra {set(candidates)-expected}')
        for t in TRANSFORMS:
            base = candidates[(t, 'compile_time', dtype)]
            runtime = candidates[(t, 'runtime', dtype)]
            ratios[(key[0], t)].append((runtime / base - 1.0) * 100)
    print(f'{len(plans)} geometry/dtype cells; {len(rows)} timed transform/output configurations')
    print('The A/B ratio compares identical FP8 output and finalization; the only specialization difference is input transform.')
    print('Positive percentages mean runtime selection is slower. Small event differences may be noise.')
    for (family, transform), values in sorted(ratios.items()):
        print(f'{family:12} {transform:8} mean={sum(values)/len(values):+7.2f}% '
              f'max={max(values):+7.2f}% min={min(values):+7.2f}% '
              f'>5%={sum(v>5 for v in values)}/{len(values)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    analyze(args.csv)


if __name__ == '__main__':
    main()
