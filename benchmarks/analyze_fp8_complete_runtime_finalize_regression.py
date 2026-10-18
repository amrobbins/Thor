#!/usr/bin/env python3
"""Validate coverage and summarize the FP8 Complete runtime-finalizer A/B benchmark."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FAMILIES = ('one_window', 'runtime_k')
DTYPES = ('fp8_e4m3', 'fp8_e5m2')
FINALIZERS = ('identity', 'divide', 'sqrt')


def analyze(path: Path):
    lines = path.read_text().splitlines()
    if '# fp8_complete_runtime_finalize_regression_complete' not in lines:
        raise ValueError('Incomplete benchmark: completion marker is missing')
    data = [line for line in lines if line and not line.startswith('#')]
    rows = list(csv.DictReader(data))
    if len(rows) != 4 * 2 * 7:
        raise ValueError(f'Expected 56 physical/output/epilogue rows, found {len(rows)}')
    grouped = defaultdict(dict)
    for row in rows:
        family = row['family']
        dtype = row['input_dtype']
        output = row['output_dtype']
        mode = row['finalize']
        version = row['version']
        if family not in FAMILIES or dtype not in DTYPES or mode not in FINALIZERS:
            raise ValueError(f'Unexpected physical combination: {row}')
        key = (family, row['outer'], row['R'], row['K'], dtype)
        variant = (output, mode, version)
        if variant in grouped[key]:
            raise ValueError(f'Duplicate measured variant: {key}, {variant}')
        latency_ms = float(row['median_ms'])
        if latency_ms <= 0:
            raise ValueError(f'Invalid latency: {row}')
        grouped[key][variant] = latency_ms
    if len(grouped) != 8:
        raise ValueError(f'Expected 8 shape/dtype combinations, found {len(grouped)}')

    ratios = defaultdict(list)
    for key, plan in sorted(grouped.items()):
        dtype = key[-1]
        legacy = plan.get((dtype, 'identity', 'legacy_fp8'))
        if legacy is None:
            raise ValueError(f'Missing legacy baseline for {key}')
        expected = {(dtype, 'identity', 'legacy_fp8')}
        for output in (dtype, 'fp32'):
            for finalizer in FINALIZERS:
                expected.add((output, finalizer, 'runtime'))
        if set(plan) != expected:
            raise ValueError(f'Incomplete set of finalizers for {key}: missing {expected - set(plan)}, unexpected {set(plan) - expected}')
        fp8_identity = plan[(dtype, 'identity', 'runtime')]
        for output in (dtype, 'fp32'):
            identity = plan[(output, 'identity', 'runtime')]
            for finalizer in FINALIZERS:
                candidate = plan[(output, finalizer, 'runtime')]
                ratios[(key[0], output == 'fp32', finalizer)].append((candidate / identity - 1.0) * 100.0)
        ratios[(key[0], False, 'identity_vs_legacy')].append((fp8_identity / legacy - 1.0) * 100.0)
    print(f'{len(grouped)} geometry/dtype cells; {len(rows)} timed plan/finalizer combinations')
    print('Percent overhead is relative to runtime identity with the same output dtype; identity_vs_legacy compares FP8 with the original identity epilogue.')
    print('A negative percentage is faster; small GPU-event differences may be measurement noise.')
    for (family, fp32, finalizer), values in sorted(ratios.items()):
        label = 'fp32' if fp32 else 'fp8'
        print(f'{family:12} {label:4} {finalizer:20} mean={sum(values)/len(values):+7.2f}% '
              f'max={max(values):+7.2f}% min={min(values):+7.2f}% '
              f'>5%={sum(v > 5 for v in values)}/{len(values)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    analyze(args.csv)


if __name__ == '__main__':
    main()
