#!/usr/bin/env python3
"""Summarize the focused FP8 Complete original-vs-runtime store benchmark.

Requires the four fixed geometries, both FP8 input types, and all four runtime
output dtypes. This is a benchmark summary, NOT automatic acceptance: timing
noise and any SASS/register-growth differences still require review.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    with args.csv.open(newline='') as handle:
        rows = list(csv.DictReader(line for line in handle if line.strip() and not line.startswith('#')))
    groups = defaultdict(dict)
    for row in rows:
        group = (row['family'], row['outer'], row['R'], row['K'], row['input_dtype'])
        key = (row['output_dtype'], row['version'])
        if key in groups[group]:
            raise ValueError(f'Duplicate physical plan: {group}, {key}')
        groups[group][key] = row
    expected_outputs = {'fp16', 'bf16', 'fp32'}
    if len(groups) != 8:
        raise ValueError(f'Expected eight geometry/input dtype groups, got {len(groups)}')
    print('FP8 Complete output-epilogue A/B: 9 interleaved GPU samples per plan')
    print('ratio=runtime_fp8_median/legacy_fp8_median; >1 means slower')
    ratios = defaultdict(list)
    for group, plans in sorted(groups.items()):
        input_dtype = group[-1]
        expected = {(input_dtype, 'legacy_fp8'), (input_dtype, 'runtime_output')}
        expected |= {(output, 'runtime_output') for output in expected_outputs}
        if set(plans) != expected:
            raise ValueError(f'Incomplete/extra plans at {group}: {set(plans) ^ expected}')
        old = float(plans[input_dtype, 'legacy_fp8']['median_ms'])
        new = float(plans[input_dtype, 'runtime_output']['median_ms'])
        if old <= 0 or new <= 0:
            raise ValueError(f'Nonpositive timing at {group}')
        ratio = new / old
        ratios[group[0]].append(ratio)
        print(f'{group}: old={old * 1000:.3f}us new={new * 1000:.3f}us '
              f'ratio={ratio:.4f} slowdown={(ratio-1)*100:+.2f}%')
    for family, values in sorted(ratios.items()):
        print(f'{family}: max slowdown={(max(values)-1)*100:+.2f}% '
              f'avg slowdown={(sum(values)/len(values)-1)*100:+.2f}%')
    print('Output-type timings are in the original CSV; cross-type bandwidth is not a same-output A/B.')


if __name__ == '__main__':
    main()
