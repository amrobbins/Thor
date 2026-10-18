#!/usr/bin/env python3
"""Summarize end-to-end FP8 DeepRSmallK discovery measurements.

The winning configuration minimizes measured first-stage + *production FP32*
continuation latency. Partial results are explicitly marked incomplete.
"""
import argparse
import csv
import io
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEY = ('outer', 'R', 'K')
REQUIRED = ('packet_bytes', 'rows_per_shard', 'shards_per_output',
            'first_median_ms', 'continuation_median_ms', 'total_median_ms')


def load(path):
    text = Path(path).read_text()
    complete = '# rk_fp8_deep_r_small_k_end_to_end_sweep_complete' in text
    errors = [line for line in text.splitlines() if line.startswith('CubReductionBenchmark failed:')]
    lines = [line for line in text.splitlines() if line and not line.startswith('#')
             and not line.startswith('CubReductionBenchmark failed:')]
    if not lines:
        raise ValueError('No benchmark CSV rows found')
    reader = csv.DictReader(io.StringIO('\n'.join(lines)))
    needed = set(KEY + REQUIRED)
    missing = needed.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    by_shape = defaultdict(list)
    for row in reader:
        if row.get('family') != 'deep_r_small_k':
            continue
        try:
            for key in KEY + ('packet_bytes','rows_per_shard','shards_per_output','intermediate_bytes'):
                row[key] = int(row[key])
            for key in ('first_median_ms','continuation_median_ms','total_median_ms'):
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f'Invalid {key}')
            for i in range(5):
                field = f'sample{i}_ms'
                if field in row and row[field]:
                    row[field] = float(row[field])
                    if not math.isfinite(row[field]) or row[field] <= 0:
                        raise ValueError(f'Invalid {field}')
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'Malformed candidate row: {row}') from exc
        by_shape[tuple(row[k] for k in KEY)].append(row)
    if not by_shape:
        raise ValueError('No valid DeepRSmallK candidate rows')
    return by_shape, complete, errors


def analyse(by_shape):
    winners = []
    for shape, candidates in sorted(by_shape.items()):
        best = min(candidates, key=lambda r: (r['total_median_ms'], r['packet_bytes'], r['rows_per_shard']))
        first_only = min(candidates, key=lambda r: (r['first_median_ms'], r['packet_bytes'], r['rows_per_shard']))
        result = {k: best[k] for k in KEY}
        result.update(packet_bytes=best['packet_bytes'], rows_per_shard=best['rows_per_shard'],
                      shards_per_output=best['shards_per_output'], intermediate_bytes=best['intermediate_bytes'],
                      first_us=1000 * best['first_median_ms'],
                      continuation_us=1000 * best['continuation_median_ms'],
                      total_us=1000 * best['total_median_ms'],
                      first_only_packet_bytes=first_only['packet_bytes'],
                      first_only_rows_per_shard=first_only['rows_per_shard'],
                      first_only_total_us=1000 * first_only['total_median_ms'],
                      first_only_regret_us=1000 * (first_only['total_median_ms'] - best['total_median_ms']),
                      candidate_count=len(candidates))
        winners.append(result)
    return winners


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--details', required=True, help='Output CSV with per-geometry winning configurations')
    args = ap.parse_args()
    by_shape, complete, errors = load(args.input)
    winners = analyse(by_shape)
    Path(args.details).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.details).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(winners[0].keys()))
        writer.writeheader()
        writer.writerows(winners)
    print('status:', 'COMPLETE' if complete else 'INCOMPLETE - no sweep completion marker')
    print('geometries:', len(by_shape), 'candidate rows:', sum(map(len, by_shape.values())))
    if errors:
        print('benchmark error:', errors[-1])
    print('winning packet widths:', dict(sorted(Counter(w['packet_bytes'] for w in winners).items())))
    print('winning row depths:', dict(sorted(Counter(w['rows_per_shard'] for w in winners).items())))
    divergences = [w for w in winners if w['first_only_regret_us'] > 0.01]
    print('first-stage-only winner differs and loses >=0.01 us:', len(divergences))
    print('total first-stage-only regret (us):', round(sum(w['first_only_regret_us'] for w in winners), 3))
    print('largest first-stage-only regret (us):', round(max(w['first_only_regret_us'] for w in winners), 3))
    print('fastest end-to-end per geometry:')
    for w in winners:
        print('  outer={outer} R={R} K={K} p={packet_bytes} rows={rows_per_shard} '
              'shards={shards_per_output} first={first_us:.1f}us '
              'FP32={continuation_us:.1f}us total={total_us:.1f}us'.format(**w))
    print('details:', args.details)
    if not complete:
        print('WARNING: Incomplete sweep; do not use partial results for local selector calibration.', file=sys.stderr)


if __name__ == '__main__':
    main()
