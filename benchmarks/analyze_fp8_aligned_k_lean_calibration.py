#!/usr/bin/env python3
"""Strict inventory/oracle analyzer for the NEW lean AlignedK calibration sweep.

This is calibration data, not an independent holdout and not a selector claim.
New measurements must not be used to claim independent validation of a model
that is subsequently tuned on them.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import statistics

RK = ((160, 18), (160, 384), (384, 36), (384, 768), (640, 56), (640, 384),
      (1536, 120), (1536, 1536), (3072, 18), (3072, 192), (6144, 56),
      (6144, 768), (12288, 36), (12288, 384), (24576, 120), (24576, 1536))
PACKETS = (2, 4, 8)
CTAS = (32, 64, 128, 256)
TARGET_ROWS = (32, 128)
SAMPLE_COLS = tuple(f'sample{i}_ms' for i in range(5))


def ceildiv(n, d):
    return (n + d - 1) // d


def expected_shapes():
    result = set()
    for r, k in RK:
        for outer in {11, *(max(1, (target + r*k//2)//(r*k))
                            for target in (16*1024**2, 96*1024**2))}:
            result.add((outer, r, k))
    return result


def expected_configs(r, k):
    return {(p, cta, shards)
            for p in PACKETS if k % p == 0
            for cta in CTAS
            for shards in {1, *(ceildiv(r, depth) for depth in TARGET_ROWS if depth < r)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--winners', type=Path, help='Optional complete per-shape winner comparison')
    args = ap.parse_args()
    raw = args.input.read_text()
    if not re.search(r'^# mode=rk_fp8_aligned_k_lean_calibration\b', raw, re.M):
        raise ValueError('Wrong benchmark mode; expected new AlignedK lean calibration')
    match = re.search(r'^# rk_fp8_aligned_k_lean_calibration_complete shapes=(\d+) candidate_configurations=(\d+)\s*$', raw, re.M)
    if not match:
        raise ValueError('Run incomplete: completion marker missing')
    rows = list(csv.DictReader(line for line in raw.splitlines() if not line.startswith('#')))
    required = {'family', 'outer', 'R', 'K', 'topology', 'packet_bytes', 'block_threads',
                'rows_per_shard', 'shards_per_output', 'component_tiles', 'first_stage_ctas',
                'intermediate_bytes', 'resident_ctas_per_sm', 'first_median_ms',
                'continuation_median_ms', 'total_median_ms', *SAMPLE_COLS}
    if len(rows) != int(match.group(2)) or not rows or not required <= rows[0].keys():
        raise ValueError('Candidate count mismatch or missing benchmark columns')
    grouped = defaultdict(dict)
    for row in rows:
        if row['family'] != 'aligned_k':
            raise ValueError('Unexpected physical family')
        o, r, k = (int(row[f]) for f in ('outer', 'R', 'K'))
        p, cta, shards = (int(row[f]) for f in ('packet_bytes', 'block_threads', 'shards_per_output'))
        identity = (p, cta, shards)
        if (o, r, k) not in expected_shapes() or identity not in expected_configs(r, k):
            raise ValueError(f'Illegal or unexpected candidate {o,r,k,identity}')
        if identity in grouped[o, r, k]:
            raise ValueError(f'Duplicate candidate {o,r,k,identity}')
        if (row['topology'] == 'complete') != (shards == 1):
            raise ValueError(f'Topology-shards disagreement at {o,r,k,identity}')
        if int(row['rows_per_shard']) != ceildiv(r, shards):
            raise ValueError('Incorrect maximum row count')
        tiles = ceildiv(k, p*cta)
        if int(row['component_tiles']) != tiles or int(row['first_stage_ctas']) != o*shards*tiles:
            raise ValueError('Invalid component tile / CTA count')
        if int(row['intermediate_bytes']) != (0 if shards == 1 else o*shards*k*4):
            raise ValueError('FP32 intermediate footprint inconsistent')
        if int(row['resident_ctas_per_sm']) <= 0:
            raise ValueError('Nonresident compiled kernel')
        samples = [float(row[c]) for c in SAMPLE_COLS]
        total = float(row['total_median_ms'])
        if total <= 0 or abs(statistics.median(samples)-total) > 1e-6:
            raise ValueError('Total median disagrees with samples')
        if shards == 1 and abs(float(row['continuation_median_ms'])) > 1e-7:
            raise ValueError('Complete must have no FP32 continuation')
        grouped[o, r, k][identity] = row
    if set(grouped) != expected_shapes() or len(grouped) != int(match.group(1)):
        raise ValueError(f'Incomplete geometric coverage: {len(grouped)}/{len(expected_shapes())}')
    results = []
    for (outer, r, k), candidates in sorted(grouped.items()):
        if set(candidates) != expected_configs(r, k):
            raise ValueError(f'Incomplete candidate inventory at {(outer,r,k)}')
        def us(row):
            return 1000*float(row['total_median_ms'])
        vals = list(candidates.values())
        best = min(vals, key=us)
        complete = min((v for v in vals if int(v['shards_per_output']) == 1), key=us)
        staged = min((v for v in vals if int(v['shards_per_output']) > 1), key=us)
        no256 = min((v for v in vals if int(v['block_threads']) != 256), key=us)
        row = {'outer': outer, 'R': r, 'K': k, 'input_mib': outer*r*k/(1024**2),
               'best_topology': best['topology'], 'best_packet': best['packet_bytes'],
               'best_cta': best['block_threads'], 'best_shards': best['shards_per_output'],
               'best_total_us': us(best), 'complete_regret_us': us(complete)-us(best),
               'staged_regret_us': us(staged)-us(best),
               'without_cta256_regret_us': us(no256)-us(best),
               'without_cta256_regret_pct': 100*(us(no256)/us(best)-1)}
        results.append(row)
    print(f'COMPLETE AlignedK LEAN CALIBRATION: {len(results)} shapes, {len(rows)} candidate rows, five samples each')
    print('Oracle topology:', dict(Counter(row['best_topology'] for row in results)))
    print('Oracle packet:', dict(sorted(Counter(int(row['best_packet']) for row in results).items())))
    print('Oracle CTA:', dict(sorted(Counter(int(row['best_cta']) for row in results).items())))
    print('Oracle shards:', dict(sorted(Counter(int(row['best_shards']) for row in results).items())))
    print('Remove CTA256: >5% in', sum(r['without_cta256_regret_pct'] > 5.000001 for r in results),
          'max absolute loss', round(max(r['without_cta256_regret_us'] for r in results), 3), 'us')
    print('Top complete losses:')
    for row in sorted(results, key=lambda r: r['complete_regret_us'], reverse=True)[:5]:
        print(f"  outer={row['outer']} R={row['R']} K={row['K']} "
              f"complete loses {row['complete_regret_us']:.3f}us")
    if args.winners:
        with args.winners.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]));w.writeheader();w.writerows(results)
        print('Saved:', args.winners)
    print('Calibration only. No selector is fitted, accepted, or silently dispatched.')


if __name__ == '__main__':
    main()
