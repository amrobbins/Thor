#!/usr/bin/env python3
"""Analyze AlignedK local-inventory discovery measurements (not a selector holdout).

Run after:
  ./build-release/thor_cub_reduction_benchmark \
    --rk-fp8-aligned-k-end-to-end-sweep > rk-fp8-aligned-k-end-to-end.csv

Required dimensions: the existing compiled production KParallel p2/p4/p8/p16,
CTA32/64/128/256, and Complete or staged sharding. Staged total includes the
stamped production FP32 continuation. Reports only measured inventory/oracles;
no selector, fitting, or production-routing claims are made.
"""
import argparse
import collections
import csv
from pathlib import Path
import re
import statistics

RK = ((96, 32), (96, 256), (256, 64), (256, 1024), (768, 32), (768, 512),
      (2048, 128), (2048, 2048), (8192, 32), (8192, 1024), (16384, 64), (16384, 256))
PACKETS = (2, 4, 8, 16)
CTAS = (32, 64, 128, 256)
ROWS = (32, 128, 512, 2048)
SAMPLES = tuple(f'sample{i}_ms' for i in range(5))


def expected_shapes():
    result = set()
    for r, k in RK:
        for outer in {17, *[max(1, (target + r*k//2)//(r*k))
                             for target in (20*1024**2, 96*1024**2)]}:
            result.add((outer, r, k))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--winners', type=Path, help='Optional per-geometry winner CSV')
    args = parser.parse_args()
    raw = args.input.read_text()
    if not re.search(r'^# mode=rk_fp8_aligned_k_end_to_end_sweep\b', raw, re.M):
        raise ValueError('Wrong benchmark mode')
    complete = re.search(r'^# rk_fp8_aligned_k_end_to_end_sweep_complete shapes=(\d+) candidate_configurations=(\d+)\s*$', raw, re.M)
    if not complete:
        raise ValueError('Incomplete run: missing completion marker')
    with args.input.open(newline='') as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    required = {'family', 'outer', 'R', 'K', 'topology', 'packet_bytes', 'block_threads',
                'rows_per_shard', 'shards_per_output', 'first_stage_ctas', 'component_tiles',
                'resident_ctas_per_sm', 'first_median_ms', 'continuation_median_ms',
                'total_median_ms', *SAMPLES}
    if not rows or not required <= rows[0].keys():
        raise ValueError('Missing benchmark columns')
    if len(rows) != int(complete.group(2)):
        raise ValueError('Candidate-count mismatch against completion marker')
    grouped = collections.defaultdict(list)
    seen = set()
    for row in rows:
        if row['family'] != 'aligned_k':
            raise ValueError('Unexpected physical mode in candidate rows')
        outer, r, k = [int(row[x]) for x in ('outer', 'R', 'K')]
        p, cta, shards = [int(row[x]) for x in ('packet_bytes', 'block_threads', 'shards_per_output')]
        topology = row['topology']
        if p not in PACKETS or cta not in CTAS or k % p or shards < 1:
            raise ValueError('Illegal packet/CTA/geometry in measured candidate')
        if (topology == 'complete') != (shards == 1):
            raise ValueError('Topology/shard mismatch')
        if int(row['resident_ctas_per_sm']) <= 0 or int(row['first_stage_ctas']) < 1:
            raise ValueError('Invalid compiled kernel occupancy or CTA supply')
        identity = (outer, r, k, p, cta, shards)
        if identity in seen:
            raise ValueError(f'Duplicate candidate {identity}')
        seen.add(identity)
        totals = [float(row[col]) for col in SAMPLES]
        elapsed = float(row['total_median_ms'])
        if elapsed <= 0 or abs(statistics.median(totals) - elapsed) > 1e-6:
            raise ValueError('Invalid or inconsistent total median')
        if topology == 'complete' and abs(float(row['continuation_median_ms'])) > 1e-9:
            raise ValueError('Complete should have no continuation')
        row['_elapsed'] = elapsed
        grouped[(outer, r, k)].append(row)
    shapes = expected_shapes()
    if set(grouped) != shapes or len(grouped) != int(complete.group(1)):
        raise ValueError('Shape count or intended R/K/outer coverage mismatch')
    winners = []
    for (outer, r, k), candidates in sorted(grouped.items()):
        expected = {(p, cta, shards)
                    for p in PACKETS if k % p == 0
                    for cta in CTAS
                    for shards in {1, *[(r + depth - 1)//depth for depth in ROWS if depth < r]}}
        actual = {(int(c['packet_bytes']), int(c['block_threads']), int(c['shards_per_output']))
                  for c in candidates}
        if actual != expected:
            raise ValueError(f'Missing or extra candidates at {(outer,r,k)}: {expected^actual}')
        best = min(candidates, key=lambda c: c['_elapsed'])
        complete_best = min((c for c in candidates if c['topology'] == 'complete'),
                            key=lambda c: c['_elapsed'])
        staged_best = min((c for c in candidates if c['topology'] == 'staged'),
                          key=lambda c: c['_elapsed'])
        winners.append({'outer': outer, 'R': r, 'K': k, 'input_mib': outer*r*k/1024**2,
                        'winner_topology': best['topology'],
                        'winner_packet_bytes': best['packet_bytes'],
                        'winner_block_threads': best['block_threads'],
                        'winner_shards': best['shards_per_output'],
                        'winner_rows_per_shard': best['rows_per_shard'],
                        'winner_total_us': best['_elapsed']*1000,
                        'complete_best_us': complete_best['_elapsed']*1000,
                        'staged_best_us': staged_best['_elapsed']*1000,
                        'complete_over_oracle_us': (complete_best['_elapsed']-best['_elapsed'])*1000,
                        'staged_over_oracle_us': (staged_best['_elapsed']-best['_elapsed'])*1000})
    print(f'AlignedK E2E inventory census COMPLETE: shapes={len(grouped)}, candidates={len(rows)}')
    print('Winner topology:', dict(collections.Counter(w['winner_topology'] for w in winners)))
    print('Winner packets:', dict(sorted(collections.Counter(int(w['winner_packet_bytes']) for w in winners).items())))
    print('Winner CTA widths:', dict(sorted(collections.Counter(int(w['winner_block_threads']) for w in winners).items())))
    print('Winner staged shards:', dict(sorted(collections.Counter(int(w['winner_shards']) for w in winners
                                                        if w['winner_topology'] == 'staged').items())))
    print('Largest Complete-vs-staged gaps (positive means Complete loses):')
    for row in sorted(winners, key=lambda x: x['complete_over_oracle_us'], reverse=True)[:10]:
        print(f"  outer={row['outer']} R={row['R']} K={row['K']} {row['input_mib']:.1f}MiB "
              f"best={row['winner_topology']} p{row['winner_packet_bytes']} CTA{row['winner_block_threads']} "
              f"shards={row['winner_shards']} total={row['winner_total_us']:.3f}us "
              f"complete_regret={row['complete_over_oracle_us']:.3f}us")
    if args.winners:
        with args.winners.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(winners[0]))
            writer.writeheader()
            writer.writerows(winners)
        print('Saved:', args.winners)
    print('Inventory discovery only: local AlignedK score, pruning and independent holdout remain pending.')


if __name__ == '__main__':
    main()
