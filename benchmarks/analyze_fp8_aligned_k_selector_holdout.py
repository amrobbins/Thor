#!/usr/bin/env python3
"""Strict evaluation of the *experimental* nine-combination AlignedK selector.

The selector is not production-ready and is not enabled in the general selector.
No fitting/tuning of any threshold is performed by this script.
"""
import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

RK = [(112, 26), (112, 320), (192, 42), (192, 576), (352, 74), (352, 840),
      (704, 50), (704, 672), (1408, 106), (1408, 960), (3584, 66),
      (3584, 768), (7168, 114), (7168, 448), (14336, 82), (14336, 896)]
TARGETS = [24 * 2**20, 80 * 2**20, 160 * 2**20]


def ceildiv(n, d):
    return (n + d - 1) // d


def shapes():
    for r, k in RK:
        outers = [37]
        for goal in TARGETS:
            outer = max(1, (goal + r * k // 2) // (r * k))
            if outer not in outers:
                outers.append(outer)
        for outer in outers:
            yield outer, r, k


def inventory(o, r, k):
    out = set()
    for p in (2, 8):
        if k % p:
            continue
        for threads in (32, 64, 128):
            if (p, threads) in ((2, 32), (8, 32), (8, 64)):
                out.add((p, threads, 1))
            for target in (32, 128):
                if r > target:
                    out.add((p, threads, ceildiv(r, target)))
    return out


def choose(o, r, k, allowed):
    """Exactly mirrors Fp8AlignedKPolicy::experimentalSelect for resident kernels."""
    output = o * k
    complete = r <= 32 or (r <= 256 and
                                (output >= 300000 or (k <= 64 and output >= 180000)))
    target = 128 if o * r * k >= 50 * 2**20 else 32
    p = 8 if k >= 128 and k % 8 == 0 else 2
    per_row = ceildiv(k, p)
    th = 32 if per_row < 32 else 64 if per_row < 80 else 128
    if complete:
        th = 64 if p == 8 and per_row >= 32 else 32
    shards = 1 if complete else ceildiv(r, target)
    preferred = (p, th, shards)
    if preferred in allowed:
        return preferred
    return min(allowed, key=lambda v: (0 if v[2] == shards else 1_000_000) +
               (0 if v[0] == p else 10_000) + abs(v[1] - th))


def analyze(source, details):
    raw = Path(source).read_text()
    metadata = [line for line in raw.splitlines() if line.startswith('#')]
    if not any('mode=rk_fp8_aligned_k_selector_holdout ' in line for line in metadata):
        raise ValueError('wrong benchmark mode or missing benchmark header')
    marker = re.search(r'^# rk_fp8_aligned_k_selector_holdout_complete shapes=(\d+) candidate_configurations=(\d+)\s*$',
                       raw, flags=re.MULTILINE)
    if marker is None:
        raise ValueError('missing completion marker: incomplete run')
    reader = csv.DictReader(line for line in raw.splitlines() if line and not line.startswith('#'))
    groups = {}
    for row in reader:
        if row['family'] != 'aligned_k':
            raise ValueError('unexpected family')
        o, r, k = (int(row[x]) for x in ('outer', 'R', 'K'))
        v = (int(row['packet_bytes']), int(row['block_threads']), int(row['shards_per_output']))
        if v not in inventory(o, r, k):
            raise ValueError(f'illegal variant {v} for {(o, r, k)}')
        topology = 'complete' if v[2] == 1 else 'staged'
        if row['topology'] != topology:
            raise ValueError('inconsistent topology')
        if int(row['rows_per_shard']) != ceildiv(r, v[2]):
            raise ValueError('effective rows_per_shard mismatch')
        tiles = ceildiv(k, v[0] * v[1])
        if int(row['component_tiles']) != tiles or int(row['first_stage_ctas']) != o * v[2] * tiles:
            raise ValueError('CTA/component geometry mismatch')
        if int(row['intermediate_bytes']) != (0 if v[2] == 1 else o * v[2] * k * 4):
            raise ValueError('partial footprint mismatch')
        if int(row['resident_ctas_per_sm']) < 1:
            raise ValueError('nonresident kernel')
        samples = [float(row[f'sample{i}_ms']) for i in range(5)]
        observed = float(row['total_median_ms'])
        if not all(math.isfinite(x) and x > 0 for x in samples + [observed]):
            raise ValueError('bad timing')
        if abs(sorted(samples)[2] - observed) > 0.000001:
            raise ValueError('sample median mismatch')
        if float(row['first_median_ms']) < 0 or float(row['continuation_median_ms']) < 0:
            raise ValueError('negative stage timing')
        key = (o, r, k)
        grouped = groups.setdefault(key, {})
        if v in grouped:
            raise ValueError(f'duplicate candidate {key}, {v}')
        grouped[v] = row
    expected = set(shapes())
    if set(groups) != expected:
        raise ValueError(f'geometry coverage mismatch: missing={expected-set(groups)}, extra={set(groups)-expected}')
    nrows = sum(len(g) for g in groups.values())
    for key, group in groups.items():
        legal = inventory(*key)
        if group.keys() != legal:
            raise ValueError(f'inventory mismatch for {key}: missing={legal-group.keys()}, extra={group.keys()-legal}')
    if int(marker[1]) != len(groups) or int(marker[2]) != nrows:
        raise ValueError('completion marker differs from actual rows')
    rows = []
    for key, group in groups.items():
        pick = choose(*key, group.keys())
        winner = min(group, key=lambda v: float(group[v]['total_median_ms']))
        v = float(group[pick]['total_median_ms']) * 1000
        best = float(group[winner]['total_median_ms']) * 1000
        r = v - best
        rows.append(dict(outer=key[0], R=key[1], K=key[2],
                         input_mib=key[0]*key[1]*key[2]/2**20,
                         chosen_packet=pick[0], chosen_cta=pick[1], chosen_shards=pick[2],
                         oracle_packet=winner[0], oracle_cta=winner[1], oracle_shards=winner[2],
                         chosen_us=v, oracle_us=best, selector_regret_us=r,
                         selector_regret_pct=100*r/best))
    pct = [r['selector_regret_pct'] for r in rows]
    losses = [r['selector_regret_us'] for r in rows]
    print(f'COMPLETE: shapes={len(groups)}, candidates={nrows}; allowed inventory only; ' \
          'the full-inventory oracle is not measured in this sweep')
    print(f'Selector regret vs nine-inventory oracle: >5%={sum(x>5.0001 for x in pct)}/{len(rows)}, '
          f'>10%={sum(x>10.0001 for x in pct)}/{len(rows)}, '
          f'max_abs_us={max(losses):.3f}, total_us={sum(losses):.3f}, max_pct={max(pct):.3f}')
    for label, pred in [('under24MiB', lambda x:x < 24),
                        ('24to96MiB',lambda x:24 <= x < 96),
                        ('96MiBplus', lambda x:x >= 96)]:
        arr = [r for r in rows if pred(r['input_mib'])]
        if arr:
            print(f'  {label}: {len(arr)} shapes, >5%={sum(x["selector_regret_pct"]>5.0001 for x in arr)}, '
                  f'max_abs_us={max(x["selector_regret_us"] for x in arr):.3f}')
    print('Largest misses:')
    for r in sorted(rows, key=lambda x:x['selector_regret_us'], reverse=True)[:10]:
        print(' ', r['outer'], r['R'], r['K'], 'selected',
              (r['chosen_packet'], r['chosen_cta'], r['chosen_shards']), 'oracle',
              (r['oracle_packet'], r['oracle_cta'], r['oracle_shards']),
              f'{r["selector_regret_us"]:.3f}us {r["selector_regret_pct"]:.2f}%')
    if details:
        with open(details, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)
        print('Saved:', details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--details')
    args = parser.parse_args()
    analyze(args.input, args.details)


if __name__ == '__main__':
    main()
