#!/usr/bin/env python3
"""Strict fresh RuntimeK top-five holdout analyzer (complete FP8 reductions).

From build-release:
  python3 ../benchmarks/analyze_rk_fp8_runtime_k_shortlist.py \
      rk-fp8-runtime-k-shortlist-holdout.csv

Top-N figures assume an ideal choice of the fastest *measured* candidate from
that shortlist. They measure shortlist recall, not autotuning overhead/success.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

KEY = ('outer', 'R', 'K', 'dtype')
RK = {(48,505), (88,265), (176,502), (448,515), (704,1023),
      (1408,257), (3584,509), (6656,496), (11776,1031), (28672,511)}
DTYPES = {'fp8_e4m3', 'fp8_e5m2'}


def load(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    if not any('independent_shortlist_holdout=1' in s for s in lines if s.startswith('#')):
        raise ValueError('Not a RuntimeK shortlist holdout')
    ends = [s for s in lines if s.startswith('# runtime_k_shortlist_holdout_complete shapes=')]
    if len(ends) != 1 or 'shapes=40 ' not in ends[0]:
        raise ValueError('Missing complete 40-cell holdout marker; do not analyze a partial run')
    noncomments = [s for s in lines if s.strip() and not s.startswith('#')]
    rows = list(csv.DictReader(noncomments))
    if not rows:
        raise ValueError('No timed candidates')
    return rows


def result(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['family'] != 'runtime_k':
            raise ValueError('Unexpected benchmark family')
        key = tuple(row[x] for x in KEY)
        if (int(key[1]), int(key[2])) not in RK or key[3] not in DTYPES:
            raise ValueError(f'Unexpected geometry/dtype {key}')
        groups[key].append(row)
    if len(groups) != 40:
        raise ValueError(f'Incomplete coverage: {len(groups)} of 40 geometry/dtype cells')
    for (r,k) in RK:
        for dtype in DTYPES:
            keys = [key for key in groups if int(key[1]) == r and int(key[2]) == k
                    and key[3] == dtype]
            if len(keys) != 2 or len({x[0] for x in keys}) != 2:
                raise ValueError(f'Expected both input targets: R={r} K={k} {dtype}')
    print(f'{len(groups)} workload/dtype cells; {len(rows)} physical candidates')
    metrics = {i: [] for i in range(1,6)}
    inventory_loss = []
    inventory_percent = []
    detail = []
    selected_layouts = Counter()
    for key, candidates in sorted(groups.items()):
        ranked = [row for row in candidates if int(row['shortlist_rank']) > 0]
        ranks = sorted(int(row['shortlist_rank']) for row in ranked)
        if ranks != list(range(1,len(ranked)+1)) or not 1 <= len(ranked) <= 5:
            raise ValueError(f'Illegal, duplicate or missing shortlist rank: {key}: {ranks}')
        chosen = [row for row in candidates if row['policy_selected'] == '1']
        if len(chosen) != 1 or chosen[0]['shortlist_rank'] != '1':
            raise ValueError(f'First shortlist entry disagrees with baseline selector: {key}')
        if any(row['lean_eligible'] != '1' for row in ranked):
            raise ValueError(f'Unpromoted physical variant in shortlist: {key}')
        lean = [row for row in candidates if row['lean_eligible'] == '1']
        if not lean:
            raise ValueError(f'No legal candidate: {key}')
        if len(ranked) != min(5, len(lean)):
            raise ValueError(f'Incomplete five-plan shortlist: {key}')
        val = lambda x: float(x['total_median_ms'])
        oracle = min(candidates,key=val)
        lean_oracle = min(lean,key=val)
        inventory_loss.append(max(0., val(lean_oracle)-val(oracle))*1000)
        inventory_percent.append(max(0., 100*(val(lean_oracle)/val(oracle)-1)))
        selected_layouts[(chosen[0]['layout'],chosen[0]['topology'])] += 1
        for n in range(1,6):
            best = min((row for row in ranked if int(row['shortlist_rank']) <= n),key=val)
            pct = max(0., 100*(val(best)/val(lean_oracle)-1))
            us = max(0., (val(best)-val(lean_oracle))*1000)
            metrics[n].append((pct,us))
        detail.append((metrics[1][-1], metrics[5][-1], key,
                       chosen[0], min(ranked,key=val), lean_oracle))
    print('Baseline layout choices:', dict(selected_layouts))
    print(f'Measured six-depth inventory vs broad oracle: '
          f'>5%={sum(p>5 for p in inventory_percent)}/40 '
          f'max={max(inventory_percent):.2f}% cumulative={sum(inventory_loss):.3f}us')
    for n, regrets in metrics.items():
        pct, us = zip(*regrets)
        print(f'Top {n} oracle within measured shortlist: '
              f'>5%={sum(x>5 for x in pct):2d}/40, >10%={sum(x>10 for x in pct):2d}, '
              f'max={max(pct):5.2f}%, max_abs={max(us):7.3f}us, '
              f'cumulative={sum(us):8.3f}us')
    print('Largest first-choice misses (vs six-depth oracle):')
    for (pct,us), five, key, selected, shortlist_best, oracle in sorted(
            detail,key=lambda v:v[0],reverse=True)[:12]:
        def ident(v):
            return f'{v["layout"]}/{v["topology"]}/rows{v["rows_per_shard"]}'
        print(f'  O={key[0]} R={key[1]} K={key[2]} {key[3]}: '
              f'first={pct:.2f}% ({us:.3f}us) top5={five[0]:.2f}% ({five[1]:.3f}us) '
              f'first={ident(selected)} best5={ident(shortlist_best)} '
              f'oracle={ident(oracle)}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    args=parser.parse_args()
    result(load(args.csv))


if __name__=='__main__':
    main()
