#!/usr/bin/env python3
"""Analyze fresh RuntimeK mode-local holdout, separating inventory vs policy regret.

Usage: python3 benchmarks/analyze_rk_fp8_runtime_k_selector.py <selector-holdout.csv>
The broad oracle is measured, NOT every physically possible RuntimeK geometry.
"""
import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path

KEY = ('outer', 'R', 'K', 'dtype')


def best(rows):
    return min(rows, key=lambda x: float(x['total_median_ms']))


def label(row):
    return f"O={row['outer']} R={row['R']} K={row['K']} {row['dtype']}"


def metrics(name, regrets):
    percentages = [v[0] for v in regrets]
    absolute = [v[1] for v in regrets]
    print(f"{name:27s} >5%={sum(v>5 for v in percentages):3d}/{len(regrets)} "
          f">10%={sum(v>10 for v in percentages):3d} "
          f"max={max(percentages):6.2f}% max_abs={max(absolute):7.3f}us "
          f"sum_abs={sum(absolute):8.3f}us")


def diff(chosen, oracle):
    a, b = float(chosen['total_median_ms']), float(oracle['total_median_ms'])
    return max(0,100*(a/b-1)), max(0,(a-b)*1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    comments, body = [], []
    with args.file.open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('#'):
                comments.append(line.strip())
            elif line.strip():
                body.append(line)
    if not any('independent_selector_holdout=1' in c for c in comments):
        raise ValueError('Not an independent RuntimeK selector holdout')
    groups = defaultdict(list)
    for row in csv.DictReader(body):
        if row['family'] != 'runtime_k':
            raise ValueError('Non-RuntimeK row')
        groups[tuple(row[x] for x in KEY)].append(row)
    if not groups:
        raise ValueError('No rows')
    inventory_loss, selector_loss, overall_loss, details = [], [], [], []
    for group, rows in sorted(groups.items()):
        broad = best(rows)
        lean_rows = [row for row in rows if row['lean_eligible'] == '1']
        selected = [row for row in rows if row['policy_selected'] == '1']
        if not lean_rows or len(selected) != 1 or selected[0] not in lean_rows:
            raise ValueError(f"Incomplete or illegal selection for {group}: {len(selected)}")
        lean = best(lean_rows)
        inventory_loss.append(diff(lean, broad))
        selector_loss.append(diff(selected[0], lean))
        overall_loss.append(diff(selected[0], broad))
        details.append((diff(selected[0], broad), selected[0], lean, broad))
    print(f"{len(groups)} geometry/dtype cells; {sum(map(len,groups.values()))} physical candidates")
    print('Chosen layouts:',dict(Counter(item[1]['layout']+'_'+item[1]['topology'] for item in details)))
    metrics('Inventory vs broad oracle',inventory_loss)
    metrics('Local selector vs lean oracle',selector_loss)
    metrics('Combined vs broad oracle',overall_loss)
    print('Largest combined misses:')
    for (pct,us),selected,lean,broad in sorted(details,key=lambda x:x[0],reverse=True)[:16]:
        print(f" {label(selected):42s} regret={pct:6.2f}% ({us:7.3f}us) "
              f"selected={selected['layout']}/{selected['topology']}/{selected['rows_per_shard']} "
              f"lean={lean['layout']}/{lean['topology']}/{lean['rows_per_shard']} "
              f"broad={broad['layout']}/{broad['topology']}/{broad['rows_per_shard']}")


if __name__ == '__main__':
    main()
