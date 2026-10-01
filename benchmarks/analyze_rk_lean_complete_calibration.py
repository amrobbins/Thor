#!/usr/bin/env python3
"""Summarize --rk-lean-complete-calibration output."""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def rows(path):
    with path.open(newline='') as f:
        lines = (line for line in f if line.strip() and not line.startswith('#'))
        yield from csv.DictReader(lines)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-lean-complete-calibration.csv>")
    data = list(rows(Path(sys.argv[1])))
    print(f"rows={len(data)}")
    grouped = defaultdict(list)
    prod = {}
    for r in data:
        key = (int(r['outer']), int(r['R']), int(r['K']), r['dtype'])
        if r['implementation'] == 'production_selector':
            prod[key] = float(r['median_ms'])
        else:
            grouped[key].append(r)

    comparisons = []
    print('BEST IDENTICAL-GEOMETRY CURRENT VS LEAN')
    for key in sorted(grouped):
        by_geom = defaultdict(dict)
        for r in grouped[key]:
            geom = (int(r['packet_bytes']), int(r['block_threads']))
            by_geom[geom][r['implementation']] = r
        best = None
        for geom, pair in by_geom.items():
            if 'current_kparallel_complete' not in pair or 'lean_kparallel_complete' not in pair:
                continue
            cur = float(pair['current_kparallel_complete']['median_ms'])
            lean = float(pair['lean_kparallel_complete']['median_ms'])
            speedup = cur / lean
            item = (speedup, geom, cur, lean)
            if best is None or item[0] > best[0]:
                best = item
        if best is None:
            continue
        speedup, geom, cur, lean = best
        p = prod.get(key)
        lean_prod = lean / p if p else float('nan')
        comparisons.append((speedup, lean_prod, key, geom, cur, lean, p))
        o, rr, k, dtype = key
        print(f"o{o}_r{rr}_i{k} {dtype}: current/lean={speedup:.4f} "
              f"packet={geom[0]} threads={geom[1]} current={cur:.4f}ms lean={lean:.4f}ms "
              f"production={p:.4f}ms lean/production={lean_prod:.4f}")

    if not comparisons:
        return
    speedups = sorted(x[0] for x in comparisons)
    prod_ratios = sorted(x[1] for x in comparisons if x[1] == x[1])
    def median(v): return v[len(v)//2]
    print('\nSUMMARY')
    print(f"paired_shapes={len(comparisons)} current_over_lean_median={median(speedups):.4f} "
          f"current_over_lean_best={max(speedups):.4f}")
    if prod_ratios:
        print(f"lean_over_production_median={median(prod_ratios):.4f} "
              f"lean_over_production_p95={prod_ratios[min(len(prod_ratios)-1, int(len(prod_ratios)*0.95))]:.4f} "
              f"lean_over_production_worst={max(prod_ratios):.4f}")
    print('\nLARGEST CURRENT -> LEAN SPEEDUPS')
    for speedup, lean_prod, key, geom, cur, lean, p in sorted(comparisons, reverse=True)[:25]:
        o, rr, k, dtype = key
        print(f"{speedup:.4f}  o{o}_r{rr}_i{k}/{dtype} packet={geom[0]} threads={geom[1]} "
              f"lean/production={lean_prod:.4f}")


if __name__ == '__main__':
    main()
