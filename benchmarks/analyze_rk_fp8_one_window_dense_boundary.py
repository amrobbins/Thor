#!/usr/bin/env python3
"""Analyze ONLY the promoted-vs-demoted OneWindow physical frontier, not cross-mode routing.

No trained model, no shape lookup table. Summaries highlight continuous R/K/outer regimes
where promoted topologies are within 5% and where a demoted oracle falsifies coverage.
"""
import argparse
import collections
import csv
import math
import re

PROMOTED = 'one_window_promoted_'
ORACLE = 'ORACLE_one_window_demoted_'
THRESHOLD = 1.05
SHAPE = ('family', 'case', 'outer', 'R', 'K', 'dtype')


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def band_k(k):
    if k <= 8: return 'K<=8'
    if k <= 32: return '9<=K<=32'
    if k <= 128: return '33<=K<=128'
    if k <= 256: return '129<=K<=256'
    return '257<=K<=512'


def band_r(r):
    if r <= 8: return 'R<=8'
    if r <= 32: return '9<=R<=32'
    if r <= 128: return '33<=R<=128'
    if r <= 512: return '129<=R<=512'
    return 'R>512'


def ratio(a, b):
    return (a / b - 1.0) * 100.0 if b > 0 else float('inf')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--cells', required=True)
    p.add_argument('--near', required=True)
    p.add_argument('--regions', required=True)
    args = p.parse_args()
    groups = collections.defaultdict(list)
    with open(args.input, newline='') as file:
        for row in csv.DictReader(line for line in file if line.strip() and not line.startswith('#')):
            impl = row['implementation']
            if not (impl.startswith(PROMOTED) or impl.startswith(ORACLE)):
                continue
            try:
                row['median'] = float(row['median_ms'])
                row['best'] = float(row['best_ms'])
                row['worst'] = float(row['worst_ms'])
            except (ValueError, KeyError):
                continue
            row['stable'] = row['median'] <= THRESHOLD * row['best']
            row['role'] = 'promoted' if impl.startswith(PROMOTED) else 'demoted_oracle'
            row['variant'] = impl[len(PROMOTED):] if row['role'] == 'promoted' else impl[len(ORACLE):]
            groups[tuple(row[f] for f in SHAPE)].append(row)
    if not groups:
        raise SystemExit('No recognized OneWindow promoted/oracle benchmark rows in CSV.')
    cell_rows, near_rows = [], []
    regions = collections.defaultdict(list)
    win_variants = collections.Counter()
    near_variants = collections.Counter()
    oracle_alerts = []
    incomplete = []
    for key, rows in sorted(groups.items(), key=lambda kv: tuple(int(x) if x.isdigit() else x for x in kv[0])):
        prom = [r for r in rows if r['role'] == 'promoted']
        ctrl = [r for r in rows if r['role'] == 'demoted_oracle']
        if not prom:
            incomplete.append(key)
            continue
        # The raw median determines the primary frontier; stability is a diagnostic and NEVER legality.
        best_p = min(prom, key=lambda r: r['median'])
        best_c = min(ctrl, key=lambda r: r['median']) if ctrl else None
        best_any = min(rows, key=lambda r: r['median'])
        reg = (band_r(int(key[3])), band_k(int(key[4])), key[0])
        gain = ratio(best_p['median'], best_any['median'])
        within = [r for r in prom if r['median'] <= THRESHOLD * best_any['median']]
        stable_prom = [r for r in prom if r['stable']]
        stable_ctrl = [r for r in ctrl if r['stable']]
        best_stable_p = min(stable_prom, key=lambda r: r['median']) if stable_prom else None
        best_stable_c = min(stable_ctrl, key=lambda r: r['median']) if stable_ctrl else None
        stable_gain = ratio(best_stable_p['median'], min(best_stable_p['median'], best_stable_c['median'])) if (best_stable_p and best_stable_c) else None
        win_variants[best_p['variant']] += 1
        for r in within:
            near_variants[r['variant']] += 1
        for r in rows:
            near_rows.append(dict(zip(SHAPE, key)) | {
                'variant':r['variant'], 'role':r['role'], 'median_ms':r['median'],
                'relative_to_best_promoted_pct':round(ratio(r['median'],best_p['median']),3),
                'relative_to_fastest_any_pct':round(ratio(r['median'],best_any['median']),3),
                'within_5pct_promoted':int(r in within),
                'stability_median_over_best_pct':round(ratio(r['median'],r['best']),3),
                'stable_5pct':int(r['stable'])})
        data = dict(zip(SHAPE, key)) | {
            'R_band':reg[0], 'K_band':reg[1],
            'best_promoted':best_p['variant'], 'best_promoted_ms':best_p['median'],
            'promoted_near_5pct':','.join(sorted(r['variant'] for r in within)),
            'promoted_available':len(prom), 'demoted_oracle_available':len(ctrl),
            'best_demoted_oracle':best_c['variant'] if best_c else '',
            'best_demoted_ms':best_c['median'] if best_c else '',
            'best_overall':best_any['variant'],
            'demoted_win_pct':round(gain,3),
            'demoted_falsifies_5pct':int(gain > 5.0),
            'best_promoted_stable_5pct':int(best_p['stable']),
            'best_oracle_stable_5pct':int(best_c['stable']) if best_c else '',
            'stable_regret_pct':round(stable_gain,3) if stable_gain is not None else ''}
        cell_rows.append(data)
        regions[reg].append(data)
        if gain > 5.0:
            oracle_alerts.append(data)
    fields_cells = list(cell_rows[0])
    fields_near = list(near_rows[0])
    write_csv(args.cells,cell_rows,fields_cells)
    write_csv(args.near,near_rows,fields_near)
    region_rows = []
    for reg, rows in sorted(regions.items()):
        bad = [r for r in rows if r['demoted_falsifies_5pct']]
        region_rows.append({
            'R_band':reg[0], 'K_band':reg[1], 'work_target':reg[2],
            'cells':len(rows), 'oracle_better_by_over_5pct':len(bad),
            'oracle_regret_max_pct':round(max(r['demoted_win_pct'] for r in rows),3),
            'best_promoted_counts':';'.join(f'{k}:{v}' for k,v in sorted(collections.Counter(r['best_promoted'] for r in rows).items()))})
    write_csv(args.regions,region_rows,list(region_rows[0]))
    print('ONEWINDOW-ONLY 5% FRONTIER (raw median, no production selector)')
    print(f'cells={len(cell_rows)} promoted_variants=9 demoted_oracle_variants=12 ' 
          f'cases_missing_all_promoted={len(incomplete)}')
    print(f'cells_demoted_control_>5pct_faster={len(oracle_alerts)} ' 
          f'cells_demoted_control_>2pct_faster={sum(r["demoted_win_pct"]>2 for r in cell_rows)}')
    print(f'promoted_best_unstable_cells={sum(not r["best_promoted_stable_5pct"] for r in cell_rows)}')
    print('PROMOTED BEST COUNTS', dict(sorted(win_variants.items())))
    print('PROMOTED WITHIN-5% COUNTS', dict(sorted(near_variants.items())))
    print('BY PHYSICAL R/K REGIME')
    for row in region_rows:
        print(f'{row["R_band"]:14s} {row["K_band"]:14s} {row["work_target"]:13s} '
              f'{row["oracle_better_by_over_5pct"]:3d}/{row["cells"]:3d} '
              f'max_oracle_gain_pct={row["oracle_regret_max_pct"]:.2f}')
    print('LARGEST DEMOTED-ORACLE CHALLENGES')
    for row in sorted(oracle_alerts,key=lambda x:-x['demoted_win_pct'])[:30]:
        print(f'R={row["R"]:<5s} K={row["K"]:<4s} outer={row["outer"]:<9s} work={row["case"]:22s} '
              f'{row["best_promoted"]} -> {row["best_demoted_oracle"]} '
              f'gain_pct={row["demoted_win_pct"]:.2f} stable_P={row["best_promoted_stable_5pct"]} stable_O={row["best_oracle_stable_5pct"]}')
    print('These are ONEWINDOW-INTERNAL relative wins; no RuntimeK/AlignedK/DeepRSmallK conclusions are possible.')
    if incomplete:
        print('INCOMPLETE CELLS', incomplete[:15])

if __name__ == '__main__':
    main()
