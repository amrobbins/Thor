#!/usr/bin/env python3
"""OneWindow-only 10-variant selector regret and 5% frontier validation.

Use raw median timings for the primary decision; stability is a separate diagnostic.
No generalized-mode routing conclusion can be drawn from these measurements.
"""
import argparse
import collections
import csv
import math
import statistics

SELECT = 'one_window_selected_'
CANDIDATE = 'one_window_candidate_'
ORACLE = 'ORACLE_one_window_demoted_'
SHAPE = ('family', 'outer', 'R', 'K', 'dtype')


def pct(a, b):
    return (a / b - 1.0) * 100 if b > 0 else float('inf')


def write(path, rows, fields):
    with open(path, 'w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def rband(r):
    if r <= 8: return 'R<=8'
    if r <= 32: return '9<=R<=32'
    if r <= 128: return '33<=R<=128'
    if r <= 512: return '129<=R<=512'
    return 'R>512'


def kband(k):
    if k <= 8: return 'K<=8'
    if k <= 32: return '9<=K<=32'
    if k <= 128: return '33<=K<=128'
    if k <= 256: return '129<=K<=256'
    return 'K>256'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--cells', required=True)
    parser.add_argument('--regions', required=True)
    args = parser.parse_args()
    grouped = collections.defaultdict(list)
    with open(args.input, newline='') as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.startswith('#'))
        required = {*SHAPE, 'implementation', 'median_ms', 'best_ms', 'worst_ms'}
        if not required.issubset(reader.fieldnames or ()):
            raise SystemExit(f'Wrong CSV schema: missing {required - set(reader.fieldnames or ())}')
        for row in reader:
            name = row['implementation']
            if name.startswith(SELECT):
                role, variant = 'selected', name[len(SELECT):]
            elif name.startswith(CANDIDATE):
                role, variant = 'candidate', name[len(CANDIDATE):]
            elif name.startswith(ORACLE):
                role, variant = 'oracle', name[len(ORACLE):]
            else:
                continue
            row['role'] = role
            row['variant'] = variant
            try:
                row['latency'] = float(row['median_ms'])
                row['best'] = float(row['best_ms'])
                row['worst'] = float(row['worst_ms'])
            except ValueError:
                raise SystemExit(f'Invalid timing data in {name}')
            if row['latency'] <= 0 or not math.isfinite(row['latency']):
                raise SystemExit(f'Invalid latency for {name}')
            grouped[tuple(row[field] for field in SHAPE)].append(row)
    if not grouped:
        raise SystemExit('No selector-validation results found; check benchmark flag and CSV.')
    report, misses = [], []
    for key, rows in sorted(grouped.items()):
        selected = [r for r in rows if r['role']=='selected']
        candidates = [r for r in rows if r['role'] in ('selected', 'candidate')]
        oracle = [r for r in rows if r['role']=='oracle']
        if len(selected)!=1:
            raise SystemExit(f'Expected exactly one selector choice, got {len(selected)} at {key}')
        if not candidates: raise SystemExit(f'No promoted candidates at {key}')
        if len({r['variant'] for r in rows})!=len(rows):
            raise SystemExit(f'Duplicate variant measurements at {key}')
        winner = min(candidates, key=lambda x:x['latency'])
        fastest = min(rows, key=lambda x:x['latency'])
        chosen = selected[0]
        regret = pct(chosen['latency'],winner['latency'])
        all_regret = pct(chosen['latency'],fastest['latency'])
        oracle_gain = pct(winner['latency'], fastest['latency'])
        near = [r['variant'] for r in candidates if r['latency'] <= winner['latency']*1.05]
        def stable(r): return r['latency'] <= 1.05 * r['best']
        data = dict(zip(SHAPE,key)) | {
            'selected':chosen['variant'], 'selected_ms':chosen['latency'],
            'best_of_ten':winner['variant'],'best_of_ten_ms':winner['latency'],
            'best_of_all':fastest['variant'],'best_of_all_ms':fastest['latency'],
            'selector_regret_pct':round(regret,4),
            'selector_vs_all_regret_pct':round(all_regret,4),
            'oracle_best_advantage_pct':round(oracle_gain,4),
            'selector_over_5pct':int(regret>5.0+1e-8),
            'oracle_falsifies_ten_5pct':int(oracle_gain>5.0+1e-8),
            'promoted_within_5pct':','.join(sorted(near)),
            'selected_stable_5pct':int(stable(chosen)),
            'winner_stable_5pct':int(stable(winner)),
            'oracle_rows':len(oracle),'candidate_rows':len(candidates),
            'R_band':rband(int(key[2])),'K_band':kband(int(key[3]))}
        report.append(data)
        if regret>5: misses.append(data)
    regions = collections.defaultdict(list)
    for row in report:
        regions[(row['family'],row['R_band'],row['K_band'])].append(row)
    region_rows=[]
    for key, rows in sorted(regions.items()):
        region_rows.append(dict(zip(('family','R_band','K_band'),key)) | {
            'cells':len(rows),
            'selector_regret_over_5pct':sum(r['selector_over_5pct'] for r in rows),
            'oracle_falsifies_ten_5pct':sum(r['oracle_falsifies_ten_5pct'] for r in rows),
            'selector_regret_p95_pct':round(sorted(r['selector_regret_pct'] for r in rows)[int(.95*(len(rows)-1))],3),
            'most_common_selection':collections.Counter(r['selected'] for r in rows).most_common(1)[0][0]})
    write(args.cells,report,list(report[0]))
    write(args.regions,region_rows,list(region_rows[0]))
    regrets=sorted(r['selector_regret_pct'] for r in report)
    print('ONEWINDOW-ONLY VALIDATION: 10 promoted candidates + 11 demoted oracles; raw median')
    print(f'cells={len(report)} selected_missing=0 selected_illegal=0 selector_>5pct={len(misses)} '
          f'selector_>10pct={sum(r["selector_regret_pct"]>10 for r in report)} '
          f'oracle_>5pct={sum(r["oracle_falsifies_ten_5pct"] for r in report)}')
    print(f'regret median={statistics.median(regrets):.2f}% p90={regrets[int(.9*(len(regrets)-1))]:.2f}% '
          f'p95={regrets[int(.95*(len(regrets)-1))]:.2f}% max={regrets[-1]:.2f}%')
    print(f'nonstable_selected={sum(not r["selected_stable_5pct"] for r in report)} '
          f'nonstable_best_of_ten={sum(not r["winner_stable_5pct"] for r in report)} '
          '(stability never controls selection eligibility)')
    print('selector_counts=',dict(collections.Counter(r['selected'] for r in report).most_common()))
    print('REGIONS')
    for r in region_rows:
        print(f'{r["family"]:13s} {r["R_band"]:15s} {r["K_band"]:14s} '
              f'>5%={r["selector_regret_over_5pct"]:3d}/{r["cells"]:3d} '
              f'oracle>5%={r["oracle_falsifies_ten_5pct"]:3d} '
              f'p95={r["selector_regret_p95_pct"]:6.2f}%')
    print('WORST SELECTOR MISSES')
    for r in sorted(misses,key=lambda x:-x['selector_regret_pct'])[:30]:
        print(f'R={r["R"]:<5s} K={r["K"]:<4s} outer={r["outer"]:<9s} '
              f'{r["family"]:13s} {r["selected"]} -> {r["best_of_ten"]} '
              f'regret={r["selector_regret_pct"]:.2f}%')
    print('No global-mode or production-dispatch claims follow from this benchmark.')

if __name__=='__main__':
    main()
