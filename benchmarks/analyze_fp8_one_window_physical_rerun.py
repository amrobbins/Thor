#!/usr/bin/env python3
"""Reassess the unchanged 891-calibrated OneWindow physical score on an interleaved 10-sample rerun.

The benchmark CSV contains summary rows in the original 892 schema and commented
``# one_window_samples,...`` lines preserving chronological timings. No model
calibration or new selection thresholds are performed by this analyzer.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_onewindow_natural_fit import add_physics, evaluate

KEY = ('family', 'outer', 'R', 'K', 'dtype')
PREFIX = 'one_window_samples'


def load_samples(path):
    out = {}
    with open(path, newline='') as f:
        for line in f:
            if not line.startswith('# ' + PREFIX + ','):
                continue
            fields = next(csv.reader([line[2:]]))
            # The column legend has a nonnumeric `sample_0_ms` field and is skipped.
            if fields[1] == 'family':
                continue
            if len(fields) != 17:
                raise ValueError(f'Expected ten round samples, got {len(fields)-7}: {line[:120]}')
            _, family, outer, r, k, dtype, variant, *values = fields
            key = (family, int(outer), int(r), int(k), dtype, variant)
            if key in out:
                raise ValueError(f'Duplicate timing sample series: {key}')
            out[key] = np.array([float(x) for x in values], dtype=np.float64)
    return out


def describe(label, df):
    if df.empty:
        print(f'{label}: no cases')
        return
    misses = df[df.regret_pct > 5.0001]
    regret = df.regret_pct.to_numpy()
    lost = df.lost_us.to_numpy()
    print(f'{label}: geometries={len(df)} >5%={len(misses)} >10%={(regret > 10.0001).sum()} '
          f'p90={np.percentile(regret, 90):.2f}% p95={np.percentile(regret,95):.2f}% '
          f'max_regret={regret.max():.2f}% total_lost={lost.sum():.1f}us '
          f'max_lost={lost.max():.1f}us')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='Fresh 10-sample interleaved benchmark CSV')
    ap.add_argument('--details', default='one_window_interleaved_physical_details.csv')
    ap.add_argument('--min-selected-us', type=float, default=10.0,
                    help='Focus on model-selected kernels taking at least this many microseconds')
    ap.add_argument('--significant-loss-us', type=float, default=2.0,
                    help='Also report absolute latency losses exceeding this threshold')
    args = ap.parse_args()
    if args.min_selected_us < 0 or args.significant_loss_us < 0:
        ap.error('Thresholds must be nonnegative')

    raw = pd.read_csv(args.input, comment='#')
    d = add_physics(raw)
    result = evaluate(d)
    result['selected_us'] = 1000 * result.median_ms_selected
    result['best_us'] = 1000 * result.median_ms_best
    result['lost_us'] = result.selected_us - result.best_us
    # The benchmark's original policy choice remains a useful control, but the
    # 379-case model is an offline physical score, not Fp8OneWindowSelector.h.
    if 'old_ms' in result:
        result['old_lost_us'] = 1000 * (result.old_ms - result.median_ms_best)

    samples = load_samples(args.input)
    if not samples:
        raise SystemExit('Missing interleaved timing series. Was the patched benchmark run?')
    paired_spreads = []
    missing = []
    sampled_nonoverlap = []
    round_selected_slower = []
    for row in result.itertuples():
        key = (row.family, int(row.outer), int(row.R), int(row.K), row.dtype)
        sk = key + (row.variant_selected,)
        bk = key + (row.variant_best,)
        if sk not in samples or bk not in samples:
            missing.append((sk, bk))
            paired_spreads.append(float('nan'))
            sampled_nonoverlap.append(False)
            round_selected_slower.append(float('nan'))
            continue
        st = samples[sk]
        bt = samples[bk]
        paired_spreads.append(float(np.median((st - bt) * 1000)))
        sampled_nonoverlap.append(bool(min(st) > max(bt)))
        round_selected_slower.append(float(np.mean(st > 1.05 * bt)))
    if missing:
        raise SystemExit(f'Missing ten-sample data for {len(missing)} cells; first {missing[0]}')
    result['paired_median_loss_us'] = paired_spreads
    result['nonoverlapping_ranges'] = sampled_nonoverlap
    result['fraction_rounds_selected_over_5pct'] = round_selected_slower

    Path(args.details).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.details, index=False)

    print('Physical model: frozen 891-calibrated coefficients; ten promoted variants; NO re-fitting')
    print('10 samples/variant; sequential launches interleaved across candidates by round')
    describe('ALL',result)
    focus = result[result.selected_us >= args.min_selected_us]
    describe(f'FOCUS selected >= {args.min_selected_us:g}us',focus)
    for family, part in focus.groupby('family',sort=True):
        describe(f'  {family}',part)
    for limit in (20,50,100):
        describe(f'  selected >= {limit}us',result[result.selected_us >= limit])
    important = focus[(focus.regret_pct > 5.0001) & (focus.lost_us >= args.significant_loss_us)]
    describe(f'FOCUS >5% and >= {args.significant_loss_us:g}us wasted',important)
    print(f'FOCUS misses with disjoint 10-sample timing ranges: '
          f'{int(focus.loc[focus.regret_pct > 5.0001, "nonoverlapping_ranges"].sum())}'
          f'/{int((focus.regret_pct > 5.0001).sum())}')
    print('Largest absolute latency losses in focused region:')
    for row in focus.sort_values('lost_us',ascending=False).head(25).itertuples():
        print(f'  {row.family} R={row.R} K={row.K} outer={row.outer} '
              f'{row.variant_selected}->{row.variant_best} '
              f'selected={row.selected_us:.1f}us best={row.best_us:.1f}us '
              f'loss={row.lost_us:.1f}us regret={row.regret_pct:.1f}% '
              f'nonoverlap={row.nonoverlapping_ranges}')
    print(f'Full per-shape results: {args.details}')
    print('This measures OneWindow-only regret; it does not validate generalized dispatch.')


if __name__ == '__main__':
    main()
