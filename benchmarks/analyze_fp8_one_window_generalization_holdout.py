#!/usr/bin/env python3
"""Evaluate a FROZEN OneWindow-only physical selector on the fresh five-sample holdout.

No fitting/tuning. Source of truth is the 891-calibrated model plus the previously
agreed beta=0.20 CTA-work-wave correction and single-row staging safeguard.
This does not establish whether OneWindow wins against another generalized mode.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_fp8_one_window_score_correction import evaluate

KEY = ['family', 'outer', 'R', 'K', 'dtype']
HOLDOUT_R = {9, 11, 13, 18, 22, 26, 36, 44, 56, 72, 112, 144, 224}


def timing_samples(path: Path):
    lookup = {}
    with path.open() as f:
        for line in f:
            if not line.startswith('# one_window_samples,') or line.startswith('# one_window_samples,family,'):
                continue
            parts = next(csv.reader([line[2:].rstrip()]))
            if len(parts) != 12:  # marker + 6 key columns + 5 samples
                raise ValueError(f'Five-sample holdout expected, got {len(parts)-7} samples: {parts[:7]}')
            _, family, outer, r, k, dtype, variant, *numbers = parts
            key = (family, int(outer), int(r), int(k), dtype, variant)
            if key in lookup:
                raise ValueError(f'Duplicate samples for {key}')
            lookup[key] = np.asarray([float(x)*1000 for x in numbers])
    return lookup


def report(label, frame):
    if frame.empty:
        print(f'{label}: no geometries')
        return
    focus = frame[frame.baseline_us >= 10]
    reg = frame.regret_pct
    loss = frame.loss_us
    print(f'{label}: n={len(frame)} focus>=10us={len(focus)} '
          f'misses>5%={(reg>5.0001).sum()} focus_misses>5%={(focus.regret_pct>5.0001).sum()} '
          f'focus_misses>10%={(focus.regret_pct>10.0001).sum()} '
          f'focus_>5%_and_>=2us={((focus.regret_pct>5.0001)&(focus.loss_us>=2)).sum()} '
          f'total_lost={loss.sum():.1f}us focus_lost={focus.loss_us.sum():.1f}us '
          f'max_lost={loss.max():.1f}us p95_focus_regret='
          f'{np.percentile(focus.regret_pct,95) if len(focus) else float("nan"):.2f}%')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--details', default='rk-fp8-one-window-holdout-details.csv')
    args = p.parse_args()
    path = Path(args.input)
    with path.open() as f:
        header = ''.join(line for line in f if line.startswith('#'))
    if '# fresh_holdout=1' not in header or 'samples_per_variant=5' not in header:
        p.error('Input is not a fresh five-sample OneWindow generalization holdout')
    raw = pd.read_csv(path, comment='#')
    required = set(KEY) | {'implementation', 'median_ms'}
    if required - set(raw):
        p.error(f'Missing CSV columns {required-set(raw)}')
    if not raw.R.isin(HOLDOUT_R).all():
        p.error('Holdout contains out-of-scope reduction depths')
    selected = evaluate(raw, beta=0.20, guard=True)
    if raw[raw.implementation.str.contains('one_window_candidate_|one_window_selected_', regex=True)].empty:
        p.error('No promoted OneWindow variants found')
    sample_lookup = timing_samples(path)
    if len(sample_lookup) != len(raw):
        p.error(f'Expected 5 samples for every row ({len(raw)}), got {len(sample_lookup)}')
    selected['nonoverlap'] = False
    for index, case in selected.iterrows():
        stem = tuple(case[k] for k in KEY)
        measured = sample_lookup[stem + (case.variant_selected,)]
        best = sample_lookup[stem + (case.variant_best,)]
        selected.at[index, 'nonoverlap'] = bool(measured.min() > best.max())
    promoted = raw[raw.implementation.str.startswith(('one_window_candidate_', 'one_window_selected_'))]
    demoted = raw[raw.implementation.str.startswith('ORACLE_one_window_demoted_')]
    if promoted.empty or demoted.empty:
        p.error('Expected both promoted candidates and demoted oracle controls')
    best_promoted = promoted.groupby(KEY, sort=False).median_ms.min()
    best_demoted = demoted.groupby(KEY, sort=False).median_ms.min()
    gaps = best_promoted.to_frame('promoted').join(best_demoted.to_frame('demoted'))
    gaps['demoted_advantage_pct'] = 100*(gaps.promoted/gaps.demoted - 1)
    selected['regime'] = np.where((selected.K <= 32) & (selected.R <= 144),
                                  'narrow_core', 'one_window_shoulder')
    print('FROZEN POLICY: 891-coefficient score + beta=0.20 * work_waves; single-row guard ON')
    print('HOLDOUT: new reduction depths, independently varied outer, 5 interleaved samples/variant')
    print('No calibration or re-fitting from these data. No generalized-mode result.')
    print('Number of promoted/demoted timing rows:', len(promoted), len(demoted))
    print('Best demoted oracle >5% faster than best promoted:',
          int((gaps.demoted_advantage_pct > 5.0001).sum()), 'of', len(gaps))
    print('BASELINE vs CORRECTED, focus anchored to baseline-selected >=10us:')
    baseline = selected.copy()
    baseline['loss_us'] = baseline.baseline_loss_us
    baseline['regret_pct'] = 100*baseline.baseline_loss_us/baseline.best_us
    report('  baseline', baseline)
    report('  corrected', selected)
    for family, group in selected.groupby('family',sort=True): report('  '+family,group)
    for regime, group in selected.groupby('regime',sort=True): report('  '+regime,group)
    miss = selected[(selected.baseline_us>=10) & (selected.regret_pct>5.0001)]
    print('Focused >5% misses with disjoint sample ranges:', int(miss.nonoverlap.sum()), '/', len(miss))
    changed = selected[selected.variant_selected != selected.variant_baseline]
    print('Selection changes:',len(changed),'improved:',int((changed.change_us<-.001).sum()),
          'regressed:',int((changed.change_us>.001).sum()),
          'worst_new_regression_us:',f'{selected.change_us.max():.2f}')
    print('Largest absolute losses, baseline>=10us:')
    for case in selected[selected.baseline_us>=10].nlargest(15,'loss_us').itertuples():
        print(f'  {case.family} R={case.R} K={case.K} outer={case.outer}: '
              f'{case.variant_baseline}->{case.variant_selected} vs fastest {case.variant_best}, '
              f'{case.selected_us:.1f}us vs {case.best_us:.1f}us; '
              f'loss={case.loss_us:.1f}us nonoverlap={case.nonoverlap}')
    output = Path(args.details)
    output.parent.mkdir(parents=True,exist_ok=True)
    selected.to_csv(output,index=False)
    print('Details:',output)

if __name__=='__main__': main()
