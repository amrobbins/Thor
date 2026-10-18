#!/usr/bin/env python3
"""Offline OneWindow idle-lane-work selector experiment (post-896).

Compare the existing *corrected* ten-candidate heuristic against a bounded
additional cost for consumer lanes that do not participate usefully over the
CTA-equivalent work waves:

    trial_score = frozen_891_score + 0.20 * work_waves
                + 1.0 * work_waves * (1 - useful_lane_fraction)

The existing 0.20-wave correction and staging safeguard remain unchanged.
To avoid extrapolating the adjustment to the short-stage O8 behavior, accept
new proposals only for O<=4; an unchanged O8 choice remains available. Reject
new higher-O choices staging only one row per warp group as before.

No production selector or kernel changes. Results on the supplied 896 holdout
are now DEVELOPMENT/DIAGNOSTIC data, not a fresh acceptance holdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analyze_onewindow_natural_fit import add_physics
from analyze_fp8_one_window_score_correction import evaluate, select_by

KEY = ['family', 'outer', 'R', 'K', 'dtype']
IDLE_LANE_WEIGHT = 1.0
WAVE_WEIGHT = 0.20


def compare(raw: pd.DataFrame) -> pd.DataFrame:
    physics = add_physics(raw)
    # The existing policy includes a guard against short-staging moves.
    current = evaluate(raw, beta=WAVE_WEIGHT, guard=True).set_index(KEY).sort_index()
    physics['trial_score'] = (physics.cost + WAVE_WEIGHT * physics.work_waves
                              + IDLE_LANE_WEIGHT * physics.work_waves
                              * (1.0 - physics.useful_lane_fraction))
    proposal = select_by(physics, 'trial_score')
    if not proposal.index.equals(current.index):
        raise ValueError('Candidate geometry set differs from current policy')

    k = proposal.index.get_level_values('K').to_numpy(dtype=np.int64)
    o = proposal.O.to_numpy(dtype=np.int64)
    packet = proposal.P.to_numpy(dtype=np.int64)
    rows_per_group = ((32 // o) * packet - np.where(k % packet == 0, 0, packet - 1)) // k
    selected_o = current.variant_selected.str.extract(r'o(\d+)')[0].astype(int).to_numpy()
    no_new_one_row = ~((o > selected_o) & (rows_per_group <= 1))
    # Scope this correction to the O1/O2/O4 balance. Do not promote a new O8.
    no_new_o8 = (o <= 4) | (proposal.variant.to_numpy() == current.variant_selected.to_numpy())
    accept = no_new_one_row & no_new_o8

    out = current[['variant_selected', 'variant_best', 'selected_us', 'best_us']].rename(
        columns={'variant_selected': 'current_variant', 'selected_us': 'current_us'}).copy()
    out['trial_variant'] = proposal.variant
    out['accepted'] = accept
    out['proposed_variant'] = np.where(accept, proposal.variant, out.current_variant)
    out['proposed_us'] = np.where(accept, 1000.0 * proposal.median_ms, out.current_us)
    out['current_loss_us'] = out.current_us - out.best_us
    out['proposed_loss_us'] = out.proposed_us - out.best_us
    out['current_regret_pct'] = 100.0 * out.current_loss_us / out.best_us
    out['proposed_regret_pct'] = 100.0 * out.proposed_loss_us / out.best_us
    out['delta_us'] = out.proposed_us - out.current_us
    return out.reset_index()


def summarize(label: str, d: pd.DataFrame) -> None:
    if d.empty:
        print(f'{label}: empty'); return
    focus = d[d.current_us >= 10.0]
    print(f'{label}: n={len(d)} current>=10us={len(focus)}')
    for s, loss, pct in [('CURRENT','current_loss_us','current_regret_pct'),
                         ('PROPOSED','proposed_loss_us','proposed_regret_pct')]:
        r=focus
        print(f'  {s}: >5%={int((r[pct]>5.0001).sum())} '
              f'>10%={int((r[pct]>10.0001).sum())} '
              f'>5% and >=2us={int(((r[pct]>5.0001)&(r[loss]>=2)).sum())} '
              f'>=10us loss={int((r[loss]>=10).sum())} '
              f'cumulative_loss={r[loss].sum():.1f}us '
              f'worst_loss={r[loss].max():.1f}us')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--details',default='rk-fp8-one-window-idle-lane-trial-details.csv')
    args=p.parse_args()
    raw=pd.read_csv(args.input, comment='#')
    result=compare(raw)
    Path(args.details).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.details,index=False)
    print('OFFLINE EXPERIMENT ONLY: no kernel or production scheduler changes')
    print('Existing beta=0.20 single-row-guard score + 1.0 * wave * idle-lane-fraction; new O8 choices excluded')
    print('Existing 896 holdout is now diagnostic data; future acceptance needs a fresh holdout.')
    summarize('ALL',result)
    for name,part in result.groupby('family',sort=True): summarize(name,part)
    changed=result[result.proposed_variant!=result.current_variant]
    print(f'Changed={len(changed)} faster={int((changed.delta_us<-.001).sum())} '
          f'slower={int((changed.delta_us>.001).sum())} '
          f'largest_introduced_regression={result.delta_us.max():.1f}us')
    print('Largest improvements:')
    for case in result.nsmallest(8,'delta_us').itertuples():
        if case.delta_us>=0:break
        print(f'  {case.family} R={case.R} K={case.K} outer={case.outer}: '
              f'{case.current_variant}->{case.proposed_variant} '
              f'{case.current_us:.1f}->{case.proposed_us:.1f}us '
              f'({case.delta_us:+.1f}us)')
    print('Largest newly introduced regressions:')
    for case in result.nlargest(5,'delta_us').itertuples():
        if case.delta_us<=0:break
        print(f'  {case.family} R={case.R} K={case.K} outer={case.outer}: '
              f'{case.current_variant}->{case.proposed_variant} '
              f'{case.current_us:.1f}->{case.proposed_us:.1f}us '
              f'({case.delta_us:+.1f}us)')
    print('Per-case CSV:',args.details)

if __name__=='__main__': main()
