#!/usr/bin/env python3
"""Offline OneWindow physical-score correction study (Thor 894).

Compare the frozen 891 physical score with ONE extra CTA-work-wave term:

    new_score = frozen_score + beta * total_CTA_work_waves

To avoid an unstable crossover at the physical staging limit, reject a *new*
move toward larger output parallelism when the proposed physical variant can
stage only one row per warp group. Existing selections and legality are not
changed. This gate is an experimental safeguard, not an accepted GPU law.

The input must be an existing 892/894 OneWindow validation CSV, containing all
promoted legal candidates; this is not a production selector. No GPU required.
Metrics anchor focus to BASELINE-selected >=10us to permit fair comparisons.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_onewindow_natural_fit import add_physics

KEY=['family','outer','R','K','dtype']


def select_by(df:pd.DataFrame, score:str)->pd.DataFrame:
    # Tie behavior mirrors the frozen score analyzer.
    return (df.sort_values([score,'variant'],kind='stable')
              .drop_duplicates(KEY).set_index(KEY).sort_index())


def describe(label:str, records:pd.DataFrame)->None:
    if len(records)==0:
        print(f'{label}: no geometries');return
    loss=records.loss_us.to_numpy()
    regret=records.regret_pct.to_numpy()
    print(f'{label}: n={len(records)} >5%={(regret>5.0001).sum()} '
          f'>10%={(regret>10.0001).sum()} >5%+>=2us={((regret>5.0001)&(loss>=2)).sum()} '
          f'total_loss={loss.sum():.1f}us max_loss={loss.max():.1f}us '
          f'p95_regret={np.percentile(regret,95):.2f}%')


def evaluate(raw:pd.DataFrame,beta:float,guard:bool):
    df=add_physics(raw)
    df['trial_score']=df.cost+beta*df.work_waves
    baseline=select_by(df,'cost')
    trial=select_by(df,'trial_score')
    oracle=select_by(df,'median_ms')
    base_us=baseline.median_ms*1000
    accept=pd.Series(True,index=baseline.index)
    if guard:
        r=trial.index.get_level_values('R').to_numpy(dtype=np.int64)
        k=trial.index.get_level_values('K').to_numpy(dtype=np.int64)
        o=trial.O.to_numpy(dtype=np.int64)
        p=trial.P.to_numpy(dtype=np.int64)
        rows_per_window=((32//o)*p-np.where(k%p==0,0,p-1))//k
        reject=(o>baseline.O.to_numpy())&(rows_per_window<=1)
        accept[:]=~reject
    out=pd.DataFrame(index=baseline.index)
    out['variant_baseline']=baseline.variant
    out['variant_trial']=trial.variant
    out['trial_accepted']=accept
    out['variant_selected']=np.where(accept,trial.variant,baseline.variant)
    out['variant_best']=oracle.variant
    out['baseline_us']=base_us
    out['trial_us']=1000*trial.median_ms
    out['selected_us']=np.where(accept,1000*trial.median_ms,base_us)
    out['best_us']=1000*oracle.median_ms
    out['baseline_loss_us']=out.baseline_us-out.best_us
    out['loss_us']=out.selected_us-out.best_us
    out['regret_pct']=100*out.loss_us/out.best_us
    out['change_us']=out.selected_us-out.baseline_us
    out=out.reset_index()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--details',default='one_window_score_correction_details.csv')
    parser.add_argument('--beta',type=float,default=.2,help='Additive CTA-wave price; default=0.2')
    parser.add_argument('--no-single-row-guard',action='store_true')
    args=parser.parse_args()
    if args.beta<0:parser.error('--beta must be >=0')
    d=evaluate(pd.read_csv(args.input,comment='#'),args.beta,not args.no_single_row_guard)
    Path(args.details).parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(args.details,index=False)
    print('ONEWINDOW-ONLY offline trial; no source kernel/production selector changes')
    print('Beta:',args.beta,'Guard single-row staging:',not args.no_single_row_guard)
    print('Benchmark sizes are repeated measurements of the same R/K/outer geometry grid, NOT independent shape holdouts.')
    base=d.copy()
    base['loss_us']=base.baseline_loss_us
    base['regret_pct']=100*base.loss_us/base.best_us
    focus=d.baseline_us>=10
    print('BASELINE:')
    describe('  all',base)
    describe('  baseline>=10us',base[focus])
    for family in sorted(d.family.unique()):describe('  '+family,base[base.family==family])
    print('CORRECTED:')
    describe('  all',d)
    describe('  baseline>=10us',d[focus])
    for family in sorted(d.family.unique()):describe('  '+family,d[d.family==family])
    print('CHANGES:',(d.variant_selected!=d.variant_baseline).sum(),
          'beneficial',(d.change_us<-.001).sum(),
          'harmful',(d.change_us>.001).sum(),
          'worst_new_regression_us',round(d.change_us.max(),3))
    print('Worst residual losses among baseline >=10us:')
    for row in d[focus].nlargest(12,'loss_us').itertuples():
        print(f'  {row.family} R={row.R} K={row.K} outer={row.outer} '
              f'{row.variant_baseline} -> {row.variant_selected}; fastest {row.variant_best}: '
              f'{row.baseline_us:.1f}us -> {row.selected_us:.1f}us vs {row.best_us:.1f}us; '
              f'loss {row.loss_us:.1f}us')
    print('Worst new regressions:')
    for row in d.nlargest(8,'change_us').itertuples():
        if row.change_us<=0:break
        print(f'  {row.family} R={row.R} K={row.K} outer={row.outer} '
              f'{row.variant_baseline} -> {row.variant_selected}: +{row.change_us:.1f}us')
    print('Per-case details:',args.details)


if __name__=='__main__':main()
