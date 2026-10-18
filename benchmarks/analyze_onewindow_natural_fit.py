#!/usr/bin/env python3
"""Reproduce a provisional physical OneWindow policy comparison for Thor 891/892.

This is offline analysis only: no production selector, profiling, or GPU execution.
The constants below were calibrated using 891, before examining 892 outcomes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARS={'o1_w1_p16','o1_w2_p16','o1_w4_p16','o1_w8_p16',
      'o2_w1_p16','o4_w1_p16','o8_w1_p16',
      'o8_w1_p4','o4_w1_p8','o8_w1_p8'}
KEY=['family','outer','R','K','dtype']

def add_physics(raw):
    d=raw.copy()
    d['variant']=d.implementation.str.extract(r'(o\d+_w\d+_p\d+)')[0]
    d=d[d.variant.isin(VARS)].copy()
    for name,regex in [('O',r'o(\d+)'),('W',r'_w(\d+)'),('P',r'_p(\d+)')]:
        d[name]=d.variant.str.extract(regex)[0].astype(int)
    r=d.R.to_numpy(); k=d.K.to_numpy(); o=d.O.to_numpy(); w=d.W.to_numpy(); p=d.P.to_numpy()
    lanes=32//o
    workers=np.where(k<=lanes,np.maximum(1,lanes//k),1)
    components=np.where(k<=lanes,1,(k+lanes-1)//lanes)
    rows=(lanes*p-np.where(k%p==0,0,p-1))//k
    if np.any(rows<=0): raise ValueError('Illegal staged group in input')
    rounds=(r+w*rows-1)//(w*rows)
    outputs_per_cta=8*o//w
    ctas=(d.outer.to_numpy()+outputs_per_cta-1)//outputs_per_cta
    waves=np.maximum(1.,ctas/(d.device_sms.to_numpy()*d.occupancy_limit_ctas_per_sm.to_numpy()))
    lane_util=np.where(k>lanes,k/(lanes*components),k*workers/lanes)
    # Critical worker's exact count of FP8x4 decode-iteration groups, preserving full/tail windows.
    full_cycles=r//(w*rows)
    leftover=np.minimum(r%(w*rows),rows)
    per_full=(rows+4*workers-1)//(4*workers)
    per_tail=(leftover+4*workers-1)//(4*workers)
    decode_iters=components*(full_cycles*per_full+per_tail)
    staged_util=r*k/(rounds*w*lanes*p)
    # Approximate packet inflation from unaligned staging windows.
    avg_prefix=np.where(k%p==0,0,(p-1)/2)
    packet_amp=np.maximum(1.,np.ceil((rows*k+avg_prefix)/p)*p/(rows*k))
    packet_lanes=np.minimum(1.,o*np.ceil((rows*k+avg_prefix)/p)/32)
    supplied_warps=np.minimum(ctas/d.device_sms.to_numpy(),d.occupancy_limit_ctas_per_sm.to_numpy())*8*(1-.499*(1-packet_lanes))
    # Sufficient-parallelism gate: stop penalizing once threshold is met.
    gate=1+1.284*np.maximum(0.,1.094/np.maximum(supplied_warps,.1)-1)
    # Exploratory shared geometry-cost coefficients calibrated on 891. Not hardware constants.
    cost=(waves**.4018)*(3.9502+
        rounds*(1+1.2989*(1-lane_util)+.03016*(1-staged_util)+.00309*(packet_amp-1))+
        .19559*(decode_iters/(1+rounds))+
        .00784*np.log2(w))*gate
    d['cost']=cost
    d['useful_lane_fraction']=lane_util
    d['decode_iterations']=decode_iters
    d['rounds']=rounds
    d['work_waves']=waves
    d['stage_utilization']=staged_util
    d['packet_amplification']=packet_amp
    d['parallelism_gate']=gate
    return d

def evaluate(d):
    best=d.sort_values(['median_ms','variant'],kind='stable').drop_duplicates(KEY).set_index(KEY)
    selected=d.sort_values(['cost','variant'],kind='stable').drop_duplicates(KEY).set_index(KEY)
    out=selected[['variant','median_ms','cost','useful_lane_fraction','decode_iterations','rounds','work_waves','stage_utilization','packet_amplification','parallelism_gate']].join(
        best[['variant','median_ms']],lsuffix='_selected',rsuffix='_best')
    out['regret_pct']=100*(out.median_ms_selected/out.median_ms_best-1)
    if d.implementation.str.startswith('one_window_selected_').any():
        existing=d[d.implementation.str.startswith('one_window_selected_')].set_index(KEY)
        out=out.join(existing[['variant','median_ms']].rename(columns={'variant':'old_variant','median_ms':'old_ms'}))
        out['old_regret_pct']=100*(out.old_ms/out.median_ms_best-1)
    return out.reset_index()

def stat(label,data):
    x=data.regret_pct.to_numpy()
    print(f'{label}: cells={len(x)}, misses>5%={(x>5.0001).sum()}, misses>10%={(x>10.0001).sum()}, '
          f'p90={np.percentile(x,90):.2f}%, p95={np.percentile(x,95):.2f}%, max={max(x):.2f}%')
    if 'old_regret_pct' in data.columns:
        o=data.old_regret_pct.to_numpy()
        print(f'   original selector: misses>5%={(o>5.0001).sum()}, '
              f'misses>10%={(o>10.0001).sum()}, p90={np.percentile(o,90):.2f}%, p95={np.percentile(o,95):.2f}%')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True,help='Thor 891 dense-boundary benchmark CSV')
    p.add_argument('--validation',required=True,help='Thor 892 OneWindow validation CSV')
    p.add_argument('--details',required=True,help='Output per-cell comparison CSV')
    a=p.parse_args()
    train=add_physics(pd.read_csv(a.train,comment='#'))
    test=add_physics(pd.read_csv(a.validation,comment='#'))
    ta=evaluate(train); tb=evaluate(test)
    stat('891 calibration',ta);stat('892 independent work-size validation',tb)
    rks=set(zip(ta.R,ta.K))
    unseen=tb[[z not in rks for z in zip(tb.R,tb.K)]]
    stat('892 unseen R/K',unseen)
    for family,part in tb.groupby('family'):
        stat(f'892 {family}',part)
    print('Gated candidates (891/892)',int((train.parallelism_gate>1.000001).sum()),int((test.parallelism_gate>1.000001).sum()))
    Path(a.details).parent.mkdir(parents=True,exist_ok=True)
    tb.to_csv(a.details,index=False)

if __name__=='__main__':main()
