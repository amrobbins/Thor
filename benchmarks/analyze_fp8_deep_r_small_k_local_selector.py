#!/usr/bin/env python3
"""Offline FP8 DeepRSmallK P16 four-depth selector diagnostics.

Only source-defined compiled first-stage candidates and their real FP32
continuation are compared. CUDA occupancy was not logged by the 900 sweep:
--resident-per-sm is therefore a PROXY, not a claim of actual kernel occupancy.

Example:
  python benchmarks/analyze_fp8_deep_r_small_k_local_selector.py \
    --input rk-fp8-deep-r-small-k-end-to-end.csv --details deep-four-way.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEPTHS = (64, 128, 256, 512)

def cost(row, sms: int, resident: int) -> float:
    outer, r, k, depth = (int(row[x]) for x in ('outer','R','K','rows_per_shard'))
    shards = r // depth + (r % depth != 0)
    group_rows = min(depth // 8, 497 // k)
    rows_per_round = 8 * group_rows
    full_rounds = depth // rows_per_round + (depth % rows_per_round != 0)
    last_rows = r - (shards - 1) * depth
    tail_rounds = last_rows // rows_per_round + (last_rows % rows_per_round != 0)
    total_rounds = outer * ((shards - 1) * full_rounds + tail_rounds)
    staging_waves = max(full_rounds, total_rounds / (sms * resident))
    partial_mb = outer * shards * k * 4 / 1e6
    return staging_waves + .5 * full_rounds + .25 * partial_mb

def metrics(df: pd.DataFrame, reference: str) -> dict:
    regret = (df['total_median_ms'] - df[reference]) * 1e3
    relative = regret / (df[reference] * 1e3)
    return {'shapes':int(len(df)), '>5%':int((relative > .05 + 1e-10).sum()),
        '>10%':int((relative > .10 + 1e-10).sum()),
        'max_us':round(float(regret.max()),3),
        'sum_us':round(float(regret.sum()),3),
        'max_relative_pct':round(float(relative.max()*100),3)}

def run(source: pd.DataFrame, sms: int, resident: int):
    keys=['outer','R','K']
    full=source.groupby(keys, as_index=False)['total_median_ms'].min().rename(columns={'total_median_ms':'full_oracle_ms'})
    legal=source.loc[(source.packet_bytes == 16) & source.rows_per_shard.isin(DEPTHS)
        & (source.rows_per_shard < source.R)].copy()
    if legal.empty: raise ValueError('no compiled P16 four-depth observations')
    legal['score'] = legal.apply(cost,axis=1,args=(sms,resident))
    ix=legal.groupby(keys).total_median_ms.idxmin()
    four=legal.loc[ix,keys+['total_median_ms']].rename(columns={'total_median_ms':'four_oracle_ms'})
    idx=legal.groupby(keys).score.idxmin()
    selected=legal.loc[idx].merge(full,on=keys).merge(four,on=keys)
    if len(selected) != len(full):
        raise ValueError('source contains shapes without legal four-depth candidates')
    selected['regret_vs_four_us']=(selected.total_median_ms-selected.four_oracle_ms)*1000
    selected['inventory_loss_us']=(selected.four_oracle_ms-selected.full_oracle_ms)*1000
    print(f'Geometries {len(selected)}; P16 compiled depths {DEPTHS}; sms={sms}; residency PROXY={resident}')
    print('Selector regret vs legal FOUR-depth oracle: ', metrics(selected,'four_oracle_ms'))
    # Compare best-four to broader full ladder. Selector loss and inventory loss are separate.
    inv=selected.copy();inv['total_median_ms']=inv.four_oracle_ms
    print('Inventory loss FOUR-depth oracle vs unrestricted sweep:',metrics(inv,'full_oracle_ms'))
    print('Combined loss selected vs unrestricted sweep: ',metrics(selected,'full_oracle_ms'))
    for name,sub in [('24MiB',selected[selected.outer*selected.R*selected.K < 50*2**20]),
            ('96MiB',selected[(selected.outer*selected.R*selected.K > 50*2**20)&(selected.outer*selected.R*selected.K < 140*2**20)]),
            ('192MiB volume slice (not independent)',selected[selected.outer*selected.R*selected.K > 140*2**20])]:
        print(name,metrics(sub,'four_oracle_ms'))
    print('Selected depths: ', selected.rows_per_shard.value_counts().sort_index().to_dict())
    return selected.sort_values(['R','K','outer'])

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--details',type=Path)
    ap.add_argument('--sms',type=int,default=170)
    ap.add_argument('--resident-per-sm',type=int,default=4)
    args=ap.parse_args()
    if args.sms < 1 or args.resident_per_sm < 1: ap.error('SM and residency counts must be positive')
    result=run(pd.read_csv(args.input,comment='#'),args.sms,args.resident_per_sm)
    if args.details: result.to_csv(args.details,index=False)
