#!/usr/bin/env python3
"""Strict independent AlignedK topology/CTA256 holdout analysis (no complete P/CTA selector).

The geometry-dependent Complete/Staged32/Staged128 rule was derived from the
84 discovery/calibration shapes and is frozen here BEFORE this holdout.
The packet/CTA choice inside the chosen topology is still an oracle. Do not
interpret this as end-to-end local selector validation.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import statistics

RK = ((128, 26), (128, 640), (224, 52), (224, 296), (448, 88), (448, 636),
      (896, 30), (896, 288), (1792, 148), (1792, 1288), (4608, 54),
      (4608, 644), (10240, 84), (10240, 296), (20480, 152), (20480, 1288))
PACKETS = (2, 4, 8)
CTAS = (32, 64, 128, 256)
TARGET_ROWS = (32, 128)
SAMPLES = tuple(f'sample{i}_ms' for i in range(5))


def ceildiv(a, b):
    return (a + b - 1) // b


def expected_shapes():
    return {(o, r, k) for r, k in RK for o in
            {23, *(max(1, (t + r*k//2)//(r*k))
                   for t in (32*1024**2, 64*1024**2, 128*1024**2))}}


def expected_configs(r, k):
    return {(p, cta, s) for p in PACKETS if k % p == 0
            for cta in CTAS
            for s in {1, *(ceildiv(r, n) for n in TARGET_ROWS if n < r)}}


def proposed_topology(outer, r, k):
    """Frozen coarse topology rule; not a packet/CTA selector."""
    if r <= 256 and (outer * k >= 300000 or (k <= 64 and outer * k >= 180000)):
        return 1  # Complete
    mib = outer * r * k / (1024**2)
    target = 128 if mib >= 50 else 32
    return ceildiv(r, target)


def analyze(path):
    raw = path.read_text()
    if not re.search(r'^# mode=rk_fp8_aligned_k_topology_holdout\b', raw, re.M):
        raise ValueError('Wrong AlignedK holdout benchmark mode')
    end = re.search(r'^# rk_fp8_aligned_k_topology_holdout_complete shapes=(\d+) '
                    r'candidate_configurations=(\d+)\s*$', raw, re.M)
    if not end:
        raise ValueError('Incomplete benchmark: missing completion marker')
    rows = list(csv.DictReader(line for line in raw.splitlines() if not line.startswith('#')))
    needed = {'family','outer','R','K','topology','packet_bytes','block_threads',
              'rows_per_shard','shards_per_output','component_tiles','first_stage_ctas',
              'intermediate_bytes','resident_ctas_per_sm','first_median_ms',
              'continuation_median_ms','total_median_ms',*SAMPLES}
    if not rows or not needed.issubset(rows[0]) or len(rows) != int(end.group(2)):
        raise ValueError('Wrong candidate count or CSV columns')
    expected = expected_shapes()
    groups = defaultdict(dict)
    for v in rows:
        o,r,k,p,c,s = (int(v[n]) for n in ('outer','R','K','packet_bytes','block_threads',
                                            'shards_per_output'))
        key=(o,r,k);conf=(p,c,s)
        if v['family'] != 'aligned_k' or key not in expected or conf not in expected_configs(r,k):
            raise ValueError(f'Unexpected geometry/configuration: {key} {conf}')
        if conf in groups[key]:
            raise ValueError(f'Duplicate candidate: {key} {conf}')
        if (v['topology']=='complete') != (s==1):
            raise ValueError('Topology mismatch')
        if int(v['rows_per_shard']) != ceildiv(r,s):
            raise ValueError('Rows per shard mismatch')
        tiles = ceildiv(k,p*c)
        if int(v['component_tiles']) != tiles or int(v['first_stage_ctas'])!=o*s*tiles:
            raise ValueError('Incorrect physical CTA geometry')
        if int(v['intermediate_bytes']) != (0 if s==1 else o*s*k*4):
            raise ValueError('Incorrect FP32 intermediate footprint')
        if int(v['resident_ctas_per_sm']) < 1:
            raise ValueError('Invalid kernel residency')
        timing=[float(v[name]) for name in SAMPLES]
        median=float(v['total_median_ms'])
        if median <= 0 or any(t<=0 for t in timing) or abs(statistics.median(timing)-median)>1e-6:
            raise ValueError('Invalid median timing')
        if s==1 and abs(float(v['continuation_median_ms']))>1e-7:
            raise ValueError('Unexpected Complete continuation')
        groups[key][conf]=v
    if set(groups)!=expected or int(end.group(1)) != len(expected):
        raise ValueError(f'Incomplete shape inventory: got {len(groups)}, expected {len(expected)}')
    output=[]
    for key,variants in sorted(groups.items()):
        o,r,k=key
        if set(variants)!=expected_configs(r,k):
            raise ValueError(f'Incomplete candidate inventory at {key}')
        def best(predicate):
            return min((float(v['total_median_ms'])*1000 for (p,c,s),v in variants.items()
                        if predicate(p,c,s)),default=float('inf'))
        oracle=best(lambda p,c,s:True)
        no256=best(lambda p,c,s:c<256)
        no_p4=best(lambda p,c,s:p!=4)
        topology_shards=proposed_topology(o,r,k)
        top=best(lambda p,c,s:s==topology_shards)
        top_no256=best(lambda p,c,s:s==topology_shards and c<256)
        if any(x==float('inf') for x in (oracle,no256,no_p4,top,top_no256)):
            raise ValueError(f'No candidate for rule at {key}')
        output.append({'outer':o,'R':r,'K':k,'input_mib':o*r*k/2**20,
                       'oracle_us':oracle,'without_cta256_us':no256,
                       'without_p4_us':no_p4, 'frozen_topology_shards':topology_shards,
                       'topology_oracle_packet_cta_us':top,
                       'topology_and_no256_oracle_packet_cta_us':top_no256,
                       'cta256_inventory_regret_us':no256-oracle,
                       'cta256_inventory_regret_pct':100*(no256/oracle-1),
                       'p4_inventory_regret_us':no_p4-oracle,
                       'topology_only_regret_us':top-oracle,
                       'topology_only_regret_pct':100*(top/oracle-1),
                       'topology_plus_no256_regret_us':top_no256-oracle,
                       'topology_plus_no256_regret_pct':100*(top_no256/oracle-1)})
    return output,len(rows)


def summary(data, prefix):
    pct=[row[prefix+'_pct'] for row in data]
    us=[row[prefix+'_us'] for row in data]
    print(prefix+': >5%='+str(sum(v>5.000001 for v in pct))+'/'+str(len(data))+
          ', >10%='+str(sum(v>10.000001 for v in pct))+
          f', max_us={max(us):.3f}, cumulative_us={sum(us):.3f}, max_pct={max(pct):.3f}')
    for row in sorted(data,key=lambda v:-v[prefix+'_us'])[:5]:
        if row[prefix+'_us'] > .001:
            print(f'  outer={row["outer"]} R={row["R"]} K={row["K"]} '
                  f'input={row["input_mib"]:.2f}MiB '
                  f'loss={row[prefix+"_us"]:.3f}us ({row[prefix+"_pct"]:.2f}%)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--details',type=Path)
    a=ap.parse_args()
    rows,n=analyze(a.input)
    print(f'COMPLETE independent AlignedK holdout: {len(rows)} shapes, {n} legal candidate results')
    print(f'Complete chosen by frozen topology rule: '
          f'{sum(row["frozen_topology_shards"]==1 for row in rows)}/{len(rows)}')
    summary(rows,'cta256_inventory_regret')
    # P4 deletion remains a diagnostic only; its percent penalty is printed separately.
    print('P4 deletion: max_us=%.3f, cumulative_us=%.3f' %
          (max(row['p4_inventory_regret_us'] for row in rows),
           sum(row['p4_inventory_regret_us'] for row in rows)))
    summary(rows,'topology_only_regret')
    summary(rows,'topology_plus_no256_regret')
    print('IMPORTANT: packet and CTA are still oracle-selected inside the frozen topology. '
          'This is NOT a complete AlignedK selector holdout.')
    if a.details:
        with a.details.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('Saved:',a.details)

if __name__=='__main__':
    main()
