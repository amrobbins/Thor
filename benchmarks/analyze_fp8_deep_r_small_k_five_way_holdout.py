#!/usr/bin/env python3
"""Verify the second independent DeepRSmallK compiled-P16 five-depth holdout.

Usage:
  python benchmarks/analyze_fp8_deep_r_small_k_five_way_holdout.py \
    --input rk-fp8-deep-r-five-way-holdout.csv \
    --details rk-fp8-deep-r-five-way-holdout-details.csv

Selector coefficients are unchanged from the earlier four-depth policy. Evaluate
only the allowed depths {64,128,256,512,1024}; also report five-vs-four oracle
improvement separately from selector regret. No fitting or model updates occur.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEPTHS = (64, 128, 256, 512, 1024)
OLD_DEPTHS = (64, 128, 256, 512)
KEYS = ["outer", "R", "K"]
SAMPLES = [f"sample{i}_ms" for i in range(5)]
EXPECTED_GEOMETRIES = 80
EXPECTED_CANDIDATES = 400


def score(row, sms):
    outer, r, k, depth, active = (int(row[col]) for col in
                                   ("outer", "R", "K", "rows_per_shard", "resident_ctas_per_sm"))
    if active <= 0 or k <= 0 or k > 256 or r <= depth:
        raise ValueError("Illegal candidate or zero actual CUDA residency")
    shards = (r + depth - 1) // depth
    group_rows = min(depth // 8, 497 // k)
    if group_rows < 1:
        raise ValueError("Invalid logical P16 group width")
    rows_per_round = 8 * group_rows
    full_rounds = (depth + rows_per_round - 1) // rows_per_round
    last_rows = r - (shards - 1) * depth
    last_rounds = (last_rows + rows_per_round - 1) // rows_per_round
    all_rounds = outer * ((shards - 1) * full_rounds + last_rounds)
    staging_waves = max(full_rounds, all_rounds / (sms * active))
    partial_mb = outer * shards * k * 4 / 1e6
    return staging_waves + 0.5 * full_rounds + 0.25 * partial_mb


def summarize(label, chosen, reference):
    regret_us = (chosen.chosen_ms - chosen[reference]) * 1000
    rel = regret_us / (chosen[reference] * 1000)
    print(f"{label}: shapes={len(chosen)}, >5%={(rel > 0.05+1e-10).sum()}, "
          f">10%={(rel > 0.10+1e-10).sum()}, max_us={regret_us.max():.3f}, "
          f"total_us={regret_us.sum():.3f}, max_relative_pct={100*rel.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    args = parser.parse_args()
    raw = args.input.read_text()
    if not re.search(r'^# mode=rk_fp8_deep_r_small_k_five_way_holdout\b', raw, re.MULTILINE):
        raise ValueError('Not a five-way holdout CSV')
    sm_match = re.search(r'^# holdout_sm_count=(\d+)\b', raw, re.MULTILINE)
    complete = re.search(
        r'^# rk_fp8_deep_r_small_k_five_way_holdout_complete shapes=(\d+) '
        r'candidate_configurations=(\d+)\s*$', raw, re.MULTILINE)
    if sm_match is None or complete is None:
        raise ValueError('Incomplete holdout: missing actual SM count or completion marker')
    sms = int(sm_match.group(1))
    n_shapes, n_candidates = map(int, complete.groups())
    if sms < 1 or (n_shapes, n_candidates) != (EXPECTED_GEOMETRIES, EXPECTED_CANDIDATES):
        raise ValueError('Incorrect SM count or unexpectedly missing geometry/candidate rows')
    data = pd.read_csv(args.input, comment='#')
    required = set(KEYS + ["rows_per_shard", "packet_bytes", "candidate_role", "total_median_ms",
                           "first_median_ms", "continuation_median_ms", "resident_ctas_per_sm",
                           "selector_score", "selector_chosen"] + SAMPLES)
    if not required.issubset(data.columns):
        raise ValueError(f'Missing CSV columns: {required - set(data.columns)}')
    if len(data) != n_candidates or data.groupby(KEYS).ngroups != n_shapes:
        raise ValueError('Completion count mismatch')
    if data.duplicated(KEYS + ["rows_per_shard", "packet_bytes"]).any():
        raise ValueError('Duplicate geometry/candidate')
    if not (data.packet_bytes == 16).all() or not (data.candidate_role == 'allowed').all():
        raise ValueError('Holdout must contain ONLY the five legal P16 candidates')
    for (outer, r, k), group in data.groupby(KEYS):
        expected = {d for d in DEPTHS if d < r}
        if set(group.rows_per_shard) != expected or len(group) != len(expected):
            raise ValueError(f'Wrong compiled depth inventory for {(outer,r,k)}')
        if group.selector_chosen.sum() != 1:
            raise ValueError(f'Expected one selector choice for {(outer,r,k)}')
    if (data.total_median_ms <= 0).any() or (data.resident_ctas_per_sm <= 0).any():
        raise ValueError('Invalid elapsed time or kernel residency')
    if not np.allclose(data[SAMPLES].median(axis=1), data.total_median_ms, rtol=0, atol=1e-7):
        raise ValueError('Total medians do not match the five interleaved samples')
    data['independent_score'] = data.apply(score, axis=1, sms=sms)
    if not np.allclose(data.independent_score, data.selector_score, rtol=1e-5, atol=1e-5):
        raise ValueError('Frozen-score calculation differs from the recorded C++ score')
    g = data.groupby(KEYS)
    predicted = data.loc[g.independent_score.idxmin(), KEYS + ["rows_per_shard"]]
    chosen = data[data.selector_chosen == 1].copy()
    chosen = chosen.merge(predicted, on=KEYS, suffixes=('', '_independent'), validate='one_to_one')
    if (chosen.rows_per_shard != chosen.rows_per_shard_independent).any():
        raise ValueError('Host selector choice disagrees with independent score')
    best_five = g.total_median_ms.min().rename('five_oracle_ms')
    best_four = data[data.rows_per_shard.isin(OLD_DEPTHS)].groupby(KEYS).total_median_ms.min().rename('four_oracle_ms')
    oracle_depth = data.loc[g.total_median_ms.idxmin(), KEYS + ["rows_per_shard"]].rename(
        columns={'rows_per_shard':'five_oracle_depth'})
    chosen = chosen.merge(best_five.reset_index(), on=KEYS, validate='one_to_one')
    chosen = chosen.merge(best_four.reset_index(), on=KEYS, validate='one_to_one')
    chosen = chosen.merge(oracle_depth, on=KEYS, validate='one_to_one')
    chosen['chosen_ms'] = chosen.total_median_ms
    chosen['selector_regret_us'] = (chosen.chosen_ms - chosen.five_oracle_ms) * 1000
    chosen['four_to_five_oracle_gain_us'] = (chosen.four_oracle_ms - chosen.five_oracle_ms) * 1000
    chosen['input_mib'] = chosen.outer * chosen.R * chosen.K / (1024*1024)
    print(f'Independent FIVE-way holdout: shapes={len(chosen)}, candidates={len(data)}, '
          f'SM_count={sms}; actual compiled P16 occupancy')
    summarize('Selector regret vs FIVE-choice oracle', chosen, 'five_oracle_ms')
    for label, mask in [('<24MiB', chosen.input_mib < 24),
                        ('24-96MiB', (chosen.input_mib >= 24) & (chosen.input_mib < 96)),
                        ('>=96MiB', chosen.input_mib >= 96),
                        ('tiny K<=11', chosen.K <= 11),
                        ('long R>=7168', chosen.R >= 7168)]:
        sub = chosen[mask]
        if len(sub): summarize('  '+label, sub, 'five_oracle_ms')
    print('Chosen depths:', chosen.rows_per_shard.value_counts().sort_index().to_dict())
    print('Oracle depths:', chosen.five_oracle_depth.value_counts().sort_index().to_dict())
    gains = chosen.four_to_five_oracle_gain_us
    print(f'Inventory gain from admitting 1024: shapes_improved={(gains > 0.001).sum()}/{len(gains)}, '
          f'>5% improvement={int(((chosen.four_oracle_ms / chosen.five_oracle_ms-1) > 0.05).sum())}, '
          f'max_gain_us={gains.max():.3f}, total_gain_us={gains.sum():.3f}')
    print('Top selector regret:')
    print(chosen.nlargest(8, 'selector_regret_us')
          [KEYS + ['rows_per_shard', 'five_oracle_depth', 'five_oracle_ms', 'chosen_ms',
                   'selector_regret_us', 'resident_ctas_per_sm']].to_string(index=False))
    print('Top inventory improvements:')
    print(chosen.nlargest(8, 'four_to_five_oracle_gain_us')
          [KEYS + ['four_oracle_ms', 'five_oracle_ms', 'five_oracle_depth',
                   'four_to_five_oracle_gain_us']].to_string(index=False))
    if args.details:
        chosen.sort_values(KEYS).to_csv(args.details, index=False)
        print(f'Saved: {args.details}')


if __name__ == '__main__':
    main()
