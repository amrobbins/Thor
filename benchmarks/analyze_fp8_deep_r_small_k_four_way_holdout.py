#!/usr/bin/env python3
"""Read-only independent GPU holdout evaluation for the frozen DeepRSmallK P16 policy.

Usage:
  python benchmarks/analyze_fp8_deep_r_small_k_four_way_holdout.py \
    --input rk-fp8-deep-r-four-way-holdout.csv --details deep-r-four-way-holdout-details.csv

No training, fitting, or selector changes are performed. The 1024-row calibration
kernel, present only for selected deep/tiny-K shapes, is NOT a selectable depth
and is NOT an unrestricted ladder oracle.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEPTHS = (64, 128, 256, 512)
KEYS = ["outer", "R", "K"]
SAMPLE_COLS = [f"sample{i}_ms" for i in range(5)]
COMPLETE = "# rk_fp8_deep_r_small_k_four_way_holdout_complete"


def score(row, sms):
    outer, r, k, depth, active = (int(row[col]) for col in
                                   ("outer", "R", "K", "rows_per_shard", "resident_ctas_per_sm"))
    if active < 1:
        raise ValueError("zero actual CUDA kernel residency")
    shards = (r + depth - 1) // depth
    group_rows = min(depth // 8, 497 // k)
    assert group_rows > 0
    round_rows = 8 * group_rows
    full_rounds = (depth + round_rows - 1) // round_rows
    last_rows = r - (shards - 1) * depth
    last_rounds = (last_rows + round_rows - 1) // round_rows
    all_rounds = outer * ((shards - 1) * full_rounds + last_rounds)
    staging_waves = max(full_rounds, all_rounds / (sms * active))
    partial_mb = outer * shards * k * 4 / 1e6
    return staging_waves + 0.5 * full_rounds + 0.25 * partial_mb


def report_metrics(label, data, reference):
    regret_us = (data["chosen_ms"] - data[reference]) * 1000.0
    relative = regret_us / (data[reference] * 1000.0)
    print(f"{label}: shapes={len(data)}, >5%={int((relative > 0.05 + 1e-10).sum())}, "
          f">10%={int((relative > 0.10 + 1e-10).sum())}, "
          f"max_us={regret_us.max():.3f}, total_us={regret_us.sum():.3f}, "
          f"max_relative_pct={relative.max()*100:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--sms", type=int,
                        help="Override actual device SM count; by default read the GPU benchmark header")
    args = parser.parse_args()
    raw = args.input.read_text()
    sm_match = re.search(r"^# holdout_sm_count=(\d+)\b", raw, re.MULTILINE)
    if sm_match is None:
        raise ValueError("Holdout is missing an actual device SM count header")
    sms = int(sm_match.group(1))
    if args.sms is not None and args.sms != sms:
        raise ValueError("--sms differs from the actual device SM count in the holdout file")
    if sms < 1:
        raise ValueError("Invalid GPU SM count")
    completion = re.search(
        r"^# rk_fp8_deep_r_small_k_four_way_holdout_complete shapes=(\d+) "
        r"candidate_configurations=(\d+)\s*$", raw, re.MULTILINE)
    if completion is None:
        raise ValueError("Holdout run is incomplete: expected completion marker missing")
    geometry_count, candidate_count = map(int, completion.groups())
    data = pd.read_csv(args.input, comment="#")
    if data.empty:
        raise ValueError("Empty holdout")
    if len(data) != candidate_count or len(data.groupby(KEYS)) != geometry_count:
        raise ValueError("Holdout CSV row/geometry counts do not match completion marker")
    required = set(KEYS + ["rows_per_shard", "packet_bytes", "total_median_ms", "candidate_role",
                           "selector_score", "selector_chosen", "resident_ctas_per_sm"] + SAMPLE_COLS)
    if not required.issubset(data.columns):
        raise ValueError(f"Missing holdout CSV fields: {required - set(data.columns)}")
    if data.duplicated(KEYS + ["rows_per_shard", "packet_bytes"]).any():
        raise ValueError("Duplicate shape/candidate pairs")
    if (data.packet_bytes != 16).any():
        raise ValueError("Holdout must be P16 only")
    if not np.allclose(data[SAMPLE_COLS].median(axis=1), data.total_median_ms, rtol=0, atol=1e-7):
        raise ValueError("Reported total medians disagree with five actual samples")
    allowed = data[data.candidate_role == "allowed"].copy()
    controls = data[data.candidate_role == "inventory_control"].copy()
    if len(data) != len(allowed) + len(controls):
        raise ValueError("Unknown candidate roles")
    if not allowed.rows_per_shard.isin(DEPTHS).all():
        raise ValueError("Non-four-depth selectable kernel in holdout")
    g = allowed.groupby(KEYS)
    for (_, r, _), group in g:
        legal_depths = {d for d in DEPTHS if d < r}
        if set(group.rows_per_shard) != legal_depths:
            raise ValueError(f"Missing/extra allowed depth for R={r}")
    if not (g.selector_chosen.sum() == 1).all():
        raise ValueError("Expected exactly one selector choice per geometry")
    allowed["independent_score"] = allowed.apply(score, axis=1, sms=sms)
    if not np.allclose(allowed.independent_score, allowed.selector_score, rtol=1e-5, atol=1e-5):
        raise ValueError("Actual occupancy hardware score disagrees with independent calculation")
    score_min = allowed.loc[g.independent_score.idxmin(), KEYS + ["rows_per_shard"]]
    chosen = allowed[allowed.selector_chosen == 1].copy()
    chosen = chosen.merge(score_min, on=KEYS, suffixes=("", "_by_score"), validate="one_to_one")
    if (chosen.rows_per_shard != chosen.rows_per_shard_by_score).any():
        raise ValueError("Recorded selector choice disagrees with min cost score")
    best_four = g.total_median_ms.min().rename("four_oracle_ms")
    chosen = chosen.merge(best_four.reset_index(), on=KEYS, validate="one_to_one")
    chosen["chosen_ms"] = chosen.total_median_ms
    chosen["selector_regret_us"] = (chosen.chosen_ms - chosen.four_oracle_ms)*1e3
    chosen["input_mib"] = chosen.outer*chosen.R*chosen.K/(1024*1024)
    print(f"Independent holdout: {len(chosen)} shapes, {len(allowed)} allowed candidate rows, "
          f"{len(controls)} nonselectable 1024-row controls; SM count={sms}")
    report_metrics("Selector regret vs best permitted four", chosen, "four_oracle_ms")
    for label, condition in (("<24MiB", chosen.input_mib < 24),
                             ("24-96MiB", (chosen.input_mib >= 24)&(chosen.input_mib < 96)),
                             (">=96MiB", chosen.input_mib >= 96),
                             ("long R>=6144", chosen.R >= 6144)):
        group = chosen.loc[condition]
        if not group.empty:
            report_metrics(f"  {label}", group, "four_oracle_ms")
    print("Chosen compiled depths:", chosen.rows_per_shard.value_counts().sort_index().to_dict())
    if not controls.empty:
        if (controls.rows_per_shard != 1024).any() or (controls.selector_chosen != 0).any():
            raise ValueError("Unexpected selection or depth in diagnostic inventory controls")
        controls = controls[KEYS + ["total_median_ms"]].rename(columns={"total_median_ms": "control_1024_ms"})
        chosen = chosen.merge(controls, on=KEYS, how="left", validate="one_to_one")
        comparable = chosen[chosen.control_1024_ms.notna()].copy()
        comparable["inventory_gap_1024_us"] = (comparable.four_oracle_ms -
                                                comparable.control_1024_ms).clip(lower=0)*1e3
        print(f"1024-control comparisons={len(comparable)}, "
              f"control faster than four-way oracle={(comparable.inventory_gap_1024_us > 0).sum()}, "
              f"max demonstrated gap={comparable.inventory_gap_1024_us.max():.3f}us, "
              "NOT full-inventory regret")
    print("Top selector-regret shapes:")
    print(chosen.nlargest(8, "selector_regret_us")
          [KEYS + ["rows_per_shard", "four_oracle_ms", "chosen_ms", "selector_regret_us", "resident_ctas_per_sm"]]
          .to_string(index=False))
    if args.details:
        chosen.sort_values(KEYS).to_csv(args.details, index=False)
        print(f"Saved: {args.details}")


if __name__ == "__main__":
    main()
