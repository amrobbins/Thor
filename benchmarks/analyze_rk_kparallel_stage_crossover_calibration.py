#!/usr/bin/env python3
"""Summarize --rk-kparallel-stage-crossover-calibration output."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    with path.open(newline="") as f:
        lines = (line for line in f if line and not line.startswith("#"))
        return list(csv.DictReader(lines))


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-kparallel-stage-crossover-calibration.csv>")
    rows = load_rows(Path(sys.argv[1]))
    print(f"rows={len(rows)}")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["dtype"])].append(row)

    stage_over_lean = []
    prod_ratios = []
    by_outer_reduction = defaultdict(list)
    winners = defaultdict(lambda: {"lean": 0, "staged": 0})

    print("BEST LEAN COMPLETE VS STAGED KPARALLEL FIRST PASS")
    for key in sorted(groups, key=lambda k: (i(groups[k][0], "outer"), i(groups[k][0], "R"), i(groups[k][0], "K"), k[1])):
        group = groups[key]
        prod = min((r for r in group if r["implementation"] == "production_selector"), key=lambda r: f(r, "median_ms"))
        lean = min((r for r in group if r["implementation"] == "lean_kparallel_complete"), key=lambda r: f(r, "median_ms"))
        staged = min((r for r in group if r["implementation"] == "kparallel_staged"), key=lambda r: f(r, "median_ms"))
        ratio = f(staged, "median_ms") / f(lean, "median_ms")
        prod_ratio = min(f(staged, "median_ms"), f(lean, "median_ms")) / f(prod, "median_ms")
        stage_over_lean.append(ratio)
        prod_ratios.append(prod_ratio)
        o, r, k = i(lean, "outer"), i(lean, "R"), i(lean, "K")
        winner = "staged" if ratio < 1.0 else "lean"
        winners[(o, r)][winner] += 1
        by_outer_reduction[(o, r)].append(ratio)
        print(
            f"{lean['case']} {lean['dtype']}: winner={winner} "
            f"lean={f(lean,'median_ms'):.4f}ms(packet={lean['packet_bytes']},threads={lean['block_threads']}) "
            f"staged={f(staged,'median_ms'):.4f}ms(packet={staged['packet_bytes']},threads={staged['block_threads']},"
            f"rows={staged['rows_per_shard']},shards={staged['shards_per_output']},blocks={staged['blocks']},"
            f"intermediate={staged['output_bytes']}B) staged/lean={ratio:.4f} "
            f"production={f(prod,'median_ms'):.4f}ms best/production={prod_ratio:.4f}"
        )

    print("\nCROSSOVER SUMMARY BY O,R")
    for (o, r) in sorted(by_outer_reduction):
        vals = by_outer_reduction[(o, r)]
        counts = winners[(o, r)]
        print(
            f"O={o} R={r}: staged_wins={counts['staged']} lean_wins={counts['lean']} "
            f"staged/lean_median={statistics.median(vals):.4f} "
            f"min={min(vals):.4f} max={max(vals):.4f}"
        )

    if stage_over_lean:
        print("\nOVERALL")
        print(
            f"staged_over_lean_median={statistics.median(stage_over_lean):.4f} "
            f"staged_win_cases={sum(x < 1.0 for x in stage_over_lean)}/{len(stage_over_lean)}"
        )
        print(
            f"best_of_lean_or_staged_over_production_median={statistics.median(prod_ratios):.4f} "
            f"p95={sorted(prod_ratios)[int(0.95 * (len(prod_ratios)-1))]:.4f} "
            f"worst={max(prod_ratios):.4f}"
        )


if __name__ == "__main__":
    main()
