#!/usr/bin/env python3
"""Summarize --rk-kparallel-end-to-end-crossover-calibration output."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(line for line in f if not line.startswith("#"))]
    if not rows:
        raise SystemExit("no benchmark rows found")
    return rows


def best(rows, impl):
    candidates = [r for r in rows if r["implementation"] == impl]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["median_ms"]))


def fmt(row):
    if row is None:
        return "n/a"
    extra = ""
    if row["implementation"].startswith("staged"):
        extra = f",rows={row['first_rows_per_shard']},shards={row['first_shards']},stages={row['total_stages']}"
    elif row["implementation"] == "lean_kparallel_complete":
        extra = ",stages=1"
    return f"{float(row['median_ms']):.4f}ms(packet={row['packet_bytes']},threads={row['block_threads']}{extra})"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-kparallel-end-to-end-crossover-calibration.csv>")
    rows = load(Path(sys.argv[1]))
    print(f"rows={len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["outer"]), int(row["R"]), int(row["K"]), row["dtype"])].append(row)

    ratios = []
    best_vs_prod = []
    winner_counts = defaultdict(int)
    by_or = defaultdict(lambda: defaultdict(int))
    print("BEST END-TO-END LEAN COMPLETE VS STAGED-FIRST + REPLAN")
    for key in sorted(groups):
        o, r, k, dtype = key
        g = groups[key]
        prod = best(g, "production_selector")
        lean = best(g, "lean_kparallel_complete")
        s16 = best(g, "staged16_then_replan")
        s32 = best(g, "staged32_then_replan")
        candidates = [x for x in (lean, s16, s32) if x is not None]
        winner = min(candidates, key=lambda x: float(x["median_ms"]))
        winner_name = winner["implementation"]
        winner_counts[winner_name] += 1
        by_or[(o, r)][winner_name] += 1
        if lean is not None and winner_name != "lean_kparallel_complete":
            ratios.append(float(winner["median_ms"]) / float(lean["median_ms"]))
        if prod is not None:
            best_vs_prod.append(float(winner["median_ms"]) / float(prod["median_ms"]))
        ratio = float(winner["median_ms"]) / float(lean["median_ms"]) if lean else float("nan")
        prod_ratio = float(winner["median_ms"]) / float(prod["median_ms"]) if prod else float("nan")
        print(
            f"o{o}_r{r}_i{k} {dtype}: winner={winner_name} lean={fmt(lean)} "
            f"staged16={fmt(s16)} staged32={fmt(s32)} best/lean={ratio:.4f} "
            f"production={float(prod['median_ms']):.4f}ms best/production={prod_ratio:.4f}"
        )

    print("\nCROSSOVER SUMMARY BY O,R")
    for (o, r) in sorted(by_or):
        c = by_or[(o, r)]
        print(
            f"O={o} R={r}: lean={c['lean_kparallel_complete']} "
            f"staged16={c['staged16_then_replan']} staged32={c['staged32_then_replan']}"
        )

    print("\nGLOBAL SUMMARY")
    total = sum(winner_counts.values())
    print(
        "winner_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(winner_counts.items())) + f" total={total}"
    )
    if ratios:
        sr = sorted(ratios)
        print(
            f"staged_winner_over_lean median={statistics.median(sr):.4f} "
            f"p95={sr[min(len(sr)-1, int(0.95*(len(sr)-1)))]:.4f} best={min(sr):.4f}"
        )
    if best_vs_prod:
        bp = sorted(best_vs_prod)
        print(
            f"best_modern_over_production median={statistics.median(bp):.4f} "
            f"p95={bp[min(len(bp)-1, int(0.95*(len(bp)-1)))]:.4f} worst={max(bp):.4f}"
        )


if __name__ == "__main__":
    main()
