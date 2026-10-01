#!/usr/bin/env python3
"""Analyze realistic RK end-to-end progress calibration by wall-clock value.

Primary objective: absolute full-reduction microseconds saved. Ratios are printed only as context.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        lines = (line for line in f if not line.startswith("#"))
        return list(csv.DictReader(lines))


def ms(row: dict[str, str]) -> float:
    return float(row["median_ms"])


def describe(row: dict[str, str]) -> str:
    return (
        f'{row["implementation"]}/{row["first_strategy"]} '
        f'rows={row["first_rows_per_shard"]} shards={row["first_shards"]} '
        f'stages={row["total_stages"]} chain={row["continuation_chain"]}'
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-realistic-end-to-end-progress-calibration.csv>")
    rows = load_rows(Path(sys.argv[1]))
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)

    summaries = []
    for case, group in by_case.items():
        production = next((r for r in group if r["implementation"] == "production_selector"), None)
        current = next((r for r in group if r["implementation"] == "current_modern"), None)
        forced = [r for r in group if r["implementation"] == "forced_first_pass"]
        if production is None or current is None or not forced:
            continue
        best = min(forced, key=ms)
        current_ms = ms(current)
        production_ms = ms(production)
        best_ms = ms(best)
        summaries.append(
            {
                "case": case,
                "outer": int(current["outer"]),
                "R": int(current["R"]),
                "K": int(current["K"]),
                "dtype": current["dtype"],
                "current": current,
                "production": production,
                "best": best,
                "current_ms": current_ms,
                "production_ms": production_ms,
                "best_ms": best_ms,
                "save_vs_current_us": (current_ms - best_ms) * 1000.0,
                "gap_vs_production_us": (current_ms - production_ms) * 1000.0,
            }
        )

    print(f"cases={len(summaries)} rows={len(rows)}")
    print("\nFULL REDUCTIONS RANKED BY CURRENT MODERN WALL TIME")
    for s in sorted(summaries, key=lambda x: x["current_ms"], reverse=True):
        ratio = s["current_ms"] / s["production_ms"] if s["production_ms"] else float("inf")
        print(
            f'{s["case"]}: current={s["current_ms"]*1000:.1f}us '
            f'production={s["production_ms"]*1000:.1f}us gap={s["gap_vs_production_us"]:+.1f}us '
            f'ratio={ratio:.3f} best_forced={s["best_ms"]*1000:.1f}us '
            f'save={s["save_vs_current_us"]:+.1f}us {describe(s["best"])}'
        )

    print("\nRANKED BY ABSOLUTE MICROSECONDS SAVED VS CURRENT MODERN")
    for s in sorted(summaries, key=lambda x: x["save_vs_current_us"], reverse=True):
        print(
            f'{s["case"]}: save={s["save_vs_current_us"]:+.1f}us '
            f'{s["current_ms"]*1000:.1f}->{s["best_ms"]*1000:.1f}us '
            f'production={s["production_ms"]*1000:.1f}us best={describe(s["best"])}'
        )

    meaningful = [s for s in summaries if s["save_vs_current_us"] >= 10.0]
    print("\nMEANINGFUL OPPORTUNITIES (>=10us ABSOLUTE SAVING)")
    if not meaningful:
        print("none")
    else:
        total = sum(s["save_vs_current_us"] for s in meaningful)
        print(f"count={len(meaningful)} aggregate_save_us={total:.1f}")
        for s in sorted(meaningful, key=lambda x: x["save_vs_current_us"], reverse=True):
            print(
                f'{s["case"]}: {s["save_vs_current_us"]:+.1f}us '
                f'best_strategy={s["best"]["first_strategy"]} '
                f'rows={s["best"]["first_rows_per_shard"]} shards={s["best"]["first_shards"]} '
                f'stages={s["best"]["total_stages"]}'
            )


if __name__ == "__main__":
    main()
