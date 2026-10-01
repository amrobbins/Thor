#!/usr/bin/env python3
"""Analyze --rk-census-gate CSV output.

After DenseRK legacy deletion the census is production-modern-only: ordinary dense RK production itself is the
KParallel/RCooperative family. Older pre-delete captures that contain both `production` and
`candidate:dense_rk_family_shadow` rows from archived captures are still accepted for historical comparison.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

CANDIDATE_EXECUTORS = {"candidate:dense_rk_family", "candidate:dense_rk_family_shadow"}


@dataclass(frozen=True)
class Key:
    family: str
    case: str
    dimensions: str
    axes: str
    dtype: str
    operation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="stdout captured from thor_cub_reduction_benchmark --rk-census-gate")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=None,
        help="Pre-delete captures only: fail when modern / legacy-production median exceeds this ratio.",
    )
    parser.add_argument("--show-worst", type=int, default=25, help="Pre-delete captures only: rows to print.")
    return parser.parse_args()


def row_key(row: dict[str, str]) -> Key:
    return Key(row["family"], row["case"], row["dimensions"], row["axes"], row["dtype"], row["operation"])


def inventory_markers(rows: list[dict[str, str]]) -> dict[str, bool]:
    strategies = [row["strategy"] for row in rows]
    return {
        "k_parallel": any("k_parallel" in strategy for strategy in strategies),
        "r_cooperative": any("r_cooperative" in strategy for strategy in strategies),
        "rotated": any(":rotated:" in strategy for strategy in strategies),
        "flat_rows": any(":flat_rows:" in strategy for strategy in strategies),
        "staged": any(":staged" in strategy for strategy in strategies),
        "complete": any(":complete" in strategy for strategy in strategies),
    }


def main() -> int:
    args = parse_args()
    if args.max_regression is not None and args.max_regression <= 0:
        raise SystemExit("--max-regression must be positive")

    text = args.input.read_text()
    coverage_gaps = [line for line in text.splitlines() if line.startswith("# rk_gate_coverage_gap ")]
    csv_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not csv_lines:
        print("No RK census CSV rows found.", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(io.StringIO("\n".join(csv_lines))))
    required = {"executor", "family", "case", "dimensions", "axes", "dtype", "operation", "median_ms", "strategy"}
    if not rows or not required.issubset(rows[0]):
        print("Input does not look like --rk-census-gate output.", file=sys.stderr)
        return 2

    production = {row_key(row): row for row in rows if row["executor"] == "production"}
    shadow = {row_key(row): row for row in rows if row["executor"] in CANDIDATE_EXECUTORS}

    # Post-delete mode: production is the modern inventory. There is intentionally no legacy timing side compiled.
    if production and not shadow:
        markers = inventory_markers(list(production.values()))
        missing_inventory = [name for name, observed in markers.items() if not observed]
        invalid = []
        for key, row in production.items():
            value = float(row["median_ms"])
            if not math.isfinite(value) or value <= 0:
                invalid.append(key)

        print(f"modern_rows={len(production)}")
        print(f"coverage_gaps={len(coverage_gaps)}")
        print("legacy_reference_rows=0 (historical VALUE RK kernels are not compiled)")
        print("inventory_coverage=" + ",".join(f"{k}:{'yes' if v else 'no'}" for k, v in markers.items()))
        print("performance_gate=modern_only (use an archived pre-delete CSV for historical timing comparison)")

        if coverage_gaps:
            print("\nCOVERAGE GAPS")
            for line in coverage_gaps:
                print(line)
        if missing_inventory:
            print("\nMISSING MODERN INVENTORY COVERAGE")
            for marker in missing_inventory:
                print(marker)
        if invalid:
            print("\nINVALID MODERN TIMINGS")
            for key in invalid[:100]:
                print(key)
        return 1 if coverage_gaps or missing_inventory or invalid else 0

    # Historical pre-delete mode retained so archived captures remain analyzable.
    keys = sorted(set(production) | set(shadow), key=lambda k: (k.family, k.case, k.dtype, k.operation))
    missing = [key for key in keys if key not in production or key not in shadow]
    ratios: list[tuple[float, Key, dict[str, str], dict[str, str]]] = []
    for key in keys:
        if key not in production or key not in shadow:
            continue
        prod_ms = float(production[key]["median_ms"])
        modern_ms = float(shadow[key]["median_ms"])
        if not math.isfinite(prod_ms) or not math.isfinite(modern_ms) or prod_ms <= 0 or modern_ms <= 0:
            print(f"Invalid timing for {key}", file=sys.stderr)
            return 2
        ratios.append((modern_ms / prod_ms, key, production[key], shadow[key]))
    ratios.sort(key=lambda item: item[0], reverse=True)

    markers = inventory_markers(list(shadow.values()))
    missing_inventory = [name for name, observed in markers.items() if not observed]
    regressions = [] if args.max_regression is None else [item for item in ratios if item[0] > args.max_regression]

    print(f"paired_rows={len(ratios)}")
    print(f"coverage_gaps={len(coverage_gaps)}")
    print(f"missing_pairs={len(missing)}")
    print("inventory_coverage=" + ",".join(f"{k}:{'yes' if v else 'no'}" for k, v in markers.items()))
    if ratios:
        values = sorted(item[0] for item in ratios)
        median = values[len(values) // 2]
        p95 = values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)]
        print(f"modern_over_production median={median:.4f} p95={p95:.4f} worst={values[-1]:.4f}")
    if args.max_regression is None:
        print("performance_gate=report_only (historical paired capture)")
    else:
        print(f"performance_gate max_regression={args.max_regression:.4f} failures={len(regressions)}")

    if ratios and args.show_worst > 0:
        print("\nWORST MODERN/PRODUCTION RATIOS")
        for ratio, key, prod, cand in ratios[: args.show_worst]:
            print(
                f"{ratio:.4f}  {key.family}/{key.case} {key.dtype} {key.operation} "
                f"production={prod['strategy']} modern={cand['strategy']} "
                f"prod_ms={float(prod['median_ms']):.4f} modern_ms={float(cand['median_ms']):.4f}"
            )

    return 1 if coverage_gaps or missing or missing_inventory or regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
