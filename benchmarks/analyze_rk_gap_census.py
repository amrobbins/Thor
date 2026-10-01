#!/usr/bin/env python3
"""Summarize --rk-modern-gap-census output without imposing a performance threshold."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#") and line.strip()]
    return list(csv.DictReader(lines))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def minmax_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["family"] != "minmax":
            continue
        key = tuple(
            row[name]
            for name in (
                "case",
                "dtype",
                "strategy",
                "access",
                "progress",
                "implementation",
                "packet_bytes",
                "block_threads",
                "rows_per_shard",
                "shards_per_output",
            )
        )
        grouped[key][row["operation"]] = row

    min_ratios: list[float] = []
    max_ratios: list[float] = []
    comparisons: list[tuple[float, str]] = []
    for key, by_op in grouped.items():
        if not {"sum", "min", "max"}.issubset(by_op):
            continue
        sum_gbps = f(by_op["sum"], "logical_GBps")
        if sum_gbps <= 0:
            continue
        min_ratio = f(by_op["min"], "logical_GBps") / sum_gbps
        max_ratio = f(by_op["max"], "logical_GBps") / sum_gbps
        min_ratios.append(min_ratio)
        max_ratios.append(max_ratio)
        label = "/".join(key[:2]) + f" {key[4]} packet={key[6]} threads={key[7]} rows={key[8]}"
        comparisons.append((min(min_ratio, max_ratio), label))

    print("MIN/MAX SAME-GEOMETRY THROUGHPUT / SUM")
    if not min_ratios:
        print("no complete SUM/MIN/MAX geometry triplets")
        return
    print(
        f"triplets={len(min_ratios)} "
        f"min_median={statistics.median(min_ratios):.4f} "
        f"max_median={statistics.median(max_ratios):.4f} "
        f"min_best={max(min_ratios):.4f} max_best={max(max_ratios):.4f}"
    )
    print("lowest same-geometry ratios:")
    for ratio, label in sorted(comparisons)[:12]:
        print(f"  {ratio:.4f}  {label}")


def narrow_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["family"] == "narrow_awkward":
            grouped[(row["case"], row["dtype"], row["operation"])].append(row)

    print("\nNARROW AWKWARD BEST SINGLE-PASS THROUGHPUT")
    for key in sorted(grouped):
        case, dtype, op = key
        candidates = grouped[key]
        best_by_impl: dict[str, dict[str, str]] = {}
        for row in candidates:
            impl = row["implementation"]
            if impl not in best_by_impl or f(row, "logical_GBps") > f(best_by_impl[impl], "logical_GBps"):
                best_by_impl[impl] = row
        flat = best_by_impl.get("narrow_low_precision_flat_r_cooperative_first_stage")
        rotated = best_by_impl.get("awkward_alignment_rotated_sharded_first_stage")
        if flat is None or rotated is None:
            continue
        flat_gbps = f(flat, "logical_GBps")
        rotated_gbps = f(rotated, "logical_GBps")
        ratio = flat_gbps / rotated_gbps if rotated_gbps > 0 else float("inf")
        line = (
            f"{case} {dtype} {op}: flat={flat_gbps:.2f}GB/s(rows={flat['rows_per_shard']}) "
            f"rotated={rotated_gbps:.2f}GB/s(rows={rotated['rows_per_shard']}) flat/rotated={ratio:.4f}"
        )
        k_parallel = best_by_impl.get("k_parallel_tiled_stage")
        if k_parallel is not None:
            k_parallel_gbps = f(k_parallel, "logical_GBps")
            flat_over_k_parallel = flat_gbps / k_parallel_gbps if k_parallel_gbps > 0 else float("inf")
            line += (
                f" kparallel={k_parallel_gbps:.2f}GB/s(packet={k_parallel['packet_bytes']},"
                f"threads={k_parallel['block_threads']},rows={k_parallel['rows_per_shard']}) "
                f"flat/kparallel={flat_over_k_parallel:.4f}"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    print(f"rows={len(rows)}")
    minmax_summary(rows)
    narrow_summary(rows)


if __name__ == "__main__":
    main()
