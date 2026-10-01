#!/usr/bin/env python3
"""Summarize --rk-narrow-low-precision-calibration into a selector-envelope report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#") and line.strip()]
    return list(csv.DictReader(lines))


def gbps(row: dict[str, str]) -> float:
    return float(row["logical_GBps"])


def best(rows: list[dict[str, str]], implementation: str) -> dict[str, str] | None:
    candidates = [row for row in rows if row["implementation"] == implementation]
    return max(candidates, key=gbps) if candidates else None


def geometry(row: dict[str, str] | None) -> str:
    if row is None:
        return "n/a"
    if row["strategy"] == "k_parallel":
        return f"packet={row['packet_bytes']},threads={row['block_threads']},rows={row['rows_per_shard']}"
    return f"rows={row['rows_per_shard']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    print(f"rows={len(rows)}")

    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("family") != "narrow_low_precision_calibration":
            continue
        grouped[(int(row["K"]), row["dtype"], row["operation"])].append(row)

    print("NARROW LOW-PRECISION K=1..32 BEST SINGLE-PASS THROUGHPUT")
    winner_sets: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    flat_over_k: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)

    for key in sorted(grouped):
        k, dtype, op = key
        candidates = grouped[key]
        flat = best(candidates, "narrow_low_precision_flat_r_cooperative_first_stage")
        rotated = best(candidates, "awkward_alignment_rotated_sharded_first_stage")
        kparallel = best(candidates, "k_parallel_tiled_stage")
        if flat is None or rotated is None:
            continue

        ranked = [("flat", flat), ("rotated", rotated)]
        if kparallel is not None:
            ranked.append(("kparallel", kparallel))
        winner, winner_row = max(ranked, key=lambda item: gbps(item[1]))
        winner_sets[(dtype, op)][winner].append(k)

        line = (
            f"K={k:2d} {dtype} {op}: "
            f"flat={gbps(flat):.2f}({geometry(flat)}) "
            f"rotated={gbps(rotated):.2f}({geometry(rotated)}) "
            f"flat/rotated={gbps(flat) / gbps(rotated):.4f}"
        )
        if kparallel is not None:
            ratio = gbps(flat) / gbps(kparallel)
            flat_over_k[(dtype, op)].append((k, ratio))
            line += f" kparallel={gbps(kparallel):.2f}({geometry(kparallel)}) flat/kparallel={ratio:.4f}"
        line += f" winner={winner} winner_GBps={gbps(winner_row):.2f}"
        print(line)

    print("\nWINNER K SETS")
    for dtype, op in sorted(winner_sets):
        by_winner = winner_sets[(dtype, op)]
        parts = []
        for winner in ("flat", "kparallel", "rotated"):
            ks = by_winner.get(winner, [])
            if ks:
                parts.append(f"{winner}=" + ",".join(map(str, ks)))
        print(f"{dtype} {op}: " + " ".join(parts))

    print("\nFLAT / KPARALLEL FOR PACKET-LEGAL K")
    for dtype, op in sorted(flat_over_k):
        pairs = flat_over_k[(dtype, op)]
        wins = [k for k, ratio in pairs if ratio >= 1.0]
        losses = [(k, ratio) for k, ratio in pairs if ratio < 1.0]
        best_ratio = max((ratio for _, ratio in pairs), default=float("nan"))
        worst_ratio = min((ratio for _, ratio in pairs), default=float("nan"))
        print(
            f"{dtype} {op}: flat_wins={','.join(map(str, wins)) or '-'} "
            f"flat_losses=" + (",".join(f"{k}({ratio:.4f})" for k, ratio in losses) or "-") +
            f" worst_ratio={worst_ratio:.4f} best_ratio={best_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
