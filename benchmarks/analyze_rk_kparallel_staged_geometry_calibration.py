#!/usr/bin/env python3
"""Summarize --rk-kparallel-staged-geometry-calibration output."""

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path):
    with path.open(newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-kparallel-staged-geometry-calibration.csv>")
    rows = load_rows(Path(sys.argv[1]))
    print(f"rows={len(rows)}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(i(row, "outer"), i(row, "R"), i(row, "K"), row["dtype"])].append(row)

    winners = []
    print("BEST FIXED-TOPOLOGY STAGED KPARALLEL GEOMETRY")
    for key in sorted(grouped):
        candidates = grouped[key]
        best = min(candidates, key=lambda r: f(r, "median_ms"))
        o, r, k, dtype = key
        winners.append(best)
        print(
            f"o{o}_r{r}_i{k} {dtype}: "
            f"packet={i(best,'packet_bytes')} threads={i(best,'block_threads')} "
            f"warps={i(best,'warps_per_cta')} owners={i(best,'packet_owners')} "
            f"tiles={i(best,'component_tiles')} blocks={i(best,'blocks')} "
            f"time={f(best,'median_ms'):.4f}ms gbps={f(best,'logical_GBps'):.2f}"
        )

    print("\nWINNER CTA WIDTH COUNTS")
    for dtype in ("fp16", "bf16", "fp32"):
        subset = [w for w in winners if w["dtype"] == dtype]
        counts = Counter(i(w, "block_threads") for w in subset)
        detail = " ".join(f"t{threads}={counts[threads]}" for threads in (32, 64, 128, 256))
        print(f"{dtype}: {detail}")

    print("\nWINNER PACKET WIDTH COUNTS")
    for dtype in ("fp16", "bf16", "fp32"):
        subset = [w for w in winners if w["dtype"] == dtype]
        counts = Counter(i(w, "packet_bytes") for w in subset)
        print(f"{dtype}: p4={counts[4]} p8={counts[8]} p16={counts[16]}")

    print("\nCTA WIDTH BY K")
    for k in sorted({i(w, "K") for w in winners}):
        subset = [w for w in winners if i(w, "K") == k]
        counts = Counter(i(w, "block_threads") for w in subset)
        detail = " ".join(f"t{threads}={counts[threads]}" for threads in (32, 64, 128, 256) if counts[threads])
        print(f"K={k}: {detail}")

    print("\nCTA WIDTH BY PACKET OWNERS")
    by_owners = defaultdict(list)
    for w in winners:
        by_owners[i(w, "packet_owners")].append(w)
    for owners in sorted(by_owners):
        subset = by_owners[owners]
        counts = Counter(i(w, "block_threads") for w in subset)
        detail = " ".join(f"t{threads}={counts[threads]}" for threads in (32, 64, 128, 256) if counts[threads])
        print(f"owners={owners}: n={len(subset)} {detail}")

    ratios = []
    for key, candidates in grouped.items():
        best = min(candidates, key=lambda r: f(r, "median_ms"))
        # Compare the existing production-like CTA width for each packet to the global best.
        for packet in (4, 8, 16):
            pc = [r for r in candidates if i(r, "packet_bytes") == packet]
            if not pc:
                continue
            # Approximate current helper: smallest power-of-two warp count up to 8 needed to cover packet owners.
            owners = i(pc[0], "packet_owners")
            required_warps = (owners + 31) // 32
            warps = 1
            while warps < 8 and warps < required_warps:
                warps *= 2
            current_threads = warps * 32
            current = next((r for r in pc if i(r, "block_threads") == current_threads), None)
            if current is not None:
                ratios.append(f(current, "median_ms") / f(best, "median_ms"))
    if ratios:
        print("\nCURRENT-HELPER / BEST GEOMETRY")
        print(
            f"samples={len(ratios)} median={statistics.median(ratios):.4f} "
            f"p95={sorted(ratios)[int(0.95*(len(ratios)-1))]:.4f} max={max(ratios):.4f}"
        )


if __name__ == "__main__":
    main()
