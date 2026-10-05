#!/usr/bin/env python3
"""Analyze the FP8 RCooperative single-pass calibration sweep.

Calibration rule:
    consume as much R as possible while the physical pass sustains the calibrated line-rate floor.

The default scheduling floor is 1500 GB/s input-read bandwidth. 1600 GB/s remains the peak/reference bar and is
reported separately. Every CSV row is exactly one physical CUDA reduction pass; there is no fallback, production
selector, or continuation timing in this calibration.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    family: str
    case: str
    outer: int
    reduction: int
    inner: int
    dtype: str
    access: str
    progress: str
    implementation: str
    packet_bytes: int
    block_threads: int
    rows_per_shard: int
    shards_per_output: int
    first_stage_blocks: int
    launch_warps_per_sm: float
    occupancy_waves: float
    input_bytes: int
    output_bytes: int
    median_ms: float
    best_ms: float
    worst_ms: float
    logical_gbps: float

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.family, self.case, self.dtype)

    @property
    def variant(self) -> str:
        if self.access.startswith("rotated_"):
            return f"{self.access}_p{self.packet_bytes}"
        if self.access == "flat_rows":
            return f"flat_rows_p{self.packet_bytes}"
        return f"{self.access}_p{self.packet_bytes}"

    @property
    def variant_key(self) -> tuple[str, str, str, str]:
        return (*self.key, self.variant)

    @property
    def spread(self) -> float:
        return self.worst_ms / self.best_ms - 1.0 if self.best_ms > 0 else math.inf

    @property
    def r_consumed(self) -> int:
        return min(self.rows_per_shard, self.reduction)

    @property
    def input_gbps(self) -> float:
        return self.input_bytes / (self.median_ms * 1.0e6)


def read_rows(path: Path) -> tuple[list[Row], dict[str, str]]:
    raw_lines = path.read_text().splitlines()
    metadata: dict[str, str] = {}
    for line in raw_lines:
        if not line.startswith("#"):
            continue
        for token in line[1:].strip().split():
            if "=" in token:
                key, value = token.split("=", 1)
                metadata[key] = value

    lines = [line for line in raw_lines if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: no CSV rows found")
    reader = csv.DictReader(lines)
    required = {
        "family", "case", "outer", "R", "K", "dtype", "access", "progress", "implementation",
        "packet_bytes", "block_threads", "rows_per_shard", "shards_per_output", "first_stage_blocks",
        "launch_warps_per_sm", "occupancy_waves", "input_bytes", "output_bytes", "median_ms", "best_ms",
        "worst_ms", "logical_GBps",
    }
    missing = required.difference(reader.fieldnames or ())
    if missing:
        raise SystemExit(f"{path}: missing columns: {sorted(missing)}")

    rows: list[Row] = []
    for raw in reader:
        rows.append(Row(
            family=raw["family"],
            case=raw["case"],
            outer=int(raw["outer"]),
            reduction=int(raw["R"]),
            inner=int(raw["K"]),
            dtype=raw["dtype"],
            access=raw["access"],
            progress=raw["progress"],
            implementation=raw["implementation"],
            packet_bytes=int(raw["packet_bytes"]),
            block_threads=int(raw["block_threads"]),
            rows_per_shard=int(raw["rows_per_shard"]),
            shards_per_output=int(raw["shards_per_output"]),
            first_stage_blocks=int(raw["first_stage_blocks"]),
            launch_warps_per_sm=float(raw["launch_warps_per_sm"]),
            occupancy_waves=float(raw["occupancy_waves"]),
            input_bytes=int(raw["input_bytes"]),
            output_bytes=int(raw["output_bytes"]),
            median_ms=float(raw["median_ms"]),
            best_ms=float(raw["best_ms"]),
            worst_ms=float(raw["worst_ms"]),
            logical_gbps=float(raw["logical_GBps"]),
        ))
    return rows, metadata


def describe_row(row: Row) -> str:
    return (
        f"{row.variant:20s} rows={row.rows_per_shard:<5d} shards={row.shards_per_output:<4d} "
        f"cta={row.block_threads:<3d} blocks={row.first_stage_blocks:<6d} "
        f"warps/SM={row.launch_warps_per_sm:5.2f} waves={row.occupancy_waves:6.2f} "
        f"input={row.input_gbps:7.1f}GB/s logical={row.logical_gbps:7.1f}GB/s {row.median_ms:.4f}ms"
    )


def deepest_at(rows: list[Row], threshold: float) -> Row | None:
    qualifying = [r for r in rows if r.input_gbps >= threshold]
    return max(qualifying, key=lambda r: (r.r_consumed, -r.median_ms)) if qualifying else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--line-rate-gbps", type=float, default=1500.0,
                        help="primary input-read GB/s scheduling floor (default: 1500)")
    parser.add_argument("--peak-reference-gbps", type=float, default=1600.0,
                        help="strong line-rate reference reported separately (default: 1600)")
    parser.add_argument("--unstable-spread", type=float, default=0.05,
                        help="worst/best timing spread above which a row is unstable (default: 0.05)")
    args = parser.parse_args()

    rows, metadata = read_rows(args.csv)
    duplicates = Counter((r.key, r.access, r.packet_bytes, r.rows_per_shard, r.block_threads) for r in rows)
    duplicate_count = sum(count - 1 for count in duplicates.values() if count > 1)
    if duplicate_count:
        raise SystemExit(f"duplicate physical rows={duplicate_count}")

    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    variant_groups: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.key].append(row)
        variant_groups[row.variant_key].append(row)

    timing_samples = metadata.get("timing_samples", "unknown")
    print(
        f"rows={len(rows)} exact_shape_dtypes={len(groups)} timing_samples={timing_samples} "
        f"line_rate_floor={args.line_rate_gbps:.1f}GB/s peak_reference={args.peak_reference_gbps:.1f}GB/s "
        f"unstable_spread>{args.unstable_spread * 100:.1f}%"
    )

    floor_frontier: dict[tuple[str, str, str, str], Row | None] = {}
    peak_frontier: dict[tuple[str, str, str, str], Row | None] = {}
    best_throughput: dict[tuple[str, str, str, str], Row] = {}
    for key, candidates in variant_groups.items():
        best_throughput[key] = max(candidates, key=lambda r: (r.input_gbps, r.r_consumed))
        floor_frontier[key] = deepest_at(candidates, args.line_rate_gbps)
        peak_frontier[key] = deepest_at(candidates, args.peak_reference_gbps)

    print("\nDEEPEST R CONSUMPTION BY PHYSICAL VARIANT")
    for variant in sorted({r.variant for r in rows}):
        keys = [key for key in variant_groups if key[3] == variant]
        floor_rows = [floor_frontier[k] for k in keys if floor_frontier[k] is not None]
        peak_rows = [peak_frontier[k] for k in keys if peak_frontier[k] is not None]
        if floor_rows:
            depths = [r.r_consumed for r in floor_rows]
            floor_text = (
                f"floor={len(floor_rows):2d}/{len(keys):2d} depth_med={statistics.median(depths):6.0f} "
                f"depth_max={max(depths):5d} throughput_med={statistics.median(r.input_gbps for r in floor_rows):7.1f}"
            )
        else:
            floor_text = f"floor= 0/{len(keys):2d}"
        if peak_rows:
            peak_text = f"peak={len(peak_rows):2d}/{len(keys):2d}"
        else:
            peak_text = f"peak= 0/{len(keys):2d}"
        print(f"{variant:27s}: {floor_text}  {peak_text}")

    print("\nBEST SCHEDULABLE SINGLE PASS PER EXACT SHAPE (DEEPEST >= FLOOR)")
    winner_counts = Counter()
    no_floor: list[tuple[tuple[str, str, str], Row]] = []
    for key, group in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[1][0].inner)):
        candidates = [floor_frontier[(*key, variant)] for variant in {r.variant for r in group}]
        candidates = [r for r in candidates if r is not None]
        if candidates:
            chosen = max(candidates, key=lambda r: (r.r_consumed, -r.median_ms))
            winner_counts[chosen.variant] += 1
            peak = " PEAK" if chosen.input_gbps >= args.peak_reference_gbps else ""
            unstable = " UNSTABLE" if chosen.spread > args.unstable_spread else ""
            print(
                f"{key[2]:8s} O={chosen.outer:<6d} R={chosen.reduction:<6d} K={chosen.inner:<4d} "
                + describe_row(chosen) + peak + unstable
            )
        else:
            best = max(group, key=lambda r: r.input_gbps)
            no_floor.append((key, best))
            print(
                f"{key[2]:8s} O={best.outer:<6d} R={best.reduction:<6d} K={best.inner:<4d} "
                f"NO_FLOOR best={describe_row(best)}"
            )
    print(f"\nwinner_counts={dict(winner_counts)}")

    print("\nROTATED P16 LAYOUT AT EQUAL ROW DEPTH")
    layout_wins = Counter()
    layout_pairs = 0
    exact_saves_component_tile = 0
    prefix_wins_when_same_tile_count = 0
    exact_wins_when_saves_tile = 0
    for _, group in groups.items():
        by_depth: dict[int, list[Row]] = defaultdict(list)
        for r in group:
            if r.access.startswith("rotated_"):
                by_depth[r.rows_per_shard].append(r)
        for depth_rows in by_depth.values():
            stable = [r for r in depth_rows if r.spread <= args.unstable_spread]
            by_access = {r.access: r for r in stable}
            if "rotated_prefix_reserved" not in by_access or "rotated_exact" not in by_access:
                continue
            prefix = by_access["rotated_prefix_reserved"]
            exact = by_access["rotated_exact"]
            layout_pairs += 1
            best = prefix if prefix.median_ms <= exact.median_ms else exact
            layout_wins[best.access] += 1
            prefix_tiles = math.ceil(prefix.inner / 497)
            exact_tiles = math.ceil(exact.inner / 512)
            if exact_tiles < prefix_tiles:
                exact_saves_component_tile += 1
                if best is exact:
                    exact_wins_when_saves_tile += 1
            elif best is prefix:
                prefix_wins_when_same_tile_count += 1
    print(f"pairs={layout_pairs} wins={dict(layout_wins)}")
    print(
        f"exact_saves_component_tile={exact_saves_component_tile} "
        f"exact_wins_when_saves_tile={exact_wins_when_saves_tile} "
        f"prefix_wins_when_same_tile_count={prefix_wins_when_same_tile_count}"
    )

    print("\nODD-K COVERAGE (KPARALLEL GAP)")
    odd_groups = {key: group for key, group in groups.items() if group[0].inner % 2 == 1}
    odd_union = 0
    odd_variant_counts = Counter()
    for key, group in odd_groups.items():
        variants = {r.variant for r in group}
        qualified = []
        for variant in variants:
            row = floor_frontier.get((*key, variant))
            if row is not None:
                qualified.append(row)
                odd_variant_counts[variant] += 1
        if qualified:
            odd_union += 1
    print(f"union_floor_coverage={odd_union}/{len(odd_groups)}")
    print(f"variant_floor_coverage={dict(sorted(odd_variant_counts.items()))}")

    def report_flat_vs_rotated(label: str, min_k: int, max_k: int) -> None:
        print(f"\n{label}")
        deltas: list[float] = []
        wins = Counter()
        k_values = set()
        for _, group in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[1][0].inner)):
            inner = group[0].inner
            if inner < min_k or inner > max_k:
                continue
            flat = {r.rows_per_shard: r for r in group if r.access == "flat_rows" and r.spread <= args.unstable_spread}
            rotated_by_depth: dict[int, list[Row]] = defaultdict(list)
            for r in group:
                if r.access.startswith("rotated_") and r.spread <= args.unstable_spread:
                    rotated_by_depth[r.rows_per_shard].append(r)
            for depth in sorted(flat.keys() & rotated_by_depth.keys()):
                f = flat[depth]
                r = min(rotated_by_depth[depth], key=lambda row: row.median_ms)
                delta = (r.median_ms / f.median_ms - 1.0) * 100.0
                deltas.append(delta)
                k_values.add(inner)
                wins["flat_rows" if f.median_ms < r.median_ms else r.variant] += 1
        if deltas:
            print(
                f"k_values={len(k_values)} pairs={len(deltas)} flat_speedup_vs_best_rotated "
                f"median={statistics.median(deltas):.2f}% best={max(deltas):.2f}% worst={min(deltas):.2f}% "
                f"wins={dict(wins)}"
            )
        else:
            print("no stable equal-depth pairs")

    report_flat_vs_rotated("NARROW K<=32: ADAPTIVE P16 FLATROWS VS BEST ROTATED AT EQUAL DEPTH", 1, 32)
    report_flat_vs_rotated("MEDIUM 33<=K<=256: ADAPTIVE P16 FLATROWS VS BEST ROTATED AT EQUAL DEPTH", 33, 256)

    misses = [(key, best_throughput[key]) for key in variant_groups if floor_frontier[key] is None]
    if misses:
        print(f"\nSHAPES/VARIANTS THAT NEVER REACH THE {args.line_rate_gbps / 1000.0:.2f} TB/s FLOOR")
        for key, best in sorted(misses, key=lambda item: (item[0][3], item[1].inner, item[1].dtype)):
            print(
                f"{key[3]:27s} {best.dtype:8s} O={best.outer:<6d} R={best.reduction:<6d} K={best.inner:<4d} "
                f"best_input={best.input_gbps:7.1f}GB/s rows={best.rows_per_shard:<5d} "
                f"cta={best.block_threads:<3d} warps/SM={best.launch_warps_per_sm:5.2f}"
            )

    unstable_rows = [r for r in rows if r.spread > args.unstable_spread]
    print(f"\nmeasurement_spread_gt_{args.unstable_spread * 100:.1f}pct={len(unstable_rows)}/{len(rows)}")
    for row in sorted(unstable_rows, key=lambda r: r.spread, reverse=True)[:20]:
        print(
            f"{row.spread * 100:6.2f}% {row.dtype:8s} K={row.inner:<4d} {row.variant:20s} "
            f"rows={row.rows_per_shard:<5d} best/median/worst="
            f"{row.best_ms:.4f}/{row.median_ms:.4f}/{row.worst_ms:.4f}ms"
        )

    print("\nCALIBRATION STATUS")
    print(
        f"shape_dtypes_with_any_floor_candidate={len(groups) - len(no_floor)}/{len(groups)} "
        f"floor={args.line_rate_gbps:.0f}GB/s peak_reference={args.peak_reference_gbps:.0f}GB/s"
    )
    print("policy remains disabled: optimize/characterize physical kernels first, then derive scheduling boundaries")


if __name__ == "__main__":
    main()
