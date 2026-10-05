#!/usr/bin/env python3
"""Analyze the FP8 RK full-space coverage census.

Production KParallel competes wherever its real selector is legal. Calibration-only RCooperative candidates then show
where they fill a genuine hole, win outright, or create a latency/progress tradeoff that needs end-to-end follow-up.
The report retains both absolute-latency and saturated-bandwidth views rather than inventing a single size threshold.
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
    strategy: str
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
        if self.strategy == "k_parallel":
            return f"production_kparallel_p{self.packet_bytes}"
        if self.access in ("outer_grouped", "experimental_deprecated_outer_grouped"):
            return f"DEPRECATED_EXPERIMENT_outer_grouped_p{self.packet_bytes}"
        if self.access == "multi_output_warp":
            return f"legacy_multi_output_warp_p{self.packet_bytes}"
        if self.access == "multi_output_compact":
            return f"compact_multi_output_p{self.packet_bytes}"
        if self.access == "r_parallel_warp":
            return f"r_parallel_p{self.packet_bytes}"
        if self.access.startswith("r_parallel_multi_output_"):
            outputs_per_warp = self.access.removeprefix("r_parallel_multi_output_")
            return f"r_parallel_multi_o{outputs_per_warp}_p{self.packet_bytes}"
        if self.access.startswith("r_parallel_multi_warp_"):
            warps_per_output = self.access.removeprefix("r_parallel_multi_warp_")
            return f"r_parallel_multi_w{warps_per_output}_p{self.packet_bytes}"
        if self.access.startswith("r_parallel_general_w"):
            warps_per_output = self.access.removeprefix("r_parallel_general_w")
            return f"r_parallel_general_w{warps_per_output}_p{self.packet_bytes}"
        if self.access.startswith("r_parallel_general_o"):
            ownership = self.access.removeprefix("r_parallel_general_")
            return f"r_parallel_general_{ownership}_p{self.packet_bytes}"
        if self.access == "r_parallel_k2_paired_rows":
            return f"k2_paired_rows_p{self.packet_bytes}"
        if self.access == "r_parallel_k3_phase_stream":
            return f"k3_phase_stream_p{self.packet_bytes}"
        if self.access == "r_parallel_k3_phase_stream_p8_input":
            return "k3_phase_stream_p8in"
        if self.access == "r_parallel_k3_phase_stream_runtime_r_p4":
            return "k3_runtime_r_p4"
        if self.access == "r_parallel_k3_phase_stream_runtime_r_p8_input":
            return "k3_runtime_r_p8in"
        if self.access == "r_parallel_k7_phase_stream_p8_input":
            return "k7_phase_stream_p8in"
        if self.access == "r_parallel_k7_phase_stream_p16_input":
            return "k7_phase_stream_p16in"
        if self.access == "direct_component_warp":
            return f"direct_component_p{self.packet_bytes}"
        if self.access == "p4_read_write_roofline":
            return f"p4_rw_roofline_p{self.packet_bytes}"
        if self.access == "flat_rows":
            return f"flat_rows_p{self.packet_bytes}"
        if self.access == "rotated_prefix_reserved":
            return f"rotated_prefix_p{self.packet_bytes}"
        if self.access == "rotated_exact":
            return f"rotated_exact_p{self.packet_bytes}"
        if self.progress == "complete":
            return f"complete_p{self.packet_bytes}"
        return f"{self.access}_p{self.packet_bytes}"

    @property
    def spread(self) -> float:
        """Robust timing spread used for calibration stability.

        The benchmark reports best/median/worst over an odd sample count. A single interrupted slow sample should not
        invalidate an otherwise repeatable geometry, so calibration stability is based on median/best rather than
        worst/best. The raw worst/best spread remains available separately for diagnostics.
        """
        return self.median_ms / self.best_ms - 1.0 if self.best_ms > 0.0 else math.inf

    @property
    def worst_spread(self) -> float:
        return self.worst_ms / self.best_ms - 1.0 if self.best_ms > 0.0 else math.inf

    @property
    def input_gbps(self) -> float:
        return self.input_bytes / (self.median_ms * 1.0e6)

    @property
    def physical_gbps(self) -> float:
        """Physical read+write bandwidth for this pass.

        Deep-R calibration could use input bandwidth because output traffic was negligible. In the full-space
        sweep that is false for shallow R, and staged versus complete candidates can have materially different write
        traffic. Use input+output bytes so the line-rate test remains physically meaningful.
        """
        return (self.input_bytes + self.output_bytes) / (self.median_ms * 1.0e6)

    @property
    def r_consumed(self) -> int:
        return min(self.rows_per_shard, self.reduction)


def read_rows(path: Path) -> tuple[list[Row], dict[str, str]]:
    raw_lines = path.read_text().splitlines()
    metadata: dict[str, str] = {}
    for line in raw_lines:
        if line.startswith("#"):
            for token in line[1:].strip().split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    metadata[key] = value
    lines = [line for line in raw_lines if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: no CSV rows found")
    reader = csv.DictReader(lines)
    required = {
        "family", "case", "outer", "R", "K", "dtype", "strategy", "access", "progress", "implementation",
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
            family=raw["family"], case=raw["case"], outer=int(raw["outer"]), reduction=int(raw["R"]),
            inner=int(raw["K"]), dtype=raw["dtype"], strategy=raw["strategy"], access=raw["access"], progress=raw["progress"],
            implementation=raw["implementation"], packet_bytes=int(raw["packet_bytes"]),
            block_threads=int(raw["block_threads"]), rows_per_shard=int(raw["rows_per_shard"]),
            shards_per_output=int(raw["shards_per_output"]), first_stage_blocks=int(raw["first_stage_blocks"]),
            launch_warps_per_sm=float(raw["launch_warps_per_sm"]), occupancy_waves=float(raw["occupancy_waves"]),
            input_bytes=int(raw["input_bytes"]), output_bytes=int(raw["output_bytes"]),
            median_ms=float(raw["median_ms"]), best_ms=float(raw["best_ms"]), worst_ms=float(raw["worst_ms"]),
            logical_gbps=float(raw["logical_GBps"]),
        ))
    return rows, metadata


def fmt_bytes(value: int) -> str:
    if value >= 1024 ** 3:
        return f"{value / 1024**3:.2f}GiB"
    if value >= 1024 ** 2:
        return f"{value / 1024**2:.2f}MiB"
    return f"{value / 1024:.1f}KiB"


def fastest(rows: list[Row]) -> Row:
    return min(rows, key=lambda r: (r.median_ms, r.spread))


SMALL_R_NICE = {2, 4, 8, 16, 32, 64}
SMALL_R_AWKWARD = {3, 7, 15, 31, 63}
MEDIUM_R_NICE = {256, 512, 1024}
MEDIUM_R_AWKWARD = {255, 511, 1023}
LARGE_R_NICE = {4096, 8192, 32768}
LARGE_R_AWKWARD = {4095, 8191, 32767}

SMALL_K_NICE = {2, 4, 8, 16, 32}
SMALL_K_AWKWARD = {3, 5, 7, 9, 11, 13, 15, 17, 19, 31}
MEDIUM_K_NICE = {64, 128, 256}
MEDIUM_K_AWKWARD = {63, 127, 129, 255}
LARGE_K_NICE = {512, 1024, 4096}
LARGE_K_AWKWARD = {497, 511, 993, 1023, 1493, 4095}


def r_regime(value: int) -> str:
    if value in SMALL_R_NICE | SMALL_R_AWKWARD:
        return "smallR"
    if value in MEDIUM_R_NICE | MEDIUM_R_AWKWARD:
        return "mediumR"
    if value in LARGE_R_NICE | LARGE_R_AWKWARD:
        return "largeR"
    return "otherR"


def k_regime(value: int) -> str:
    if value in SMALL_K_NICE | SMALL_K_AWKWARD:
        return "smallK"
    if value in MEDIUM_K_NICE | MEDIUM_K_AWKWARD:
        return "mediumK"
    if value in LARGE_K_NICE | LARGE_K_AWKWARD:
        return "largeK"
    return "otherK"


def r_shape(value: int) -> str:
    if value in SMALL_R_NICE | MEDIUM_R_NICE | LARGE_R_NICE:
        return "niceR"
    if value in SMALL_R_AWKWARD | MEDIUM_R_AWKWARD | LARGE_R_AWKWARD:
        return "awkwardR"
    return "otherR"


def k_shape(value: int) -> str:
    if value in SMALL_K_NICE | MEDIUM_K_NICE | LARGE_K_NICE:
        return "niceK"
    if value in SMALL_K_AWKWARD | MEDIUM_K_AWKWARD | LARGE_K_AWKWARD:
        return "awkwardK"
    return "otherK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--line-rate-gbps", type=float, default=1500.0)
    parser.add_argument("--peak-reference-gbps", type=float, default=1600.0)
    parser.add_argument(
        "--unstable-spread", type=float, default=0.05,
        help="maximum median/best timing spread for a stable measurement; worst/best is reported separately",
    )
    parser.add_argument("--meaningful-small-packet-win", type=float, default=0.03,
                        help="minimum latency win over the best p16 RCooperative candidate to count as a meaningful p4/p8 niche")
    parser.add_argument("--coverage-tie", type=float, default=0.03,
                        help="latency tolerance used when deciding whether production KParallel already covers a cell")
    args = parser.parse_args()

    rows, metadata = read_rows(args.csv)
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.key].append(row)

    duplicates = Counter((r.key, r.variant, r.rows_per_shard, r.block_threads) for r in rows)
    duplicate_count = sum(count - 1 for count in duplicates.values() if count > 1)
    if duplicate_count:
        raise SystemExit(f"duplicate physical rows={duplicate_count}")

    stable_all_by_key: dict[tuple[str, str, str], list[Row]] = {
        key: [r for r in group if r.spread <= args.unstable_spread]
        for key, group in groups.items()
    }
    # The old one-output-per-warp outer-grouped kernel is deliberately retained only as an experimental baseline.
    # The raw p4 read/write row is a physical roofline reference rather than a reduction candidate. Exclude both from
    # every coverage, winner, line-rate, packet-policy, and promotion-oriented decision.
    def promotion_candidate(row: Row) -> bool:
        return (
            row.strategy != "reference"
            and row.access not in ("outer_grouped", "experimental_deprecated_outer_grouped")
        )

    eligible_by_key: dict[tuple[str, str, str], list[Row]] = {
        key: [r for r in group if promotion_candidate(r)]
        for key, group in groups.items()
    }
    stable_by_key: dict[tuple[str, str, str], list[Row]] = {
        key: [r for r in stable_all_by_key[key] if promotion_candidate(r)]
        for key in groups
    }
    no_stable = [key for key, stable in stable_by_key.items() if not stable]
    abs_winner: dict[tuple[str, str, str], Row] = {
        key: fastest(stable if stable else eligible_by_key[key])
        for key, stable in stable_by_key.items()
        if stable or eligible_by_key[key]
    }

    print(
        f"rows={len(rows)} exact_shape_dtypes={len(groups)} timing_samples={metadata.get('timing_samples', 'unknown')} "
        f"stable_shape_dtypes={len(groups)-len(no_stable)}/{len(groups)} "
        f"physical_line_rate_floor={args.line_rate_gbps:.1f}GB/s physical_peak_reference={args.peak_reference_gbps:.1f}GB/s "
        f"unstable_median_over_best>{args.unstable_spread*100:.1f}%"
    )

    focused_holes_only = metadata.get("focused_holes_only") == "1"

    # Classify whether production KParallel already solves an exact geometry. Keep legality separate from
    # measurement stability: a production KParallel candidate that exists but is noisy is not the same thing as an
    # unsupported/unavailable KParallel geometry. We only call KParallel covered when it is stable, within the
    # latency tie threshold of the best stable RCooperative pass, and makes at least as much reduction progress.
    # Crossed latency/progress cases are intentionally left for end-to-end continuation calibration.
    coverage_status: dict[tuple[str, str, str], str] = {}
    line_rate_status: dict[tuple[str, str, str], str] = {}
    coverage_detail: dict[tuple[str, str, str], tuple[Row | None, Row | None]] = {}
    for key, stable in stable_by_key.items():
        group = eligible_by_key[key]
        kp_all = [r for r in group if r.strategy == "k_parallel"]
        rcoop_all = [r for r in group if r.strategy == "r_cooperative"]
        kp = [r for r in stable if r.strategy == "k_parallel"]
        rcoop = [r for r in stable if r.strategy == "r_cooperative"]
        best_kp = fastest(kp) if kp else None
        best_rcoop = fastest(rcoop) if rcoop else None
        coverage_detail[key] = (best_kp, best_rcoop)

        if focused_holes_only:
            coverage_status[key] = "RCOOP_FOCUSED" if best_rcoop is not None else "NO_STABLE_RCOOP"
            line_rate_status[key] = (
                "RCOOP_LINE_RATE" if any(r.physical_gbps >= args.line_rate_gbps for r in rcoop) else "NO_LINE_RATE"
            )
            continue

        kp_legal = bool(kp_all)
        if best_kp is None and best_rcoop is None:
            coverage_status[key] = (
                "KPARALLEL_UNSTABLE_NO_STABLE_RCOOP" if kp_legal
                else "KPARALLEL_UNAVAILABLE_NO_STABLE_RCOOP"
            )
        elif best_kp is None:
            coverage_status[key] = (
                "KPARALLEL_UNSTABLE_RCOOP_COVERED" if kp_legal
                else "KPARALLEL_UNAVAILABLE_RCOOP_COVERED"
            )
        elif best_rcoop is None:
            coverage_status[key] = "COVERED_BY_KPARALLEL"
        else:
            kp_latency_tied = best_kp.median_ms <= best_rcoop.median_ms * (1.0 + args.coverage_tie)
            kp_progress_at_least = best_kp.output_bytes <= best_rcoop.output_bytes
            rcoop_meaningfully_faster = best_rcoop.median_ms < best_kp.median_ms / (1.0 + args.coverage_tie)
            rcoop_progress_at_least = best_rcoop.output_bytes <= best_kp.output_bytes
            if kp_latency_tied and kp_progress_at_least:
                coverage_status[key] = "COVERED_BY_KPARALLEL"
            elif rcoop_meaningfully_faster and rcoop_progress_at_least:
                coverage_status[key] = "RCOOPERATIVE_WIN"
            else:
                coverage_status[key] = "PROGRESS_LATENCY_TRADEOFF"

        kp_floor = any(r.physical_gbps >= args.line_rate_gbps for r in kp)
        rcoop_floor = any(r.physical_gbps >= args.line_rate_gbps for r in rcoop)
        if kp_floor:
            line_rate_status[key] = "KPARALLEL_LINE_RATE"
        elif rcoop_floor:
            line_rate_status[key] = "RCOOP_UNIQUE_LINE_RATE"
        else:
            line_rate_status[key] = "NO_LINE_RATE"

    print("\nCOVERAGE MAP BY ACTUAL-WORK TARGET x R-REGIME x K-REGIME")
    coverage_cells: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    actual_sizes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for key, status in coverage_status.items():
        sample = groups[key][0]
        cell = (sample.family, r_regime(sample.reduction), k_regime(sample.inner))
        coverage_cells[cell][status] += 1
        actual_sizes[cell].append(sample.input_bytes)
    for cell in sorted(coverage_cells):
        family, rr, kr = cell
        med = int(statistics.median(actual_sizes[cell]))
        print(f"{family:14s} {rr:7s} x {kr:7s} actual_input_med={fmt_bytes(med):>9s} statuses={dict(coverage_cells[cell])}")

    print("\nNICE/AWKWARD COVERAGE WITHIN EACH R x K CELL")
    shape_cells: dict[tuple[str, str, str, str, str], Counter] = defaultdict(Counter)
    for key, status in coverage_status.items():
        sample = groups[key][0]
        cell = (sample.family, r_regime(sample.reduction), k_regime(sample.inner), r_shape(sample.reduction), k_shape(sample.inner))
        shape_cells[cell][status] += 1
    for cell in sorted(shape_cells):
        family, rr, kr, rs, ks = cell
        print(f"{family:14s} {rr:7s} x {kr:7s} {rs:8s}/{ks:8s} statuses={dict(shape_cells[cell])}")

    print("\nPHYSICAL LINE-RATE COVERAGE SOURCE BY R x K CELL")
    floor_cells: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for key, status in line_rate_status.items():
        sample = groups[key][0]
        floor_cells[(sample.family, r_regime(sample.reduction), k_regime(sample.inner))][status] += 1
    for cell in sorted(floor_cells):
        family, rr, kr = cell
        print(f"{family:14s} {rr:7s} x {kr:7s} statuses={dict(floor_cells[cell])}")

    print("\nFOCUSED RCOOPERATIVE CELLS" if focused_holes_only else
          "\nEXACT CELLS THAT ARE NOT ALREADY CLEANLY COVERED BY PRODUCTION KPARALLEL")
    interesting = {
        "RCOOPERATIVE_WIN",
        "PROGRESS_LATENCY_TRADEOFF",
        "KPARALLEL_UNAVAILABLE_RCOOP_COVERED",
        "KPARALLEL_UNSTABLE_RCOOP_COVERED",
        "KPARALLEL_UNAVAILABLE_NO_STABLE_RCOOP",
        "KPARALLEL_UNSTABLE_NO_STABLE_RCOOP",
        "RCOOP_FOCUSED",
        "NO_STABLE_RCOOP",
    }
    interesting_rows = []
    for key, status in coverage_status.items():
        if status not in interesting:
            continue
        sample = groups[key][0]
        kp, rc = coverage_detail[key]
        interesting_rows.append((sample.family, sample.reduction, sample.inner, sample.dtype, status, kp, rc))
    for family, reduction, inner, dtype, status, kp, rc in sorted(interesting_rows)[:400]:
        kp_text = "none" if kp is None else f"{kp.variant}:{kp.median_ms*1000:.2f}us->{fmt_bytes(kp.output_bytes)}"
        rc_text = "none" if rc is None else f"{rc.variant}:{rc.median_ms*1000:.2f}us->{fmt_bytes(rc.output_bytes)}"
        print(f"{family:14s} R={reduction:<6d} K={inner:<4d} {dtype:8s} {status:38s} KP={kp_text} RCO={rc_text}")
    if len(interesting_rows) > 400:
        print(f"... {len(interesting_rows)-400} additional non-KParallel-covered cells omitted from detail; aggregate maps above remain complete")

    print("\nDEPRECATED EXPERIMENT BASELINE (EXCLUDED FROM ALL PROMOTION/COVERAGE DECISIONS)")
    deprecated_rows = [
        row for stable in stable_all_by_key.values() for row in stable
        if row.access in ("outer_grouped", "experimental_deprecated_outer_grouped")
    ]
    replacement_rows = [
        row for stable in stable_all_by_key.values() for row in stable
        if row.access in ("multi_output_compact", "r_parallel_warp", "direct_component_warp")
        or row.access.startswith("r_parallel_general_w")
        or row.access.startswith("r_parallel_general_o")
    ]
    replacement_by_key: dict[tuple[str, str, str], Row] = {}
    for row in replacement_rows:
        current = replacement_by_key.get(row.key)
        if current is None or row.median_ms < current.median_ms:
            replacement_by_key[row.key] = row
    paired = [(row, replacement_by_key[row.key]) for row in deprecated_rows if row.key in replacement_by_key]
    if paired:
        replacement_wins = sum(new.median_ms < old.median_ms for old, new in paired)
        deprecated_wins = sum(old.median_ms < new.median_ms for old, new in paired)
        ties = len(paired) - replacement_wins - deprecated_wins
        ratios = [old.median_ms / new.median_ms for old, new in paired]
        print(
            f"deprecated_rows={len(deprecated_rows)} paired_with_replacement={len(paired)} "
            f"replacement_faster={replacement_wins} deprecated_faster={deprecated_wins} ties={ties} "
            f"deprecated_over_best_replacement_median={statistics.median(ratios):.4f}x"
        )
    else:
        print(f"deprecated_rows={len(deprecated_rows)} paired_with_replacement=0")

    print("\nP4 READ+WRITE ROOFLINE REFERENCE (EXCLUDED FROM PROMOTION/COVERAGE)")
    roofline_by_key: dict[tuple[str, str, str], Row] = {}
    for stable in stable_all_by_key.values():
        refs = [row for row in stable if row.strategy == "reference" and row.access == "p4_read_write_roofline"]
        if refs:
            roofline_by_key[refs[0].key] = fastest(refs)
    if roofline_by_key:
        roofline_gbps = [row.physical_gbps for row in roofline_by_key.values()]
        efficiencies = []
        by_r: dict[int, list[float]] = defaultdict(list)
        by_k: dict[int, list[float]] = defaultdict(list)
        for key, roofline in roofline_by_key.items():
            candidates = stable_by_key.get(key, [])
            if not candidates:
                continue
            best = fastest(candidates)
            if roofline.physical_gbps > 0.0:
                efficiency = best.physical_gbps / roofline.physical_gbps
                efficiencies.append(efficiency)
                by_r[best.reduction].append(efficiency)
                by_k[best.inner].append(efficiency)
        print(
            f"roofline_rows={len(roofline_by_key)} physical_GBps_median={statistics.median(roofline_gbps):.1f} "
            f"p10={statistics.quantiles(roofline_gbps, n=10)[0]:.1f} p90={statistics.quantiles(roofline_gbps, n=10)[8]:.1f} "
            f"best_reduction_over_roofline_median={statistics.median(efficiencies):.4f}x"
        )
        print("efficiency_by_R=" + ", ".join(
            f"R{r}:{statistics.median(values):.3f}x" for r, values in sorted(by_r.items())
        ))
        print("efficiency_by_K=" + ", ".join(
            f"K{k}:{statistics.median(values):.3f}x" for k, values in sorted(by_k.items())
        ))
    else:
        print("roofline_rows=0")

    print("\nABSOLUTE-TIME WINNERS BY TARGET-SIZE / R REGIME")
    regimes: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for key, winner in abs_winner.items():
        regimes[(key[0], winner.reduction)].append(winner)
    for (family, reduction), winners in sorted(regimes.items()):
        counts = Counter(r.variant for r in winners)
        median_input = statistics.median(r.input_bytes for r in winners)
        median_us = statistics.median(r.median_ms for r in winners) * 1000.0
        print(
            f"{family:14s} R={reduction:<6d} shapes={len(winners):3d} "
            f"actual_input_med={fmt_bytes(int(median_input)):>9s} latency_med={median_us:8.2f}us winners={dict(counts)}"
        )

    print("\nABSOLUTE-TIME WINNERS BY K")
    by_k: dict[int, list[Row]] = defaultdict(list)
    for winner in abs_winner.values():
        by_k[winner.inner].append(winner)
    for inner, winners in sorted(by_k.items()):
        counts = Counter(r.variant for r in winners)
        print(f"K={inner:<4d} shapes={len(winners):3d} winners={dict(counts)}")

    print("\nPACKET-WIDTH WINNERS WITHIN EACH PHYSICAL ACCESS")
    access_shape_rows: dict[tuple[tuple[str, str, str], str], list[Row]] = defaultdict(list)
    for key, stable in stable_by_key.items():
        for row in stable:
            if row.strategy == "r_cooperative":
                access_shape_rows[(key, row.access)].append(row)
    access_packet_wins: dict[str, Counter] = defaultdict(Counter)
    for (_, access), candidates in access_shape_rows.items():
        best = fastest(candidates)
        access_packet_wins[access][best.packet_bytes] += 1
    for access, counts in sorted(access_packet_wins.items()):
        print(f"{access:27s} {dict(sorted(counts.items()))}")

    print("\nP4/P8 COVERAGE UNIQUE TO SMALL PACKETS")
    if focused_holes_only:
        print("skipped_in_focused_holes_only_mode: p16/FlatRows/Rotated baselines were intentionally not run")
    else:
        small_only_stable: list[Row] = []
        small_only_floor: list[Row] = []
        for key, stable in stable_by_key.items():
            small = [r for r in stable if r.strategy == "r_cooperative" and r.packet_bytes in (4, 8)]
            p16 = [r for r in stable if r.strategy == "r_cooperative" and r.packet_bytes == 16]
            if small and not p16:
                small_only_stable.append(fastest(small))
            small_floor = [r for r in small if r.physical_gbps >= args.line_rate_gbps]
            p16_floor = [r for r in p16 if r.physical_gbps >= args.line_rate_gbps]
            if small_floor and not p16_floor:
                small_only_floor.append(max(small_floor, key=lambda r: (r.r_consumed, -r.median_ms)))
        print(f"small_packet_only_stable={len(small_only_stable)} small_packet_unique_floor={len(small_only_floor)}")
        for row in small_only_floor[:30]:
            print(
                f"UNIQUE_FLOOR {row.family:14s} {row.dtype:8s} O={row.outer:<6d} R={row.reduction:<6d} K={row.inner:<4d} "
                f"input={fmt_bytes(row.input_bytes):>9s} {row.variant:20s} rows={row.rows_per_shard:<5d} "
                f"physical_GBps={row.physical_gbps:7.1f} input_GBps={row.input_gbps:7.1f}"
            )

    print("\nMEANINGFUL P4/P8 ABSOLUTE-LATENCY NICHES VS BEST P16")
    if focused_holes_only:
        print("skipped_in_focused_holes_only_mode: p16 baseline was intentionally not run")
    else:
        small_packet_wins: list[tuple[float, Row, Row]] = []
        for key, stable in stable_by_key.items():
            p16 = [r for r in stable if r.strategy == "r_cooperative" and r.packet_bytes == 16]
            small = [r for r in stable if r.strategy == "r_cooperative" and r.packet_bytes in (4, 8)]
            if not p16 or not small:
                continue
            best16 = fastest(p16)
            bestsmall = fastest(small)
            speedup = best16.median_ms / bestsmall.median_ms - 1.0
            if speedup >= args.meaningful_small_packet_win:
                small_packet_wins.append((speedup, bestsmall, best16))
        print(
            f"meaningful_small_packet_wins={len(small_packet_wins)}/{len(groups)} "
            f"threshold={args.meaningful_small_packet_win*100:.1f}%"
        )
        for speedup, small, p16 in sorted(small_packet_wins, reverse=True, key=lambda x: x[0])[:40]:
            print(
                f"{speedup*100:6.2f}% {small.family:14s} {small.dtype:8s} O={small.outer:<6d} R={small.reduction:<6d} "
                f"K={small.inner:<4d} input={fmt_bytes(small.input_bytes):>9s} winner={small.variant:20s} "
                f"rows={small.rows_per_shard:<5d} {small.median_ms*1000:7.2f}us vs_p16={p16.variant:20s} "
                f"rows={p16.rows_per_shard:<5d} {p16.median_ms*1000:7.2f}us"
            )

    print("\nPHYSICAL LINE-RATE REACHABILITY BY TARGET-SIZE / R REGIME")
    for (family, reduction), winners in sorted(regimes.items()):
        keys = [key for key in groups if key[0] == family and groups[key][0].reduction == reduction]
        reached = 0
        peak = 0
        for key in keys:
            stable = stable_by_key[key]
            reached += any(r.physical_gbps >= args.line_rate_gbps for r in stable)
            peak += any(r.physical_gbps >= args.peak_reference_gbps for r in stable)
        print(f"{family:14s} R={reduction:<6d} floor={reached:3d}/{len(keys):3d} peak={peak:3d}/{len(keys):3d}")

    print("\nDEEPEST STABLE >= PHYSICAL FLOOR WINNERS (SATURATED VIEW)")
    saturated_winners: list[Row] = []
    saturated_counts = Counter()
    for key, stable in stable_by_key.items():
        qualifying = [r for r in stable if r.physical_gbps >= args.line_rate_gbps]
        if not qualifying:
            continue
        chosen = max(qualifying, key=lambda r: (r.r_consumed, -r.median_ms))
        saturated_winners.append(chosen)
        saturated_counts[chosen.variant] += 1
    print(f"shape_dtypes_with_floor={len(saturated_winners)}/{len(groups)} winners={dict(saturated_counts)}")

    print("\nFIRST OBSERVED PHYSICAL-LINE-RATE INPUT SIZE BY R/K/DTYPE")
    rk_dtype: dict[tuple[int, int, str], list[tuple[int, bool]]] = defaultdict(list)
    for key, group in groups.items():
        sample = group[0]
        stable = stable_by_key[key]
        rk_dtype[(sample.reduction, sample.inner, sample.dtype)].append(
            (sample.input_bytes, any(r.physical_gbps >= args.line_rate_gbps for r in stable)))
    transitions = Counter()
    for (reduction, inner, dtype), values in sorted(rk_dtype.items()):
        reached_sizes = sorted(size for size, reached in values if reached)
        if reached_sizes:
            first = reached_sizes[0]
            transitions[fmt_bytes(first)] += 1
            print(f"R={reduction:<6d} K={inner:<4d} {dtype:8s} first_floor_input={fmt_bytes(first)}")
    print(f"first_floor_input_histogram={dict(transitions)}")

    unstable = [r for r in rows if r.spread > args.unstable_spread]
    raw_outliers = [r for r in rows if r.worst_spread > args.unstable_spread]
    print(
        f"\nmeasurement_median_over_best_gt_{args.unstable_spread*100:.1f}pct={len(unstable)}/{len(rows)} "
        f"measurement_worst_over_best_gt_{args.unstable_spread*100:.1f}pct={len(raw_outliers)}/{len(rows)}"
    )
    for row in sorted(unstable, key=lambda r: r.spread, reverse=True)[:20]:
        print(
            f"robust={row.spread*100:6.2f}% raw={row.worst_spread*100:7.2f}% "
            f"{row.family:14s} {row.dtype:8s} R={row.reduction:<6d} K={row.inner:<4d} "
            f"{row.variant:20s} rows={row.rows_per_shard:<5d} "
            f"best/median/worst={row.best_ms:.4f}/{row.median_ms:.4f}/{row.worst_ms:.4f}ms"
        )

    print("\nCALIBRATION STATUS")
    if focused_holes_only:
        print("focused_holes_only=1: already-covered production KParallel, generic packet-adaptive p16, deprecated, FlatRows, and Rotated kernels were intentionally not run; K=7 phase-stream p16 is an explicit calibration candidate")
        print("production policy remains unchanged; this sweep compares only unresolved RCooperative mechanisms")
    else:
        print("production policy remains unchanged; production KParallel is measured as the existing-solution baseline")
        print("KPARALLEL_UNAVAILABLE means no production KParallel candidate exists; KPARALLEL_UNSTABLE means it exists but failed the robust timing gate")
    print("line-rate decisions use physical input+output traffic; stability uses median/best while raw worst/best outliers remain diagnostic")
    print("p4 read+write roofline rows are physical references only and are excluded from every promotion/coverage decision")


if __name__ == "__main__":
    main()
