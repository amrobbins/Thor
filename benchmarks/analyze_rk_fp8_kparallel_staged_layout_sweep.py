#!/usr/bin/env python3
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PACKETS = (16, 8, 4, 2)
CTA_WIDTHS = (32, 64, 128, 256)
DTYPES = ("fp8_e4m3", "fp8_e5m2")
TARGET_GBPS = 1600.0
NEAR_TARGET_GBPS = 1550.0
POLICY_MISS_RATIO = 1.03
TIMING_INSTABILITY_RATIO = 1.03

SHAPES = (
    ("clean_512mib", 32, 32768, 512),
    ("clean_512mib", 16, 32768, 1024),
    ("clean_512mib", 8, 32768, 2048),
    ("clean_512mib", 4, 32768, 4096),
    ("clean_512mib", 2, 32768, 8192),
    ("clean_512mib", 1, 32768, 16384),
    ("depth_512mib", 16, 16384, 2048),
    ("depth_512mib", 4, 65536, 2048),
    ("depth_512mib", 2, 131072, 2048),
    ("awkward_512mib", 8, 32771, 2032),
    ("awkward_512mib", 2, 32771, 8176),
    ("awkward_512mib", 4, 65521, 2032),
    ("awkward_512mib", 1, 65521, 8176),
    ("clean_1gib", 16, 32768, 2048),
    ("clean_1gib", 4, 32768, 8192),
)


def percentile(values, q):
    vals = sorted(values)
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))]



def read_metadata(path):
    metadata = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            for key in ("sm_count", "warp_size", "max_threads_per_sm", "timing_samples"):
                match = re.search(rf"\b{key}=(\d+)", line)
                if match:
                    metadata[key] = int(match.group(1))
    return metadata


def ceil_div(numerator, denominator):
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator + int(numerator % denominator != 0)


def next_power_of_two(value):
    result = 1
    while result < value:
        result <<= 1
    return result


def production_saturation_warps_per_sm(packet, hardware_warps_per_sm):
    base = ceil_div(hardware_warps_per_sm, 4)
    multiplier = 1 if packet >= 8 else 8 // packet
    return base * multiplier


def production_saturation_target(packet, sm_count, hardware_warps_per_sm):
    return sm_count * production_saturation_warps_per_sm(packet, hardware_warps_per_sm)


def production_layout_can_claim_saturation(row):
    return row["packet_bytes"] != 2 or row["block_threads"] >= 128


def production_layout_is_saturated(row, sm_count, hardware_warps_per_sm):
    return (
        production_layout_can_claim_saturation(row)
        and row["launch_warp_supply"]
        >= production_saturation_target(row["packet_bytes"], sm_count, hardware_warps_per_sm)
    )


def timing_instability_ratio(row):
    if row["best_ms"] <= 0 or row["median_ms"] <= 0:
        return float("inf")
    return max(row["median_ms"] / row["best_ms"], row["worst_ms"] / row["median_ms"])


def timing_is_stable(row):
    return timing_instability_ratio(row) <= TIMING_INSTABILITY_RATIO


def fastest_stable_layout(group):
    stable = [row for row in group if timing_is_stable(row)]
    if not stable:
        return None
    return min(stable, key=lambda row: (row["median_ms"], -row["logical_GBps"], row["output_bytes"]))


def production_row_depth_distance(row):
    return min(abs(row["rows_per_shard"] - depth) for depth in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16))


def production_shard_slack(row):
    return row["rows_per_shard"] * row["shards_per_output"] - row["R"]


def production_preferred_wide_packet_cta(row, warp_size):
    owner_warps = ceil_div(row["packet_owners"], warp_size)
    preferred_warps = min(4, max(2, next_power_of_two(owner_warps)))
    return preferred_warps * warp_size


def production_saturated_layout_key(row, warp_size):
    if row["packet_bytes"] >= 8:
        preferred_cta = production_preferred_wide_packet_cta(row, warp_size)
        first = abs(row["block_threads"] - preferred_cta)
    else:
        # Narrow packet4/2 paths retain the production resident-capacity discriminator.
        first = -row["useful_resident_warps"]
    return (
        first,
        production_row_depth_distance(row),
        production_shard_slack(row),
        -row["lane_fill"],
        row["output_bytes"],
        row["block_threads"],
    )


def choose_production_saturated_layout(group, sm_count, hardware_warps_per_sm, warp_size):
    saturated = [
        row for row in group
        if production_layout_is_saturated(row, sm_count, hardware_warps_per_sm)
    ]
    if not saturated:
        return None
    minimum_saturated_shards = min(row["shards_per_output"] for row in saturated)
    envelope = [
        row for row in saturated
        if row["shards_per_output"] <= minimum_saturated_shards + 1
    ]
    return min(envelope, key=lambda row: production_saturated_layout_key(row, warp_size))


def print_production_policy_replay(rows, expected_shapes, by_packet, sm_count, hardware_warps_per_sm, warp_size):
    print("\nPRODUCTION SATURATED LAYOUT REPLAY")
    print(
        f"hardware_warps_per_sm={hardware_warps_per_sm} "
        f"targets=p16:{production_saturation_target(16, sm_count, hardware_warps_per_sm)} "
        f"p8:{production_saturation_target(8, sm_count, hardware_warps_per_sm)} "
        f"p4:{production_saturation_target(4, sm_count, hardware_warps_per_sm)} "
        f"p2:{production_saturation_target(2, sm_count, hardware_warps_per_sm)}"
    )

    selections = {}
    references = {}
    for packet in PACKETS:
        packet_selected = []
        packet_references = []
        unstable_selected = 0
        no_stable_reference = 0
        for shape in sorted(expected_shapes):
            group = by_packet[shape + (packet,)]
            selected = choose_production_saturated_layout(
                group, sm_count, hardware_warps_per_sm, warp_size
            )
            if selected is None:
                continue
            reference = fastest_stable_layout(group)
            if reference is None:
                no_stable_reference += 1
                continue
            selections[shape + (packet,)] = selected
            references[shape + (packet,)] = reference
            packet_selected.append(selected)
            packet_references.append(reference)
            unstable_selected += int(not timing_is_stable(selected))
        if not packet_selected:
            print(f"packet{packet:2d}: no saturated production layouts with a stable measured reference")
            continue
        ratios = [
            selected["median_ms"] / reference["median_ms"]
            for selected, reference in zip(packet_selected, packet_references)
        ]
        stable_miss_ratios = [
            ratio
            for selected, ratio in zip(packet_selected, ratios)
            if timing_is_stable(selected) and ratio > POLICY_MISS_RATIO
        ]
        print(
            f"packet{packet:2d}: selected={len(packet_selected):2d} "
            f">=1600={sum(row['logical_GBps'] >= TARGET_GBPS for row in packet_selected):2d}/{len(packet_selected):2d} "
            f"logical_median={statistics.median(row['logical_GBps'] for row in packet_selected):7.1f} "
            f"logical_min={min(row['logical_GBps'] for row in packet_selected):7.1f} "
            f"selected/stable-best_time median={statistics.median(ratios):.4f} worst={max(ratios):.4f} "
            f"stable_>3%={len(stable_miss_ratios)} unstable_selected={unstable_selected} "
            f"no_stable_reference={no_stable_reference}"
        )

    packet = 16
    stable_misses = []
    unstable_misses = []
    for shape in sorted(expected_shapes):
        selected = selections.get(shape + (packet,))
        reference = references.get(shape + (packet,))
        if selected is None or reference is None:
            continue
        ratio = selected["median_ms"] / reference["median_ms"]
        if ratio <= POLICY_MISS_RATIO:
            continue
        item = (ratio, shape, selected, reference)
        if timing_is_stable(selected) and timing_is_stable(reference):
            stable_misses.append(item)
        else:
            unstable_misses.append(item)

    print("\nPACKET16 MATERIAL PRODUCTION MISSES (>3% slower than fastest stable measured layout)")
    if not stable_misses:
        print("stable_misses=none")
    else:
        for ratio, shape, selected, reference in sorted(stable_misses, reverse=True):
            family, dtype, outer, reduction, inner = shape
            print(
                f"STABLE {family:15s} {dtype:10s} O={outer:<3d} R={reduction:<6d} K={inner:<6d} "
                f"selected=CTA{selected['block_threads']}/s{selected['shards_per_output']}/r{selected['rows_per_shard']} "
                f"{selected['logical_GBps']:.1f}GB/s reference=CTA{reference['block_threads']}/"
                f"s{reference['shards_per_output']}/r{reference['rows_per_shard']} "
                f"{reference['logical_GBps']:.1f}GB/s time_ratio={ratio:.4f}"
            )
    if unstable_misses:
        print("unstable_measurements_require_rerun:")
        for ratio, shape, selected, reference in sorted(unstable_misses, reverse=True):
            family, dtype, outer, reduction, inner = shape
            print(
                f"UNSTABLE {family:15s} {dtype:10s} O={outer:<3d} R={reduction:<6d} K={inner:<6d} "
                f"selected=CTA{selected['block_threads']}/s{selected['shards_per_output']}/r{selected['rows_per_shard']} "
                f"time_ratio={ratio:.4f} selected_instability={timing_instability_ratio(selected):.4f} "
                f"reference_instability={timing_instability_ratio(reference):.4f}"
            )

    packet16_selected = [row for key, row in selections.items() if key[-1] == packet]
    unstable_packet16 = sum(not timing_is_stable(row) for row in packet16_selected)
    if stable_misses:
        policy_gate = "FAIL"
    elif unstable_misses:
        policy_gate = "INCONCLUSIVE"
    else:
        policy_gate = "PASS"
    print(
        f"packet16_policy_gate={policy_gate} "
        f"stable_material_misses={len(stable_misses)} "
        f"unstable_material_misses={len(unstable_misses)} "
        f"unstable_selected_measurements={unstable_packet16}"
    )
    return policy_gate, unstable_packet16


def first_bandwidth_frontier(rows, packet, target_gbps, sm_count, families=None):
    selected = []
    groups = defaultdict(list)
    for row in rows:
        if row["packet_bytes"] != packet:
            continue
        if families is not None and row["family"] not in families:
            continue
        key = (row["family"], row["dtype"], row["outer"], row["R"], row["K"])
        groups[key].append(row)

    for key, group in groups.items():
        line_rate = [row for row in group if row["logical_GBps"] >= target_gbps]
        if not line_rate:
            continue
        # This is a capability frontier, not a policy simulation: find the least useful concurrent warp supply among
        # the measured layouts that actually reaches the requested bandwidth, then prefer the faster layout on ties.
        selected.append(min(
            line_rate,
            key=lambda row: (row["launch_warp_supply"], row["physical_block_waves"], row["median_ms"]),
        ))

    total_groups = len(groups)
    if not selected:
        return total_groups, []

    frontier = []
    for row in selected:
        useful_per_sm = row["launch_warp_supply"] / sm_count
        frontier.append({
            "row": row,
            "useful_per_sm": useful_per_sm,
            "equivalent_16b_per_sm": useful_per_sm * row["packet_bytes"] / 16.0,
            "physical_waves": row["physical_block_waves"],
            "resident_fraction": (
                row["launch_warp_supply"] / row["resident_device_warps"]
                if row["resident_device_warps"] else float("nan")
            ),
        })
    return total_groups, frontier


def print_frontier(rows, target_gbps, sm_count, families=None, title=None):
    if title is None:
        title = f"MEASURED SATURATION FRONTIER >= {target_gbps:.0f} GB/s"
    print(f"\n{title}")
    print(
        "packet reachable useful_warps/SM[min med p95 max] "
        "16B_equiv/SM[med p95] physical_waves[med p95] resident_fraction[med p95]"
    )
    for packet in PACKETS:
        total, frontier = first_bandwidth_frontier(rows, packet, target_gbps, sm_count, families)
        if not frontier:
            print(f"packet{packet:2d}: 0/{total} (target not reached)")
            continue
        useful = [item["useful_per_sm"] for item in frontier]
        equivalent = [item["equivalent_16b_per_sm"] for item in frontier]
        waves = [item["physical_waves"] for item in frontier]
        resident = [item["resident_fraction"] for item in frontier]
        print(
            f"packet{packet:2d}: {len(frontier):2d}/{total:<2d} "
            f"useful={min(useful):5.2f}/{statistics.median(useful):5.2f}/{percentile(useful,0.95):5.2f}/{max(useful):5.2f} "
            f"eq16={statistics.median(equivalent):5.2f}/{percentile(equivalent,0.95):5.2f} "
            f"waves={statistics.median(waves):5.2f}/{percentile(waves,0.95):5.2f} "
            f"resident={statistics.median(resident):5.2f}/{percentile(resident,0.95):5.2f}"
        )



def print_cta_capability(rows, target_gbps, sm_count):
    print(f"\nCTA CAPABILITY AT {target_gbps:.0f} GB/s TARGET")
    print("packet CTA activeBlocks/SM activeWarps/SM max_GBps minUsefulWarps/SM_at_target")
    for packet in PACKETS:
        for cta in CTA_WIDTHS:
            group = [
                row for row in rows
                if row["packet_bytes"] == packet and row["block_threads"] == cta
            ]
            if not group:
                continue
            active_blocks = {row["max_active_blocks_per_sm"] for row in group}
            active_warps = {row["max_active_warps_per_sm"] for row in group}
            if len(active_blocks) != 1 or len(active_warps) != 1:
                raise SystemExit(
                    f"Occupancy drift inside packet{packet}/CTA{cta}: "
                    f"blocks={sorted(active_blocks)} warps={sorted(active_warps)}"
                )
            line_rate = [row for row in group if row["logical_GBps"] >= target_gbps]
            min_supply = (
                min(row["launch_warp_supply"] for row in line_rate) / sm_count
                if line_rate else float("nan")
            )
            supply_text = f"{min_supply:7.2f}" if line_rate else "    n/a"
            print(
                f"packet{packet:2d} {cta:3d} {next(iter(active_blocks)):15d} {next(iter(active_warps)):14d} "
                f"{max(row['logical_GBps'] for row in group):8.1f} {supply_text}"
            )

def print_frontier_exceptions(rows, target_gbps, sm_count):
    print(f"\nPACKET SPECIALIZATIONS THAT DO NOT REACH {target_gbps:.0f} GB/s ON EVERY SHAPE")
    for packet in PACKETS:
        groups = defaultdict(list)
        for row in rows:
            if row["packet_bytes"] != packet:
                continue
            key = (row["family"], row["dtype"], row["outer"], row["R"], row["K"])
            groups[key].append(row)
        misses = []
        for key, group in groups.items():
            best = max(group, key=lambda row: row["logical_GBps"])
            if best["logical_GBps"] < target_gbps:
                misses.append((key, best))
        if not misses:
            print(f"packet{packet:2d}: none")
            continue
        print(f"packet{packet:2d}: {len(misses)}/{len(groups)} shapes below target even at measured best layout")
        for (family, dtype, outer, reduction, inner), best in sorted(
            misses, key=lambda item: item[1]["logical_GBps"]
        )[:8]:
            print(
                f"  {family:15s} {dtype:10s} O={outer:<3d} R={reduction:<6d} K={inner:<6d} "
                f"best={best['logical_GBps']:.1f}GB/s p{packet} CTA={best['block_threads']} "
                f"shards={best['shards_per_output']} useful/SM={best['launch_warp_supply']/sm_count:.2f} "
                f"waves={best['physical_block_waves']:.2f}"
            )

def read_rows(path):
    with open(path, newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise SystemExit("No data rows found.")
    int_fields = (
        "outer", "R", "K", "packet_bytes", "items_per_lane", "block_threads", "shards_per_output",
        "rows_per_shard", "packet_owners", "component_tiles", "blocks", "warps_per_cta", "physical_warps",
        "useful_warps", "registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes",
        "max_active_blocks_per_sm", "max_active_warps_per_sm", "resident_device_blocks", "resident_device_warps",
        "useful_resident_warps", "launch_warp_supply", "comfort_target", "input_bytes", "output_bytes",
    )
    float_fields = (
        "comfort_fraction", "physical_block_waves", "lane_fill", "median_ms", "best_ms", "worst_ms",
        "input_GBps", "logical_GBps",
    )
    for row in rows:
        for field in int_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-fp8-kparallel-staged-layout-sweep.csv>")
    metadata = read_metadata(sys.argv[1])
    rows = read_rows(sys.argv[1])
    sm_count = metadata.get("sm_count")
    if not sm_count:
        raise SystemExit("Missing # sm_count=<N> benchmark metadata; saturation calibration requires the actual device SM count.")
    warp_size = metadata.get("warp_size", 32)

    expected_shapes = {(family, dtype, o, r, k) for family, o, r, k in SHAPES for dtype in DTYPES}
    by_layout = defaultdict(list)
    by_packet = defaultdict(list)
    seen = set()
    duplicate_rows = 0
    unexpected_rows = 0

    for row in rows:
        shape = (row["family"], row["dtype"], row["outer"], row["R"], row["K"])
        if shape not in expected_shapes:
            unexpected_rows += 1
        key = shape + (row["packet_bytes"], row["block_threads"], row["shards_per_output"])
        if key in seen:
            duplicate_rows += 1
        seen.add(key)
        by_layout[shape + (row["packet_bytes"], row["block_threads"])].append(row)
        by_packet[shape + (row["packet_bytes"],)].append(row)

    missing_layout_groups = 0
    non_apples_groups = 0
    for shape in expected_shapes:
        for packet in PACKETS:
            for cta in CTA_WIDTHS:
                group = by_layout.get(shape + (packet, cta), [])
                if not group:
                    missing_layout_groups += 1
                    continue
                # Within one packet/CTA layout the exact reduction and compiled specialization must remain fixed;
                # only the R shard count (hence partial-output size/progress) is intentionally swept.
                fixed = (
                    "family", "case", "dtype", "topology", "outer", "R", "K", "packet_bytes", "items_per_lane",
                    "block_threads", "packet_owners", "component_tiles", "warps_per_cta", "registers_per_thread",
                    "static_shared_bytes", "dynamic_shared_bytes", "max_active_blocks_per_sm", "max_active_warps_per_sm",
                    "resident_device_blocks", "resident_device_warps", "comfort_target", "lane_fill", "input_bytes",
                )
                first = group[0]
                if any(any(row[field] != first[field] for field in fixed) for row in group[1:]):
                    non_apples_groups += 1

    missing_packet_groups = 0
    best_by_packet = {}
    best_by_shape = {}
    for shape in sorted(expected_shapes):
        packet_bests = {}
        for packet in PACKETS:
            group = by_packet.get(shape + (packet,), [])
            if not group:
                missing_packet_groups += 1
                continue
            # Absolute first-stage kernel time is the primary objective. logical_GBps is reported separately as the
            # bandwidth capability including the FP32 partial writes intrinsic to the chosen shard layout.
            best = min(group, key=lambda row: (row["median_ms"], -row["logical_GBps"], row["output_bytes"]))
            packet_bests[packet] = best
            best_by_packet[shape + (packet,)] = best
        if len(packet_bests) == len(PACKETS):
            best_by_shape[shape] = min(packet_bests.values(), key=lambda row: (row["median_ms"], -row["packet_bytes"]))

    print(
        f"rows={len(rows)} expected_shapes={len(expected_shapes)} layout_groups={len(by_layout)} "
        f"duplicate_rows={duplicate_rows} unexpected_rows={unexpected_rows} "
        f"missing_layout_groups={missing_layout_groups} non_apples_groups={non_apples_groups} "
        f"missing_packet_groups={missing_packet_groups}"
    )
    if duplicate_rows or unexpected_rows or missing_layout_groups or non_apples_groups or missing_packet_groups:
        raise SystemExit(2)

    max_threads_per_sm = metadata.get("max_threads_per_sm")
    if max_threads_per_sm is None:
        # Older sweep files did not print max_threads_per_sm. The calibration population contains multiple CTA/register
        # specializations, so its maximum observed active-warps value recovers the hardware warp ceiling on the
        # existing RTX 5090 sweep while keeping old CSVs replayable. New benchmark output records the value directly.
        hardware_warps_per_sm = max(row["max_active_warps_per_sm"] for row in rows)
        max_threads_per_sm = hardware_warps_per_sm * warp_size
        print(
            f"production_policy_metadata=max_threads_per_sm_missing; derived {max_threads_per_sm} "
            f"from max observed active warps ({hardware_warps_per_sm}) * warp_size ({warp_size})"
        )
    else:
        hardware_warps_per_sm = max_threads_per_sm // warp_size

    timing_samples = metadata.get("timing_samples")
    if timing_samples is not None:
        print(f"production_policy_timing_samples={timing_samples}")

    packet16_policy_gate, packet16_unstable_selected = print_production_policy_replay(
        rows, expected_shapes, by_packet, sm_count, hardware_warps_per_sm, warp_size
    )

    print("\nBEST MEASURED STAGED CAPABILITY BY PACKET")
    for packet in PACKETS:
        bests = [row for key, row in best_by_packet.items() if key[-1] == packet]
        logical = [row["logical_GBps"] for row in bests]
        input_bw = [row["input_GBps"] for row in bests]
        times = [row["median_ms"] for row in bests]
        near = sum(v >= NEAR_TARGET_GBPS for v in logical)
        target = sum(v >= TARGET_GBPS for v in logical)
        print(
            f"packet{packet:2d}: groups={len(bests):2d} "
            f"logical median={statistics.median(logical):7.1f} p95={percentile(logical,0.95):7.1f} "
            f"max={max(logical):7.1f} GB/s  input median={statistics.median(input_bw):7.1f} "
            f"time median={statistics.median(times):.4f} ms  >=1550={near}/{len(bests)} >=1600={target}/{len(bests)}"
        )

    print("\nEXACT-SHAPE BEST PACKET/LAYOUT")
    print("family          dtype      O      R      K   p16_GBps p8_GBps p4_GBps p2_GBps winner  CTA shards rows  blocks  logical_GBps")
    for shape in sorted(expected_shapes):
        family, dtype, o, r, k = shape
        p = {packet: best_by_packet[shape + (packet,)] for packet in PACKETS}
        winner = best_by_shape[shape]
        print(
            f"{family:15s} {dtype:10s} {o:3d} {r:6d} {k:6d} "
            f"{p[16]['logical_GBps']:8.1f} {p[8]['logical_GBps']:7.1f} {p[4]['logical_GBps']:7.1f} {p[2]['logical_GBps']:7.1f} "
            f"p{winner['packet_bytes']:<2d} {winner['block_threads']:4d} {winner['shards_per_output']:6d} "
            f"{winner['rows_per_shard']:4d} {winner['blocks']:7d} {winner['logical_GBps']:11.1f}"
        )

    print("\nPACKET TIME RATIO VS PACKET16 AT EACH PACKET'S OWN BEST LAYOUT (lower is faster)")
    for packet in (8, 4, 2):
        ratios = []
        for shape in expected_shapes:
            base = best_by_packet[shape + (16,)]
            other = best_by_packet[shape + (packet,)]
            ratios.append(other["median_ms"] / base["median_ms"])
        faster = sum(r < 1.0 for r in ratios)
        print(
            f"p{packet}/p16 median={statistics.median(ratios):.3f} p95={percentile(ratios,0.95):.3f} "
            f"best={min(ratios):.3f} worst={max(ratios):.3f} faster={faster}/{len(ratios)}"
        )

    print("\nTOP THROUGHPUT LAYOUTS")
    print("family          dtype      O      R      K packet CTA shards rows blocks usefulSupply/Target waves logical_GBps input_GBps median_ms")
    all_best = list(best_by_packet.values())
    for row in sorted(all_best, key=lambda r: r["logical_GBps"], reverse=True)[:40]:
        print(
            f"{row['family']:15s} {row['dtype']:10s} {row['outer']:3d} {row['R']:6d} {row['K']:6d} "
            f"p{row['packet_bytes']:<2d} {row['block_threads']:3d} {row['shards_per_output']:6d} {row['rows_per_shard']:4d} "
            f"{row['blocks']:7d} {row['launch_warp_supply']}/{row['comfort_target']} "
            f"{row['physical_block_waves']:.2f} {row['logical_GBps']:11.1f} {row['input_GBps']:10.1f} {row['median_ms']:.4f}"
        )

    print_frontier(rows, NEAR_TARGET_GBPS, sm_count)
    print_frontier(rows, TARGET_GBPS, sm_count)
    print_frontier(
        rows,
        TARGET_GBPS,
        sm_count,
        families={"clean_512mib", "clean_1gib"},
        title=f"CLEAN-SHAPE SATURATION FRONTIER >= {TARGET_GBPS:.0f} GB/s",
    )
    print_frontier_exceptions(rows, TARGET_GBPS, sm_count)

    print_cta_capability(rows, TARGET_GBPS, sm_count)

    overall = list(best_by_shape.values())
    logical = [row["logical_GBps"] for row in overall]
    near = sum(v >= NEAR_TARGET_GBPS for v in logical)
    target = sum(v >= TARGET_GBPS for v in logical)
    clean_large = [row for shape, row in best_by_shape.items() if shape[0] in ("clean_512mib", "clean_1gib")]
    clean_bw = [row["logical_GBps"] for row in clean_large]
    print("\nSATURATION PERFORMANCE GATE")
    print(
        f"best-of-all-packets exact-shape logical median={statistics.median(logical):.1f} "
        f"p95={percentile(logical,0.95):.1f} max={max(logical):.1f} GB/s "
        f">=1550={near}/{len(logical)} >=1600={target}/{len(logical)}"
    )
    print(
        f"clean 512MiB/1GiB best logical median={statistics.median(clean_bw):.1f} "
        f"min={min(clean_bw):.1f} max={max(clean_bw):.1f} GB/s"
    )
    if max(clean_bw) < TARGET_GBPS:
        print("performance_gate=NOT_MET (no clean saturated shape reached 1600 GB/s; optimize FP8 staged kernels/layout before packet policy)")
    elif statistics.median(clean_bw) < NEAR_TARGET_GBPS:
        print("performance_gate=PARTIAL (some layouts reach line-rate regime, but clean-shape median is still below 1550 GB/s)")
    else:
        print("performance_gate=REACHED (clean saturated shapes are in the ~1.6 TB/s regime; packet-policy calibration can proceed after consistency review)")
    print(
        f"fp8_kparallel_layout_policy_final={packet16_policy_gate} "
        f"packet16_unstable_selected_measurements={packet16_unstable_selected}"
    )


if __name__ == "__main__":
    main()
