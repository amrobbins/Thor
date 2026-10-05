#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PACKETS = (16, 8, 4, 2)
CTA_WIDTHS = (32, 64, 128, 256)
COMFORT_WARPS_PER_SM = 8
MAX_INPUT_BYTES = 64 * 1024 * 1024

SUPPLY_OUTERS = (1, 4, 16, 64, 256, 1024)
SUPPLY_REDUCTIONS = (32, 128, 512)
SUPPLY_INNERS = (16, 64, 256, 1024)
AWKWARD_OUTERS = (1, 8, 64, 512)
AWKWARD_REDUCTIONS = (17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513)
AWKWARD_R_INNERS = (64, 256, 1024)
AWKWARD_K_P16 = (48, 80, 112, 144, 240, 496, 1008)
AWKWARD_K_P8 = (24, 40, 72, 120, 248, 504, 1016)
AWKWARD_K_P4 = (20, 36, 68, 124, 252, 508, 1020)
AWKWARD_K_REDUCTIONS = (32, 128, 512)
AWKWARD_BOTH = ((33, 48), (65, 120), (129, 124), (257, 240), (511, 504), (513, 508))
DTYPES = ("fp8_e4m3", "fp8_e5m2")


def legal_packets(k):
    return tuple(p for p in PACKETS if k % p == 0)


def expected_shapes():
    shapes = []
    for o in SUPPLY_OUTERS:
        for r in SUPPLY_REDUCTIONS:
            for k in SUPPLY_INNERS:
                shapes.append(("supply", o, r, k))
    for o in AWKWARD_OUTERS:
        for r in AWKWARD_REDUCTIONS:
            for k in AWKWARD_R_INNERS:
                shapes.append(("awkward_r", o, r, k))
    for o in AWKWARD_OUTERS:
        for r in AWKWARD_K_REDUCTIONS:
            for k in AWKWARD_K_P16 + AWKWARD_K_P8 + AWKWARD_K_P4:
                shapes.append(("awkward_k", o, r, k))
    for o in AWKWARD_OUTERS:
        for r, k in AWKWARD_BOTH:
            shapes.append(("awkward_both", o, r, k))
    return [s for s in shapes if s[1] * s[2] * s[3] <= MAX_INPUT_BYTES and len(legal_packets(s[3])) >= 2]


def read_rows(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#") and line.strip()]
    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        raise SystemExit("No data rows found.")
    ints = (
        "outer", "R", "K", "widest_legal_packet", "legal_packet_count", "packet_bytes", "items_per_lane",
        "block_threads", "packet_owners", "component_tiles", "blocks", "warps_per_cta", "physical_warps",
        "useful_warps", "registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes",
        "max_active_blocks_per_sm", "max_active_warps_per_sm", "resident_device_blocks",
        "resident_device_warps", "useful_resident_warps", "launch_warp_supply", "comfort_target",
        "input_bytes", "output_bytes",
    )
    floats = ("comfort_fraction", "lane_fill", "median_ms", "best_ms", "worst_ms", "logical_GBps")
    for row in rows:
        for key in ints:
            row[key] = int(row[key])
        for key in floats:
            row[key] = float(row[key])
    return rows


def occupancy_choice(layouts):
    comfortable = [r for r in layouts if r["launch_warp_supply"] >= r["comfort_target"]]
    if comfortable:
        # Mirrors the existing Complete KParallel ordering once the candidate is inside its hardware-relative
        # comfort envelope: maximize useful lane fill, then prefer the smaller CTA.
        return max(comfortable, key=lambda r: (r["lane_fill"], -r["block_threads"]))
    # If no layout reaches comfort, maximize the fraction of the exact specialization-relative target supplied;
    # retain the same lane-fill / smaller-CTA tie breakers.
    return max(layouts, key=lambda r: (r["comfort_fraction"], r["lane_fill"], -r["block_threads"]))


def ratio_stats(values):
    if not values:
        return "n/a"
    vals = sorted(values)
    p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]
    return f"median={statistics.median(vals):.3f} p95={p95:.3f} best={vals[0]:.3f} worst={vals[-1]:.3f}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-fp8-kparallel-complete-layout-sweep.csv>")
    rows = read_rows(sys.argv[1])

    expected = expected_shapes()
    expected_shape_keys = {(fam, dtype, o, r, k) for fam, o, r, k in expected for dtype in DTYPES}
    expected_rows = sum(len(legal_packets(k)) * len(CTA_WIDTHS) for _, o, r, k in expected for _dtype in DTYPES)

    shape_packet = defaultdict(list)
    duplicates = 0
    seen = set()
    unexpected = 0
    for row in rows:
        shape_key = (row["family"], row["dtype"], row["outer"], row["R"], row["K"])
        if shape_key not in expected_shape_keys:
            unexpected += 1
        key = shape_key + (row["packet_bytes"], row["block_threads"])
        if key in seen:
            duplicates += 1
        seen.add(key)
        shape_packet[shape_key + (row["packet_bytes"],)].append(row)

    missing_layout_groups = 0
    non_apples_layout_groups = 0
    policy_hits = 0
    policy_groups = 0
    policy_ratios = []
    policy_misses = []
    measured_best_by_shape = defaultdict(dict)

    for shape_key in sorted(expected_shape_keys):
        fam, dtype, o, r, k = shape_key
        for packet in legal_packets(k):
            layouts = shape_packet.get(shape_key + (packet,), [])
            if sorted(x["block_threads"] for x in layouts) != list(CTA_WIDTHS):
                missing_layout_groups += 1
                continue
            fixed_fields = ("family", "case", "dtype", "topology", "outer", "R", "K", "widest_legal_packet",
                            "legal_packet_count", "packet_bytes", "items_per_lane", "packet_owners", "input_bytes",
                            "output_bytes")
            first = layouts[0]
            if any(any(x[f] != first[f] for f in fixed_fields) for x in layouts[1:]):
                non_apples_layout_groups += 1
                continue

            selected = occupancy_choice(layouts)
            best = min(layouts, key=lambda x: (x["median_ms"], x["block_threads"]))
            policy_groups += 1
            ratio = selected["median_ms"] / best["median_ms"]
            policy_ratios.append(ratio)
            if selected["block_threads"] == best["block_threads"]:
                policy_hits += 1
            else:
                policy_misses.append((ratio, shape_key, packet, selected, best))
            measured_best_by_shape[shape_key][packet] = best

    missing_shape_groups = 0
    packet_winner_counts = Counter()
    packet_ratios = defaultdict(list)
    narrower_wins = []
    shape_winners = {}
    for shape_key in sorted(expected_shape_keys):
        fam, dtype, o, r, k = shape_key
        packets = legal_packets(k)
        bests = measured_best_by_shape.get(shape_key, {})
        if any(p not in bests for p in packets):
            missing_shape_groups += 1
            continue
        winner_packet = min(packets, key=lambda p: (bests[p]["median_ms"], -p))
        winner = bests[winner_packet]
        shape_winners[shape_key] = winner
        packet_winner_counts[(fam, packets[0], winner_packet)] += 1
        baseline = bests[packets[0]]
        for p in packets[1:]:
            ratio = bests[p]["median_ms"] / baseline["median_ms"]
            packet_ratios[(fam, p, packets[0])].append(ratio)
            if ratio < 1.0:
                narrower_wins.append((ratio, shape_key, packets[0], p, baseline, bests[p]))

    print(f"rows={len(rows)} expected_rows={expected_rows} shape_groups={len(expected_shape_keys)} "
          f"layout_groups={policy_groups} duplicate_rows={duplicates} unexpected_rows={unexpected} "
          f"missing_layout_groups={missing_layout_groups} non_apples_layout_groups={non_apples_layout_groups} "
          f"missing_shape_groups={missing_shape_groups}")
    if duplicates or unexpected or missing_layout_groups or non_apples_layout_groups or missing_shape_groups or len(rows) != expected_rows:
        raise SystemExit(2)

    print("\nOCCUPANCY CTA POLICY VS MEASURED BEST")
    print(f"exact_layout_groups={policy_groups} policy_best_cta_hits={policy_hits}/{policy_groups} "
          f"({100.0 * policy_hits / policy_groups:.1f}%) selected/best {ratio_stats(policy_ratios)}")
    by_packet = defaultdict(list)
    hits_by_packet = Counter()
    totals_by_packet = Counter()
    for shape_packet_key, layouts in shape_packet.items():
        if len(layouts) != 4:
            continue
        packet = shape_packet_key[-1]
        selected = occupancy_choice(layouts)
        best = min(layouts, key=lambda x: (x["median_ms"], x["block_threads"]))
        by_packet[packet].append(selected["median_ms"] / best["median_ms"])
        totals_by_packet[packet] += 1
        if selected["block_threads"] == best["block_threads"]:
            hits_by_packet[packet] += 1
    for p in PACKETS:
        if totals_by_packet[p]:
            print(f"packet{p:2d}: hits={hits_by_packet[p]}/{totals_by_packet[p]} "
                  f"selected/best {ratio_stats(by_packet[p])}")

    print("\nMEASURED-BEST PACKET WINNER COUNTS BY FAMILY / WIDEST LEGAL")
    for fam in ("supply", "awkward_r", "awkward_k", "awkward_both"):
        for widest in PACKETS:
            entries = [(win, count) for (f, w, win), count in packet_winner_counts.items() if f == fam and w == widest]
            if not entries:
                continue
            total = sum(c for _, c in entries)
            pieces = " ".join(f"p{win}={count}" for win, count in sorted(entries, reverse=True))
            print(f"{fam:13s} widest=p{widest:<2d} groups={total:4d} {pieces}")

    print("\nMEASURED-BEST TIME RATIO: NARROWER / WIDEST LEGAL (lower is faster)")
    for key in sorted(packet_ratios):
        fam, p, base = key
        vals = packet_ratios[key]
        faster = sum(v < 1.0 for v in vals)
        print(f"{fam:13s} p{p}/p{base} {ratio_stats(vals)} faster={faster}/{len(vals)}")

    print("\nSUPPLY WINNER GRID (packet/CTA from measured-best layout)")
    for dtype in DTYPES:
        print(f"dtype={dtype}")
        print("R     K    " + " ".join(f"O={o:<8d}" for o in SUPPLY_OUTERS))
        for r in SUPPLY_REDUCTIONS:
            for k in SUPPLY_INNERS:
                cells = []
                for o in SUPPLY_OUTERS:
                    key = ("supply", dtype, o, r, k)
                    row = shape_winners.get(key)
                    cells.append("-" if row is None else f"p{row['packet_bytes']}/c{row['block_threads']}")
                print(f"{r:<5d} {k:<4d} " + " ".join(f"{c:<10s}" for c in cells))

    print(f"\nOCCUPANCY POLICY MISSES count={len(policy_misses)} (largest selected/best slowdown first)")
    print("family        dtype      O     R     K packet selCTA bestCTA ratio selSupply/Target bestSupply/Target")
    for ratio, shape_key, packet, selected, best in sorted(policy_misses, reverse=True)[:80]:
        fam, dtype, o, r, k = shape_key
        print(f"{fam:13s} {dtype:10s} {o:5d} {r:5d} {k:5d} p{packet:<2d} {selected['block_threads']:6d} "
              f"{best['block_threads']:7d} {ratio:5.3f} "
              f"{selected['launch_warp_supply']}/{selected['comfort_target']} "
              f"{best['launch_warp_supply']}/{best['comfort_target']}")

    print(f"\nNARROWER PACKET WINS count={len(narrower_wins)} (largest measured speedup first)")
    print("family        dtype      O     R     K base win ratio baseCTA winCTA baseBlocks winBlocks")
    for ratio, shape_key, basep, winp, base, win in sorted(narrower_wins)[:100]:
        fam, dtype, o, r, k = shape_key
        print(f"{fam:13s} {dtype:10s} {o:5d} {r:5d} {k:5d} p{basep:<2d} p{winp:<2d} {ratio:5.3f} "
              f"{base['block_threads']:7d} {win['block_threads']:6d} {base['blocks']:10d} {win['blocks']:9d}")


if __name__ == "__main__":
    main()
