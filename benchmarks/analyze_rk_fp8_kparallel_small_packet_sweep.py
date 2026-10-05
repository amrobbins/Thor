#!/usr/bin/env python3
"""Analyze exact-shape small-reduction FP8 KParallel packet sweep output.

Every comparison group fixes dtype/O/R/K/input/output/topology/CTA/shards and varies only the
legal packet width. The baseline is the widest packet that can represent that K exactly.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_INPUT_BYTES = 64 * 1024 * 1024
PACKETS = (16, 8, 4, 2)
DTYPES = ("fp8_e4m3", "fp8_e5m2")
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

FIXED_FIELDS = (
    "family",
    "case",
    "dtype",
    "topology",
    "outer",
    "R",
    "K",
    "widest_legal_packet",
    "legal_packet_count",
    "block_threads",
    "shards_per_output",
    "input_bytes",
    "output_bytes",
)


def legal_packets(inner: int) -> tuple[int, ...]:
    return tuple(packet for packet in PACKETS if inner % packet == 0)


def expected_shapes() -> dict[tuple[str, str, str], tuple[int, ...]]:
    shapes: dict[tuple[str, str, str], tuple[int, ...]] = {}

    def add(family: str, outer: int, reduction: int, inner: int) -> None:
        if outer * reduction * inner > MAX_INPUT_BYTES:
            return
        packets = legal_packets(inner)
        if len(packets) < 2:
            raise AssertionError(f"expected sweep shape has fewer than two legal packets: {family} O={outer} R={reduction} K={inner}")
        case = f"o{outer}_r{reduction}_k{inner}"
        for dtype in DTYPES:
            key = (family, case, dtype)
            if key in shapes:
                raise AssertionError(f"duplicate expected group: {key}")
            shapes[key] = packets

    for outer in SUPPLY_OUTERS:
        for reduction in SUPPLY_REDUCTIONS:
            for inner in SUPPLY_INNERS:
                add("supply", outer, reduction, inner)
    for outer in AWKWARD_OUTERS:
        for reduction in AWKWARD_REDUCTIONS:
            for inner in AWKWARD_R_INNERS:
                add("awkward_r", outer, reduction, inner)
    for outer in AWKWARD_OUTERS:
        for reduction in AWKWARD_K_REDUCTIONS:
            for inner in (*AWKWARD_K_P16, *AWKWARD_K_P8, *AWKWARD_K_P4):
                add("awkward_k", outer, reduction, inner)
    for outer in AWKWARD_OUTERS:
        for reduction, inner in AWKWARD_BOTH:
            add("awkward_both", outer, reduction, inner)
    return shapes


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: no CSV rows found")
    return list(csv.DictReader(lines))


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-fp8-kparallel-small-packet-sweep.csv>")

    expected = expected_shapes()
    expected_rows = sum(len(packets) for packets in expected.values())
    rows = load_rows(Path(sys.argv[1]))
    by_group: dict[tuple[str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    duplicates: list[str] = []
    unexpected: list[str] = []

    for row in rows:
        key = (row["family"], row["case"], row["dtype"])
        packet = int(row["packet_bytes"])
        if key not in expected or packet not in expected.get(key, ()):
            unexpected.append(f"{key}/packet={packet}")
        if packet in by_group[key]:
            duplicates.append(f"{key}/packet={packet}")
        by_group[key][packet] = row

    missing_groups = sorted(set(expected) - set(by_group))
    extra_groups = sorted(set(by_group) - set(expected))
    missing_packets: list[str] = []
    non_apples: list[str] = []
    for key, packets in expected.items():
        actual = by_group.get(key, {})
        missing = set(packets) - set(actual)
        if missing:
            missing_packets.append(f"{key}: missing={sorted(missing, reverse=True)}")
            continue
        baseline = actual[packets[0]]
        for packet in packets:
            row = actual[packet]
            differences = [field for field in FIXED_FIELDS if row[field] != baseline[field]]
            if differences:
                non_apples.append(f"{key}/packet={packet}: changed={','.join(differences)}")

    print(
        f"rows={len(rows)} expected_rows={expected_rows} groups={len(by_group)} expected_groups={len(expected)} "
        f"duplicate_rows={len(duplicates)} unexpected_rows={len(unexpected)} missing_groups={len(missing_groups)} "
        f"extra_groups={len(extra_groups)} missing_packet_groups={len(missing_packets)} non_apples_groups={len(non_apples)}"
    )
    for label, values in (
        ("DUPLICATE", duplicates),
        ("UNEXPECTED", unexpected),
        ("MISSING_GROUP", [str(value) for value in missing_groups]),
        ("EXTRA_GROUP", [str(value) for value in extra_groups]),
        ("MISSING_PACKET", missing_packets),
        ("NON_APPLES", non_apples),
    ):
        for value in values[:20]:
            print(f"{label} {value}")

    winner_counts: Counter[tuple[str, int, int]] = Counter()
    ratio_by_family_baseline_packet: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    narrower_wins: list[tuple[float, tuple[str, str, str], int, int, dict[str, str], dict[str, str]]] = []

    for key, packets in expected.items():
        actual = by_group.get(key)
        if actual is None or any(packet not in actual for packet in packets):
            continue
        baseline_packet = packets[0]
        baseline = actual[baseline_packet]
        baseline_ms = float(baseline["median_ms"])
        winner_packet = min(packets, key=lambda packet: float(actual[packet]["median_ms"]))
        winner = actual[winner_packet]
        winner_ratio = float(winner["median_ms"]) / baseline_ms
        winner_counts[(key[0], baseline_packet, winner_packet)] += 1
        for packet in packets[1:]:
            ratio_by_family_baseline_packet[(key[0], baseline_packet, packet)].append(
                float(actual[packet]["median_ms"]) / baseline_ms
            )
        if winner_packet != baseline_packet:
            narrower_wins.append((winner_ratio, key, baseline_packet, winner_packet, baseline, winner))

    print("\nWINNER COUNTS BY FAMILY / WIDEST LEGAL PACKET")
    for family in ("supply", "awkward_r", "awkward_k", "awkward_both"):
        for baseline_packet in PACKETS:
            counts = {
                winner: winner_counts[(family, baseline_packet, winner)]
                for winner in PACKETS
                if winner_counts[(family, baseline_packet, winner)]
            }
            total = sum(counts.values())
            if total:
                rendered = " ".join(f"p{winner}={count}" for winner, count in sorted(counts.items(), reverse=True))
                print(f"{family:13s} widest=p{baseline_packet:<2d} groups={total:4d} {rendered}")

    print("\nMEDIAN TIME RATIO: NARROWER / WIDEST LEGAL (lower is faster)")
    for key in sorted(ratio_by_family_baseline_packet):
        family, baseline_packet, packet = key
        values = ratio_by_family_baseline_packet[key]
        faster = sum(value < 1.0 for value in values)
        print(
            f"{family:13s} p{packet}/p{baseline_packet} median={median(values):.3f} "
            f"best={min(values):.3f} worst={max(values):.3f} faster={faster}/{len(values)}"
        )

    print("\nSUPPLY WINNER GRID")
    print("dtype      R     K   " + "  ".join(f"O={outer}" for outer in SUPPLY_OUTERS))
    for dtype in DTYPES:
        for reduction in SUPPLY_REDUCTIONS:
            for inner in SUPPLY_INNERS:
                cells = []
                for outer in SUPPLY_OUTERS:
                    case = f"o{outer}_r{reduction}_k{inner}"
                    key = ("supply", case, dtype)
                    actual = by_group.get(key)
                    packets = expected.get(key)
                    if actual is None or packets is None:
                        cells.append("-")
                        continue
                    winner = min(packets, key=lambda packet: float(actual[packet]["median_ms"]))
                    cells.append(f"p{winner}")
                print(f"{dtype:9s} {reduction:5d} {inner:5d}  " + "  ".join(f"{cell:>6s}" for cell in cells))

    print(f"\nNARROWER WINS count={len(narrower_wins)} (sorted by largest measured speedup)")
    print("family        dtype      O     R     K  base win  ratio  base_blocks win_blocks  input_KiB")
    for ratio, key, baseline_packet, winner_packet, baseline, winner in sorted(narrower_wins)[:100]:
        print(
            f"{key[0]:13s} {key[2]:9s} {int(baseline['outer']):5d} {int(baseline['R']):5d} {int(baseline['K']):5d} "
            f"p{baseline_packet:<2d}  p{winner_packet:<2d} {ratio:6.3f} {int(baseline['blocks']):11d} "
            f"{int(winner['blocks']):10d} {int(baseline['input_bytes']) / 1024.0:10.1f}"
        )

    failed = (
        len(rows) != expected_rows
        or duplicates
        or unexpected
        or missing_groups
        or extra_groups
        or missing_packets
        or non_apples
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
