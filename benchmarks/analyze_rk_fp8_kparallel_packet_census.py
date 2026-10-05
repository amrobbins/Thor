#!/usr/bin/env python3
"""Analyze exact-shape FP8 KParallel packet A/B census output.

Every quartet is literally the same reduction and physical topology. Only packet_bytes changes.
Therefore median-time ratios in this analyzer are direct apples-to-apples kernel comparisons.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_PHASES = {"complete_64mib", "staged_64mib", "deep_staged_512mib"}
EXPECTED_K = {512, 1024, 2048, 8192}
EXPECTED_DTYPES = {"fp8_e4m3", "fp8_e5m2"}
EXPECTED_PACKETS = {16, 8, 4, 2}
EXPECTED_GROUPS = len(EXPECTED_PHASES) * len(EXPECTED_K) * len(EXPECTED_DTYPES)
EXPECTED_ROWS = EXPECTED_GROUPS * len(EXPECTED_PACKETS)

FIXED_FIELDS = (
    "phase",
    "case",
    "dtype",
    "topology",
    "outer",
    "R",
    "K",
    "block_threads",
    "shards_per_output",
    "rows_per_shard",
    "input_bytes",
    "output_bytes",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: no CSV rows found")
    return list(csv.DictReader(lines))


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <rk-fp8-kparallel-packet-census.csv>")

    rows = load_rows(Path(sys.argv[1]))
    by_group: dict[tuple[str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    duplicates: list[str] = []
    unexpected: list[str] = []

    for row in rows:
        phase = row["phase"]
        dtype = row["dtype"]
        inner = int(row["K"])
        packet = int(row["packet_bytes"])
        if phase not in EXPECTED_PHASES or dtype not in EXPECTED_DTYPES or inner not in EXPECTED_K or packet not in EXPECTED_PACKETS:
            unexpected.append(f"{phase}/{row['case']}/{dtype}/K={inner}/packet={packet}")
        key = (phase, row["case"], dtype)
        if packet in by_group[key]:
            duplicates.append(f"{key}/packet={packet}")
        by_group[key][packet] = row

    bad_groups: list[str] = []
    missing_packets: list[str] = []
    for key, packet_rows in sorted(by_group.items()):
        missing = EXPECTED_PACKETS - set(packet_rows)
        if missing:
            missing_packets.append(f"{key}: missing={sorted(missing)}")
            continue
        baseline = packet_rows[16]
        for packet, row in packet_rows.items():
            differences = [field for field in FIXED_FIELDS if row[field] != baseline[field]]
            if differences:
                bad_groups.append(f"{key}/packet={packet}: changed={','.join(differences)}")

    print(f"rows={len(rows)} expected_rows={EXPECTED_ROWS} groups={len(by_group)} expected_groups={EXPECTED_GROUPS}")
    print(
        f"duplicate_rows={len(duplicates)} unexpected_rows={len(unexpected)} "
        f"missing_packet_groups={len(missing_packets)} non_apples_groups={len(bad_groups)}"
    )
    for label, values in (
        ("DUPLICATE", duplicates),
        ("UNEXPECTED", unexpected),
        ("MISSING", missing_packets),
        ("NON_APPLES", bad_groups),
    ):
        for value in values[:20]:
            print(f"{label} {value}")

    print("\nEXACT-SHAPE PACKET COMPARISON")
    print("phase                 dtype      O      R     K   p16_ms  p8/p16  p4/p16  p2/p16  winner")
    ratios_by_topology: dict[tuple[str, int], list[float]] = defaultdict(list)
    ratios_all: dict[int, list[float]] = defaultdict(list)

    for key in sorted(by_group):
        packet_rows = by_group[key]
        if set(packet_rows) != EXPECTED_PACKETS:
            continue
        p16 = float(packet_rows[16]["median_ms"])
        ratios = {packet: float(packet_rows[packet]["median_ms"]) / p16 for packet in (8, 4, 2)}
        topology = packet_rows[16]["topology"]
        for packet, ratio in ratios.items():
            ratios_by_topology[(topology, packet)].append(ratio)
            ratios_all[packet].append(ratio)
        winner = min(EXPECTED_PACKETS, key=lambda packet: float(packet_rows[packet]["median_ms"]))
        row = packet_rows[16]
        print(
            f"{row['phase']:21s} {row['dtype']:9s} {int(row['outer']):6d} {int(row['R']):6d} {int(row['K']):5d} "
            f"{p16:8.4f} {ratios[8]:8.3f} {ratios[4]:8.3f} {ratios[2]:8.3f}  p{winner}"
        )

    print("\nRATIO SUMMARY: PACKET / PACKET16 (lower is faster)")
    for topology in ("complete", "staged"):
        pieces = []
        for packet in (8, 4, 2):
            values = ratios_by_topology.get((topology, packet), [])
            if values:
                pieces.append(
                    f"p{packet}/p16 median={statistics.median(values):.3f} "
                    f"p95={percentile(values, 0.95):.3f} best={min(values):.3f} worst={max(values):.3f}"
                )
        print(f"{topology:8s}  " + "  ".join(pieces))

    print("\nALL EXACT-SHAPE RATIOS")
    for packet in (8, 4, 2):
        values = ratios_all.get(packet, [])
        if values:
            faster = sum(value < 1.0 for value in values)
            print(
                f"packet{packet}/packet16 median={statistics.median(values):.3f} "
                f"p95={percentile(values, 0.95):.3f} best={min(values):.3f} worst={max(values):.3f} "
                f"faster_than_p16={faster}/{len(values)}"
            )

    failed = (
        len(rows) != EXPECTED_ROWS
        or len(by_group) != EXPECTED_GROUPS
        or duplicates
        or unexpected
        or missing_packets
        or bad_groups
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
