#!/usr/bin/env python3
"""Analyze the focused DenseRK family-boundary end-to-end calibration.

Primary metric is absolute microseconds. Ratios are intentionally secondary.
The script compares the fastest forced KParallel first pass with the fastest
forced RCooperative first pass for each shape and groups the result by the
packet width available to FP32 KParallel.
"""

import csv
import sys
from collections import defaultdict


def packet_bytes_for_fp32(k: int) -> int:
    if k % 4 == 0:
        return 16
    if k % 2 == 0:
        return 8
    return 4


def main(path: str) -> int:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]

    by_case = defaultdict(list)
    for row in rows:
        if row.get("family") == "realistic_end_to_end":
            by_case[row["case"]].append(row)

    comparisons = []
    for case, case_rows in by_case.items():
        kp = [r for r in case_rows if r["first_strategy"] == "k_parallel"]
        rc = [r for r in case_rows if r["first_strategy"].startswith("r_cooperative_")]
        if not kp or not rc:
            continue
        best_kp = min(kp, key=lambda r: float(r["median_ms"]))
        best_rc = min(rc, key=lambda r: float(r["median_ms"]))
        k = int(best_kp["K"])
        delta_us = (float(best_kp["median_ms"]) - float(best_rc["median_ms"])) * 1000.0
        comparisons.append((delta_us, packet_bytes_for_fp32(k), case, best_kp, best_rc))

    print(f"cases={len(comparisons)}")
    print("\nABSOLUTE FAMILY CROSSOVERS (positive means RCooperative wins)")
    for delta_us, packet_bytes, case, kp, rc in sorted(comparisons, reverse=True):
        print(
            f"{delta_us:+8.2f} us  packet={packet_bytes:2d}B  {case}  "
            f"KParallel rows={kp['first_rows_per_shard']} {float(kp['median_ms'])*1000.0:.2f} us  "
            f"RCoop={rc['first_strategy']} rows={rc['first_rows_per_shard']} {float(rc['median_ms'])*1000.0:.2f} us"
        )

    print("\nBY KPARALLEL PACKET WIDTH")
    for packet_bytes in (16, 8, 4):
        bucket = [x for x in comparisons if x[1] == packet_bytes]
        if not bucket:
            continue
        wins = [x for x in bucket if x[0] > 0]
        material = [x for x in bucket if x[0] >= 10.0]
        total_us = sum(max(0.0, x[0]) for x in bucket)
        max_us = max(x[0] for x in bucket)
        print(
            f"packet={packet_bytes:2d}B cases={len(bucket)} rcoop_wins={len(wins)} "
            f"material_wins_ge_10us={len(material)} positive_savings_sum_us={total_us:.2f} max_saving_us={max_us:.2f}"
        )

    material = [x for x in comparisons if x[0] >= 10.0]
    print("\nMATERIAL RCOOPERATIVE WINS (>=10 us)")
    if not material:
        print("none")
    else:
        for delta_us, packet_bytes, case, kp, rc in sorted(material, reverse=True):
            print(
                f"{delta_us:.2f} us  packet={packet_bytes}B  {case}  "
                f"kp_rows={kp['first_rows_per_shard']}  rc={rc['first_strategy']} rc_rows={rc['first_rows_per_shard']}"
            )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <rk-family-boundary-calibration.csv>")
    raise SystemExit(main(sys.argv[1]))
