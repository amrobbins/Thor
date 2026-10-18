#!/usr/bin/env python3
"""Summarize a complete, cache-cold FP16 RK production/forced-physical benchmark.

The 'oracle' is strictly the minimum measured candidate, not the theoretical
memory-bandwidth roofline or the optimum over uncompiled physical alternatives.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PAIR_COUNT = 24
GEOMETRIES_PER_PAIR = 3
NUM_GEOMETRIES = PAIR_COUNT * GEOMETRIES_PER_PAIR


def load(path):
    cases = defaultdict(list)
    closed = set()
    footer = False
    rows = 0
    with open(path, newline="") as f:
        data = (line for line in f if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(data)
        needed = {"family", "case", "outer", "R", "K", "dtype", "operation", "implementation",
                  "packet_bytes", "block_threads", "first_rows_per_shard", "first_shards",
                  "input_bytes", "output_bytes", "median_ms", "best_ms", "worst_ms"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError("Input lacks required FP16 autotune census CSV columns")
        for row in reader:
            if row["dtype"] != "fp16" or row["operation"] != "sum":
                raise ValueError("Unexpected dtype/operation")
            key = (int(row["outer"]), int(row["R"]), int(row["K"]))
            if row["case"] != f"o{key[0]}_r{key[1]}_k{key[2]}":
                raise ValueError("Geometry/case mismatch")
            for name in ("median_ms", "best_ms", "worst_ms"):
                value = float(row[name])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid timing {name} at {key}")
            if not float(row["best_ms"]) <= float(row["median_ms"]) <= float(row["worst_ms"]):
                raise ValueError(f"Inconsistent sample median at {key}")
            if int(row["input_bytes"]) != 2 * key[0] * key[1] * key[2]:
                raise ValueError(f"Input byte count mismatch at {key}")
            if int(row["output_bytes"]) != 2 * key[0] * key[2]:
                raise ValueError(f"Output byte count mismatch at {key}")
            signature = (row["implementation"], row["packet_bytes"], row["block_threads"],
                         row["first_rows_per_shard"], row["first_shards"])
            if signature in {r["signature"] for r in cases[key]}:
                raise ValueError(f"Duplicate candidate at {key}: {signature}")
            row["signature"] = signature
            cases[key].append(row)
            rows += 1
    with open(path) as f:
        for line in f:
            if line.startswith("# fp16_autotune_case_complete "):
                tokens = dict(part.split("=", 1) for part in line.split()[2:])
                key = (int(tokens["outer"]), int(tokens["R"]), int(tokens["K"]))
                if key in closed:
                    raise ValueError(f"Duplicate case completion marker {key}")
                closed.add(key)
                if int(tokens["candidates"]) != len(cases[key]):
                    raise ValueError(f"Wrong number of candidates for {key}")
            elif line.startswith("# rk_fp16_production_autotune_census_complete "):
                if footer:
                    raise ValueError("Duplicate completion footer")
                footer = True
                if int(line.split("geometries=")[1].split()[0]) != NUM_GEOMETRIES:
                    raise ValueError("Unexpected declared geometry total")
    if not footer or len(cases) != NUM_GEOMETRIES or closed != set(cases):
        raise ValueError(f"Incomplete census: footer={footer} cases={len(cases)}/{NUM_GEOMETRIES} closed={len(closed)}")
    for key, group in cases.items():
        if sum(row["implementation"] == "production" for row in group) != 1 or len(group) < 2:
            raise ValueError(f"Missing or duplicated production/forced candidate at {key}")
    return cases, rows


def summarize(cases, details, bandwidth_gbps):
    report = []
    for key, options in sorted(cases.items()):
        prod = next(row for row in options if row["implementation"] == "production")
        best = min(options, key=lambda row: float(row["median_ms"]))
        p = float(prod["median_ms"])
        o = float(best["median_ms"])
        us = max(0.0, (p - o) * 1000)
        rel = us / (o * 1000) * 100
        physical = f'{best["implementation"]}/P{best["packet_bytes"]}/CTA{best["block_threads"]}/rows{best["first_rows_per_shard"]}'
        input_bytes = int(prod["input_bytes"])
        effective_gbps = (input_bytes + int(prod["output_bytes"])) / (p * 1e6)
        report.append({"outer": key[0], "R": key[1], "K": key[2],
                       "input_MiB": round(input_bytes / (1024 * 1024), 3),
                       "production_us": round(p * 1000, 4), "sampled_oracle_us": round(o * 1000, 4),
                       "improvement_us": round(us, 4), "regret_pct": round(rel, 4),
                       "winner": physical, "production_logical_GBps": round(effective_gbps, 2),
                       "roofline_efficiency_pct": (round(effective_gbps / bandwidth_gbps * 100, 2)
                                                    if bandwidth_gbps else ""),
                       "candidate_count": len(options)})
    with open(details, "w", newline="") as f:
        w = csv.DictWriter(f, report[0].keys())
        w.writeheader()
        w.writerows(report)
    n = len(report)
    cases_gt5 = [r for r in report if r["regret_pct"] > 5]
    cases_gt10 = [r for r in report if r["regret_pct"] > 10]
    print(f"FP16 RK production vs sampled physical oracle: {n} geometries;"
          f" >5%={len(cases_gt5)}, >10%={len(cases_gt10)}")
    print(f"Maximum potential saving={max(r['improvement_us'] for r in report):.3f} us,"
          f" sum per-case={sum(r['improvement_us'] for r in report):.3f} us,"
          f" median={statistics.median(r['improvement_us'] for r in report):.3f} us")
    print("NOTE: summed per-case latency is a diagnostic, not a workload-weighted application speedup.")
    for lower, upper, label in ((0,24,"<24MiB"),(24,96,"24–96MiB"),(96,float('inf'),">=96MiB")):
        group = [r for r in report if lower <= r["input_MiB"] < upper]
        if group:
            print(f"  {label}: cases={len(group)} >5%={sum(r['regret_pct']>5 for r in group)}"
                  f" max_savings_us={max(r['improvement_us'] for r in group):.3f}")
    print("Largest absolute tuning opportunities:")
    for r in sorted(report, key=lambda r: r["improvement_us"], reverse=True)[:12]:
        print(f"  outer={r['outer']} R={r['R']} K={r['K']} input={r['input_MiB']:.1f}MiB"
              f" prod={r['production_us']:.3f}us best={r['sampled_oracle_us']:.3f}us"
              f" gain={r['improvement_us']:.3f}us ({r['regret_pct']:.2f}%) {r['winner']}")
    print(f"Saved per-geometry comparison: {details}")
    print("Caution: the oracle searches only these compiled RK candidates. High effective GB/s does not prove"
          " saturation; benchmark other datatype, operation, layouts, and sustained device bandwidth separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--details", default="rk-fp16-production-autotune-details.csv", type=Path)
    parser.add_argument("--bandwidth-gbps", type=float,
                        help="Optional independently measured sustainable GPU memory GB/s (not theoretical peak)")
    args = parser.parse_args()
    if args.bandwidth_gbps is not None and args.bandwidth_gbps <= 0:
        parser.error("--bandwidth-gbps must be positive")
    cases, rows = load(args.input)
    print(f"Validated: {rows} rows, {len(cases)} complete geometries")
    summarize(cases, args.details, args.bandwidth_gbps)


if __name__ == "__main__":
    main()
