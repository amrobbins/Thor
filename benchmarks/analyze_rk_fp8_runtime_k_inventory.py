#!/usr/bin/env python3
"""Compare lean RuntimeK compiled-depth inventories against measured end-to-end oracle.

Usage: python3 benchmarks/analyze_rk_fp8_runtime_k_inventory.py <csv>

The oracle is the best *measured* complete reduction, not an unmeasured global
physical oracle. A 912 discovery CSV only includes neighboring rotated depths;
use the separate --rk-fp8-runtime-k-inventory-holdout sweep for deletion decisions.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

LEAN_DEPTHS = frozenset((32, 64, 128, 256, 512))
KEY = ("outer", "R", "K", "dtype")


def fmt(row):
    return f"O={row['outer']} R={row['R']} K={row['K']} {row['dtype']}"


def select_best(candidates):
    return min(candidates, key=lambda row: float(row["total_median_ms"]))


def evaluate(name, groups, predicate):
    misses = []
    for key, rows in sorted(groups.items()):
        oracle = select_best(rows)
        pool = [row for row in rows if predicate(row)]
        if not pool:
            raise ValueError(f"Inventory {name!r} leaves geometry {key} with no candidate")
        best = select_best(pool)
        regret_us = 1000 * (float(best["total_median_ms"]) - float(oracle["total_median_ms"]))
        regret_pct = 100 * regret_us / (1000 * float(oracle["total_median_ms"]))
        misses.append((regret_pct, regret_us, key, best, oracle))
    over5 = sum(pct > 5 for pct, *_ in misses)
    over10 = sum(pct > 10 for pct, *_ in misses)
    sum_us = sum(us for _, us, *_ in misses)
    max_pct = max(pct for pct, *_ in misses)
    max_us = max(us for _, us, *_ in misses)
    print(f"{name:28s} >5%={over5:3d}/{len(misses):3d} >10%={over10:3d} "
          f"max={max_pct:6.2f}% max_abs={max_us:7.3f}us sum_abs={sum_us:9.3f}us")
    return misses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8") as handle:
        comments = []
        data = []
        for line in handle:
            if line.startswith("#"):
                comments.append(line.strip())
            elif line.strip():
                data.append(line)
    if not data:
        raise ValueError("Empty benchmark CSV")
    rows = list(csv.DictReader(data))
    if not all(row.get("family") == "runtime_k" for row in rows):
        raise ValueError("Expected RuntimeK-only end-to-end benchmark rows")
    if not any("independent_inventory_holdout=1" in line for line in comments):
        print("WARNING: This is discovery/partial-inventory data, NOT an independent full-depth holdout.")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in KEY)].append(row)
    print(f"{len(groups)} geometry/dtype cells; {len(rows)} timed physical candidates")
    winners = Counter((r["layout"], int(r["rows_per_shard"])) for rs in groups.values()
                      for r in (select_best(rs),))
    print("Measured oracle winners by layout: ", dict(Counter(layout for layout, _ in winners.elements())))
    print("Rotated winning compiled depths:",
          dict(sorted(Counter(depth for (layout, depth), count in winners.items()
                              if layout.startswith("rotated") for _ in range(count)).items())))
    any_candidate = lambda row: True
    lean = lambda row: row["layout"] == "packetstaged" or int(row["rows_per_shard"]) in LEAN_DEPTHS
    only_prefix = lambda row: row["layout"] == "packetstaged" or (row["layout"] == "rotated_prefix" and int(row["rows_per_shard"]) in LEAN_DEPTHS)
    no32 = lambda row: row["layout"] == "packetstaged" or int(row["rows_per_shard"]) in (64, 128, 256, 512)
    evaluate("All measured", groups, any_candidate)
    results = evaluate("Both layouts, depths 32-512", groups, lean)
    evaluate("Both layouts, depths 64-512", groups, no32)
    evaluate("Prefix only, depths 32-512", groups, only_prefix)
    evaluate("PacketStaged only", groups, lambda row: row["layout"] == "packetstaged")
    print("Worst lean-inventory misses:")
    for pct, us, key, selected, oracle in sorted(
            (r for r in results if r[1] > 0.0001), reverse=True)[:12]:
        print(f"  {fmt(selected):43s} regret={pct:6.2f}% ({us:7.3f}us) "
              f"chosen={selected['layout']}/{selected['rows_per_shard']} "
              f"oracle={oracle['layout']}/{oracle['rows_per_shard']}")


if __name__ == "__main__":
    main()
