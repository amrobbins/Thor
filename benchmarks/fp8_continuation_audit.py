#!/usr/bin/env python3
"""Summarise `thor_fp8_retained_cold_l2_sweep --continuation-audit` (1134+). Standard library only.

For each staged candidate: continuation = full - first. Its reference is a fitted copy model, not the candidate's own
first_copy - first: single differences of two ~70 us medians are noisy (+-2 us), and at small sizes they often fall
below what a kernel can do. 1137: the model is max(launch floor, partial bytes / B), where B is the median
bytes/us of the measured partial copies of 4 MiB and more (a device-to-device copy reads and writes every partial
byte from L2, so a bandwidth-bound continuation should not be slower). The launch floor defaults to 4 us.

Reports:
  1. continuation / reference and continuation share of the end-to-end time, per kind and continuation
     implementation, and per kind by shard count;
  2. the staged candidates with the largest continuation excess over the reference (us);
  3. ranking impact: per cell, the oracle over every candidate as measured ("today") against the oracle if no
     continuation took longer than its reference ("fast": min(full, first + reference), so negative noise in
     full - first never makes a candidate slower). How often the winner changes, and the regret of today's winner
     under the fast costs: how much slow continuations distort the training labels.

Usage: fp8_continuation_audit.py audit.csv[.gz] [--worst=25] [--floor-us=4]
"""

import argparse
import collections
import csv
import gzip
import math
import re
import sys


def rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        yield from csv.DictReader(lines)


def geomean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--worst", type=int, default=25)
    parser.add_argument("--floor-us", type=float, default=4.0, help="kernel launch floor of the reference (us)")
    args = parser.parse_args()

    cells = collections.defaultdict(dict)  # cell -> candidate -> {variant: us}
    meta = {}
    for row in rows(args.csv):
        cell = (row["kind"], row["operation"], row["dtype"], int(row["outer"]), int(row["R"]), int(row["K"]))
        candidate = (row["family"], row["configuration"])
        cells[cell].setdefault(candidate, {})[row["variant"]] = 1000.0 * float(row["total_median_ms"])
        meta[(cell, candidate)] = (int(row["shards"]), int(row["partial_bytes"]), row["continuation"])
    if not cells:
        sys.exit("no audit rows")

    # Copy bandwidth from the measured partial copies of >= 4 MiB (bytes per us).
    rates = [meta[(cell, cand)][1] / (t["first_copy"] - t["first"])
             for cell, cands in cells.items() for cand, t in cands.items()
             if "first_copy" in t and meta[(cell, cand)][1] >= 4 << 20 and t["first_copy"] - t["first"] > 0.5]
    bandwidth = quantile(rates, 0.5) if len(rates) >= 20 else 1.5e6

    def reference(partial_bytes):
        return max(args.floor_us, partial_bytes / bandwidth)

    staged = []  # (cell, candidate, first, full, cont, ref)
    for cell, candidates in cells.items():
        for candidate, times in candidates.items():
            if "full" in times and "first" in times:
                first, full = times["first"], times["full"]
                staged.append((cell, candidate, first, full, max(full - first, 0.0),
                               reference(meta[(cell, candidate)][1])))

    print(f"cells {len(cells)}, staged candidates {len(staged)}; reference = max({args.floor_us:g} us, bytes / "
          f"{bandwidth / 1e6:.2f} TB/s) from {len(rates)} partial copies >= 4 MiB")
    print("\n1. Continuation vs reference (ratio > 1: slower than a launch-floored copy of the same partials)")
    print(f"{'kind':6} {'continuation':48} {'n':>6} {'ratio_gm':>8} {'ratio_p90':>9} {'cont_us_med':>11} "
          f"{'share_gm':>8} {'share_p90':>9}")
    groups = collections.defaultdict(list)
    for cell, candidate, first, full, cont, ref in staged:
        # Group by implementation, not by its packet / CTA / shard geometry (_p8_t32_s1 etc.).
        groups[(cell[0], re.sub(r"_[pts]\d+", "", meta[(cell, candidate)][2]))].append((cont, ref, full))
    for (kind, name), items in sorted(groups.items()):
        ratios = [max(c, 0.1) / r for c, r, _ in items]
        shares = [c / f for c, _, f in items if f > 0]
        print(f"{kind:6} {name[:48]:48} {len(items):6d} {geomean(ratios):8.2f} {quantile(ratios, 0.9):9.2f} "
              f"{quantile([c for c, _, _ in items], 0.5):11.1f} {geomean(shares):8.3f} {quantile(shares, 0.9):9.3f}")

    print("\n   by shard count")
    print(f"{'kind':6} {'shards':>10} {'n':>6} {'ratio_gm':>8} {'ratio_p90':>9} {'cont_us_med':>11} {'share_gm':>8}")
    buckets = collections.defaultdict(list)
    for cell, candidate, first, full, cont, ref in staged:
        shards = meta[(cell, candidate)][0]
        bucket = 1 << max(0, shards.bit_length() - 1)
        buckets[(cell[0], bucket)].append((cont, ref, full))
    for (kind, bucket), items in sorted(buckets.items()):
        ratios = [max(c, 0.1) / r for c, r, _ in items]
        shares = [c / f for c, _, f in items if f > 0]
        print(f"{kind:6} {f'{bucket}-{2 * bucket - 1}':>10} {len(items):6d} {geomean(ratios):8.2f} "
              f"{quantile(ratios, 0.9):9.2f} {quantile([c for c, _, _ in items], 0.5):11.1f} {geomean(shares):8.3f}")

    print(f"\n2. Worst {args.worst} staged candidates by continuation excess over the reference")
    worst = sorted(staged, key=lambda s: s[4] - s[5], reverse=True)
    for cell, candidate, first, full, cont, ref in worst[: args.worst]:
        shards, partial_bytes, name = meta[(cell, candidate)]
        print(f"  {cell[0]} {cell[1]} outer={cell[3]} R={cell[4]} K={cell[5]} {candidate[0]}:{candidate[1]} "
              f"shards={shards} partials={partial_bytes / 2**20:.1f}MiB first={first:.1f}us cont={cont:.1f}us "
              f"ref={ref:.1f}us excess={cont - ref:.1f}us share={cont / full:.2f} [{name}]")

    print("\n3. Ranking impact if no continuation took longer than its reference")
    impact = collections.defaultdict(lambda: [0, 0, []])  # kind -> [cells, winner changed, regrets]
    for cell, candidates in cells.items():
        today, fast = {}, {}
        for candidate, times in candidates.items():
            if "complete" in times:
                today[candidate] = fast[candidate] = times["complete"]
            elif "full" in times and "first" in times:
                today[candidate] = times["full"]
                fast[candidate] = min(times["full"], times["first"] + reference(meta[(cell, candidate)][1]))
        if not today:
            continue
        winner = min(today, key=today.get)
        fast_best = min(fast.values())
        entry = impact[cell[0]]
        entry[0] += 1
        entry[1] += min(fast, key=fast.get) != winner
        entry[2].append(fast[winner] / fast_best)
    for kind, (count, changed, regrets) in sorted(impact.items()):
        print(f"  {kind}: {count} cells, winner changes in {changed} ({100 * changed / count:.1f}%); regret of today's "
              f"winner under fast continuations: geomean {geomean(regrets):.4f}, p95 {quantile(regrets, 0.95):.4f}, "
              f"p99 {quantile(regrets, 0.99):.4f}, max {max(regrets):.4f}")


if __name__ == "__main__":
    main()
