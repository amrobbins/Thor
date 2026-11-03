#!/usr/bin/env python3
"""Summarise `thor_fp8_kr_training_sweep` (1151-1152). Standard library only. Bandwidth in TB/s (moved = input +
output). 1152 sweeps time every launch on cold input and output ring views (the RK sweep's method), so moved TB/s is
bounded by the copy rate in the header; 1151 sweeps shared one output, which L2 could keep at short rows.

For every cell (one [K, R] site with one semantics) the sweep times the production candidate list (in_list=1, by
list_position) and the legal shapes around it (in_list=0). Per cell:
  oracle    the fastest timed shape of any kind: what the retained KR kernels can do at this site;
  list      the fastest listed shape: what an exhaustive production autotune can find;
  list@N    the fastest of list positions 1..N: an idealised tuning depth N (Quick 6, Regular 12, Ultra 24; the real
            autotuner also trims by rounds, so this is the depth's ceiling).
A cell whose oracle is below the target has a kernel gap (no shape reaches it); one whose list is more than --miss
slower than the oracle has a list gap (the generator misses a shape that exists).

Reports:
  0. grid cells: oracle moved TB/s by row length R (rows) and working set (columns), VALUE and ARG;
  1. coverage: cells by kind and input size, oracle and list@12 TB/s (geomean, p10, min), share at the target, and
     (1153) the oracle as a share of a copy moving the same bytes ("of copy": 100% = as fast as a memcpy at this
     size; read-heavy cells can pass 100%, since reads alone outrun a copy's mixed read / write traffic);
  2. large cells (>= --large-mib) by row-length bucket, VALUE and ARG: oracle TB/s, list@12, the winning families;
  3. list regret by depth N (time over the oracle and over the full list; all, large, held-out);
  4. list gaps: large cells where an unlisted shape beats the list by more than --miss, grouped and the worst ones;
  5. kernel gaps: the large cells furthest below the target, by time lost against the target;
  6. by operation and dtype (large cells): oracle TB/s;
  7. measurement: sample spread by size, close calls (top two shapes within 2%), copy rate at start and end, and
     (1153) the copy curve: what a device-to-device copy achieves by size, launch and ramp included;
  8. production (1152 sweeps): the production stamp's pick against the oracle and the full list;
  9. with --compare=other.csv[.gz] (the same seed and options): run-to-run agreement of every (cell, shape) median,
     of each cell's oracle time and shape, and of the production pick;
 10. (1155) fixed cost and rate: per kind, row class and winning family, a least-squares fit of the oracle time
     t = a + moved / B over large cells (a: the per-operation cost in us; B: the streaming rate), against a memcpy fit
     of the copy curve; at 16-64 MiB, a decides whether 1.5 TB/s is reachable;
 11. (1155, --split-stages sweeps) the Split finishing kernel's cost: full - partials time per Split shape, and the
     oracle TB/s of Split-won cells if it were free.

Usage: fp8_kr_training.py kr-training.csv[.gz] [--target=1.5] [--large-mib=16] [--miss=0.03] [--worst=25]
       [--cells-csv=out.csv] [--compare=other.csv[.gz]]
"""

import argparse
import collections
import csv
import gzip
import math
import sys

DEPTHS = (1, 2, 3, 6, 12, 24)


def read(path):
    opener = gzip.open if path.endswith(".gz") else open
    header, unsupported, lines = [], [], []
    with opener(path, "rt", newline="") as handle:
        for line in handle:
            if line.startswith("# unsupported"):
                unsupported.append(line.strip())
            elif line.startswith("#"):
                header.append(line.strip())
            elif line.strip():
                lines.append(line)
    return header, unsupported, list(csv.DictReader(lines))


def geomean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def quantile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def size_bucket(nbytes):
    mib = nbytes / 2**20
    for limit, name in ((16, "<16M"), (64, "16-64M"), (256, "64-256M")):
        if mib < limit:
            return name
    return ">=256M"


SIZE_ORDER = ("<16M", "16-64M", "64-256M", ">=256M")


def row_bucket(k, r):
    if k == 1:
        return "pure R"
    for limit, name in ((8, "R 2-8"), (32, "R 9-32"), (128, "R 33-128"), (1024, "R 129-1K"), (8192, "R 1K-8K"),
                        (65536, "R 8K-64K"), (1 << 20, "R 64K-1M")):
        if r <= limit:
            return name
    return "R >1M"


ROW_ORDER = ("R 2-8", "R 9-32", "R 33-128", "R 129-1K", "R 1K-8K", "R 8K-64K", "R 64K-1M", "R >1M", "pure R")


class Cell:
    def __init__(self, rows):
        first = rows[0]
        self.id = int(first["cell_id"])
        self.kind = first["kind"]
        self.kind_group = "VALUE" if self.kind == "value" else "ARG"
        self.operation = first["operation"]
        self.dtype = first["dtype"]
        self.k = int(first["K"])
        self.r = int(first["R"])
        self.bytes = int(first["input_bytes"])
        self.heldout = first["heldout"] == "1"
        self.random = first["random_geometry"] == "1"
        self.reference = float(first["reference_us"])
        self.target_mib = int(first.get("target_mib") or 0)
        self.production = first.get("production") or None
        self.copy_us = float(first.get("copy_us") or 0)  # 1153: a copy moving the same bytes
        self.stamp_ms = first.get("stamp_ms") or ""
        # 1155: variant=partials rows (--split-stages) time a Split shape's partial kernel alone; not candidates.
        self.partials = {row["shape"]: float(row["median_us"]) for row in rows
                         if row.get("variant") == "partials" and row["abandoned"] != "1"}
        rows = [row for row in rows if row.get("variant", "full") != "partials"]
        timed = [row for row in rows if row["abandoned"] != "1"]
        self.abandoned = len(rows) - len(timed)
        self.shapes = {row["shape"]: row for row in timed}
        self.time = {row["shape"]: float(row["median_us"]) for row in timed}
        self.moved = {row["shape"]: float(row["moved_tbps"]) for row in timed}
        self.listed = sorted((int(row["list_position"]), row["shape"]) for row in timed if row["in_list"] == "1")
        self.oracle = min(self.time, key=self.time.get)
        self.list_best = min((s for _, s in self.listed), key=self.time.get) if self.listed else None
        self.spreads = []
        for row in timed:
            samples = [float(row[key]) for key in row if key.startswith("sample") and row[key] not in ("", None)]
            if len(samples) >= 2 and float(row["median_us"]) > 0:
                self.spreads.append((max(samples) - min(samples)) / float(row["median_us"]))

    def tbps(self, shape):
        return self.moved[shape] if shape else float("nan")

    def at_depth(self, n):
        prefix = [s for position, s in self.listed if position <= n]
        return min(prefix, key=self.time.get) if prefix else None

    def regret(self, shape, against=None):
        against = against or self.oracle
        return self.time[shape] / self.time[against] - 1.0 if shape else float("nan")

    def family(self, shape):
        return self.shapes[shape]["family"]

    def production_shape(self):
        return self.production if self.production in self.time else None


def table(title, header, rows):
    print(title)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    for line in [header] + rows:
        print("  " + "  ".join(str(x).rjust(w) for x, w in zip(line, widths)))
    print()


def fmt(x, digits=2):
    return "-" if x != x else f"{x:.{digits}f}"


def pct(x):
    return "-" if x != x else f"{100 * x:.0f}%"


def summary(cells, target, depth=12):
    oracle = [c.tbps(c.oracle) for c in cells]
    listed = [c.tbps(c.at_depth(depth)) for c in cells]
    of_copy = [c.copy_us / c.time[c.oracle] for c in cells if c.copy_us]
    return [len(cells), fmt(geomean(oracle)), fmt(quantile(oracle, 0.1)), fmt(min(oracle) if oracle else float("nan")),
            pct(sum(v >= target for v in oracle) / len(oracle)) if oracle else "-",
            fmt(geomean(listed)), fmt(quantile(listed, 0.1)),
            pct(sum(v >= target for v in listed) / len(listed)) if listed else "-",
            pct(geomean(of_copy)) if of_copy else "-", pct(quantile(of_copy, 0.1)) if of_copy else "-"]


def load_cells(path):
    header, unsupported, rows = read(path)
    by_cell = collections.defaultdict(list)
    for row in rows:
        by_cell[int(row["cell_id"])].append(row)
    cells = [Cell(cell_rows) for cell_rows in by_cell.values() if any(row["abandoned"] != "1" for row in cell_rows)]
    cells.sort(key=lambda c: c.id)
    return header, unsupported, rows, cells


def header_value(header, key):
    for line in header:
        for token in line.split():
            if token.startswith(key + "="):
                return token.split("=", 1)[1]
    return None


def compare(cells, other_path, large_mib, worst):
    _, _, _, others = load_cells(other_path)
    other = {c.id: c for c in others}
    pairs = collections.defaultdict(list)
    oracle_ratio = collections.defaultdict(list)
    same_oracle = collections.Counter()
    close_oracle = collections.Counter()
    production_same = collections.Counter()
    production_cells = collections.Counter()
    shared = 0
    worst_pairs = []
    for c in cells:
        o = other.get(c.id)
        if not o or (o.k, o.r, o.operation, o.dtype) != (c.k, c.r, c.operation, c.dtype):
            continue
        shared += 1
        bucket = size_bucket(c.bytes)
        for shape, t in c.time.items():
            if shape in o.time:
                ratio = math.log(o.time[shape] / t)
                pairs[bucket].append(ratio)
                worst_pairs.append((abs(ratio), c, shape, t, o.time[shape]))
        oracle_ratio[bucket].append(math.log(o.time[o.oracle] / c.time[c.oracle]))
        same_oracle[bucket] += c.oracle == o.oracle
        close_oracle[bucket] += o.time.get(c.oracle, float("inf")) <= 1.01 * o.time[o.oracle]
        if c.production and o.production:
            production_cells[bucket] += 1
            production_same[bucket] += c.production == o.production
    print(f"9. Run-to-run agreement with {other_path} ({shared} shared cells; ratios are other / this)")
    out = []
    for bucket in SIZE_ORDER:
        d = pairs.get(bucket)
        if not d:
            continue
        a = [abs(x) for x in d]
        orc = [abs(x) for x in oracle_ratio[bucket]]
        n = len(oracle_ratio[bucket])
        out.append([bucket, len(d), pct2(math.expm1(sum(d) / len(d))), pct2(math.expm1(quantile(a, 0.5))),
                    pct2(math.expm1(quantile(a, 0.9))), pct2(math.expm1(quantile(a, 0.99))), pct2(math.expm1(max(a))),
                    n, pct2(math.expm1(quantile(orc, 0.5))), pct2(math.expm1(quantile(orc, 0.9))),
                    pct(same_oracle[bucket] / n), pct(close_oracle[bucket] / n),
                    f"{production_same[bucket]}/{production_cells[bucket]}" if production_cells[bucket] else "-"])
    table("   |median ratio - 1| of every (cell, shape), and of each cell's oracle time",
          ["size", "pairs", "mean", "p50", "p90", "p99", "max", "cells", "oracle p50", "oracle p90", "same oracle",
           "oracle within 1%", "same production"], out)
    worst_pairs.sort(key=lambda x: -x[0])
    out = [[c.kind, c.operation, c.k, c.r, f"{c.bytes / 2**20:.0f}", shape, fmt(t, 1), fmt(u, 1), pct2(math.expm1(d))]
           for d, c, shape, t, u in worst_pairs[:worst] if c.bytes >= large_mib * 2**20]
    if out:
        table("   largest disagreements, large cells", ["kind", "op", "K", "R", "MiB", "shape", "us", "other us",
                                                         "|ratio-1|"], out)


def least_squares(points):
    """t = a + x / B; returns (a us, B TB/s)."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = n * sxx - sx * sx
    if n < 3 or denominator <= 0:
        return float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denominator
    return (sy - slope * sx) / n, (1 / slope / 1e6 if slope > 0 else float("nan"))


def row_class(c):
    if c.k == 1:
        return "pure R"
    if c.r >= 65536:
        return "R >= 64K"
    if c.r >= 1024:
        return "R 1K-64K"
    return "R 33-1K" if c.r > 32 else "R <= 32"


def moved_bytes(c, shape):
    return c.tbps(shape) * c.time[shape] * 1e6


def fixed_cost(cells, header, large_mib, target):
    groups = collections.defaultdict(list)
    for c in cells:
        if c.bytes >= large_mib * 2**20:
            groups[(c.kind_group, row_class(c), c.family(c.oracle))].append((moved_bytes(c, c.oracle), c.time[c.oracle]))
    out = []
    for (kind, rows, family), points in sorted(groups.items()):
        if len(points) < 20:
            continue
        a, rate = least_squares(points)
        # The largest a that still reaches the target at 16 and 32 MiB moved at this rate.
        budget = [m * 2**20 / (target * 1e6) - m * 2**20 / (rate * 1e6) for m in (16, 32)]
        out.append([kind, rows, family, len(points), fmt(a), fmt(rate, 3), fmt(budget[0]), fmt(budget[1])])
    curve = header_value(header, "copy_curve_moved_mib_us")
    if curve:
        points = [(float(m) * 2**20, float(us)) for m, us in (item.split(":") for item in curve.split(","))
                  if float(m) >= 16]
        a, rate = least_squares(points)
        out.append(["memcpy", "copy curve", "-", len(points), fmt(a), fmt(rate, 3), "-", "-"])
    table(f"10. Fixed cost and rate, large cells: oracle us = a + moved / B; 'a for {target:g} @16M / @32M' is the most "
          f"a can be for {target:g} TB/s at 16 / 32 MiB moved at that rate",
          ["kind", "rows", "family", "n", "a us", "B TB/s", f"a for {target:g} @16M", "@32M"], out)


def split_stages(cells, large_mib, target):
    pairs = collections.defaultdict(list)
    free = collections.defaultdict(list)
    for c in cells:
        if not c.partials or c.bytes < large_mib * 2**20:
            continue
        key = (c.kind_group, row_class(c), size_bucket(c.bytes))
        for shape, partial in c.partials.items():
            if shape in c.time:
                pairs[key].append((c.time[shape] - partial, (c.time[shape] - partial) / c.time[shape]))
        if c.family(c.oracle) == "split" and c.oracle in c.partials:
            moved = moved_bytes(c, c.oracle)
            free[key].append((c.tbps(c.oracle), moved / c.partials[c.oracle] / 1e6))
    if not pairs:
        return
    out = []
    for key in sorted(pairs):
        finish = sorted(p[0] for p in pairs[key])
        share = sorted(p[1] for p in pairs[key])
        f = free.get(key, [])
        out.append(list(key) + [len(finish), fmt(quantile(finish, 0.5)), fmt(quantile(finish, 0.9)),
                                pct(quantile(share, 0.5)), len(f), fmt(geomean([x for x, _ in f])) if f else "-",
                                fmt(geomean([y for _, y in f])) if f else "-",
                                pct(sum(y >= target for _, y in f) / len(f)) if f else "-"])
    table("11. Split finishing kernel, large cells: finish us = full - partials time of each Split shape; for cells won "
          "by Split, the oracle TB/s as timed and with the finishing kernel free",
          ["kind", "rows", "size", "shapes", "finish p50 us", "p90", "share p50", "split-won", "TB/s", "TB/s free",
           f">={target:g} free"], out)


def pct2(x):
    return "-" if x != x else f"{100 * x:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--target", type=float, default=1.5, help="moved TB/s target")
    parser.add_argument("--large-mib", type=float, default=16)
    parser.add_argument("--miss", type=float, default=0.03, help="list gap threshold (fraction slower than oracle)")
    parser.add_argument("--worst", type=int, default=25)
    parser.add_argument("--cells-csv", help="write one summary row per cell")
    parser.add_argument("--compare", help="another sweep CSV with the same seed and options")
    args = parser.parse_args()

    header, unsupported, rows, cells = load_cells(args.csv)
    for line in header:
        print(line)
    large = [c for c in cells if c.bytes >= args.large_mib * 2**20]
    abandoned = sum(c.abandoned for c in cells)
    print(f"# cells={len(cells)} large(>={args.large_mib:g} MiB)={len(large)} rows={len(rows)} "
          f"abandoned_shapes={abandoned} unsupported={len(unsupported)} target={args.target} TB/s moved\n")
    if not cells:
        sys.exit("no timed cells")

    cols = ["n", "oracle gm", "p10", "min", f">={args.target:g}", "list@12 gm", "p10", f">={args.target:g}",
            "of copy gm", "p10"]

    # 0. grid table
    grid = [c for c in cells if c.target_mib]
    targets = sorted({c.target_mib for c in grid})
    for kind in ("VALUE", "ARG"):
        cell_of = collections.defaultdict(list)
        for c in grid:
            if c.kind_group == kind:
                cell_of[(c.r if c.k > 1 else 0, c.target_mib)].append(c)
        if not cell_of:
            continue
        lengths = sorted({r for r, _ in cell_of if r}) + ([0] if any(r == 0 for r, _ in cell_of) else [])
        out = []
        for r in lengths:
            line = ["pure R" if r == 0 else r]
            for t in targets:
                subset = cell_of.get((r, t))
                line.append(fmt(geomean([c.tbps(c.oracle) for c in subset])) if subset else "-")
            out.append(line)
        table(f"0. Grid cells, {kind}: oracle moved TB/s by row length R and working set (MiB; every op sampled there)",
              ["R"] + [f"{t}M" for t in targets], out)

    # 1. coverage by kind and size
    out = []
    for kind in ("VALUE", "ARG"):
        for bucket in SIZE_ORDER:
            subset = [c for c in cells if c.kind_group == kind and size_bucket(c.bytes) == bucket]
            if subset:
                out.append([kind, bucket] + summary(subset, args.target))
    table("1. Moved TB/s by kind and input size (oracle = fastest timed shape; list@12 = fastest of the first 12 listed)",
          ["kind", "size"] + cols, out)

    # 2. large cells by row bucket
    out = []
    for kind in ("VALUE", "ARG"):
        for bucket in ROW_ORDER:
            subset = [c for c in large if c.kind_group == kind and row_bucket(c.k, c.r) == bucket]
            if not subset:
                continue
            winners = collections.Counter(c.family(c.oracle) for c in subset)
            win = " ".join(f"{f}:{n}" for f, n in winners.most_common())
            out.append([kind, bucket] + summary(subset, args.target) + [win])
    table(f"2. Large cells (>= {args.large_mib:g} MiB) by row length: moved TB/s and the oracle's family",
          ["kind", "rows"] + cols + ["oracle family"], out)

    # 3. regret by depth
    out = []
    for name, subset in (("all", cells), ("large", large), ("large heldout", [c for c in large if c.heldout]),
                         ("large VALUE", [c for c in large if c.kind_group == "VALUE"]),
                         ("large ARG", [c for c in large if c.kind_group == "ARG"])):
        if not subset:
            continue
        for n in DEPTHS:
            regrets = [c.regret(c.at_depth(n)) for c in subset if c.at_depth(n)]
            list_regrets = [c.regret(c.at_depth(n), c.list_best) for c in subset if c.at_depth(n)]
            out.append([name, n, len(regrets), pct(sum(regrets) / len(regrets)), pct(quantile(regrets, 0.9)),
                        pct(max(regrets)), pct(sum(list_regrets) / len(list_regrets)), pct(max(list_regrets)),
                        pct(sum(r <= 0.02 for r in regrets) / len(regrets))])
    table("3. Regret of the best of list positions 1..N (time over the oracle; 'vs list' = over the full list)",
          ["cells", "N", "n", "mean", "p90", "max", "vs list mean", "vs list max", "within 2%"], out)

    # 4. list gaps
    misses = [c for c in large if c.list_best and c.regret(c.list_best) > args.miss]
    groups = collections.Counter((c.kind_group, row_bucket(c.k, c.r), c.family(c.oracle)) for c in misses)
    out = [[k, b, f, n] for (k, b, f), n in sorted(groups.items(), key=lambda x: -x[1])]
    table(f"4. List gaps: large cells where an unlisted shape beats the whole list by > {100 * args.miss:.0f}% "
          f"({len(misses)} of {len(large)})", ["kind", "rows", "oracle family", "cells"], out)
    out = []
    for c in sorted(misses, key=lambda c: -(c.time[c.list_best] - c.time[c.oracle]))[:args.worst]:
        out.append([c.kind, c.operation, c.k, c.r, f"{c.bytes / 2**20:.0f}", c.list_best, fmt(c.tbps(c.list_best)),
                    c.oracle, fmt(c.tbps(c.oracle)), pct(c.regret(c.list_best)),
                    fmt(c.time[c.list_best] - c.time[c.oracle], 1)])
    if out:
        table("   worst list gaps by time lost", ["kind", "op", "K", "R", "MiB", "list best", "TB/s", "oracle", "TB/s",
                                                  "regret", "us lost"], out)

    # 5. kernel gaps
    gaps = []
    for c in large:
        target_us = (c.bytes + c.k * (8 if c.kind == "arg_value_index" else 4)) / (args.target * 1e6)
        best = c.time[c.oracle]
        if best > target_us:
            gaps.append((best - target_us, c, target_us))
    gaps.sort(key=lambda x: -x[0])
    out = []
    for lost, c, target_us in gaps[:args.worst]:
        depth = c.at_depth(12)
        out.append([c.kind, c.operation, c.dtype[4:], c.k, c.r, f"{c.bytes / 2**20:.0f}", c.oracle,
                    fmt(c.tbps(c.oracle)), depth or "-", fmt(c.tbps(depth)), fmt(c.time[c.oracle], 1),
                    fmt(target_us, 1), fmt(lost, 1), fmt(c.copy_us, 1) if c.copy_us else "-",
                    pct(c.copy_us / c.time[c.oracle]) if c.copy_us else "-"])
    table(f"5. Kernel gaps: large cells whose fastest shape is below {args.target:g} TB/s ({len(gaps)} of {len(large)}), "
          f"by us over the target", ["kind", "op", "dtype", "K", "R", "MiB", "oracle", "TB/s", "list@12", "TB/s",
                                     "us", "target us", "us over", "copy us", "of copy"], out)
    if gaps:
        by = collections.Counter((c.kind_group, row_bucket(c.k, c.r)) for _, c, _ in gaps)
        lost_by = collections.defaultdict(float)
        for lost, c, _ in gaps:
            lost_by[(c.kind_group, row_bucket(c.k, c.r))] += lost
        out = [[k, b, n, fmt(lost_by[(k, b)], 0)] for (k, b), n in sorted(by.items(), key=lambda x: -lost_by[x[0]])]
        table("   kernel gaps grouped (us over target summed)", ["kind", "rows", "cells", "us over"], out)

    # 6. by operation / dtype
    out = []
    keyed = collections.defaultdict(list)
    for c in large:
        keyed[(c.kind, c.operation, c.dtype)].append(c)
    for (kind, op, dtype), subset in sorted(keyed.items()):
        out.append([kind, op, dtype] + summary(subset, args.target))
    table("6. Large cells by operation and dtype", ["kind", "op", "dtype"] + cols, out)

    # 7. measurement quality
    spreads = [s for c in cells for s in c.spreads]
    close = 0
    for c in cells:
        times = sorted(c.time.values())
        if len(times) >= 2 and times[1] / times[0] - 1 <= 0.02:
            close += 1
    print("7. Measurement")
    print(f"  sample spread (max-min)/median: p50 {pct2(quantile(spreads, 0.5))}  p90 {pct2(quantile(spreads, 0.9))}  "
          f"max {pct2(max(spreads) if spreads else float('nan'))}")
    for bucket in SIZE_ORDER:
        b = [s for c in cells if size_bucket(c.bytes) == bucket for s in c.spreads]
        if b:
            print(f"    {bucket:>7}: n {len(b)}  p50 {pct2(quantile(b, 0.5))}  p90 {pct2(quantile(b, 0.9))}  "
                  f"p99 {pct2(quantile(b, 0.99))}")
    print(f"  cells whose two fastest shapes are within 2%: {close} of {len(cells)}")
    start_bw, end_bw = header_value(header, "copy_bandwidth_tbps"), header_value(header, "copy_bandwidth_tbps_end")
    if start_bw:
        print(f"  copy bandwidth: start {start_bw} TB/s" + (f", end {end_bw} TB/s" if end_bw else ""))
    curve = header_value(header, "copy_curve_moved_mib_us")
    if curve:
        points = []
        for item in curve.split(","):
            mib, us = item.split(":")
            points.append(f"{mib}M:{float(mib) * 2**20 / float(us) / 1e6:.2f}")
        print("  copy TB/s (moved) by moved bytes: " + "  ".join(points))
    if unsupported:
        print(f"  unsupported cells: {len(unsupported)} (first: {unsupported[0]})")
    print()

    # 8. production pick
    with_production = [c for c in cells if c.production]
    if with_production:
        out = []
        for name, subset in (("all", with_production),
                             ("large", [c for c in with_production if c.bytes >= args.large_mib * 2**20]),
                             ("large VALUE", [c for c in with_production
                                              if c.bytes >= args.large_mib * 2**20 and c.kind_group == "VALUE"]),
                             ("large ARG", [c for c in with_production
                                            if c.bytes >= args.large_mib * 2**20 and c.kind_group == "ARG"])):
            picked = [c for c in subset if c.production_shape()]
            if not picked:
                continue
            regrets = [c.regret(c.production_shape()) for c in picked]
            list_regrets = [c.regret(c.production_shape(), c.list_best) for c in picked]
            out.append([name, len(subset), len(subset) - len(picked), pct(sum(regrets) / len(regrets)),
                        pct(quantile(regrets, 0.9)), pct(max(regrets)), pct(sum(list_regrets) / len(list_regrets)),
                        pct(max(list_regrets)), pct(sum(r <= 0.02 for r in regrets) / len(regrets)),
                        fmt(geomean([c.tbps(c.production_shape()) for c in picked]))])
        table("8. Production pick (the stamp at each site): time over the oracle and over the full list",
              ["cells", "n", "not KR", "mean", "p90", "max", "vs list mean", "vs list max", "within 2%", "TB/s gm"],
              out)
        bad = sorted((c for c in with_production if c.production_shape() and c.bytes >= args.large_mib * 2**20),
                     key=lambda c: -(c.time[c.production_shape()] - c.time[c.list_best]))[:args.worst]
        out = [[c.kind, c.operation, c.k, c.r, f"{c.bytes / 2**20:.0f}", c.production, fmt(c.tbps(c.production)),
                c.list_best, fmt(c.tbps(c.list_best)), pct(c.regret(c.production, c.list_best)),
                fmt(c.time[c.production] - c.time[c.list_best], 1), c.stamp_ms]
               for c in bad if c.regret(c.production, c.list_best) > 0.01]
        if out:
            table("   production picks more than 1% off the full list, large cells",
                  ["kind", "op", "K", "R", "MiB", "production", "TB/s", "list best", "TB/s", "regret", "us lost",
                   "stamp ms"], out)
        odd = sorted({c.production for c in with_production if not c.production_shape()})
        if odd:
            print(f"   production passes not among the timed shapes: {', '.join(odd[:10])}\n")
        # 1161: what a stamp costs (wall time of the production stamp, contest included), by input size.
        out = []
        for name, low, high in (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
                                (">=128M", 128 * 2**20, float("inf"))):
            times = [float(c.stamp_ms) for c in with_production if c.stamp_ms and low <= c.bytes < high]
            if times:
                out.append([name, len(times), fmt(quantile(times, 0.5), 1), fmt(quantile(times, 0.9), 1),
                            fmt(max(times), 1), fmt(sum(times) / 1000, 1)])
        if out:
            table("   stamp wall time by input size (ms; total in s)", ["input", "n", "p50", "p90", "max", "total s"],
                  out)

    if args.compare:
        compare(cells, args.compare, args.large_mib, args.worst)

    fixed_cost(cells, header, args.large_mib, args.target)
    split_stages(cells, args.large_mib, args.target)

    if args.cells_csv:
        with open(args.cells_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "kind", "operation", "dtype", "K", "R", "input_bytes", "heldout", "oracle",
                             "oracle_family", "oracle_us", "oracle_tbps", "list_best", "list_best_us", "list_tbps",
                             "production", "production_us"] +
                            [f"list{n}_us" for n in DEPTHS])
            for c in cells:
                writer.writerow([c.id, c.kind, c.operation, c.dtype, c.k, c.r, c.bytes, int(c.heldout), c.oracle,
                                 c.family(c.oracle), f"{c.time[c.oracle]:.3f}", f"{c.tbps(c.oracle):.4f}",
                                 c.list_best or "", f"{c.time[c.list_best]:.3f}" if c.list_best else "",
                                 f"{c.tbps(c.list_best):.4f}" if c.list_best else "", c.production or "",
                                 f"{c.time[c.production_shape()]:.3f}" if c.production_shape() else ""] +
                                [f"{c.time[c.at_depth(n)]:.3f}" if c.at_depth(n) else "" for n in DEPTHS])


if __name__ == "__main__":
    main()
