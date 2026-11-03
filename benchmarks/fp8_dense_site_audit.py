#!/usr/bin/env python3
"""Summarise `thor_fp8_dense_site_audit` (1145-1149). Standard library only. 1149 reports bandwidth in TB/s.

ratio = production median / reference, where reference = max(launch floor, bytes / copy bandwidth); 1146 counts the
output bytes too (1145 counted only the input). A ratio of 1 is line rate; the header records the bandwidth and floor.

Reports:
  1. per family, kind and path: cells, ratio geomean / p50 / p90 / max, and the share of cells within 1.25x and 2x;
  2. per family and kind, by row length bucket (KR: R; R: input size), the same ratios and the median GB/s;
  3. the worst cells by excess time over the reference, and any geometries the production stamp refused;
  4. with --baseline (an earlier audit CSV): per family / kind / operation the geomean of median / baseline median over
     the shared cells, the share faster / slower by more than 5%, and the largest regressions;
  5. with a --candidates run (1147): which retained family / shape is fastest per row-length bucket, how the production
     choice compares with the fastest retained shape, and the regret of timing only the first N ranked candidates
     (N = 6 Quick, 12 Regular, 24 Ultra).

Usage: fp8_dense_site_audit.py audit.csv[.gz] [--worst=30] [--large-mib=16] [--baseline=old.csv]
"""

import argparse
import collections
import csv
import gzip
import math
import sys


def read(path):
    opener = gzip.open if path.endswith(".gz") else open
    header, unsupported, data = [], [], []
    with opener(path, "rt", newline="") as handle:
        lines = []
        for line in handle:
            if line.startswith("# unsupported"):
                unsupported.append(line.strip())
            elif line.startswith("#"):
                header.append(line.strip())
            else:
                lines.append(line)
    data = list(csv.DictReader(lines))
    return header, unsupported, data


def geomean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def bucket(value):
    low = 1 << max(0, int(value).bit_length() - 1)
    return low


def human(n):
    for unit, size in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


TARGET_TBPS = 1.5


def summary(rows):
    rates = [r["tbps"] for r in rows]
    return (f"{len(rows):5d} {geomean(rates):7.2f} {quantile(rates, 0.1):7.2f} {min(rates):7.2f} "
            f"{quantile([r['read_tbps'] for r in rows], 0.5):9.2f} "
            f"{100 * sum(x >= TARGET_TBPS for x in rates) / len(rates):8.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--worst", type=int, default=30)
    parser.add_argument("--baseline", help="an earlier audit CSV to compare medians against")
    parser.add_argument("--large-mib", type=float, default=16.0,
                        help="cells at least this large are reported separately (launch floor matters less)")
    args = parser.parse_args()

    header, unsupported, raw = read(args.csv)
    if not raw:
        sys.exit("no audit rows")
    candidate_raw = [r for r in raw if r["path"] == "candidate"]
    raw = [r for r in raw if r["path"] != "candidate"]
    rows = []
    for r in raw:
        rows.append({
            "family": r["family"], "kind": r["kind"], "op": r["operation"], "dtype": r["dtype"],
            "K": int(r["K"]), "R": int(r["R"]), "bytes": int(r["input_bytes"]), "path": r["path"],
            "passes": r["passes"], "median": float(r["median_us"]), "ref": float(r["reference_us"]),
            "ratio": float(r["ratio"]), "gbps": float(r["gbps"]),
            "stamp_ms": float(r["stamp_ms"]) if r.get("stamp_ms") else None,
            # 1149: bandwidth in TB/s. Moved = input read + output written (FP32 value or UINT32 index per row);
            # read = input only. Output writes can stay in L2 past the timed interval, so moved can exceed DRAM rate.
            "tbps": (int(r["input_bytes"]) + 4 * int(r["K"])) / float(r["median_us"]) / 1e6,
            "read_tbps": int(r["input_bytes"]) / float(r["median_us"]) / 1e6,
        })
    for line in header:
        print(line)
    columns = f"{'n':>5} {'TB/s gm':>7} {'p10':>7} {'min':>7} {'read p50':>9} {'>=1.5':>9}"
    print(f"# bandwidth in TB/s: moved = input read + output written (4 bytes per row); read = input only. "
          f"Target {TARGET_TBPS} TB/s moved.")

    # The grid the user reads first: KR moved TB/s by row length and input size, then R by input size.
    def nearest_target(n):
        return min((256 << 10, 4 << 20, 64 << 20, 512 << 20), key=lambda t: abs(t - n))
    print("\n0. KR moved TB/s by row length R (rows) and input size (columns); R by input size")
    for kind, op in sorted({(r["kind"], r["op"]) for r in rows if r["family"] == "KR"}):
        grid = collections.defaultdict(dict)
        for r in rows:
            if r["family"] == "KR" and r["op"] == op:
                grid[r["R"]][nearest_target(r["bytes"])] = r["tbps"]
        print(f"   KR {kind} {op}: {'R':>9} {'256K':>6} {'4M':>6} {'64M':>6} {'512M':>6}")
        for rr in sorted(grid):
            cells = "".join(f"{grid[rr][t]:7.2f}" if t in grid[rr] else "      -"
                            for t in (256 << 10, 4 << 20, 64 << 20, 512 << 20))
            print(f"   {'':>{len(kind) + len(op) + 4}} {rr:>9}{cells}")
    for kind, op in sorted({(r["kind"], r["op"]) for r in rows if r["family"] == "R"}):
        items = sorted((r for r in rows if r["family"] == "R" and r["op"] == op), key=lambda r: r["bytes"])
        print(f"   R {kind} {op} (input size: TB/s):")
        for i in range(0, len(items), 8):
            print("     " + "  ".join(f"{human(r['bytes'])}B:{r['tbps']:.2f}" for r in items[i:i + 8]))

    large = args.large_mib * (1 << 20)
    for title, subset in (("all cells", rows), (f"cells >= {args.large_mib:g} MiB", [r for r in rows if r["bytes"] >= large])):
        print(f"\n1. Moved TB/s by family, kind, operation and selected pass ({title})")
        print(f"{'family':6} {'kind':5} {'op':6} {'path':28} {columns}")
        groups = collections.defaultdict(list)
        for r in subset:
            groups[(r["family"], r["kind"], r["op"], r["passes"])].append(r)
        for key, items in sorted(groups.items()):
            print(f"{key[0]:6} {key[1]:5} {key[2]:6} {key[3][:28]:28} {summary(items)}")

    print(f"\n2. Moved TB/s by row length (KR: R) or input size (R), all operations of a kind, cells >= "
          f"{args.large_mib:g} MiB")
    print(f"{'family':6} {'kind':5} {'bucket':>10} {columns}")
    groups = collections.defaultdict(list)
    for r in rows:
        if r["bytes"] < large:
            continue
        key_value = r["R"] if r["family"] == "KR" else r["bytes"]
        groups[(r["family"], r["kind"], bucket(key_value))].append(r)
    for (family, kind, low), items in sorted(groups.items()):
        label = f"R {human(low)}+" if family == "KR" else f"{human(low)}B+"
        print(f"{family:6} {kind:5} {label:>10} {summary(items)}")

    print("\n   KR by output count K (cells >= the large size)")
    print(f"{'kind':5} {'K':>10} {columns}")
    groups = collections.defaultdict(list)
    for r in rows:
        if r["family"] == "KR" and r["bytes"] >= large:
            groups[(r["kind"], bucket(r["K"]))].append(r)
    for (kind, low), items in sorted(groups.items()):
        print(f"{kind:5} {human(low) + '+':>10} {summary(items)}")

    print(f"\n3. Slowest {args.worst} cells >= {args.large_mib:g} MiB (moved TB/s; us over the time at "
          f"{TARGET_TBPS} TB/s)")
    for r in sorted((r for r in rows if r["bytes"] >= large), key=lambda r: r["tbps"])[: args.worst]:
        at_target = (r["bytes"] + 4 * r["K"]) / (TARGET_TBPS * 1e6)
        print(f"  {r['family']:2} {r['kind']:5} {r['op']:6} K={r['K']} R={r['R']} ({human(r['bytes'])}B) "
              f"{r['passes']}: {r['tbps']:.2f} TB/s, {r['median']:.1f} us (+{r['median'] - at_target:.1f} us)")
    if unsupported:
        print(f"\n   {len(unsupported)} geometries refused by the production stamp:")
        for line in unsupported[:20]:
            print("  " + line)

    stamps = [r["stamp_ms"] for r in rows if r["stamp_ms"] is not None]
    if stamps:
        print(f"\n   stamp time (includes R / KR autotuning on a cache miss): median {quantile(stamps, 0.5):.1f} ms, "
              f"p90 {quantile(stamps, 0.9):.1f} ms, max {max(stamps):.1f} ms, total {sum(stamps) / 1000:.1f} s")

    if args.baseline:
        _, _, old_raw = read(args.baseline)
        key = lambda r: (r["family"], r["kind"], r["operation"], r["dtype"], r["K"], r["R"])
        old = {key(r): float(r["median_us"]) for r in old_raw}
        groups = collections.defaultdict(list)
        changes = []
        for r in raw:
            k = key(r)
            if k in old and old[k] > 0:
                change = float(r["median_us"]) / old[k]
                groups[(r["family"], r["kind"], r["operation"])].append(change)
                changes.append((change, k, old[k], float(r["median_us"]), r["passes"]))
        print(f"\n4. Against the baseline {args.baseline}: median / baseline median ({len(changes)} shared cells)")
        print(f"{'family':6} {'kind':5} {'op':6} {'n':>5} {'gm':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'faster':>7} {'slower':>7}")
        for (family, kind, op), items in sorted(groups.items()):
            print(f"{family:6} {kind:5} {op:6} {len(items):5d} {geomean(items):7.3f} {quantile(items, 0.1):7.3f} "
                  f"{quantile(items, 0.5):7.3f} {quantile(items, 0.9):7.3f} "
                  f"{100 * sum(x < 0.95 for x in items) / len(items):6.0f}% {100 * sum(x > 1.05 for x in items) / len(items):6.0f}%")
        print(f"\n   Largest regressions (> 1.05x)")
        for change, k, before, after, passes in sorted(changes, reverse=True)[: args.worst]:
            if change <= 1.05:
                break
            print(f"  {k[0]:2} {k[1]:5} {k[2]:6} K={k[4]} R={k[5]}: {before:.1f} -> {after:.1f} us ({change:.2f}x) {passes}")

    if candidate_raw:
        cells = collections.defaultdict(list)
        for r in candidate_raw:
            cells[(r["family"], r["kind"], r["operation"], r["dtype"], int(r["K"]), int(r["R"]))].append(
                (float(r["median_us"]), r["passes"], int(r["rank"])))
        production = {(r["family"], r["kind"], r["operation"], r["dtype"], int(r["K"]), int(r["R"])):
                      (float(r["median_us"]), r["passes"]) for r in raw}
        print(f"\n5. Retained candidates ({len(cells)} cells)")
        print(f"{'family':6} {'kind':5} {'bucket':>10} {'n':>4}  fastest shape family (share)            "
              f"{'prod/best gm':>12} {'p90':>6} {'best TB/s gm':>12}")
        groups = collections.defaultdict(list)
        for key, items in cells.items():
            bucket_value = key[5] if key[0] == "KR" else key[5]
            groups[(key[0], key[1], bucket(bucket_value))].append(key)
        for (family, kind, low), keys in sorted(groups.items()):
            winners = collections.Counter()
            ratios = []
            best_rates = []
            for key in keys:
                best = min(cells[key])
                best_rates.append((key[4] * key[5] + 4 * key[4]) / best[0] / 1e6)
                winners[best[1][0]] += 1  # p / g / s
                if key in production:
                    ratios.append(production[key][0] / best[0])
            share = " ".join(f"{name}:{100 * count / len(keys):.0f}%" for name, count in winners.most_common())
            label = f"R {human(low)}+" if family == "KR" else f"{human(low)}B+"
            print(f"{family:6} {kind:5} {label:>10} {len(keys):4d}  {share:38} {geomean(ratios):12.3f} "
                  f"{quantile(ratios, 0.9):6.3f} {geomean(best_rates):12.2f}")
        print("\n   Regret of timing only ranks 1..N (best within the prefix / best overall)")
        for depth in (1, 2, 3, 6, 12, 24):
            regrets = collections.defaultdict(list)
            for key, items in cells.items():
                best = min(t for t, _, _ in items)
                prefix = [t for t, _, rank in items if rank <= depth]
                if prefix:
                    regrets[key[1]].append(min(prefix) / best)
            print(f"  N={depth:2d}: " + "  ".join(f"{kind} gm {geomean(v):.4f} p99 {quantile(v, 0.99):.4f} "
                                                 f"max {max(v):.4f}" for kind, v in sorted(regrets.items())))


if __name__ == "__main__":
    main()
