#!/usr/bin/env python3
"""Dense production audit report (1174): what production stamps achieve, by family, kind and size, in TB/s.

Reads `thor_dense_production_audit` output (one row per cell: a production stamp timed cold on rotating input and
output views). Bandwidth (moved bytes / time, TB/s) is the measure throughout (1177).

  1. moved TB/s by family, kind and input size: p50, p10, min, and the share of cells at the target
  2. what production runs: the pass signatures (kernel families, digits stripped) per family, with their TB/s
  3. the slowest large cells by TB/s, the cells to look at first
  4. stamp wall time by size (selection, tuning on a cache miss, allocation)
  5. measurement: sample spread ((max - min) / median) by size
  6. with --copy only: each cell against a device copy of the same moved bytes, for cells near the launch floor where
     TB/s alone does not say how far from achievable they are

With --compare=OTHER (another run, same seed and settings) section 7 reports both runs' TB/s and the per-cell TB/s
ratio: the before/after view for a conversion step, or, with another dtype, the two dtypes at equal input bytes.

Standard library only.
"""

import argparse
import csv
import gzip
import re
import sys

SIZES = (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
         (">=128M", 128 * 2**20, float("inf")))


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(header, cells, failed): header = dict of '# dense_production_audit' and '# end' fields; cells = list of row dicts
    with numeric fields converted; failed = list of '# failed' lines."""
    header, failed, lines = {}, [], []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("# dense_production_audit") or line.startswith("# end"):
                for token in re.findall(r'(\w+)=("[^"]*"|\S+)', line):
                    header[token[0]] = token[1].strip('"')
            elif line.startswith("# failed,"):
                failed.append(line.rstrip("\n")[len("# failed,"):])
            elif not line.startswith("#"):
                lines.append(line)
    cells = []
    for row in csv.DictReader(lines):
        for key in ("input_bytes", "moved_bytes", "outer", "R", "K", "pass_count", "launches_per_sample", "cell_id"):
            row[key] = int(row[key])
        for key in ("stamp_ms", "probe_us", "median_us", "copy_us", "tbps", "of_copy"):
            row[key] = float(row[key])
        samples = [float(v) for k, v in row.items() if k.startswith("sample") and k.endswith("_us") and v]
        row["spread"] = (max(samples) - min(samples)) / sorted(samples)[len(samples) // 2] if samples else 0.0
        cells.append(row)
    if not cells:
        detail = f"; {len(failed)} failed, first: {failed[0]}" if failed else " (see the audit's stderr log)"
        raise ValueError(f"{path}: no timed cells{detail}")
    return header, cells, failed


def quantile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def size_bucket(input_bytes):
    for name, low, high in SIZES:
        if low <= input_bytes < high:
            return name
    return SIZES[-1][0]


def signature(passes):
    """Pass families with configuration digits stripped: 'kparallel_p16+direct:tiled' -> 'kparallel_p+direct:tiled'."""
    return re.sub(r"\d+", "", passes)


def table(title, columns, rows, out):
    print(title, file=out)
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c)) for i, c in enumerate(columns)]
    print("  " + "  ".join(str(c).rjust(w) for c, w in zip(columns, widths)), file=out)
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w) for v, w in zip(r, widths)), file=out)
    print(file=out)


def pct(x):
    return f"{100.0 * x:.0f}%"


def _families(cells):
    return sorted({c["family"] for c in cells}, key=lambda f: ("R", "KR", "RK").index(f) if f in ("R", "KR", "RK")
                  else 9)


def report(header, cells, failed, args, out=None):
    out = out or sys.stdout
    print(f"# dtype={header.get('dtype')} device=\"{header.get('device')}\" sm_count={header.get('sm_count')} "
          f"cells={len(cells)} failed={len(failed)} copy start/end TB/s {header.get('copy_bandwidth_tbps')} / "
          f"{header.get('copy_bandwidth_tbps_end')}", file=out)
    print(file=out)
    families = _families(cells)
    kinds = sorted({c["kind"] for c in cells})

    rows = []
    for family in families:
        for kind in kinds:
            for name, _, _ in SIZES:
                group = [c for c in cells if c["family"] == family and c["kind"] == kind
                         and size_bucket(c["input_bytes"]) == name]
                if not group:
                    continue
                tbps = [c["tbps"] for c in group]
                rows.append([family, kind, name, len(group), f"{quantile(tbps, 0.5):.3f}", f"{quantile(tbps, 0.1):.3f}",
                             f"{min(tbps):.3f}", pct(sum(t >= args.target for t in tbps) / len(group))])
    table(f"1. Moved TB/s by family, kind and input size (target {args.target:g} TB/s)",
          ["family", "kind", "input", "n", "p50", "p10", "min", ">=target"], rows, out)

    rows = []
    for family in families:
        by_signature = {}
        for c in cells:
            if c["family"] == family:
                by_signature.setdefault(signature(c["passes"]), []).append(c)
        for sig, group in sorted(by_signature.items(), key=lambda item: -len(item[1])):
            tbps = [c["tbps"] for c in group]
            big = [c["tbps"] for c in group if c["input_bytes"] >= args.large_mib * 2**20]
            rows.append([family, sig, len(group), f"{quantile(tbps, 0.5):.3f}",
                         f"{quantile(big, 0.5):.3f}" if big else "-", f"{quantile(big, 0.1):.3f}" if big else "-"])
    table(f"2. What production runs: pass signatures (digits stripped) per family, TB/s (all cells; cells of at least "
          f"{args.large_mib:g} MiB)", ["family", "passes", "n", "p50", "large p50", "large p10"], rows, out)

    large = [c for c in cells if c["input_bytes"] >= args.large_mib * 2**20]
    large.sort(key=lambda c: c["tbps"])
    rows = [[c["cell_id"], c["family"], c["kind"], c["operation"], c["outer"], c["R"], c["K"],
             f"{c['input_bytes'] / 2**20:.1f}", f"{c['tbps']:.3f}", c["passes"]]
            for c in large[:args.worst]]
    table(f"3. Slowest cells of at least {args.large_mib:g} MiB (TB/s)",
          ["cell", "family", "kind", "op", "outer", "R", "K", "MiB", "TB/s", "passes"], rows, out)

    rows = []
    for name, _, _ in SIZES:
        group = [c["stamp_ms"] for c in cells if size_bucket(c["input_bytes"]) == name]
        if group:
            rows.append([name, len(group), f"{quantile(group, 0.5):.2f}", f"{quantile(group, 0.9):.2f}",
                         f"{max(group):.2f}"])
    table("4. Stamp wall time by input size (ms)", ["input", "n", "p50", "p90", "max"], rows, out)

    rows = []
    for name, _, _ in SIZES:
        group = [c["spread"] for c in cells if size_bucket(c["input_bytes"]) == name]
        if group:
            rows.append([name, len(group), pct(quantile(group, 0.5)), pct(quantile(group, 0.9)), pct(max(group))])
    table("5. Measurement: sample spread (max - min) / median", ["input", "n", "p50", "p90", "max"], rows, out)

    if args.copy:
        rows = []
        for family in families:
            for kind in kinds:
                for name, _, _ in SIZES:
                    group = [c["of_copy"] for c in cells if c["family"] == family and c["kind"] == kind
                             and size_bucket(c["input_bytes"]) == name]
                    if group:
                        rows.append([family, kind, name, len(group), f"{quantile(group, 0.5):.2f}",
                                     f"{quantile(group, 0.1):.2f}", f"{min(group):.2f}"])
        table("6. (--copy) Against a device copy of the same moved bytes (1.00 = copy time), for cells near the "
              "launch floor", ["family", "kind", "input", "n", "p50", "p10", "min"], rows, out)

    if failed:
        print(f"   failed cells ({len(failed)}), first: {failed[:3]}", file=out)
        print(file=out)


def compare(cells, other_cells, out=None):
    out = out or sys.stdout
    mine = {c["cell_id"]: c for c in cells}
    def same(c, o):
        if c["dtype"] != o["dtype"]:
            # Another dtype, same seed: the same cell draws the same family, operation and input bytes (element counts
            # differ by the element size), so cells compare at equal bytes.
            return (c["family"], c["kind"], c["operation"]) == (o["family"], o["kind"], o["operation"])
        return (c["outer"], c["R"], c["K"]) == (o["outer"], o["R"], o["K"])
    shared = [(mine[o["cell_id"]], o) for o in other_cells if o["cell_id"] in mine and same(mine[o["cell_id"]], o)]
    rows = []
    for family in _families(cells):
        for kind in sorted({c["kind"] for c in cells}):
            for name, _, _ in SIZES:
                pairs = [(c, o) for c, o in shared if c["family"] == family and c["kind"] == kind
                         and size_bucket(c["input_bytes"]) == name]
                if not pairs:
                    continue
                ratios = [c["tbps"] / o["tbps"] for c, o in pairs]
                rows.append([family, kind, name, len(pairs), f"{quantile([c['tbps'] for c, _ in pairs], 0.5):.3f}",
                             f"{quantile([o['tbps'] for _, o in pairs], 0.5):.3f}", f"{quantile(ratios, 0.1):.3f}",
                             f"{quantile(ratios, 0.5):.3f}", f"{min(ratios):.3f}"])
    table(f"7. This run against the other ({len(shared)} shared cells): TB/s p50 of each, and per-cell TB/s ratio "
          f"this / other (> 1: this run is faster)",
          ["family", "kind", "input", "n", "this p50", "other p50", "ratio p10", "ratio p50", "ratio min"], rows, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--compare", help="another audit of the same cells (same seed and settings)")
    parser.add_argument("--target", type=float, default=1.5, help="TB/s target (default 1.5)")
    parser.add_argument("--large-mib", type=float, default=16.0)
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--copy", action="store_true",
                        help="also report each cell against a device copy of the same bytes (section 6)")
    args = parser.parse_args(argv)
    header, cells, failed = load(args.csv)
    report(header, cells, failed, args)
    if args.compare:
        _, other, _ = load(args.compare)
        compare(cells, other)
    return 0


if __name__ == "__main__":
    sys.exit(main())
