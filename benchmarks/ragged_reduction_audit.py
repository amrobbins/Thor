#!/usr/bin/env python3
"""Ragged reduction audit report (1263): what production segmented-reduction stamps achieve, by distribution of segment
lengths, D, mean segment length and size, in TB/s of moved bytes (active input + offsets + outputs).

Reads `thor_ragged_reduction_audit` output: one row per cell, every shape timed under every distribution of segment
lengths. The gate of plans/Thor_Ragged_Reduction_Plan_2026-11-06.md is printed first: median TB/s and the distribution
spread at or above --large-mib, and the cells whose outputs failed verification.

  1. moved TB/s by kind and size: p50, p10, min, and the share of cells at the target
  2. moved TB/s by distribution and size
  3. the distribution spread: per shape, the fastest distribution's TB/s over the slowest's, overall, by D and by mean
     segment length, and the shapes with the widest spread
  4. moved TB/s by D (trailing values per row) at or above --large-mib
  5. moved TB/s by mean segment length at or above --large-mib, with the share of moved bytes that is input
  6. what production runs: the pass signatures (digits stripped), with their TB/s
  7. the slowest cells at or above --large-mib
  8. stamp wall time by size
  9. measurement: sample spread ((max - min) / median) by size
 10. with --copy only: each cell against a device copy of the same moved bytes

With --compare=OTHER (another run, same seed and settings) section 11 reports both runs' TB/s and the per-cell ratio, by
kind and size and by distribution: the before / after view of a phase.

1265: a --backend=ragged_kr run times each cell once per candidate shape (the `passes` column names it). Every section
and the gate use each cell's best candidate; section 12 lists the candidates: TB/s, how often each is the best and its
TB/s against the best per cell.

1272: section 13 fits each candidate's uniform cells (balanced CTAs) to median_us = fixed + rows x per row + segments x
per end: the fixed start-up time, the rows' marginal TB/s, an end's cost in rows and the extra weight that matches it
(RaggedKRShape::end_extra).

Standard library only.
"""

import argparse
import csv
import gzip
import re
import sys

SIZES = (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
         (">=128M", 128 * 2**20, float("inf")))
D_BUCKETS = (("1", 1, 2), ("2-16", 2, 17), ("17-255", 17, 256), ("256-4095", 256, 4096), (">=4096", 4096, float("inf")))
LENGTH_BUCKETS = (("1-2", 0, 2), ("2-16", 2, 16), ("16-256", 16, 256), ("256-4K", 256, 4096),
                  ("4K-64K", 4096, 65536), (">=64K", 65536, float("inf")))
DISTRIBUTION_ORDER = ("uniform", "random", "pareto", "half_empty", "one_giant")
INTS = ("cell_id", "shape_id", "D", "rows", "capacity_rows", "segments", "max_len", "empty_segments", "input_bytes",
        "offset_bytes", "output_bytes", "moved_bytes", "pass_count", "launches_per_sample", "verified")
FLOATS = ("mean_len", "stamp_ms", "probe_us", "median_us", "copy_us", "tbps", "of_copy")


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(header, cells, failed, verify_failed): header = dict of the '# ragged_reduction_audit' and '# end' fields; cells =
    row dicts with numeric fields converted; failed / verify_failed = the '# failed' / '# verify_failed' lines."""
    header, failed, verify_failed, lines = {"candidates": {}}, [], [], []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("# ragged_reduction_audit") or line.startswith("# end"):
                for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', line):
                    header[key] = value.strip('"')
            elif line.startswith("# ragged_kr_candidate "):
                name, *fields = line.split()[2:]
                header["candidates"][name] = dict(f.split("=", 1) for f in fields)
            elif line.startswith("# failed,"):
                failed.append(line.rstrip("\n")[len("# failed,"):])
            elif line.startswith("# verify_failed,"):
                verify_failed.append(line.rstrip("\n")[len("# verify_failed,"):])
            elif not line.startswith("#"):
                lines.append(line)
    cells = []
    for row in csv.DictReader(lines):
        for key in INTS:
            row[key] = int(row[key])
        for key in FLOATS:
            row[key] = float(row[key])
        samples = [float(v) for k, v in row.items() if k.startswith("sample") and k.endswith("_us") and v]
        row["spread"] = (max(samples) - min(samples)) / sorted(samples)[len(samples) // 2] if samples else 0.0
        cells.append(row)
    if not cells:
        detail = f"; {len(failed)} failed, first: {failed[0]}" if failed else " (see the audit's stderr log)"
        raise ValueError(f"{path}: no timed cells{detail}")
    return header, cells, failed, verify_failed


def cell_key(c):
    return (c["shape_id"], c["distribution"])


def best_candidates(cells):
    """(best, by_key): each (shape, distribution)'s fastest row, and every row of each, keyed by (shape, distribution).
    A production run has one row per key, so best == cells."""
    by_key = {}
    for c in cells:
        by_key.setdefault(cell_key(c), []).append(c)
    best = [max(group, key=lambda c: c["tbps"]) for group in by_key.values()]
    return sorted(best, key=lambda c: c["cell_id"]), by_key


def candidates(by_key, large_bytes, out=None):
    """Section 12: per candidate (the passes of a multi-row key), TB/s, wins and the ratio to the best of each key."""
    out = out or sys.stdout
    stats = {}
    for group in by_key.values():
        if len(group) < 2:
            continue
        best = max(group, key=lambda c: c["tbps"])
        for c in group:
            entry = stats.setdefault(c["passes"], {"tbps": [], "large": [], "ratio": [], "wins": 0})
            entry["tbps"].append(c["tbps"])
            if c["moved_bytes"] >= large_bytes:
                entry["large"].append(c["tbps"])
            entry["ratio"].append(c["tbps"] / best["tbps"])
            entry["wins"] += c is best
    if not stats:
        return
    rows = [[name, len(e["tbps"]), e["wins"], f3(quantile(e["tbps"], 0.5)),
             f3(quantile(e["large"], 0.5)) if e["large"] else "-", f3(quantile(e["large"], 0.1)) if e["large"] else "-",
             f3(quantile(e["ratio"], 0.1)), f3(quantile(e["ratio"], 0.5)), f3(min(e["ratio"]))]
            for name, e in sorted(stats.items(), key=lambda item: -item[1]["wins"])]
    table(f"12. Candidates (TB/s; large: moved >= {large_bytes / 2**20:g} MiB; ratio: TB/s / the cell's best)",
          ["candidate", "n", "wins", "p50", "large p50", "large p10", "ratio p10", "ratio p50", "ratio min"], rows, out)


def solve3(a, b):
    """x with a x = b (3 x 3, Gaussian elimination with partial pivoting); None when singular."""
    m = [row[:] + [v] for row, v in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-30:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][3] / m[i][i] for i in range(3)]


def cost_fit(rows, header, out=None, min_bytes=2**20):
    """Section 13 (1272): per candidate and offset width, least squares over the uniform cells at or above min_bytes of
    median_us = fixed + rows x a + segments x b. 1280: separately for D = 1 and D 2-15, rows counted in bytes (rows x D),
    the extra only at D = 1 (an end's bytes vary with D). 1284: rows x D x the input's element bytes; the D >= 2 class
    is 2-15 (FP8), 2-7 (FP16 / BF16) or 2-3 (FP32). 1285: rows of 16 bytes and more (ragged RK) are the class >=16B."""
    out = out or sys.stdout
    output_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}.get(header.get("value_output", "fp32"), 1)
    element_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}.get(header.get("dtype", "fp8_e4m3"), 1)
    wide_class = f"2-{15 // element_bytes}"
    groups = {}
    for c in rows:
        if c["distribution"] == "uniform" and c["moved_bytes"] >= min_bytes:
            # 1285: rows of 16 bytes and more are ragged RK's, fitted apart.
            d_class = "1" if c["D"] == 1 else (wide_class if c["D"] * element_bytes < 16 else ">=16B")
            groups.setdefault((c["passes"], c["offset_dtype"], d_class), []).append(c)
    table_rows = []
    title_size = f"moved >= {min_bytes / 2**20:g} MiB"
    for (passes, offsets, d_class), cells in sorted(groups.items()):
        if len(cells) < 4:
            continue
        xs = [(1.0, float(c["rows"] * c["D"] * element_bytes), float(c["segments"])) for c in cells]
        ys = [c["median_us"] for c in cells]
        ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
        aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
        fit = solve3(ata, aty)
        if fit is None or fit[1] <= 0:
            continue
        fixed, per_row, per_end = fit
        worst = max(abs(fixed + per_row * x[1] + per_end * x[2] - y) / y for x, y in zip(xs, ys))
        end_bytes = (8 if offsets == "uint64" else 4) + output_bytes
        ratio = per_end / per_row
        extra = f"{ratio - end_bytes:.1f}" if d_class == "1" else "-"
        table_rows.append([passes, offsets, d_class, len(cells), f"{fixed:.2f}", f3(1e-6 / per_row),
                           f"{per_end * 1e6:.1f}", f"{ratio:.1f}", extra, pct(worst)])
    if table_rows:
        table(f"13. Cost fit, uniform cells at {title_size} (us = fixed + row bytes x per byte + segments x per end; "
              "extra: end / byte less the end's bytes, D = 1)", ["candidate", "offsets", "D", "n", "fixed us",
                                                                  "row TB/s", "us / M ends", "end / byte", "extra",
                                                                  "max resid"], table_rows, out)


def quantile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def bucket(value, buckets):
    for name, low, high in buckets:
        if low <= value < high:
            return name
    return buckets[-1][0]


def size_bucket(moved_bytes):
    return bucket(moved_bytes, SIZES)


def signature(passes):
    """Passes with configuration digits stripped: 'vector_narrow_rl8' -> 'vector_narrow_rl'."""
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


def f3(x):
    return f"{x:.3f}"


def distributions(cells):
    found = {c["distribution"] for c in cells}
    return [d for d in DISTRIBUTION_ORDER if d in found] + sorted(found - set(DISTRIBUTION_ORDER))


def shape_spreads(cells, large_bytes):
    """Per shape timed under at least two distributions, at or above large_bytes: (shape cells, fastest / slowest)."""
    by_shape = {}
    for c in cells:
        if c["moved_bytes"] >= large_bytes:
            by_shape.setdefault(c["shape_id"], []).append(c)
    out = []
    for group in by_shape.values():
        if len({c["distribution"] for c in group}) >= 2:
            fast = max(group, key=lambda c: c["tbps"])
            slow = min(group, key=lambda c: c["tbps"])
            out.append((group, fast, slow, fast["tbps"] / slow["tbps"]))
    return out


def tbps_rows(groups, target):
    rows = []
    for labels, group in groups:
        if not group:
            continue
        tbps = [c["tbps"] for c in group]
        rows.append([*labels, len(group), f3(quantile(tbps, 0.5)), f3(quantile(tbps, 0.1)), f3(min(tbps)),
                     pct(sum(t >= target for t in tbps) / len(group))])
    return rows


def report(header, cells, failed, verify_failed, args, out=None):
    out = out or sys.stdout
    large = args.large_mib * 2**20
    print(f"# dtype={header.get('dtype')} device=\"{header.get('device')}\" sm_count={header.get('sm_count')} "
          f"backend={header.get('backend', 'production')} skipped={header.get('skipped', 0)} "
          f"cells={len(cells)} failed={len(failed)} verify_failed={len(verify_failed)} stamp={header.get('stamp')} "
          f"copy start/end TB/s {header.get('copy_bandwidth_tbps')} / {header.get('copy_bandwidth_tbps_end')}",
          file=out)
    big = [c for c in cells if c["moved_bytes"] >= large]
    spreads = shape_spreads(cells, large)
    ratios = [s[3] for s in spreads]
    unverified = sum(c["verified"] == 0 for c in cells)
    print(f"# gate (moved >= {args.large_mib:g} MiB): TB/s p50 {f3(quantile([c['tbps'] for c in big], 0.5))} "
          f"p10 {f3(quantile([c['tbps'] for c in big], 0.1))} (target {args.target:g}); distribution spread p50 "
          f"{f3(quantile(ratios, 0.5))} p90 {f3(quantile(ratios, 0.9))} max {f3(max(ratios)) if ratios else 'nan'} "
          f"over {len(spreads)} shapes (gate {args.spread_gate:g}); cells failing verification {unverified}", file=out)
    # 1275: the median clause is against the copy curve (plan, phase 2: at least 0.85 of copy), and a ragged_kr run
    # may time the main kernel alone (--ragged-kr-time=main) with the finisher as a fixed tax.
    print(f"# gate of copy (moved >= {args.large_mib:g} MiB): p50 {f3(quantile([c['of_copy'] for c in big], 0.5))} "
          f"p10 {f3(quantile([c['of_copy'] for c in big], 0.1))} (gate {args.copy_gate:g}); timed "
          f"{header.get('ragged_kr_time', 'full')}", file=out)
    print(file=out)

    kinds = sorted({c["kind"] for c in cells})
    groups = [((kind, name), [c for c in cells if c["kind"] == kind and size_bucket(c["moved_bytes"]) == name])
              for kind in kinds for name, _, _ in SIZES]
    table(f"1. Moved TB/s by kind and size (target {args.target:g} TB/s)",
          ["kind", "moved", "n", "p50", "p10", "min", ">=target"], tbps_rows(groups, args.target), out)

    groups = [((d, name), [c for c in cells if c["distribution"] == d and size_bucket(c["moved_bytes"]) == name])
              for d in distributions(cells) for name, _, _ in SIZES]
    table("2. Moved TB/s by distribution and size", ["distribution", "moved", "n", "p50", "p10", "min", ">=target"],
          tbps_rows(groups, args.target), out)

    rows = []
    for label, keep in [("all", lambda s: True)] + [
            (f"D {name}", (lambda low, high: lambda s: low <= s[0][0]["D"] < high)(low, high))
            for name, low, high in D_BUCKETS] + [
            (f"len {name}", (lambda low, high: lambda s: low <= s[0][0]["mean_len"] < high)(low, high))
            for name, low, high in LENGTH_BUCKETS]:
        r = [s[3] for s in spreads if keep(s)]
        if r:
            rows.append([label, len(r), f3(quantile(r, 0.5)), f3(quantile(r, 0.9)), f3(max(r)),
                         pct(sum(x <= args.spread_gate for x in r) / len(r))])
    table(f"3. Distribution spread per shape (fastest / slowest distribution TB/s, moved >= {args.large_mib:g} MiB)",
          ["shapes", "n", "p50", "p90", "max", "<=gate"], rows, out)
    rows = [[g[0]["shape_id"], g[0]["kind"], g[0]["operation"], g[0]["D"], g[0]["rows"], g[0]["segments"],
             f"{g[0]['moved_bytes'] / 2**20:.1f}", f"{fast['distribution']} {f3(fast['tbps'])}",
             f"{slow['distribution']} {f3(slow['tbps'])}", f3(ratio), slow["passes"]]
            for g, fast, slow, ratio in sorted(spreads, key=lambda s: -s[3])[:args.worst]]
    table("   widest spreads", ["shape", "kind", "op", "D", "rows", "segments", "MiB", "fastest", "slowest", "ratio",
                                "slowest passes"], rows, out)

    groups = [((kind, name), [c for c in big if c["kind"] == kind and bucket(c["D"], D_BUCKETS) == name])
              for kind in kinds for name, _, _ in D_BUCKETS]
    table(f"4. Moved TB/s by D, moved >= {args.large_mib:g} MiB",
          ["kind", "D", "n", "p50", "p10", "min", ">=target"], tbps_rows(groups, args.target), out)

    rows = []
    for name, _, _ in LENGTH_BUCKETS:
        group = [c for c in big if bucket(c["mean_len"], LENGTH_BUCKETS) == name]
        if group:
            tbps = [c["tbps"] for c in group]
            share = sum(c["input_bytes"] for c in group) / sum(c["moved_bytes"] for c in group)
            rows.append([name, len(group), f3(quantile(tbps, 0.5)), f3(quantile(tbps, 0.1)), f3(min(tbps)),
                         pct(share)])
    table(f"5. Moved TB/s by mean segment length, moved >= {args.large_mib:g} MiB",
          ["mean len", "n", "p50", "p10", "min", "input share"], rows, out)

    by_signature = {}
    for c in cells:
        by_signature.setdefault((c["kind"], signature(c["passes"])), []).append(c)
    rows = []
    for (kind, sig), group in sorted(by_signature.items(), key=lambda item: (item[0][0], -len(item[1]))):
        tbps = [c["tbps"] for c in group]
        large_tbps = [c["tbps"] for c in group if c["moved_bytes"] >= large]
        rows.append([kind, sig, len(group), f3(quantile(tbps, 0.5)),
                     f3(quantile(large_tbps, 0.5)) if large_tbps else "-",
                     f3(quantile(large_tbps, 0.1)) if large_tbps else "-"])
    table(f"6. What production runs: pass signatures (digits stripped), TB/s (all cells; moved >= {args.large_mib:g} MiB)",
          ["kind", "passes", "n", "p50", "large p50", "large p10"], rows, out)

    rows = [[c["cell_id"], c["shape_id"], c["distribution"], c["kind"], c["operation"], c["D"], c["rows"],
             c["segments"], c["max_len"], f"{c['moved_bytes'] / 2**20:.1f}", f3(c["tbps"]), c["passes"]]
            for c in sorted(big, key=lambda c: c["tbps"])[:args.worst]]
    table(f"7. Slowest cells, moved >= {args.large_mib:g} MiB (TB/s)",
          ["cell", "shape", "distribution", "kind", "op", "D", "rows", "segments", "max len", "MiB", "TB/s", "passes"],
          rows, out)

    for number, title, key, unit in ((8, "Stamp wall time by size (ms)", "stamp_ms", None),
                                     (9, "Measurement: sample spread (max - min) / median", "spread", "pct")):
        rows = []
        for name, _, _ in SIZES:
            group = [c[key] for c in cells if size_bucket(c["moved_bytes"]) == name]
            if group:
                fmt = pct if unit == "pct" else (lambda x: f"{x:.2f}")
                rows.append([name, len(group), fmt(quantile(group, 0.5)), fmt(quantile(group, 0.9)), fmt(max(group))])
        table(f"{number}. {title}", ["moved", "n", "p50", "p90", "max"], rows, out)

    if args.copy:
        rows = []
        for kind in kinds:
            for name, _, _ in SIZES:
                group = [c["of_copy"] for c in cells if c["kind"] == kind and size_bucket(c["moved_bytes"]) == name]
                if group:
                    rows.append([kind, name, len(group), f"{quantile(group, 0.5):.2f}", f"{quantile(group, 0.1):.2f}",
                                 f"{min(group):.2f}"])
        table("10. (--copy) Against a device copy of the same moved bytes (1.00 = copy time)",
              ["kind", "moved", "n", "p50", "p10", "min"], rows, out)

    if failed:
        print(f"   failed cells ({len(failed)}), first: {failed[:3]}", file=out)
    if verify_failed:
        print(f"   cells failing verification ({len(verify_failed)}), first: {verify_failed[:3]}", file=out)
    if failed or verify_failed:
        print(file=out)


def compare(cells, other_cells, out=None):
    out = out or sys.stdout
    key = cell_key
    mine = {key(c): c for c in cells}
    same = lambda c, o: (c["kind"], c["operation"], c["D"], c["rows"], c["segments"]) == \
        (o["kind"], o["operation"], o["D"], o["rows"], o["segments"])
    shared = [(mine[key(o)], o) for o in other_cells if key(o) in mine and same(mine[key(o)], o)]

    def rows_for(groups):
        rows = []
        for labels, pairs in groups:
            if not pairs:
                continue
            ratios = [c["tbps"] / o["tbps"] for c, o in pairs]
            rows.append([*labels, len(pairs), f3(quantile([c["tbps"] for c, _ in pairs], 0.5)),
                         f3(quantile([o["tbps"] for _, o in pairs], 0.5)), f3(quantile(ratios, 0.1)),
                         f3(quantile(ratios, 0.5)), f3(min(ratios))])
        return rows

    columns = ["this p50", "other p50", "ratio p10", "ratio p50", "ratio min"]
    kinds = sorted({c["kind"] for c in cells})
    groups = [((kind, name), [(c, o) for c, o in shared if c["kind"] == kind and size_bucket(c["moved_bytes"]) == name])
              for kind in kinds for name, _, _ in SIZES]
    table(f"11. This run against the other ({len(shared)} shared cells): TB/s p50 of each and the per-cell TB/s ratio "
          f"this / other (> 1: this run is faster), by kind and size", ["kind", "moved", "n", *columns],
          rows_for(groups), out)
    groups = [((d,), [(c, o) for c, o in shared if c["distribution"] == d]) for d in distributions(cells)]
    table("    by distribution", ["distribution", "n", *columns], rows_for(groups), out)
    # 1271: the gate's "no cell at >= 1 MiB slower than the other run", cell by cell.
    slower = sorted(((c["tbps"] / o["tbps"], c, o) for c, o in shared if c["moved_bytes"] >= 2**20), key=lambda t: t[0])
    rows = [[c["shape_id"], c["distribution"], c["operation"], c["rows"], c["segments"], f"{c['mean_len']:.4g}",
             f"{c['moved_bytes'] / 2**20:.1f}", f3(c["tbps"]), f3(o["tbps"]), f3(r)]
            for r, c, o in slower if r < 1.0][:20]
    table(f"    cells at >= 1 MiB slower than the other run ({sum(r < 1.0 for r, _, _ in slower)} of {len(slower)}; "
          "slowest 20)", ["shape", "distribution", "op", "rows", "segments", "mean len", "MiB", "this", "other",
                          "ratio"], rows, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--compare", help="another audit of the same cells (same seed and settings)")
    parser.add_argument("--target", type=float, default=1.5, help="TB/s target (default 1.5)")
    parser.add_argument("--copy-gate", type=float, default=0.85,
                        help="median share of the copy curve the plan's phase 2 gate asks (default 0.85, 1275)")
    parser.add_argument("--spread-gate", type=float, default=1.05,
                        help="distribution spread the plan's phase 2 gate allows (default 1.05)")
    parser.add_argument("--large-mib", type=float, default=16.0)
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--copy", action="store_true",
                        help="also report each cell against a device copy of the same bytes (section 10)")
    args = parser.parse_args(argv)
    header, rows, failed, verify_failed = load(args.csv)
    cells, by_key = best_candidates(rows)
    # A candidate failing verification fails its cell, even when another candidate of the cell is faster.
    for c in cells:
        if any(r["verified"] == 0 for r in by_key[cell_key(c)]):
            c["verified"] = 0
    report(header, cells, failed, verify_failed, args)
    candidates(by_key, args.large_mib * 2**20)
    if header.get("backend") == "ragged_kr":
        cost_fit(rows, header, min_bytes=args.large_mib * 2**20)
    if args.compare:
        _, other, _, _ = load(args.compare)
        compare(cells, best_candidates(other)[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
