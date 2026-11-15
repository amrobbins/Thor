#!/usr/bin/env python3
"""Dense copy audit report (1388): what Thor's dense copy of a strided span (copyToDenseAsync / launchCopyToDense,
Utilities/TensorOperations/Copy/StridedCopy.h) achieves per layout class, in TB/s of moved bytes, against the copy
curve and against what copyToDenseAsync ran through 1387 (through 1393 also, for 2D rotations, against
gpuMatrixTranspose, deleted in 1394).

Reads `thor_dense_copy_audit` output (one row per cell). Cases: d dense (one memcpy), r a rotation ([b, a] of a dense
[a, b]), g a payload rotation ([b, a, p] of [a, b, p]), h an attention head permute ([B, S, H, D] of [B, H, S, D]),
k a pitched packing ([O, R, I] rows with row gaps, slab gaps or both), s a rank-1 gather (stride 2 to 33), n a general
permutation of a dense rank-4 or rank-5 tensor. Moved bytes are the source bytes read (distinct sectors for k and s)
plus the destination's. of_copy is the copy curve's time at the moved bytes over the cell's (1.0 at copy speed);
vs_1387 is the 1387 path's time over the cell's (above 1.0 this build is faster).

The gate lines come first, per case at or above --large-mib moved: TB/s p50, of_copy p50 / p10 / min, vs_1387 p50 /
min, with the cells that failed verification or failed to run. Then:

  1. by case and size: TB/s p50, of_copy p50 / p10 / min, vs_1387 p50 / min
  2. by plan (the plan name with the row length stripped): cells, of_copy p50 / min, vs_1387 p50
  3. k by sub-kind and s by stride: of_copy p50 / min, vs_1387 p50 / min
  4. (through 1393: r against gpuMatrixTranspose; 1394 deleted it, and an older run's columns are ignored)
  5. the cells furthest below the copy curve (--slowest)
  6. measurement: sample spread ((max - min) / median) by case

With --compare=OTHER (another run of the same seed and settings), section 7 joins the cells and reports OTHER's time
over this run's by case and size (above 1.0 this run is faster).

Standard library only.
"""

import argparse
import csv
import gzip
import re
import sys

SIZES = (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
         (">=128M", 128 * 2**20, float("inf")))
INTS = ("cell_id", "useful_bytes", "touched_bytes", "moved_bytes", "verified", "launches")
FLOATS = ("probe_us", "median_us", "copy_us", "ref_1387_us", "tbps", "of_copy", "vs_1387")
CASES = ("d", "r", "g", "h", "k", "s", "n")
CASE_NAMES = {"d": "d dense", "r": "r rotation", "g": "g payload rotation", "h": "h head permute",
              "k": "k pitched packing", "s": "s rank-1 gather", "n": "n general permutation"}


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(header, cells, failed, verify_failed) from one audit output."""
    header, failed, verify_failed, lines = {}, [], [], []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("# dense_copy_audit") or line.startswith("# end"):
                for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', line):
                    header[key] = value.strip('"')
            elif line.startswith("# failed,"):
                failed.append(line.rstrip("\n")[len("# failed,"):])
            elif line.startswith("# verify_failed,"):
                verify_failed.append(line.rstrip("\n")[len("# verify_failed,"):])
            elif not line.startswith("#") and line.strip():
                lines.append(line)
    cells = []
    for row in csv.DictReader(lines):
        for key in INTS:
            row[key] = int(row[key])
        for key in FLOATS:
            row[key] = float(row[key])
        samples = [float(v) for k, v in row.items() if k.startswith("sample") and k.endswith("_us") and v]
        row["spread"] = (max(samples) - min(samples)) / row["median_us"] if samples and row["median_us"] else 0.0
        cells.append(row)
    return header, cells, failed, verify_failed


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def fmt(x, digits=3):
    return "-" if x != x else f"{x:.{digits}f}"


def size_bucket(moved):
    for name, low, high in SIZES:
        if low <= moved < high:
            return name
    return SIZES[-1][0]


def plan_kind(plan):
    """The plan name without its row length or tile (1389: rectangular, sized to the span): row_copy_o2_w4,
    transpose_w16_b1, memcpy_w4, packed_gather_o5_w1 (1389b: it has neither), matrix_transpose_w4 (1389e: likewise).
    A 1388 name (transpose_p1_w4_t32_b0, a square tile) groups the same way."""
    return re.sub(r"_t\d+(x\d+)?", "", re.sub(r"_p\d+", "", plan))


def stats(cells):
    of_copy = [c["of_copy"] for c in cells]
    vs = [c["vs_1387"] for c in cells if c["vs_1387"] > 0]
    return (f"cells {len(cells):4d} TB/s p50 {fmt(quantile([c['tbps'] for c in cells], .5))} of_copy p50 "
            f"{fmt(quantile(of_copy, .5))} p10 {fmt(quantile(of_copy, .1))} min {fmt(min(of_copy))}"
            f" vs_1387 p50 {fmt(quantile(vs, .5), 2)} min {fmt(min(vs) if vs else float('nan'), 2)}")


def gate(cells, failed, verify_failed, large_bytes):
    lines = [f"# gate (moved >= {large_bytes / 2**20:g} MiB)"]
    for case in CASES:
        group = [c for c in cells if c["case"] == case and c["moved_bytes"] >= large_bytes]
        if group:
            lines.append(f"  {CASE_NAMES[case]:22s} {stats(group)}")
    unverified = [c for c in cells if not c["verified"]]
    lines.append(f"  unverified cells: {len(unverified)} {[c['cell_id'] for c in unverified][:20]}")
    lines.append(f"  failed cells: {len(failed)}")
    for line in failed[:10]:
        lines.append(f"    {line[:160]}")
    for line in verify_failed[:10]:
        lines.append(f"    verify_failed {line[:160]}")
    return lines


def by_case_size(cells):
    lines = ["", "# 1. by case and size (moved bytes)"]
    for case in CASES:
        for name, low, high in SIZES:
            group = [c for c in cells if c["case"] == case and low <= c["moved_bytes"] < high]
            if group:
                lines.append(f"  {CASE_NAMES[case]:22s} {name:8s} {stats(group)}")
    return lines


def by_plan(cells):
    lines = ["", "# 2. by plan (row length stripped)"]
    for case in CASES:
        kinds = sorted({plan_kind(c["plan"]) for c in cells if c["case"] == case})
        for kind in kinds:
            group = [c for c in cells if c["case"] == case and plan_kind(c["plan"]) == kind]
            of_copy = [c["of_copy"] for c in group]
            vs = [c["vs_1387"] for c in group if c["vs_1387"] > 0]
            lines.append(f"  {CASE_NAMES[case]:22s} {kind:26s} cells {len(group):4d} of_copy p50 "
                         f"{fmt(quantile(of_copy, .5))} min {fmt(min(of_copy))} vs_1387 p50 {fmt(quantile(vs, .5), 2)}")
    return lines


def by_sub(cells):
    lines = ["", "# 3. k by sub-kind, s by stride"]
    for case in ("k", "s"):
        for sub in sorted({c["sub"] for c in cells if c["case"] == case}):
            group = [c for c in cells if c["case"] == case and c["sub"] == sub]
            lines.append(f"  {CASE_NAMES[case]:22s} {sub:10s} {stats(group)}")
    return lines


def slowest(cells, count):
    lines = ["", f"# 5. the {count} cells furthest below the copy curve"]
    for c in sorted(cells, key=lambda c: c["of_copy"])[:count]:
        lines.append(f"  {c['cell_id']:5d} {c['case']} {c['sub'][:9]:9s} dims {c['dims'][:24]:24s} strides "
                     f"{c['strides'][:28]:28s} {c['moved_bytes'] / 2**20:8.2f} MiB {fmt(c['median_us'], 1):>9s} us "
                     f"of_copy {fmt(c['of_copy'])} vs_1387 {fmt(c['vs_1387'], 2)} {c['plan']}")
    return lines


def spread(cells):
    lines = ["", "# 6. measurement: sample spread (max - min) / median"]
    for case in CASES:
        group = [c["spread"] for c in cells if c["case"] == case]
        if group:
            lines.append(f"  {CASE_NAMES[case]:22s} p50 {fmt(quantile(group, .5))} p90 {fmt(quantile(group, .9))} "
                         f"max {fmt(max(group))}")
    return lines


def key(c):
    return (c["cell_id"], c["case"], c["dims"], c["strides"])


def compare(cells, other, revision, other_revision):
    theirs = {key(c): c for c in other}
    joined = [(c, theirs[key(c)]) for c in cells if key(c) in theirs]
    lines = ["", f"# 7. compare: the other run's time (revision {other_revision}) over this run's (revision "
                 f"{revision}), {len(joined)} joined cells"]
    changed = sum(1 for a, b in joined if a["plan"] != b["plan"])
    lines.append(f"  plans that differ between the runs: {changed} of {len(joined)}"
                 + ("  (none: is this the rebuilt library?)" if joined and not changed else ""))
    for case in CASES:
        for name, _, _ in SIZES:
            pairs = [(a, b) for a, b in joined if a["case"] == case and size_bucket(a["moved_bytes"]) == name]
            if pairs:
                ratio = [b["median_us"] / a["median_us"] for a, b in pairs]
                lines.append(f"  {CASE_NAMES[case]:22s} {name:8s} cells {len(pairs):4d} p50 {fmt(quantile(ratio, .5))}"
                             f" min {fmt(min(ratio))} max {fmt(max(ratio))}")
    return lines


def report(options):
    header, cells, failed, verify_failed = load(options.path)
    lines = [f"# dense_copy_audit {options.path}: revision {header.get('version', '?')}, dtype "
             f"{header.get('dtype', '?')}, {len(cells)} cells, device {header.get('device', '?')}, copy "
             f"{header.get('copy_bandwidth_tbps', '?')} TB/s"]
    lines += gate(cells, failed, verify_failed, options.large_mib * 2**20)
    lines += by_case_size(cells) + by_plan(cells) + by_sub(cells)
    lines += slowest(cells, options.slowest) + spread(cells)
    if options.compare:
        other_header, other, _, _ = load(options.compare)
        lines += compare(cells, other, header.get("version", "?"), other_header.get("version", "?"))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--large-mib", type=float, default=16.0)
    parser.add_argument("--slowest", type=int, default=15)
    options = parser.parse_args(argv)
    print("\n".join(report(options)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
