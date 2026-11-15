#!/usr/bin/env python3
"""View reduction audit report (1335): what production CubReduction stamps of non-dense views achieve, in TB/s of
touched bytes, against the copy curve and against each cell's reference (plans/Thor_View_Reduction_Plan_2026-11-08.md).

Reads `thor_view_reduction_audit` output (one row per cell). Cases: A (a compact permutation, every axis reduced),
B (a compact permutation, reduced axes physically trailing), C (a rank-1 strided view; s = 0 is a broadcast view), and
the phase 5 measurement (1375): D (permutation-aware tiled with natural output, on the Thor tiled kernels) and E (a
split physical reduction; production throws, so an unsupported cell times the dense stamp of its physical tensor,
reference `view_unsupported`). The reference is the dense stamp of the same bytes (A / B / D / E) or the gather
reference kernel (C, s >= 2); of_reference is the reference's time over the cell's, so 1.0 is at the reference and below
1.0 is slower.

The gate lines come first, per case at or above --large-mib touched: TB/s p50 / p10, of_copy p50, of_reference p50 /
p10 / min, and the cells whose outputs failed verification or that failed to run. Then:

  1. by case and size: TB/s p50 / p10, of_copy p50, of_reference p50 / p10 / min
  2. C by stride: pitch, cells, of_reference p50 / min, touched and useful TB/s p50, the time of the s = 0 cells
  3. of_reference by operation, per case
  4. what production runs: the pass signatures per case (digits stripped)
  5. the cells furthest below their reference (--worst)
  6. stamp wall time by case
  7. measurement: sample spread ((max - min) / median) by case

With --compare=OTHER (another run of the same seed and settings, e.g. the phase 0 baseline) section 8 joins the cells
and reports OTHER's time over this run's (above 1.0: this run is faster) by case and size and, for C, by stride, with
the legacy clause: the cells at or above --legacy-mib touched where this run is slower than OTHER by more than
--legacy-tolerance.

Section 9, phase 5 (D, E; 1375): D's tiled time over the dense stamp's by size and by tiled family (above 1.0 the dense
planner is faster), with the cells where either side wins by more than --legacy-tolerance; E's unsupported cells and the
dense stamp's TB/s and of_copy by size.

Section 10, phase 5 (F, G, H; 1386): the three view classes still on the Thor tiled kernels. F a compact permutation
with a rotated [inner, outer] output, G VIEW-DIRECT-2B ([A, R, B, P] read as [B, A, P]), H VIEW-PITCHED-TILED (rows /
slabs with gaps). The reference is the dense stamp of the physical bytes in physical output order (F, G) or of the
packed tensor without the gaps (H). Per case: tiled / dense by size, kernel family, outputs and reduction size, with
the cells 2x and 10x slower than the reference; then the slowest cells (--slowest).

Section 11, phase 7 (H; 1391): H by gap sub-class and by how it ran (sliced, packed, 1392 pitched_complete), view /
dense. Section 12, phase 7 C (H; 1392): the in-place read (RuntimeK PacketComplete at the view's pitches) against the
split, from the alternative columns an H cell carries (split_us when the main read it in place, pitched_w<W>_us at
each forced W): by tile bytes (under 256 or not) and by the bytes in flight per SM at the reducer's W (1392a: the most
warps per item that fit one wave of 32 warps per SM, at least four rows per warp; in place from 4 KiB per SM against a
sliced split, 2 KiB against a packed one), the split's time over the best W's (above 1.0 the in-place read is
faster), the time the reducer's choice takes over the best of all five, how often each W is best, and the cells where
the choice is slower than the best by more than --legacy-tolerance.

Standard library only.
"""

import argparse
import csv
import gzip
import re
import sys

SIZES = (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
         (">=128M", 128 * 2**20, float("inf")))
INTS = ("cell_id", "count", "outputs", "stride", "pitch_bytes", "span_bytes", "useful_bytes", "touched_bytes",
        "verified", "launches")
FLOATS = ("stamp_ms", "max_rel_err", "probe_us", "median_us", "reference_us", "copy_us", "tbps", "of_copy",
          "of_reference")
CASES = ("a", "b", "c", "d", "e", "f", "g", "h")
CASE_NAMES = {"a": "A full permutation", "b": "B trailing permutation", "c": "C strided rank-1",
              "d": "D tiled natural output", "e": "E split physical", "f": "F rotated output",
              "g": "G payload transpose", "h": "H pitched"}
# 1386 (section 10): buckets of outputs and reduction size (count)
OUTPUTS = (("<=256", 0, 257), ("257-4K", 257, 4097), ("4K-64K", 4097, 65537), (">64K", 65537, float("inf")))
REDUCTIONS = (("R<64", 0, 64), ("R 64-1K", 64, 1024), ("R 1K-16K", 1024, 16384), ("R>=16K", 16384, float("inf")))


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(header, cells, failed, verify_failed) from one audit output."""
    header, failed, verify_failed, lines = {}, [], [], []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("# view_reduction_audit") or line.startswith("# end"):
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
        row["useful_tbps"] = row["useful_bytes"] / row["median_us"] / 1e6 if row["median_us"] else 0.0
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


def size_bucket(touched):
    for name, low, high in SIZES:
        if low <= touched < high:
            return name
    return SIZES[-1][0]


def with_reference(cells):
    return [c for c in cells if c["of_reference"] > 0]


def gate(cells, failed, verify_failed, large_bytes):
    lines = [f"# gate (touched >= {large_bytes / 2**20:g} MiB)"]
    for case in CASES:
        here = [c for c in cells if c["case"] == case and c["touched_bytes"] >= large_bytes]
        if not here:
            continue
        ref = [c["of_reference"] for c in with_reference(here)]
        tbps = [c["tbps"] for c in here]
        lines.append(f"  {CASE_NAMES[case]:24s} cells {len(here):4d}  TB/s p50 {fmt(quantile(tbps, .5))}"
                     f" p10 {fmt(quantile(tbps, .1))}"
                     f"  of_copy p50 {fmt(quantile([c['of_copy'] for c in here], .5))}"
                     f"  of_reference p50 {fmt(quantile(ref, .5))} p10 {fmt(quantile(ref, .1))}"
                     f" min {fmt(min(ref) if ref else float('nan'))}")
    unverified = [c for c in cells if not c["verified"]]
    lines.append(f"  unverified cells {len(unverified)} ({len(verify_failed)} verify_failed lines), "
                 f"failed cells {len(failed)}")
    for line in failed[:10]:
        lines.append(f"    failed: {line[:160]}")
    return lines


def by_case_size(cells):
    lines = ["", "# 1. by case and size (touched bytes)",
             f"  {'case':24s} {'size':8s} {'cells':>5s} {'TB/s p50':>9s} {'p10':>7s} {'of_copy':>8s} "
             f"{'ref p50':>8s} {'p10':>6s} {'min':>6s}"]
    for case in CASES:
        for name, _, _ in SIZES:
            here = [c for c in cells if c["case"] == case and size_bucket(c["touched_bytes"]) == name]
            if not here:
                continue
            ref = [c["of_reference"] for c in with_reference(here)]
            lines.append(f"  {CASE_NAMES[case]:24s} {name:8s} {len(here):5d} "
                         f"{fmt(quantile([c['tbps'] for c in here], .5)):>9s} "
                         f"{fmt(quantile([c['tbps'] for c in here], .1)):>7s} "
                         f"{fmt(quantile([c['of_copy'] for c in here], .5)):>8s} {fmt(quantile(ref, .5)):>8s} "
                         f"{fmt(quantile(ref, .1)):>6s} {fmt(min(ref) if ref else float('nan')):>6s}")
    return lines


def by_stride(cells):
    here = [c for c in cells if c["case"] == "c"]
    if not here:
        return []
    lines = ["", "# 2. C by stride",
             f"  {'stride':>7s} {'pitch':>7s} {'cells':>5s} {'ref p50':>8s} {'min':>6s} {'TB/s p50':>9s} "
             f"{'useful':>7s} {'us p50':>9s}"]
    for stride in sorted({c["stride"] for c in here}):
        group = [c for c in here if c["stride"] == stride]
        ref = [c["of_reference"] for c in with_reference(group)]
        lines.append(f"  {stride:7d} {group[0]['pitch_bytes']:7d} {len(group):5d} {fmt(quantile(ref, .5)):>8s} "
                     f"{fmt(min(ref) if ref else float('nan')):>6s} "
                     f"{fmt(quantile([c['tbps'] for c in group], .5)):>9s} "
                     f"{fmt(quantile([c['useful_tbps'] for c in group], .5)):>7s} "
                     f"{fmt(quantile([c['median_us'] for c in group], .5), 1):>9s}")
    return lines


def by_operation(cells):
    lines = ["", "# 3. of_reference p50 by operation"]
    ops = sorted({c["operation"] for c in cells})
    lines.append(f"  {'case':24s} " + " ".join(f"{op:>11s}" for op in ops))
    for case in CASES:
        here = with_reference([c for c in cells if c["case"] == case])
        if not here:
            continue
        values = [fmt(quantile([c["of_reference"] for c in here if c["operation"] == op], .5)) for op in ops]
        lines.append(f"  {CASE_NAMES[case]:24s} " + " ".join(f"{v:>11s}" for v in values))
    return lines


def signatures(cells):
    lines = ["", "# 4. pass signatures (digits stripped)"]
    for case in CASES:
        counts = {}
        for c in cells:
            if c["case"] == case:
                key = re.sub(r"\d+", "#", c["passes"])
                counts[key] = counts.get(key, 0) + 1
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"  {case} {n:5d}  {key[:110]}")
    return lines


def worst(cells, count):
    lines = ["", f"# 5. the {count} cells furthest below their reference"]
    for c in sorted(with_reference(cells), key=lambda c: c["of_reference"])[:count]:
        lines.append(f"  {c['cell_id']:5d} {c['case']} {c['operation']:11s} dims {c['dims']:24s} strides "
                     f"{c['strides']:22s} touched {c['touched_bytes'] / 2**20:9.2f} MiB of_reference "
                     f"{fmt(c['of_reference'])} {c['passes'][:40]}")
    return lines


def stamp_time(cells):
    lines = ["", "# 6. stamp wall time (ms)"]
    for case in CASES:
        here = [c["stamp_ms"] for c in cells if c["case"] == case]
        if here:
            lines.append(f"  {CASE_NAMES[case]:24s} p50 {fmt(quantile(here, .5), 2)} p90 {fmt(quantile(here, .9), 2)} "
                         f"max {fmt(max(here), 2)}")
    return lines


def spread(cells):
    lines = ["", "# 7. sample spread ((max - min) / median)"]
    for case in CASES:
        here = [c["spread"] for c in cells if c["case"] == case]
        if here:
            lines.append(f"  {CASE_NAMES[case]:24s} p50 {fmt(quantile(here, .5))} p90 {fmt(quantile(here, .9))}")
    return lines


def phase5(cells, tolerance):
    """D: the tiled stamp's time over the dense stamp's (1 / of_reference); E: what the dense stamp achieves."""
    tiled = with_reference([c for c in cells if c["case"] == "d"])
    split = [c for c in cells if c["case"] == "e"]
    if not tiled and not split:
        return []
    lines = ["", "# 9. phase 5: D tiled / dense (above 1.0 the dense planner is faster); E on the dense planner"]
    groups = [(name, [c for c in tiled if size_bucket(c["touched_bytes"]) == name]) for name, _, _ in SIZES]
    families = sorted({c["passes"] for c in tiled})
    groups += [(family, [c for c in tiled if c["passes"] == family]) for family in families]
    groups.append(("all", tiled))
    for name, group in groups:
        if not group:
            continue
        ratio = [1.0 / c["of_reference"] for c in group]
        faster = sum(1 for r in ratio if r > 1.0 + tolerance)
        slower = sum(1 for r in ratio if r < 1.0 - tolerance)
        lines.append(f"  D {name:26s} cells {len(group):4d} tiled/dense p50 {fmt(quantile(ratio, .5))}"
                     f" p10 {fmt(quantile(ratio, .1))} p90 {fmt(quantile(ratio, .9))}"
                     f"  dense faster {faster:3d} slower {slower:3d}")
    for c in sorted(tiled, key=lambda c: -c["of_reference"])[:5]:  # the dense stamp furthest behind the tiled one
        lines.append(f"    dense slowest: {c['cell_id']:5d} {c['operation']:11s} dims {c['dims']:24s} touched "
                     f"{c['touched_bytes'] / 2**20:8.2f} MiB tiled/dense {fmt(1.0 / c['of_reference'])} "
                     f"{c['reference_passes'][:40]}")
    unsupported = [c for c in split if c["reference"] == "view_unsupported"]
    lines.append(f"  E cells {len(split)}, unsupported by the view stamp {len(unsupported)}")
    for name, _, _ in SIZES:
        group = [c for c in unsupported if size_bucket(c["touched_bytes"]) == name]
        if group:
            tbps = [c["tbps"] for c in group]
            lines.append(f"  E dense {name:20s} cells {len(group):4d} TB/s p50 {fmt(quantile(tbps, .5))}"
                         f" p10 {fmt(quantile(tbps, .1))}"
                         f"  of_copy p50 {fmt(quantile([c['of_copy'] for c in group], .5))}"
                         f" min {fmt(min(c['of_copy'] for c in group))}")
    return lines


def bucket(value, buckets):
    for name, low, high in buckets:
        if low <= value < high:
            return name
    return buckets[-1][0]


def family(passes):
    """A pass chain without its tiles and sizes (digits) or the 1390 two-part prefix."""
    return re.sub(r"\d+", "#", passes).removeprefix("split_").removeprefix("tiled:")


def tiled_classes(cells, slowest):
    """F / G / H: the view's time over the dense reference's (1 / of_reference), and the slowest cells. Through 1386
    the view ran the Thor tiled kernels, in 1387-1389 the dense planner with a layout pass inside the reducer, from 1390
    the two parts a caller runs (passes "split_rotated:..." / "split_packed:...")."""
    lines = []
    for case in ("f", "g", "h"):
        here = with_reference([c for c in cells if c["case"] == case])
        if not here:
            continue
        if not lines:
            lines = ["", "# 10. F / G / H, view / dense (above 1.0 the dense reference is faster)"]
        groups = [(name, [c for c in here if size_bucket(c["touched_bytes"]) == name]) for name, _, _ in SIZES]
        families = sorted({family(c["passes"]) for c in here})
        groups += [(name, [c for c in here if family(c["passes"]) == name]) for name in families]
        groups += [(name, [c for c in here if bucket(c["outputs"], OUTPUTS) == name]) for name, _, _ in OUTPUTS]
        groups += [(name, [c for c in here if bucket(c["count"], REDUCTIONS) == name]) for name, _, _ in REDUCTIONS]
        groups.append(("all", here))
        for name, group in groups:
            if not group:
                continue
            ratio = [1.0 / c["of_reference"] for c in group]
            lines.append(f"  {case.upper()} {name[:42]:42s} cells {len(group):4d}"
                         f" view/dense p50 {fmt(quantile(ratio, .5))} p90 {fmt(quantile(ratio, .9))}"
                         f" max {fmt(max(ratio))}"
                         f"  >=2x {sum(1 for r in ratio if r >= 2.0):3d} >=10x {sum(1 for r in ratio if r >= 10.0):3d}"
                         f"  TB/s p50 {fmt(quantile([c['tbps'] for c in group], .5))}")
        for c in sorted(here, key=lambda c: c["of_reference"])[:slowest]:
            lines.append(f"    slowest: {c['cell_id']:5d} {c['operation']:11s} dims {c['dims']:24s} outputs "
                         f"{c['outputs']:9d} R {c['count']:9d} {c['touched_bytes'] / 2**20:8.2f} MiB "
                         f"{fmt(c['median_us'], 1):>9s} us dense {fmt(c['reference_us'], 1):>8s} us "
                         f"view/dense {fmt(1.0 / c['of_reference'], 1)} {family(c['passes'])[:42]}")
    return lines


ELEMENT_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}
# 1392 (phase 7 C): the PacketComplete tile cap by element bytes, and the alternative columns of an H cell
PACKET_TILE = {4: 124, 2: 248, 1: 496}
ALTERNATIVES = ("split", "pitched_w1", "pitched_w2", "pitched_w4", "pitched_w8")
IN_FLIGHT_KIB_PER_SM = (("<2K", 0, 2), ("2-4K", 2, 4), ("4-8K", 4, 8), (">=8K", 8, float("inf")))


def pitched_rows(c):
    """1391 (phase 7): an H cell as [O, R, I] rows (I contiguous, row pitch Sr, slab pitch So), from its dims, strides
    and axes in physical order; None when it does not flatten so."""
    dims = [int(x) for x in c["dims"].split("x")]
    strides = [int(x) for x in c["strides"].split("x")]
    reduced = {int(x) for x in c["axes"].split("x")}
    axes = sorted(((s, d, a in reduced) for a, (d, s) in enumerate(zip(dims, strides)) if d > 1), key=lambda t: -t[0])
    position = 0
    runs = []
    for wanted in (False, True, False):  # outer retained, reduced, inner retained
        run = []
        while position < len(axes) and axes[position][2] == wanted:
            run.append(axes[position])
            position += 1
        runs.append(run)
    outer, reduction, inner = runs
    if position != len(axes) or not reduction:
        return None

    def flatten(run, base):
        """(extent, stride of the run's innermost axis) when the run is affine, else None."""
        extent, expect = 1, base
        for stride, d, _ in reversed(run):
            if expect is not None and stride != expect:
                return None
            extent, expect = extent * d, stride * d
        return extent, run[-1][0] if run else base

    flats = (flatten(outer, None), flatten(reduction, None), flatten(inner, 1))
    if any(f is None for f in flats):
        return None
    (O, So), (R, Sr), (I, _) = flats
    return {"O": O, "R": R, "I": I, "Sr": Sr, "So": So if O > 1 else R * Sr}


def h_subclass(c):
    rows = pitched_rows(c)
    if rows is None:
        return "other"
    b = ELEMENT_BYTES.get(c["dtype"], 4)
    row_gap = (rows["Sr"] - rows["I"]) * b
    slab_gap = rows["O"] > 1 and rows["So"] != rows["R"] * rows["Sr"]
    if row_gap and slab_gap:
        return "row + slab gaps"
    if slab_gap:
        return "slab gaps"
    return "row gaps < 32 B" if row_gap < 32 else "row gaps >= 32 B"


def h_mechanisms(cells, large_bytes):
    """1391 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7): H by sub-class and by how it ran
    (split_sliced: its rows read at their pitch; split_packed: its dense copy), all sizes and from --large-mib."""
    here = with_reference([c for c in cells if c["case"] == "h"])
    if not here:
        return []
    lines = ["", "# 11. phase 7: H by gaps and path, view / dense"]
    for name in ("row gaps < 32 B", "row gaps >= 32 B", "slab gaps", "row + slab gaps", "other"):
        for path in sorted({c["passes"].split(":")[0] for c in here}):
            for label, group in (("all", [c for c in here if h_subclass(c) == name and c["passes"].startswith(path)]),
                                 ("large", [c for c in here if h_subclass(c) == name and c["passes"].startswith(path)
                                            and c["touched_bytes"] >= large_bytes])):
                if not group:
                    continue
                ratio = [1.0 / c["of_reference"] for c in group]
                lines.append(f"  {name:17s} {path.removeprefix('split_'):8s} {label:5s} cells {len(group):3d}"
                             f" view/dense p50 {fmt(quantile(ratio, .5))} max {fmt(max(ratio))}")
    return lines


def alternative_times(c):
    """1392: {name: us} of an H cell's paths: its alternative columns, and its main under the name of what it ran
    (split for a split, pitched_w<W> for the in-place read at the rule's W)."""
    times = {}
    for name in ALTERNATIVES:
        value = c.get(name + "_us")
        if value not in (None, ""):
            times[name] = float(value)
    match = re.match(r"pitched_complete:w(\d+)_t\d+", c["passes"])
    if match:
        times["pitched_w" + match.group(1)] = c["median_us"]  # the main is the rule's W, timed as the main
    elif c["passes"].startswith("split_"):
        times["split"] = c["median_us"]
    return times


def pitched_launch(c, sm_count):
    """1392a: (W, KiB in flight per SM) of the reducer's in-place launch (pitchedCompleteLaunch), or None: W the most of
    1 / 2 / 4 / 8 whose O x K tiles x W warps fit one wave of 32 per SM, at least four rows per warp."""
    rows = pitched_rows(c)
    if rows is None or not sm_count:
        return None
    b = ELEMENT_BYTES.get(c["dtype"], 4)
    tiles = -(-rows["I"] // PACKET_TILE[b])
    work, resident = rows["O"] * tiles, 32 * sm_count
    warps = 1
    while warps < 8 and work * warps * 2 <= resident:
        warps *= 2
    while warps > 1 and warps * 4 > rows["R"]:
        warps //= 2
    return warps, min(work * warps, resident) * tile_bytes(c) / sm_count / 1024


def tile_bytes(c):
    """1392: the bytes of one balanced K tile (pitchedCompleteLaunch), or None."""
    rows = pitched_rows(c)
    if rows is None:
        return None
    b = ELEMENT_BYTES.get(c["dtype"], 4)
    tiles = -(-rows["I"] // PACKET_TILE[b])
    return -(-rows["I"] // tiles) * b


def pitched_alternatives(cells, header, tolerance):
    """1392 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7 C): the in-place read of H against the
    split, by K tile bytes and work items per SM."""
    sm_count = int(header.get("sm_count", 0) or 0)
    here = [c for c in cells if c["case"] == "h"]
    timed = [(c, alternative_times(c)) for c in here]
    timed = [(c, t) for c, t in timed if "split" in t and any(k.startswith("pitched_w") for k in t)]
    if not timed:
        return []
    lines = ["", f"# 12. phase 7 C: H in place / split by K tile bytes and bytes in flight per SM ({sm_count} SMs)"]
    for wide, (name, low, high) in ((w, b) for w in (False, True) for b in IN_FLIGHT_KIB_PER_SM):
        group = [(c, t) for c, t in timed if (p := pitched_launch(c, sm_count)) is not None and low <= p[1] < high
                 and (tile_bytes(c) >= 256) == wide]
        if not group:
            continue
        split_over_best = [t["split"] / min(v for k, v in t.items() if k.startswith("pitched_w")) for _, t in group]
        chosen_over_best = [c["median_us"] / min(t.values()) for c, t in group]
        in_place = sum(1 for c, _ in group if c["passes"].startswith("pitched_complete"))
        best = {k: 0 for k in ALTERNATIVES}
        for _, t in group:
            best[min(t, key=t.get)] += 1
        wins = " ".join(f"{k.removeprefix('pitched_')} {best[k]}" for k in ALTERNATIVES)
        tile = ">=256 B" if wide else "<256 B"
        lines.append(f"  tile {tile:7s} in flight/SM {name:4s} cells {len(group):3d} in place {in_place:3d}"
                     f" split/best W p50 {fmt(quantile(split_over_best, .5))} min {fmt(min(split_over_best))}"
                     f" max {fmt(max(split_over_best))} chosen/best p50 {fmt(quantile(chosen_over_best, .5))}"
                     f" max {fmt(max(chosen_over_best))} best: {wins}")
    missed = sorted(((c["median_us"] / min(t.values()), c, t) for c, t in timed
                     if c["median_us"] / min(t.values()) > 1.0 + tolerance), key=lambda m: -m[0])
    lines.append(f"  choice slower than the best by more than {tolerance:.0%}: {len(missed)} cells")
    for ratio, c, t in missed[:10]:
        rows = pitched_rows(c) or {}
        p = pitched_launch(c, sm_count)
        lines.append(f"    {c['cell_id']:5d} {c['operation']:11s} O {rows.get('O', '?')} R {rows.get('R', '?')}"
                     f" I {rows.get('I', '?')} W {p[0] if p else '?'} in flight {fmt(p[1]) if p else '?'} KiB/SM"
                     f" ran {c['passes'].split(':')[0]} {fmt(c['median_us'], 1)} us, best {min(t, key=t.get)}"
                     f" {fmt(min(t.values()), 1)} us ({fmt(ratio)}x)")
    return lines


def key(c):
    return (c["cell_id"], c["case"], c["operation"], c["dims"], c["strides"])


def compare(cells, other, legacy_bytes, tolerance):
    theirs = {key(c): c for c in other}
    joined = [(c, theirs[key(c)]) for c in cells if key(c) in theirs]
    lines = ["", f"# 8. compare: the other run's time over this run's ({len(joined)} joined cells)"]
    for case in CASES:
        for name, _, _ in SIZES:
            pairs = [(a, b) for a, b in joined if a["case"] == case and size_bucket(a["touched_bytes"]) == name]
            if pairs:
                ratio = [b["median_us"] / a["median_us"] for a, b in pairs]
                lines.append(f"  {CASE_NAMES[case]:24s} {name:8s} cells {len(pairs):4d} p50 {fmt(quantile(ratio, .5))}"
                             f" p10 {fmt(quantile(ratio, .1))} min {fmt(min(ratio))}")
    strided = [(a, b) for a, b in joined if a["case"] == "c"]
    for stride in sorted({a["stride"] for a, _ in strided}):
        ratio = [b["median_us"] / a["median_us"] for a, b in strided if a["stride"] == stride]
        lines.append(f"  C stride {stride:<14d} cells {len(ratio):4d} p50 {fmt(quantile(ratio, .5))} "
                     f"min {fmt(min(ratio))}")
    slower = [(a, b) for a, b in joined
              if a["touched_bytes"] >= legacy_bytes and b["median_us"] / a["median_us"] < 1.0 - tolerance]
    lines.append(f"  legacy clause (touched >= {legacy_bytes / 2**20:g} MiB, slower by more than {tolerance:.0%}): "
                 f"{len(slower)} cells")
    for a, b in sorted(slower, key=lambda p: p[1]["median_us"] / p[0]["median_us"])[:10]:
        lines.append(f"    {a['cell_id']:5d} {a['case']} {a['operation']:11s} dims {a['dims']:24s} "
                     f"ratio {fmt(b['median_us'] / a['median_us'])}")
    return lines


def report(options):
    header, cells, failed, verify_failed = load(options.path)
    lines = [f"# view_reduction_audit {options.path}: dtype {header.get('dtype', '?')}, {len(cells)} cells, device "
             f"{header.get('device', '?')}, copy {header.get('copy_bandwidth_tbps', '?')} TB/s"]
    large = options.large_mib * 2**20
    lines += gate(cells, failed, verify_failed, large)
    lines += by_case_size(cells) + by_stride(cells) + by_operation(cells) + signatures(cells)
    lines += worst(cells, options.worst) + stamp_time(cells) + spread(cells)
    if options.compare:
        _, other, _, _ = load(options.compare)
        lines += compare(cells, other, options.legacy_mib * 2**20, options.legacy_tolerance)
    lines += phase5(cells, options.legacy_tolerance)
    lines += tiled_classes(cells, options.slowest)
    lines += h_mechanisms(cells, large)
    lines += pitched_alternatives(cells, header, options.legacy_tolerance)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--large-mib", type=float, default=16.0)
    parser.add_argument("--legacy-mib", type=float, default=1.0)
    parser.add_argument("--legacy-tolerance", type=float, default=0.02)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--slowest", type=int, default=8, help="section 10: the slowest cells per case")
    options = parser.parse_args(argv)
    print("\n".join(report(options)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
