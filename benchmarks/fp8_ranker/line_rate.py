#!/usr/bin/env python3
"""Line rate of a `--training` dataset (1194). Standard library only.

For every cell: the oracle (fastest candidate) in TB/s of moved bytes, the input (outer * R * K * element size; FP32 4
bytes, FP8 1; a carried-index cell adds its index width) plus the output it writes (1195: as the FP8 audits report; at
R = 3 the output is a third of the input),
and the regret of the list-prefix shortlist at the tuning depths (the untrained selector times the first
6 / 12 / 24 candidates in list_position order). Prints, by target working set and by K bucket: cells, oracle TB/s
p50 / p10 / min, the share of cells at --line-tbps (default 1.5), the families that win, and the prefix regrets.

Usage: line_rate.py DATASET.csv[.gz] [--line-tbps=1.5]
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402

ELEMENT_BYTES = {"fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1}
DEPTHS = (6, 12, 24)


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def k_bucket(k):
    for limit, name in ((8, "K<=8"), (32, "K<=32"), (125, "K<=125"), (512, "K<=512"), (4096, "K<=4096")):
        if k <= limit:
            return name
    return "K>4096"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--line-tbps", type=float, default=1.5)
    args = parser.parse_args()
    header, cells = dataset.load(args.dataset)
    # 1211: a carried-index cell (kind carried_*) streams its carried index too (the header's carried_index width).
    carried_bytes = {"uint32": 4, "uint64": 8}.get(header.get("carried_index", "none"), 0)
    rows = []
    for cell in cells:
        element_bytes = ELEMENT_BYTES[cell.dtype] + (carried_bytes if cell.kind.startswith("carried_") else 0)
        input_bytes = cell.outer * cell.r * cell.k * element_bytes
        # Outputs: VALUE an FP32 value; ARG a UINT32 index, and an FP32 value for arg_value_index.
        output_bytes = cell.outer * cell.k * {"value": 4, "arg_index": 4, "arg_value_index": 8,
                                              "carried_arg_index": 4, "carried_arg_value_index": 8}[cell.kind]
        best = min(cell.candidates, key=lambda c: c.median_ms)
        by_position = sorted(cell.candidates, key=lambda c: c.list_position)
        regrets = {d: min(c.median_ms for c in by_position[:d]) / best.median_ms for d in DEPTHS}
        seconds = best.median_ms * 1e-3
        rows.append({"cell": cell, "tbps": (input_bytes + output_bytes) / seconds / 1e12,
                     "input_tbps": input_bytes / seconds / 1e12, "winner": best.family,
                     "regrets": regrets})
    print(f"# {args.dataset}: {len(rows)} cells, dtypes {sorted({r['cell'].dtype for r in rows})}, "
          f"sm_count {header.get('sm_count', '?')}")

    def report(title, key):
        groups = collections.defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        print(f"\n## {title}")
        print(f"{'group':>12} {'cells':>6} {'p50':>6} {'p10':>6} {'min':>6} {'>=line':>7}  "
              + "  ".join(f"top{d} p90/max" for d in DEPTHS) + "  winners")
        for name in sorted(groups, key=lambda g: (len(str(g)), str(g))):
            group = groups[name]
            tbps = [r["tbps"] for r in group]
            at_line = sum(t >= args.line_tbps for t in tbps) / len(tbps)
            regret = "  ".join(f"{quantile([r['regrets'][d] for r in group], 0.9):5.3f}/"
                               f"{max(r['regrets'][d] for r in group):5.2f}" for d in DEPTHS)
            winners = collections.Counter(r["winner"] for r in group).most_common()
            print(f"{str(name):>12} {len(group):6d} {quantile(tbps, 0.5):6.3f} {quantile(tbps, 0.1):6.3f} "
                  f"{min(tbps):6.3f} {100 * at_line:6.1f}%  {regret}  "
                  + ", ".join(f"{w} {n}" for w, n in winners))

    report("by target working set (MiB)", lambda r: r["cell"].target_mib)
    report("by K", lambda r: k_bucket(r["cell"].k))
    report("by operation", lambda r: r["cell"].operation)
    slow = sorted(rows, key=lambda r: r["tbps"])[:15]
    print("\n## slowest cells (oracle, moved bytes)")
    for r in slow:
        c = r["cell"]
        print(f"  {r['tbps']:6.3f} TB/s moved ({r['input_tbps']:5.3f} input)  outer={c.outer} R={c.r} K={c.k} {c.target_mib} MiB {c.operation} "
              f"winner {r['winner']}")


if __name__ == "__main__":
    main()
