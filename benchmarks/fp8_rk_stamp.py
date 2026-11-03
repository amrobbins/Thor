#!/usr/bin/env python3
"""FP8 RK stamp audit (1162): what production's own RK contest costs and picks.

Reads `thor_fp8_retained_cold_l2_sweep --training --stamp` output. Every cell carries the timed median of every
retained candidate (the oracle is the fastest) and a '#stamp,cell_id,stamp_ms,candidates,winner' line: the wall time of
production's own contest at that cell (Regular depth) and the candidate it picked, named like the CSV configuration.

  1. stamp wall time by input size
  2. the pick against the oracle (time over the fastest timed candidate), by kind and size
  3. the contest's GPU time estimated from the timed medians, before 1162 (every one of the `candidates` top-ranked
     shapes run once per round: warmup + 7 rounds) and with the shared policy (warmup + probe for every one, then 7
     rounds of about 200 us for each one within 3x max(best, 8 us)), against the measured stamp time

Standard library only.
"""

import argparse
import csv
import gzip
import math
import sys

SIZES = (("<1M", 0, 2**20), ("1-16M", 2**20, 16 * 2**20), ("16-128M", 16 * 2**20, 128 * 2**20),
         (">=128M", 128 * 2**20, float("inf")))
ROUNDS = 7  # Regular (fp8RKTuningPlan)


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(cells, stamps): cells[id] = {kind, bytes, times{configuration: us}, margins{configuration: margin}};
    stamps[id] = (stamp_ms, candidates, winner)."""
    stamps, lines = {}, []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("#stamp,"):
                if line.startswith("#stamp,cell_id"):
                    continue
                _, cell_id, ms, count, winner = line.rstrip("\n").split(",", 4)
                stamps[int(cell_id)] = (float(ms), int(count), winner)
            elif not line.startswith("#"):
                lines.append(line)
    cells = {}
    for row in csv.DictReader(lines):
        cell_id = int(row["cell_id"])
        cell = cells.setdefault(cell_id, {"kind": row["kind"], "times": {}, "margins": {},
                                          "bytes": int(row["outer"]) * int(row["R"]) * int(row["K"])})
        cell["times"][row["configuration"]] = float(row["total_median_ms"]) * 1000.0
        cell["margins"][row["configuration"]] = float(row["frozen_margin"])
        samples = [float(row[f"sample{i}_ms"]) for i in range(5) if row.get(f"sample{i}_ms")]
        cell.setdefault("spread", {})[row["configuration"]] = (
            (max(samples) - min(samples)) / sorted(samples)[len(samples) // 2] if samples else 0.0)
    if not stamps:
        raise ValueError(f"{path}: no '#stamp' lines (run the sweep with --training --stamp)")
    return cells, stamps


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def contest_us(times, count):
    """Estimated contest GPU time (us) of the top `count` candidates (by the ranker margin) before and after 1162."""
    before = sum(t * (1 + ROUNDS) for t in times[:count])
    probes = [max(t, 2.048) for t in times[:count]]
    limit = 3.0 * max(min(probes), 8.0)
    after = 0.0
    for t, p in zip(times[:count], probes):
        after += 2 * t  # warmup + probe
        if p <= limit:
            after += ROUNDS * t * min(max(math.ceil(200.0 / p), 1), 32)
    return before, after


def table(title, header, rows):
    print(title)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    for row in [header] + rows:
        print("  " + "  ".join(str(x).rjust(w) for x, w in zip(row, widths)))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--worst", type=int, default=15)
    args = parser.parse_args(argv)
    cells, stamps = load(args.csv)
    joined = [(cid, cells[cid], stamps[cid]) for cid in sorted(stamps) if cid in cells]
    print(f"# cells={len(cells)} stamped={len(stamps)} joined={len(joined)}\n")

    rows = []
    for name, low, high in SIZES:
        ms = [s[0] for _, c, s in joined if low <= c["bytes"] < high]
        if ms:
            rows.append([name, len(ms), f"{quantile(ms, 0.5):.1f}", f"{quantile(ms, 0.9):.1f}", f"{max(ms):.1f}",
                         f"{sum(ms) / 1000:.1f}"])
    table("1. Stamp wall time by input size (ms; total in s)", ["input", "n", "p50", "p90", "max", "total s"], rows)

    rows, worst, unmatched = [], [], []
    for kind in ("value", "arg"):
        for name, low, high in SIZES:
            regrets = []
            for cid, c, (ms, count, winner) in joined:
                if (c["kind"] == "value") != (kind == "value") or not low <= c["bytes"] < high or count == 0:
                    continue
                if winner not in c["times"]:
                    unmatched.append((cid, winner))
                    continue
                oracle = min(c["times"].values())
                regret = c["times"][winner] / oracle - 1.0
                regrets.append(regret)
                worst.append((c["times"][winner] - oracle, cid, c, winner, regret, ms))
            if regrets:
                rows.append([kind.upper(), name, len(regrets), f"{100 * sum(regrets) / len(regrets):.1f}%",
                             f"{100 * quantile(regrets, 0.9):.1f}%", f"{100 * max(regrets):.1f}%",
                             f"{100 * sum(r <= 0.02 for r in regrets) / len(regrets):.0f}%"])
    table("2. The stamp's pick against the oracle (time over the fastest timed candidate)",
          ["kind", "input", "n", "mean", "p90", "max", "within 2%"], rows)
    worst.sort(key=lambda w: -w[0])
    table("   largest losses (us)",
          ["cell", "kind", "MiB", "pick", "oracle", "regret", "us lost", "stamp ms"],
          [[cid, c["kind"], f"{c['bytes'] / 2**20:.1f}", winner, min(c["times"], key=c["times"].get),
            f"{100 * regret:.1f}%", f"{lost:.1f}", f"{ms:.1f}"] for lost, cid, c, winner, regret, ms in
           worst[:args.worst] if regret > 0.01])
    if unmatched:
        print(f"   picks not among the timed configurations: {len(unmatched)} (first: {unmatched[:3]})\n")

    # 2b (1163): where a loss comes from. The ranker's top N (N = the stamp's candidate count, by frozen margin) either
    # holds the oracle or not; the contest then picks within it. The sweep's own sample spread bounds what it can tell.
    rows = []
    for kind in ("value", "arg"):
        for name, low, high in SIZES:
            ranker, contest, holds, spread = [], [], 0, []
            for cid, c, (ms, count, winner) in joined:
                if (c["kind"] == "value") != (kind == "value") or not low <= c["bytes"] < high or count == 0 \
                        or winner not in c["times"]:
                    continue
                top = sorted(c["times"], key=lambda k: -c["margins"][k])[:count]
                oracle = min(c["times"].values())
                best = min(c["times"][k] for k in top)
                ranker.append(best / oracle - 1.0)
                contest.append(c["times"][winner] / best - 1.0)
                holds += best == oracle
                spread.append(c["spread"].get(winner, 0.0))
            if ranker:
                rows.append([kind.upper(), name, len(ranker), f"{100 * holds / len(ranker):.0f}%",
                             f"{100 * max(ranker):.1f}%", f"{100 * quantile(contest, 0.9):.1f}%",
                             f"{100 * max(contest):.1f}%", f"{100 * quantile(spread, 0.5):.1f}%"])
    table("2b. Ranker vs contest: does the top N hold the oracle, how far the pick is from the top N's best, and the\n"
          "    sweep's own sample spread (max-min)/median of the picked shape",
          ["kind", "input", "n", "top N holds oracle", "top N max", "pick p90", "pick max", "sweep spread p50"], rows)

    rows = []
    for name, low, high in SIZES:
        before, after, measured = [], [], []
        for _, c, (ms, count, _) in joined:
            if not low <= c["bytes"] < high or count == 0:
                continue
            ranked = sorted(c["times"], key=lambda k: -c["margins"][k])
            b, a = contest_us([c["times"][k] for k in ranked], count)
            before.append(b / 1000)
            after.append(a / 1000)
            measured.append(ms)
        if measured:
            rows.append([name, len(measured), f"{quantile(before, 0.5):.1f}", f"{quantile(after, 0.5):.1f}",
                         f"{quantile(measured, 0.5):.1f}", f"{sum(before) / 1000:.1f}", f"{sum(after) / 1000:.1f}",
                         f"{sum(measured) / 1000:.1f}"])
    table("3. Contest GPU time estimated from the timed medians (ms; totals in s): before 1162 (one launch per\n"
          "   measurement, every candidate), with the shared policy, and the measured stamp (wall, incl. setup)",
          ["input", "n", "before p50", "policy p50", "measured p50", "before s", "policy s", "measured s"], rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
