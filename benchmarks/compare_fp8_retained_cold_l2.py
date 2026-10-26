#!/usr/bin/env python3
"""Compare retained FP8 GPU sweeps, emphasizing high-bandwidth regressions.

The full-plan time includes FP32 continuations. Historical same-FP8 and new
FP32 output contracts must never be compared as a clean kernel A/B.
"""
import argparse
import csv
import gzip
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KEY_FIELDS = ('family', 'outer', 'R', 'K', 'dtype', 'configuration', 'operation')
CELL_FIELDS = ('outer', 'R', 'K', 'dtype', 'operation')


def open_text(path):
    return gzip.open(path, 'rt', newline='') if str(path).endswith('.gz') else open(path, newline='')


def read(path):
    rows = defaultdict(list)
    cell_rows = defaultdict(list)
    output = None
    with open_text(path) as stream:
        def noncomments():
            nonlocal output
            for line in stream:
                if line.startswith('#'):
                    if 'output=' in line and output is None:
                        output = line.split('output=', 1)[1].split()[0]
                elif line.strip():
                    yield line
        for row in csv.DictReader(noncomments()):
            if not row.get('family'):
                continue
            time = float(row['total_median_ms'])
            if not math.isfinite(time) or time <= 0:
                raise ValueError(f'Nonpositive or nonfinite time in {path}: {row}')
            key = tuple(row[f] for f in KEY_FIELDS)
            rows[key].append(row)
            cell_rows[tuple(row[f] for f in CELL_FIELDS)].append(row)
    if not rows:
        raise ValueError(f'No retained candidate measurements in {path}')
    return rows, cell_rows, output


def throughput(row):
    # FP8 is one byte per input element; this is nominal input bandwidth.
    n = int(row['outer']) * int(row['R']) * int(row['K'])
    return n / (float(row['total_median_ms']) * 1e9)


def paired(before, after):
    result = defaultdict(list)
    missing_a = missing_b = 0
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key, []), after.get(key, [])
        missing_a += max(0, len(left) - len(right))
        missing_b += max(0, len(right) - len(left))
        for a, b in zip(left, right):
            pre, post = float(a['total_median_ms']), float(b['total_median_ms'])
            result['all'].append((a, b))
            result[key[0]].append((a, b))
            for tier in (1.0, 1.3, 1.5):
                if throughput(a) >= tier:
                    result[f'baseline >={tier:.1f} TB/s'].append((a, b))
    return result, missing_a, missing_b


def summarize_pairs(pairs):
    if not pairs:
        return 'N=0'
    speedups = [(float(a['total_median_ms']) / float(b['total_median_ms']) - 1) * 100
                for a, b in pairs]
    loss = [100 * (float(b['total_median_ms']) / float(a['total_median_ms']) - 1)
            for a, b in pairs]
    worst = max(float(b['total_median_ms']) - float(a['total_median_ms']) for a, b in pairs) * 1e3
    material = sum(x > 3 and float(b['total_median_ms']) - float(a['total_median_ms']) > 0.0001
                   for x, (a, b) in zip(loss, pairs))
    return (f'N={len(pairs)} median_speedup={statistics.median(speedups):+.2f}% '
            f'material_regressions={material} worst_additional_us={worst:+.3f}')


def oracle_stats(cells):
    total = exact = fast = fast_exact = misses = 0
    for rowset in cells.values():
        fastest = min(float(r['total_median_ms']) for r in rowset)
        total += 1
        fast_case = max(throughput(r) for r in rowset) >= 1.5
        fast += fast_case
        # Tie tolerance only covers timestamp representation, not a 5% window.
        has = any(int(row.get('rank28') or '0') in range(1, 29) and
                  float(row['total_median_ms']) <= fastest + 1e-9 for row in rowset)
        exact += has
        fast_exact += fast_case and has
        misses += not has
    return f'exact28={exact}/{total} misses={misses} >=1.5TB/s={fast_exact}/{fast}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    args = parser.parse_args(argv)
    before, _, before_output = read(args.before)
    after, cells, after_output = read(args.after)
    if not before_output or not after_output or before_output != after_output:
        raise SystemExit(f'Cannot compare different or unknown output semantics: '
                         f'before={before_output!r}, after={after_output!r}')
    groups, missing_before, missing_after = paired(before, after)
    print(f'FP8 retained paired full-plan GPU medians; output={after_output}')
    print('Positive speedup means after is faster. Measurement noise remains possible.')
    for label in ('all', 'one_window', 'deep_r_small_k', 'aligned_k', 'runtime_k',
                  'baseline >=1.0 TB/s', 'baseline >=1.3 TB/s', 'baseline >=1.5 TB/s'):
        print(f'{label}: {summarize_pairs(groups[label])}')
    print(f'Unpaired measurements: before_only={missing_before} after_only={missing_after}')
    if any(r.get('rank28') not in (None, '') for rows in cells.values() for r in rows):
        print(f'After retained measured oracle: {oracle_stats(cells)}')
    if missing_before or missing_after:
        print('WARNING: candidate populations differ; inspect absent candidate configurations.')


if __name__ == '__main__':
    main()
