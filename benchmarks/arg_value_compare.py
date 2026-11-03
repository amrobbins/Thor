#!/usr/bin/env python3
"""ARG-A0: FP8 RK argmin/argmax against the VALUE min/max line-rate reference.

Reads one sweep CSV (``--focus=arg``, or any run that measured argmin/argmax together with min_identity/max_identity)
and pairs every ARG cell with the VALUE cell of the same (outer, R, K, dtype):

  argmin <-> min_identity,  argmax <-> max_identity

The VALUE reference is the fastest measured candidate (``oracle``) and, separately, the fastest candidate inside the
learned 28-candidate shortlist (``top28``, what production can reach). The ratio is ARG time / VALUE time, so 1.0
means ARG runs at VALUE line rate and the A-series target is <= ~1.05.

Since ARG-A2 the ARG cells hold the stamped production plan (family arg_production; configuration = the planned
passes, e.g. fp8_arg_one_window:o8_w1_p16_c1 or direct:tiled_generic) and, with --onewindow-runtime-k, every
OneWindow ARG twin timed directly (family one_window_arg). The twin report compares each ARG twin with the VALUE
OneWindow unit of the same name (family one_window_rk) on the same cell.

Throughput is reported as achieved DRAM traffic: FP8 input bytes plus the bytes each reduction writes (VALUE: one
value per output; ARG: index, plus value unless the run used --arg-outputs=index). Use a second CSV (``--after``) to
compare two ARG builds cell for cell.
"""
import argparse
import csv
import gzip
import math
from collections import defaultdict

PAIRS = {'argmin': 'min_identity', 'argmax': 'max_identity'}
# ARG twin family -> the VALUE family whose candidate of the same configuration name it mirrors.
# RuntimeK ARG names differ from VALUE's (own tiles): best-vs-best only. AlignedK names match VALUE's.
TWIN_FAMILIES = {'one_window_arg': 'one_window_rk', 'deep_r_small_k_arg': 'deep_r_small_k',
                 'runtime_k_arg': 'runtime_k', 'aligned_k_arg': 'aligned_k'}
CELL_FIELDS = ('outer', 'R', 'K', 'dtype')
K_CLASSES = ((2, 2, 'K=2'), (3, 8, 'K 3-8'), (9, 32, 'K 9-32'), (33, 128, 'K 33-128'), (129, 512, 'K 129-512'),
             (513, 1 << 62, 'K >512'))


def open_text(path):
    return gzip.open(path, 'rt', newline='') if str(path).endswith('.gz') else open(path, newline='')


def header_options(line):
    options = {}
    for token in line[1:].split():
        if '=' in token:
            key, value = token.split('=', 1)
            options[key] = value
    return options


def read(path):
    """Return ({(cell, operation): [rows]}, header options)."""
    rows = defaultdict(list)
    options = {}
    with open_text(path) as stream:
        def noncomments():
            for line in stream:
                if line.startswith('#'):
                    if not options and 'output=' in line:
                        options.update(header_options(line))
                elif line.strip():
                    yield line
        for row in csv.DictReader(noncomments()):
            if not row.get('family'):
                continue
            time = float(row['total_median_ms'])
            if not math.isfinite(time) or time <= 0:
                raise ValueError(f'Nonpositive or nonfinite time in {path}: {row}')
            rows[(tuple(row[f] for f in CELL_FIELDS), row['operation'])].append(row)
    if not rows:
        raise ValueError(f'No measurements in {path}')
    return rows, options


def output_bytes_per_element(options, arg):
    value = 1 if options.get('output') == 'same_fp8' else 4
    if not arg:
        return value
    return 4 + (value if options.get('arg_outputs', 'value_index') == 'value_index' else 0)


def tbps(cell, ms, out_bytes):
    outer, r, k = (int(x) for x in cell[:3])
    traffic = outer * r * k + outer * k * out_bytes
    return traffic / (ms * 1e9)


def collapse_replicates(rows):
    """Average repeated measurements of one candidate.

    Large R*K shapes round several working-set targets to the same ``outer`` (e.g. outer=1 for 24 and 96 MiB), so the
    sweep measures that cell once per target. Those are replicates of one cell, not distinct cells.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['configuration'])].append(row)
    collapsed = []
    for group in groups.values():
        merged = dict(group[0])
        merged['total_median_ms'] = str(sum(float(r['total_median_ms']) for r in group) / len(group))
        merged['replicates'] = len(group)
        collapsed.append(merged)
    return collapsed


def fastest(rows, top28=False):
    eligible = [row for row in rows if not top28 or int(row['rank28']) > 0]
    return min(eligible, key=lambda row: float(row['total_median_ms'])) if eligible else None


def geomean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def k_class(k):
    for low, high, name in K_CLASSES:
        if low <= k <= high:
            return name
    return 'K=1'


def paired_cells(rows, options):
    """One record per ARG cell that has a VALUE reference."""
    records = []
    missing = 0
    arg_bytes = output_bytes_per_element(options, True)
    value_bytes = output_bytes_per_element(options, False)
    for (cell, operation), arg_rows in sorted(rows.items()):
        if operation not in PAIRS:
            continue
        value_rows = rows.get((cell, PAIRS[operation]))
        if not value_rows:
            missing += 1
            continue
        arg_rows = collapse_replicates(arg_rows)
        value_rows = collapse_replicates(value_rows)
        production = [row for row in arg_rows if row['family'] in ('arg_production', 'arg_direct')]
        if len(production) != 1:
            raise ValueError(f'Expected one production ARG configuration for {cell} {operation}, got {len(production)}')
        arg = production[0]
        twins = {(row['family'], row['configuration']): float(row['total_median_ms']) for row in arg_rows
                 if row['family'] in TWIN_FAMILIES}
        value_units = {(row['family'], row['configuration']): float(row['total_median_ms']) for row in value_rows
                       if row['family'] in TWIN_FAMILIES.values()}
        oracle = fastest(value_rows)
        top28 = fastest(value_rows, top28=True)
        arg_ms = float(arg['total_median_ms'])
        records.append({
            'cell': cell, 'operation': operation, 'backend': arg['configuration'], 'arg_ms': arg_ms,
            'oracle_ms': float(oracle['total_median_ms']), 'oracle': f"{oracle['family']}:{oracle['configuration']}",
            'top28_ms': float(top28['total_median_ms']) if top28 else float('nan'),
            'arg_tbps': tbps(cell, arg_ms, arg_bytes),
            'value_tbps': tbps(cell, float(oracle['total_median_ms']), value_bytes),
            'mib': int(cell[0]) * int(cell[1]) * int(cell[2]) / (1 << 20),
            'twins': twins,
            'value_units': value_units,
        })
    return records, missing


def summarize(records, label, key):
    groups = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    print(f'\n{label}: cells, ARG/VALUE-oracle geomean, ARG/VALUE-top28 geomean, worst, ARG TB/s median, '
          f'VALUE TB/s median')
    for name in sorted(groups, key=str):
        group = groups[name]
        ratios = sorted(r['arg_ms'] / r['oracle_ms'] for r in group)
        top = [r['arg_ms'] / r['top28_ms'] for r in group if math.isfinite(r['top28_ms'])]
        arg_t = sorted(r['arg_tbps'] for r in group)
        val_t = sorted(r['value_tbps'] for r in group)
        print(f'  {str(name):<28} {len(group):5d}  {geomean(ratios):6.3f}  {geomean(top):6.3f}  {ratios[-1]:6.2f}'
              f'  {arg_t[len(arg_t) // 2]:5.2f}  {val_t[len(val_t) // 2]:5.2f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', help='sweep CSV (.csv or .csv.gz) with ARG and VALUE min/max cells')
    parser.add_argument('--after', help='second sweep CSV: also compare ARG time cell for cell (after/before)')
    parser.add_argument('--worst', type=int, default=20, help='list the N cells furthest from VALUE')
    parser.add_argument('--min-mib', type=float, default=0.0,
                        help='ignore cells with a smaller input (24 MiB cells are ~15 us and noise-bound)')
    args = parser.parse_args(argv)

    rows, options = read(args.csv)
    records, missing = paired_cells(rows, options)
    records = [r for r in records if r['mib'] >= args.min_mib]
    if not records:
        raise SystemExit('No ARG cell has a VALUE min/max reference (run with --focus=arg).')
    print(f"output={options.get('output')} arg_outputs={options.get('arg_outputs', 'value_index')} "
          f"paired_cells={len(records)} arg_cells_without_value={missing}")
    ratios = [r['arg_ms'] / r['oracle_ms'] for r in records]
    print(f'ARG / VALUE oracle: geomean {geomean(ratios):.3f}; '
          f'<=1.05: {sum(x <= 1.05 for x in ratios)}; >1.5: {sum(x > 1.5 for x in ratios)}; '
          f'>2: {sum(x > 2 for x in ratios)}; of {len(ratios)}')
    hot = [r for r in records if r['value_tbps'] >= 1.3]
    if hot:
        print(f'Cells where VALUE reaches >=1.3 TB/s: {len(hot)}; ARG also >=1.3 TB/s on '
              f'{sum(r["arg_tbps"] >= 1.3 for r in hot)}; ARG/VALUE geomean there '
              f'{geomean(r["arg_ms"] / r["oracle_ms"] for r in hot):.3f}')
    summarize(records, 'By ARG production pass family', lambda r: '+'.join(
        p.split(':')[0] if p.startswith('fp8_arg') else p for p in r['backend'].split('+')))
    summarize(records, 'By K class', lambda r: k_class(int(r['cell'][2])))
    summarize(records, 'By operation', lambda r: r['operation'])
    summarize(records, 'By input size (MiB, rounded)', lambda r: int(round(r['mib'] / 24.0) * 24))

    for arg_family, value_family in TWIN_FAMILIES.items():
        def pick(table, family):
            return {name: ms for (fam, name), ms in table.items() if fam == family}
        twin_cells = [(r, pick(r['twins'], arg_family), pick(r['value_units'], value_family)) for r in records]
        twin_cells = [(r, t, v) for r, t, v in twin_cells if t and v]
        if not twin_cells:
            continue
        best = [min(t.values()) / min(v.values()) for _, t, v in twin_cells]
        matched = [t[name] / v[name] for _, t, v in twin_cells for name in t if name in v]
        print(f'\n{arg_family} vs {value_family}: {len(twin_cells)} cells; best ARG twin / best VALUE candidate '
              f'geomean {geomean(best):.3f} (worst {max(best):.2f}); same-name ARG/VALUE geomean '
              f'{geomean(matched):.3f} over {len(matched)} candidate-cells')
        per_unit = defaultdict(list)
        for _, t, v in twin_cells:
            for name in t:
                if name in v:
                    per_unit[name].append(t[name] / v[name])
        print('  per configuration: name, cells, ARG/VALUE geomean, worst')
        for name in sorted(per_unit):
            values = per_unit[name]
            print(f'  {name:<22} {len(values):5d}  {geomean(values):6.3f}  {max(values):6.2f}')
    all_twins = [r for r in records if r['twins']]
    if all_twins:
        print(f'Production ARG / best ARG twin (any family), {len(all_twins)} cells: geomean '
              f'{geomean(r["arg_ms"] / min(r["twins"].values()) for r in all_twins):.3f} '
              '(>1: production does not run the fastest twin)')

    print(f'\nWorst {args.worst} cells (ARG/VALUE oracle): outer,R,K,dtype op backend ratio ARG_ms VALUE_ms '
          f'ARG_TB/s VALUE_TB/s VALUE_oracle')
    for r in sorted(records, key=lambda r: r['arg_ms'] / r['oracle_ms'], reverse=True)[:args.worst]:
        print(f"  {','.join(r['cell'])} {r['operation']} {r['backend']} {r['arg_ms'] / r['oracle_ms']:.2f} "
              f"{r['arg_ms']:.4f} {r['oracle_ms']:.4f} {r['arg_tbps']:.2f} {r['value_tbps']:.2f} {r['oracle']}")

    if args.after:
        after_rows, after_options = read(args.after)
        if after_options.get('output') != options.get('output') or \
                after_options.get('arg_outputs', 'value_index') != options.get('arg_outputs', 'value_index'):
            raise SystemExit('Refusing to compare runs with different output= or arg_outputs= settings.')
        after_records = {(r['cell'], r['operation']): r for r in paired_cells(after_rows, after_options)[0]}
        changes = [(after_records[(r['cell'], r['operation'])]['arg_ms'] / r['arg_ms'], r) for r in records
                   if (r['cell'], r['operation']) in after_records]
        if changes:
            print(f'\nARG after/before: {len(changes)} cells, geomean {geomean(c for c, _ in changes):.3f}, '
                  f'>5% slower {sum(c > 1.05 for c, _ in changes)}, >5% faster {sum(c < 0.95 for c, _ in changes)}')
            after_ratio = [after_records[(r['cell'], r['operation'])]['arg_ms'] /
                           after_records[(r['cell'], r['operation'])]['oracle_ms'] for _, r in changes]
            print(f'ARG / VALUE oracle after: geomean {geomean(after_ratio):.3f}')


if __name__ == '__main__':
    main()
