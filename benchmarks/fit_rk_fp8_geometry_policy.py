#!/usr/bin/env python3
"""Explore compact, geometry-only Dense RK FP8 policies; no production dispatch.

Train on the 887 unified sweep, freeze JSON, then score an *unseen-R* GPU
holdout.  No test-shape identifiers, measured timing, or per-shape lookup data
enter a feature vector.  Report unsupported choices rather than quietly picking
an oracle fallback.

Requires pandas, numpy, scikit-learn.  This script is calibration tooling, not a
runtime dependency of Thor.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold

MODE_NAMES = ('OneWindow', 'DeepRSmallK', 'AlignedK', 'RuntimeK')
FEATURES = ('log2_R', 'log2_K', 'log2_outer', 'log2_work_bytes',
            'log2_output_per_sm', 'log2_ctas_o1_w1_per_sm',
            'k_mod_16', 'k_mod_8', 'k_mod_4')
KEY = ['outer', 'R', 'K', 'dtype']


def mode_of(access):
    if access.startswith('r_parallel_general_one_window_'):
        return 'OneWindow'
    if access.startswith('r_parallel_general_deep_r_small_k'):
        return 'DeepRSmallK'
    if access.startswith('r_parallel_general_aligned_k'):
        return 'AlignedK'
    if access.startswith('r_parallel_general_runtime_k_'):
        return 'RuntimeK'
    return None


def variant_of(access, packet):
    match = re.search(r'one_window_(?:o([248])_w1|w([1248]))$', access)
    if not match:
        return None
    topology = 'o' + match[1] + '_w1' if match[1] else 'o1_w' + match[2]
    return f'{topology}_p{int(packet)}'


def load_cells(path):
    raw = pd.read_csv(path, comment='#', low_memory=False)
    required = KEY + ['access', 'strategy', 'median_ms', 'best_ms', 'input_bytes', 'output_bytes', 'device_sms', 'packet_bytes']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'{path}: missing CSV columns {sorted(missing)}')
    # No oracle, historical standalone modes, or selector-probe rows can become a winner.
    raw = raw.loc[(raw.strategy != 'oracle_control') &
                  ~raw.access.str.contains('selector_probe', regex=False)].copy()
    raw['mode'] = raw.access.map(mode_of)
    raw = raw.loc[raw['mode'].notna() & (raw.median_ms > 0) & (raw.best_ms > 0)].copy()
    raw['variant'] = [variant_of(a, p) if m == 'OneWindow' else None
                      for a, p, m in zip(raw.access, raw.packet_bytes, raw['mode'])]
    raw['stable'] = raw.median_ms / raw.best_ms <= 1.05 + 1e-9
    cells = []
    for key, rows in raw.groupby(KEY, sort=True):
        stable_rows = rows.loc[rows.stable]
        # Preserve the established rule: if *no* candidate is stable, retain the raw pool.
        pool = stable_rows if len(stable_rows) else rows
        winner = pool.loc[pool.median_ms.idxmin()]
        modes = pool.groupby('mode').median_ms.min().to_dict()
        variants = pool.loc[pool['mode'] == 'OneWindow'].groupby('variant').median_ms.min().to_dict()
        geo = rows.iloc[0]
        cells.append(dict(zip(KEY, key), bytes=int(geo.input_bytes), output_bytes=int(geo.output_bytes), sms=int(geo.device_sms),
                          winner_mode=winner['mode'], winner_time=float(winner.median_ms),
                          mode_time=modes, variant_time=variants,
                          ow_best=min(variants.values(), default=math.inf),
                          stable_candidates=len(stable_rows)))
    return cells


def features(cell):
    r, k, outer, b, sms = (cell[x] for x in ('R', 'K', 'outer', 'bytes', 'sms'))
    return np.array([math.log2(r), math.log2(k), math.log2(outer), math.log2(b),
                     math.log2(max(outer * k / sms, 1e-9)),
                     math.log2(max(math.ceil(outer / 8) / sms, 1e-9)),
                     k % 16, k % 8, k % 4], dtype=float)


def training_cells(cells):
    return [c for c in cells if c['winner_mode'] in MODE_NAMES]


def new_tree(max_leaves, min_leaf):
    return DecisionTreeClassifier(max_leaf_nodes=max_leaves, min_samples_leaf=min_leaf,
                                  random_state=0, criterion='entropy')


def learn(cells, leaves_mode=8, leaves_ow=10):
    x = np.array([features(c) for c in cells])
    labels = [c['winner_mode'] for c in cells]
    tree_mode = new_tree(leaves_mode, max(8, len(cells) // 100))
    tree_mode.fit(x, labels)
    ow_cells = [c for c in cells if c['winner_mode'] == 'OneWindow' and c['variant_time']]
    # Fastest *stable* OneWindow variant, conditioned on OneWindow genuinely winning.
    ow_labels = [min(c['variant_time'], key=c['variant_time'].get) for c in ow_cells]
    tree_ow = new_tree(leaves_ow, max(5, len(ow_cells) // 100))
    tree_ow.fit(np.array([features(c) for c in ow_cells]), ow_labels)
    return tree_mode, tree_ow


def freeze_tree(tree):
    tr = tree.tree_
    return dict(classes=[str(s) for s in tree.classes_],
                feature=[str(FEATURES[i]) if i >= 0 else None for i in tr.feature],
                threshold=[float(t) if i >= 0 else None for i, t in zip(tr.feature, tr.threshold)],
                left=tr.children_left.tolist(), right=tr.children_right.tolist(),
                prediction=[str(tree.classes_[int(np.argmax(v[0]))]) for v in tr.value])


def predict_frozen(tree, cell):
    vector = dict(zip(FEATURES, features(cell)))
    i = 0
    while tree['left'][i] != -1:
        i = tree['left'][i] if vector[tree['feature'][i]] <= tree['threshold'][i] else tree['right'][i]
    return tree['prediction'][i]


def evaluate(cells, p, label):
    ratios, ow_ratios = [], []
    unsupported_mode = unsupported_variant = 0
    records = []
    for c in cells:
        mode = predict_frozen(p['mode'], c)
        # A missing candidate is a *policy error*, NOT zero-regret default fallback.
        mode_latency = c['mode_time'].get(mode)
        if mode_latency is None:
            unsupported_mode += 1
            regret = math.nan
        else:
            regret = 100.0 * (mode_latency / c['winner_time'] - 1.0)
            ratios.append(regret)
        variant = None
        ow_regret = math.nan
        selected_latency = mode_latency
        if mode == 'OneWindow':
            variant = predict_frozen(p['one_window'], c)
            vlatency = c['variant_time'].get(variant)
            if vlatency is None:
                unsupported_variant += 1
                selected_latency = None
            else:
                selected_latency = vlatency
                ow_regret = 100.0 * (vlatency / c['winner_time'] - 1.0)
                ow_ratios.append(ow_regret)
        oracle_gbps = (c['bytes'] + c['output_bytes']) / (c['winner_time'] * 1e6)
        selected_gbps = ((c['bytes'] + c['output_bytes']) / (selected_latency * 1e6)
                         if selected_latency is not None else math.nan)
        records.append(dict(outer=c['outer'], R=c['R'], K=c['K'], dtype=c['dtype'],
                            input_bytes=c['bytes'], output_bytes=c['output_bytes'],
                            oracle_GBps=oracle_gbps, policy_GBps=selected_gbps,
                            lost_1500_GBps=int(oracle_gbps >= 1500.0 and
                                               (not math.isfinite(selected_gbps) or selected_gbps < 1500.0)),
                            expected_mode=c['winner_mode'],
                            policy_mode=mode, policy_ow_variant=variant,
                            mode_regret_pct=regret, full_regret_pct=ow_regret if mode == 'OneWindow' else regret,
                            available=int(mode_latency is not None and (mode != 'OneWindow' or not math.isnan(ow_regret)))))
    df = pd.DataFrame(records)
    full = df.loc[df.available == 1, 'full_regret_pct']
    print(f'[{label}] cells={len(cells)} unsupported_mode={unsupported_mode} '
          f'unsupported_OneWindow_variant={unsupported_variant} scored={len(full)}')
    if len(full):
        print(f'  measured-full-policy regret median={full.median():.2f}% '
              f'p90={full.quantile(.90):.2f}% p95={full.quantile(.95):.2f}% '
              f'>2%={int((full>2).sum())} >5%={int((full>5).sum())} '
              f'>10%={int((full>10).sum())} worst={full.max():.2f}%')
        print(f"  lost >=1500 GB/s physical cells={int(df.lost_1500_GBps.sum())} "
              f"(including unsupported choices); mode selections={df.policy_mode.value_counts().to_dict()}")
        print('  by-R / median regret: ' + str(df.groupby('R').full_regret_pct.median().round(2).to_dict()))
    return df


def grouped_cv(cells, axis, mode_leaves, ow_leaves):
    group = [c[axis] for c in cells]
    split = GroupKFold(n_splits=4)
    report = []
    for train, test in split.split(np.arange(len(cells)), groups=group):
        tr = [cells[int(i)] for i in train]
        te = [cells[int(i)] for i in test]
        a, b = learn(tr, mode_leaves, ow_leaves)
        policy = dict(mode=freeze_tree(a), one_window=freeze_tree(b))
        frame = evaluate(te, policy, f'CV unseen-{axis} fold')
        report.append(frame)
    both = pd.concat(report, ignore_index=True)
    n = len(both)
    covered = both.loc[both.available == 1, 'full_regret_pct']
    print(f'CV UNSEEN {axis}: coverage={len(covered)}/{n}, >5%-of-covered={int((covered>5).sum())}, '
          f'p95={covered.quantile(.95):.2f}%, missing={n-len(covered)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='887 unified selector raw benchmark CSV')
    parser.add_argument('--policy', required=True, help='JSON freeze path (write if fitting, read if evaluating)')
    parser.add_argument('--evaluate', help='Holdout raw benchmark CSV; do NOT refit')
    parser.add_argument('--details', help='Write one row per evaluated geometry')
    parser.add_argument('--mode-leaves', type=int, default=8)
    parser.add_argument('--onewindow-leaves', type=int, default=10)
    parser.add_argument('--cross-validate', action='store_true')
    args = parser.parse_args()
    train = training_cells(load_cells(args.train))
    if not args.evaluate:
        mode, ow = learn(train, args.mode_leaves, args.onewindow_leaves)
        print('MODE POLICY\n' + export_text(mode, feature_names=list(FEATURES), decimals=2))
        print('ONEWINDOW POLICY\n' + export_text(ow, feature_names=list(FEATURES), decimals=2))
        frozen = dict(version=1, hardware='RTX5090', training='887 unified selector',
                      method='bounded CART classification / regret-gated selection',
                      features=list(FEATURES), mode=freeze_tree(mode), one_window=freeze_tree(ow),
                      no_production_authority=True)
        Path(args.policy).write_text(json.dumps(frozen, indent=2) + '\n')
        print('Frozen policy:', args.policy)
        report = evaluate(train, frozen, 'training-only (not acceptance)')
        if args.details:
            report.to_csv(args.details, index=False)
        if args.cross_validate:
            grouped_cv(train, 'R', args.mode_leaves, args.onewindow_leaves)
            grouped_cv(train, 'K', args.mode_leaves, args.onewindow_leaves)
    else:
        frozen = json.loads(Path(args.policy).read_text())
        if frozen['features'] != list(FEATURES):
            raise ValueError('Policy JSON features incompatible with this analyzer')
        holdout = load_cells(args.evaluate)
        train_shapes = {(c['outer'], c['R'], c['K'], c['dtype']) for c in train}
        overlap = sum((c['outer'], c['R'], c['K'], c['dtype']) in train_shapes for c in holdout)
        if overlap:
            raise ValueError(f'{overlap} holdout cells were in training: cannot claim holdout')
        report = evaluate(holdout, frozen, 'never-seen holdout, frozen policy')
        if args.details:
            report.to_csv(args.details, index=False)


if __name__ == '__main__':
    main()
