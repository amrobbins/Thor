#!/usr/bin/env python3
"""Train, validate and export Thor's R / KR shape rankers: FP8 (1158+) or, with --dtype=fp32, FP32 (1182). One model
per dtype (user decision 2026-11-02, so a dtype's kernels can change without retraining the others), each for VALUE
and ARG, trained only on that dtype's cells of the datasets.

The RK ranker's method (benchmarks/fp8_ranker), on the KR pool:
1. Labels: per cell, regret = time / oracle, graded (dataset.relevance; ties within 1% are equally good). Objective:
   XGBoost rank:ndcg grouped by cell. Features: the sweep's C++ fp8KRRankerFeatures (24).
2. Geometry-separated K-fold CV over the non-held-out cells (a row length R never straddles folds) gives out-of-fold
   scores; a final fit on all non-held-out cells scores the held-out row lengths.
3. Regret(N) curves per kind, for the model order and for the heuristic order (fp8KRCandidateShapes, then the rest of
   the pool), on out-of-fold and held-out scores, for all cells and for large ones (>= --large-mib).
4. Depths: Regular = smallest N with p99 regret <= --regular-p99 for VALUE and for ARG; Quick = smallest N with
   geomean <= --quick-geomean and p95 <= --quick-p95 for both; Ultra = min(pool size, 2 x Regular). Chosen on
   out-of-fold scores, confirmed on held-out.
5. Export: XGBoost JSON, the generated Fp8KRRankerModel.h (TRAINED = true) and Fp8KRRankerParityFixtures.h (margins
   checked against XGBoost's own), curves.csv, misses.csv and summary.txt.

Cut over, FP8: copy Fp8KRRankerModel.h to Utilities/TensorOperations/Cub/Fp8KR/, Fp8KRRankerParityFixtures.h to
test/Utilities/TensorOperations/, and set FP8_KR_SHORTLIST_DEPTHS in Fp8KRSegments.h to the printed depths.
Cut over, FP32: copy Fp32KRRankerModel.h to Utilities/TensorOperations/Cub/Fp32KR/, Fp32KRRankerParityFixtures.h to
test/Utilities/TensorOperations/, and set FP32_KR_SHORTLIST_DEPTHS in Fp32KR/Fp32KRStampAutotuner.h.

Usage: train.py kr-training.csv.gz [more.csv.gz ...] --out-dir=fp8_kr_ranker_out [--dtype=fp8|fp32] [--seed=0]
Requires numpy, scikit-learn and xgboost.
"""

import argparse
import csv
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fp8_ranker"))
sys.path.insert(0, HERE)  # first: this directory's dataset.py, not the RK one
import dataset  # noqa: E402  (fp8_kr_ranker/dataset.py)
import evaluate  # noqa: E402  (fp8_ranker/evaluate.py)
import export  # noqa: E402  (fp8_ranker/export.py)

REPORT_DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24)
GENERATOR = "benchmarks/fp8_kr_ranker/train.py"

# 1182: per-dtype outputs. `cells` picks the dataset's cells of the dtype (FP8: E4M3 and E5M2 together).
DTYPES = {
    "fp8": {
        "label": "FP8",
        "cells": lambda dtype: dtype.startswith("fp8"),
        "model_header": "Fp8KRRankerModel.h",
        "fixtures_header": "Fp8KRRankerParityFixtures.h",
        "model_namespace": "ThorImplementation::CubReductionInternal::Fp8KR::Fp8KRRankerModel",
        "parity_namespace": "ThorImplementation::CubReductionInternal::Fp8KR::Fp8KRRankerParity",
        "depths": "set in Fp8KRSegments.h: FP8_KR_SHORTLIST_DEPTHS",
        "json": "fp8_kr_ranker.json",
    },
    "fp32": {
        "label": "FP32",
        "cells": lambda dtype: dtype == "fp32",
        "model_header": "Fp32KRRankerModel.h",
        "fixtures_header": "Fp32KRRankerParityFixtures.h",
        "model_namespace": "ThorImplementation::CubReductionInternal::Fp32KR::Fp32KRRankerModel",
        "parity_namespace": "ThorImplementation::CubReductionInternal::Fp32KR::Fp32KRRankerParity",
        "depths": "set in Fp32KR/Fp32KRStampAutotuner.h: FP32_KR_SHORTLIST_DEPTHS",
        "json": "fp32_kr_ranker.json",
    },
}


# 1217: carried-index ARG cells (the sweep's --carried-index) are their own group.
KIND_GROUPS = ("value", "arg", "carried_arg")


def kind_group(cell):
    if cell.kind == "value":
        return "value"
    return "carried_arg" if cell.kind.startswith("carried_") else "arg"


def build_matrix(np, cells):
    features, labels, qid = [], [], []
    for q, cell in enumerate(cells):
        for candidate, regret in zip(cell.candidates, cell.regrets()):
            features.append(candidate.features)
            labels.append(dataset.relevance(regret))
            qid.append(q)
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32),
            np.asarray(qid, dtype=np.int32))


def make_model(xgboost, args):
    return xgboost.XGBRanker(objective="rank:ndcg", n_estimators=args.n_estimators, max_depth=args.max_depth,
                             learning_rate=args.learning_rate, subsample=0.85, colsample_bytree=0.9,
                             min_child_weight=4, reg_lambda=2.0, tree_method="hist", random_state=args.seed,
                             n_jobs=args.jobs)


def split_scores(cells, flat):
    out, start = [], 0
    for cell in cells:
        out.append([float(x) for x in flat[start:start + len(cell.candidates)]])
        start += len(cell.candidates)
    if start != len(flat):
        raise AssertionError("score/candidate boundary mismatch")
    return out


def curves_for(cells, scores, max_n, large_bytes):
    """{(subset, kind_group, ordering): [curve per cell]}, subset 'all' or 'large'; and the orders."""
    curves, orders = {}, []
    for cell, cell_scores in zip(cells, scores):
        cell_orders = {"model": evaluate.order_by_scores(cell, cell_scores), "list": evaluate.list_order(cell)}
        frozen = [c.frozen_margin for c in cell.candidates]
        if all(math.isfinite(m) for m in frozen):  # 1217: the installed model's order (sweeps 1217+)
            cell_orders["frozen"] = evaluate.order_by_scores(cell, frozen)
        orders.append(cell_orders)
        for name, order in cell_orders.items():
            curve = evaluate.regret_curve(cell, order, max_n)
            curves.setdefault(("all", kind_group(cell), name), []).append(curve)
            if cell.input_bytes >= large_bytes:
                curves.setdefault(("large", kind_group(cell), name), []).append(curve)
    return curves, orders


def choose_depths(curves, max_n, args):
    regular, quick = [], []
    for kind in KIND_GROUPS:
        series = curves.get(("all", kind, "model"))
        if not series:
            continue
        regular.append(evaluate.smallest_depth(series, max_n, lambda s: s["p99"] <= args.regular_p99))
        quick.append(evaluate.smallest_depth(
            series, max_n, lambda s: s["geomean"] <= args.quick_geomean and s["p95"] <= args.quick_p95))
    pick = lambda depths: None if (not depths or None in depths) else max(depths)  # noqa: E731
    return pick(regular), pick(quick)


def tuning_cost(cells, orders, n):
    """Mean over cells of (sum of the first n candidates' times) / oracle: one autotune round's timing cost."""
    total = 0.0
    for cell, cell_orders in zip(cells, orders):
        total += sum(cell.candidates[i].median_ms for i in cell_orders["model"][:n]) / cell.oracle_ms
    return total / len(cells) if cells else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp8",
                        help="the ranker to train: its dtype's cells only (default fp8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--regular-p99", type=float, default=1.02)
    parser.add_argument("--quick-geomean", type=float, default=1.01)
    parser.add_argument("--quick-p95", type=float, default=1.05)
    parser.add_argument("--large-mib", type=float, default=16)
    parser.add_argument("--fixtures", type=int, default=256)
    args = parser.parse_args()
    try:
        import numpy as np
        import xgboost
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        parser.error(f"{exc}. Install numpy, scikit-learn and xgboost in the benchmark environment.")

    spec = DTYPES[args.dtype]
    cells, provenance = [], []
    for path in args.datasets:
        header, loaded = dataset.load(path)
        loaded = [c for c in loaded if spec["cells"](c.dtype)]
        cells.extend(loaded)
        provenance.append(f"{os.path.basename(path)} seed={header.get('seed')} sm_count={header.get('sm_count')} "
                          f"version={header.get('version')} {args.dtype}_cells={len(loaded)}")
    if not cells:
        parser.error(f"no {args.dtype} cells in the datasets (sweep with --dtypes including {args.dtype})")
    train_cells = [c for c in cells if not c.heldout]
    held_cells = [c for c in cells if c.heldout]
    if not train_cells or not held_cells:
        parser.error("need both held-out and training cells (see the sweep's --heldout-percent)")
    max_n = max(len(c.candidates) for c in cells)
    large_bytes = args.large_mib * 2**20
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, qid = build_matrix(np, train_cells)
    shapes = sorted({c.shape for c in train_cells})
    shape_id = {s: i for i, s in enumerate(shapes)}
    cell_group = np.asarray([shape_id[c.shape] for c in train_cells], dtype=np.int32)
    folds = min(args.folds, len(shapes))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=folds).split(
            np.zeros(len(train_cells)), groups=cell_group)):
        if set(cell_group[train_idx]) & set(cell_group[test_idx]):
            raise AssertionError("geometry leakage between folds")
        train_rows = np.isin(qid, train_idx)
        test_rows = np.isin(qid, test_idx)
        model = make_model(xgboost, args)
        model.fit(X[train_rows], y[train_rows], qid=qid[train_rows])
        oof[test_rows] = model.predict(X[test_rows])
        print(f"fold {fold + 1}/{folds}: {len(test_idx)} unseen cells", flush=True)
    if not np.isfinite(oof).all():
        raise AssertionError("missing out-of-fold score")

    final = make_model(xgboost, args)
    final.fit(X, y, qid=qid)
    Xh, _, _ = build_matrix(np, held_cells)
    held_scores = final.predict(Xh)

    oof_curves, oof_orders = curves_for(train_cells, split_scores(train_cells, oof), max_n, large_bytes)
    held_curves, held_orders = curves_for(held_cells, split_scores(held_cells, held_scores), max_n, large_bytes)

    with open(os.path.join(args.out_dir, "curves.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fit", "subset", "kind", "ordering", "n", "cells", "geomean", "p95", "p99", "max", "exact"])
        for fit, curves in (("oof", oof_curves), ("heldout", held_curves)):
            for (subset, kind, name), series in sorted(curves.items()):
                for n in range(1, max_n + 1):
                    s = evaluate.summarize(series, n)
                    writer.writerow([fit, subset, kind, name, n, s["cells"], f"{s['geomean']:.6f}", f"{s['p95']:.6f}",
                                     f"{s['p99']:.6f}", f"{s['max']:.6f}", f"{s['exact']:.4f}"])

    regular, quick = choose_depths(oof_curves, max_n, args)
    ultra = min(max_n, 2 * regular) if regular else None
    lines = [f"{spec['label']} KR ranker: {len(cells)} cells ({len(train_cells)} train in {len(shapes)} row lengths, "
             f"{len(held_cells)} held out), {len(X) + len(Xh)} candidates, max {max_n} per cell",
             "datasets: " + "; ".join(provenance), "",
             f"depths (out-of-fold): Quick {quick}, Regular {regular}, Ultra {ultra}",
             f"{spec['depths']}{{{quick}, {regular}, {ultra}}}", ""]
    for fit, curves, fit_cells, fit_orders in (("out-of-fold", oof_curves, train_cells, oof_orders),
                                               ("held-out", held_curves, held_cells, held_orders)):
        lines.append(f"{fit} regret at depth N (geomean / p95 / p99 / max, exact-oracle fraction):")
        for (subset, kind, name), series in sorted(curves.items()):
            lines.append(f"  {subset:5} {kind:11} {name}: ({len(series)} cells)")
            for n in sorted(set(REPORT_DEPTHS) | {d for d in (regular, quick) if d}):
                if n > max_n:
                    continue
                s = evaluate.summarize(series, n)
                mark = " <- Regular" if n == regular else " <- Quick" if n == quick else ""
                lines.append(f"    N={n:3d}  {s['geomean']:.4f} / {s['p95']:.4f} / {s['p99']:.4f} / {s['max']:.4f}"
                             f"  exact {s['exact']:.3f}{mark}")
        for kind in KIND_GROUPS:
            kind_cells = [(c, o) for c, o in zip(fit_cells, fit_orders) if kind_group(c) == kind]
            for label, n in (("Quick", quick), ("Regular", regular), ("Ultra", ultra)):
                if n and kind_cells:
                    cost = tuning_cost([c for c, _ in kind_cells], [o for _, o in kind_cells], n)
                    lines.append(f"  {kind} {label} N={n}: timing cost per autotune round {cost:.1f}x one oracle run")
        lines.append("")

    if regular:
        with open(os.path.join(args.out_dir, "misses.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fit", "kind", "K", "R", "input_bytes", "dtype", "operation", "regret_at_regular",
                             "oracle_shape", "oracle_rank_in_model_order"])
            for fit, fit_cells, fit_orders in (("oof", train_cells, oof_orders), ("heldout", held_cells, held_orders)):
                for cell, cell_orders in zip(fit_cells, fit_orders):
                    order = cell_orders["model"]
                    regret = evaluate.regret_curve(cell, order, regular)[-1]
                    if regret > 1.0:
                        best = min(range(len(cell.candidates)), key=lambda i: cell.candidates[i].median_ms)
                        writer.writerow([fit, cell.kind, cell.k, cell.r, cell.input_bytes, cell.dtype, cell.operation,
                                         f"{regret:.4f}", cell.candidates[best].configuration, order.index(best) + 1])

    model_path = os.path.join(args.out_dir, spec["json"])
    final.get_booster().save_model(model_path)
    model_json, model_bytes = export.load_model_json(model_path)
    trees = export.parse_trees(model_json)
    base_score = float(str(model_json["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    all_rows = [c.features for cell in cells for c in cell.candidates]
    sample = random.Random(args.seed).sample(all_rows, min(args.fixtures, len(all_rows)))
    xgb_margin = final.get_booster().predict(xgboost.DMatrix(np.asarray(sample, dtype=np.float32)), output_margin=True)
    worst = max(abs(export.margin(trees, row) - (float(m) - base_score)) for row, m in zip(sample, xgb_margin))
    if worst > 1e-3:
        raise AssertionError(f"exported tree evaluation disagrees with XGBoost by {worst}")
    sha = export.write_header(os.path.join(args.out_dir, spec["model_header"]), trees, dataset.FEATURE_COUNT,
                              model_bytes, f"{args.dtype}, {len(cells)} cells, seed {args.seed}, Quick {quick}, "
                                           f"Regular {regular}, Ultra {ultra}", namespace=spec["model_namespace"],
                              generator=GENERATOR, trained_flag=True)
    export.write_parity_fixtures(os.path.join(args.out_dir, spec["fixtures_header"]), trees, sample,
                                 dataset.FEATURE_COUNT, sha, namespace=spec["parity_namespace"], generator=GENERATOR)
    lines.append(f"exported {len(trees)} trees, model SHA256 {sha}; parity vs XGBoost max |diff| {worst:.2e} "
                 f"over {len(sample)} fixtures")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
