#!/usr/bin/env python3
"""Train, validate and export Thor's FP8 RK candidate ranker (one model for VALUE and ARG). 1139+.

Inputs: one or more `thor_fp8_retained_cold_l2_sweep --training` CSVs. Requires numpy, scikit-learn and xgboost.

1. Labels: per cell, regret = time / oracle, graded (dataset.relevance; ties within 1% are equally good). Objective:
   XGBoost rank:ndcg grouped by cell. Features: the sweep's C++ rankerFeatures plus the aggregate kind (26).
2. Geometry-separated K-fold CV over the non-held-out cells ((R, K) shapes never straddle folds) gives out-of-fold
   scores; a final fit on all non-held-out cells scores the held-out shapes.
3. Regret(N) curves per kind and ordering (evaluate.py), on out-of-fold and held-out scores, against the frozen
   ranker (VALUE) and the frozen model / generator order (ARG).
4. Depths: Regular = smallest N with p99 regret <= --regular-p99 for VALUE and for ARG; Quick = smallest N with
   geomean <= --quick-geomean and p95 <= --quick-p95 for both. Chosen on out-of-fold scores, confirmed on held-out.
   The family-leader prefix (production's generalShortlist rule) is kept only if it lowers the Regular depth.
5. Export: XGBoost JSON, the generated Fp8GeometryRankerModel.h (FEATURE_COUNT = 26), parity fixtures checked
   against XGBoost's own margins, curves.csv, misses.csv and summary.txt.

1196: --dtype=fp32 trains the FP32 RK ranker (one ranker per dtype) on the datasets' FP32 cells (`--training
--dtypes=fp32`; VALUE and, from 1205, ARG cells), ordered by the model alone (the FP32 selector has no family-leader
prefix), exported as Fp32RKRankerModel.h (TRAINED = true) and Fp32RKRankerParityFixtures.h. It also prints an Ultra
depth (the smallest N whose out-of-fold max regret is <= --ultra-max for VALUE and for ARG); FP32_RK_SHORTLIST_DEPTHS in Fp32RK/Fp32RKCandidates.h is set from the
printed depths by hand. --dtype=fp8 (the default) is unchanged.

1252: --dtype=f16 trains the FP16 / BF16 RK ranker (one model for both formats; f6, the E4M3 slot, is the BF16 flag)
on the datasets' fp16 and bf16 cells (`--training --dtypes=fp16,bf16`, VALUE and ARG), as --dtype=fp32: the model
alone, Quick / Regular / Ultra printed, exported as F16RKRankerModel.h and F16RKRankerParityFixtures.h;
F16_RK_SHORTLIST_DEPTHS in F16RK/F16RKCandidates.h is set by hand (a trained model does not compile without them).

Usage: train.py fp8-training.csv.gz [more.csv.gz ...] --out-dir=fp8_ranker_out [--dtype=fp8|fp32|f16] [--seed=0]
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402
import evaluate  # noqa: E402
import export  # noqa: E402

REPORT_DEPTHS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64)

# 1196: per-dtype outputs. `cells` picks the dataset's cells of the dtype (FP8: E4M3 and E5M2 together).
DTYPES = {
    "fp8": {
        "label": "FP8",
        "cells": lambda dtype: dtype.startswith("fp8"),
        "model_header": "Fp8GeometryRankerModel.h",
        "fixtures_header": "Fp8GeometryRankerParityFixtures.h",
        "model_namespace": export.RK_MODEL_NAMESPACE,
        "parity_namespace": export.RK_PARITY_NAMESPACE,
        "trained_flag": False,
        "leaders": True,  # production's generalShortlist may keep the family-leader prefix
        "json": "fp8_geometry_ranker.json",
        "ultra": None,
    },
    "fp32": {
        "label": "FP32",
        "cells": lambda dtype: dtype == "fp32",
        "model_header": "Fp32RKRankerModel.h",
        "fixtures_header": "Fp32RKRankerParityFixtures.h",
        "model_namespace": "ThorImplementation::CubReductionInternal::Fp32RK::Fp32RKRankerModel",
        "parity_namespace": "ThorImplementation::CubReductionInternal::Fp32RK::Fp32RKRankerParity",
        "trained_flag": True,
        "leaders": False,  # fp32RKRankedKernels orders by the margin alone
        "json": "fp32_rk_ranker.json",
        "ultra": "FP32_RK_SHORTLIST_DEPTHS in Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h",
    },
    "f16": {  # 1252: FP16 and BF16 together (the BF16 flag is a feature)
        "label": "FP16 / BF16",
        "cells": lambda dtype: dtype in ("fp16", "bf16"),
        "model_header": "F16RKRankerModel.h",
        "fixtures_header": "F16RKRankerParityFixtures.h",
        "model_namespace": "ThorImplementation::CubReductionInternal::F16RK::F16RKRankerModel",
        "parity_namespace": "ThorImplementation::CubReductionInternal::F16RK::F16RKRankerParity",
        "trained_flag": True,
        "leaders": False,  # f16RKRankedKernels orders by the margin alone
        "json": "f16_rk_ranker.json",
        "ultra": "F16_RK_SHORTLIST_DEPTHS in Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h",
    },
}


KIND_GROUPS = ("value", "arg", "carried_arg")  # 1211: carried-index ARG cells are reported and gated as their own group


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


def orderings(cell, scores, leaders=True):
    result = {"model": evaluate.order_by_scores(cell, scores)}
    if leaders:
        result["model_leaders"] = evaluate.order_by_scores(cell, scores, leaders=True)
    if cell.kind == "value":
        result["frozen"] = evaluate.list_order(cell)
    else:
        frozen = [c.frozen_margin for c in cell.candidates]
        result["frozen"] = evaluate.order_by_scores(cell, frozen, leaders=True)
        result["list"] = evaluate.list_order(cell)
    return result


def curves_for(cells, scores, max_n, leaders=True):
    """{(kind_group, ordering): [curve per cell]} and the orders themselves."""
    curves, orders = {}, []
    for cell, cell_scores in zip(cells, scores):
        cell_orders = orderings(cell, cell_scores, leaders)
        orders.append(cell_orders)
        for name, order in cell_orders.items():
            curves.setdefault((kind_group(cell), name), []).append(evaluate.regret_curve(cell, order, max_n))
    return curves, orders


def choose_depths(curves, max_n, ordering, args):
    regular, quick = [], []
    for kind in KIND_GROUPS:
        series = curves.get((kind, ordering))
        if not series:
            continue
        regular.append(evaluate.smallest_depth(series, max_n, lambda s: s["p99"] <= args.regular_p99))
        quick.append(evaluate.smallest_depth(
            series, max_n, lambda s: s["geomean"] <= args.quick_geomean and s["p95"] <= args.quick_p95))
    pick = lambda depths: None if (not depths or None in depths) else max(depths)  # noqa: E731
    return pick(regular), pick(quick)


def tuning_cost(cells, orders, ordering, n):
    """Mean over cells of (sum of the first n candidates' times) / oracle: the autotune's timing cost per round."""
    total = 0.0
    for cell, cell_orders in zip(cells, orders):
        order = cell_orders[ordering][:n]
        total += sum(cell.candidates[i].median_ms for i in order) / cell.oracle_ms
    return total / len(cells) if cells else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--regular-p99", type=float, default=1.02)
    parser.add_argument("--quick-geomean", type=float, default=1.01)
    parser.add_argument("--quick-p95", type=float, default=1.05)
    parser.add_argument("--fixtures", type=int, default=256)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp8",
                        help="the ranker to train: its dtype's cells only (default fp8)")
    parser.add_argument("--ultra-max", type=float, default=1.01,
                        help="--dtype=fp32 / f16: the printed Ultra depth is the smallest N with max regret <= this")
    args = parser.parse_args()
    spec = DTYPES[args.dtype]
    try:
        import numpy as np
        import xgboost
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        parser.error(f"{exc}. Install numpy, scikit-learn and xgboost in the benchmark environment.")

    cells, provenance = [], []
    for path in args.datasets:
        header, loaded = dataset.load(path)
        loaded = [c for c in loaded if spec["cells"](c.dtype)]
        cells.extend(loaded)  # cells keep file order; grouping uses object order, not cell ids
        provenance.append(f"{os.path.basename(path)} seed={header.get('seed')} sm_count={header.get('sm_count')} "
                          f"{args.dtype}_cells={len(loaded)}")
    if not cells:
        parser.error(f"no {args.dtype} cells in the datasets")
    train_cells = [c for c in cells if not c.heldout]
    held_cells = [c for c in cells if c.heldout]
    if not train_cells or not held_cells:
        parser.error("need both held-out and training cells (see --training-heldout-percent)")
    max_n = max(len(c.candidates) for c in cells)
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, qid = build_matrix(np, train_cells)
    shapes = sorted({c.shape for c in train_cells})
    shape_id = {s: i for i, s in enumerate(shapes)}
    cell_group = np.asarray([shape_id[c.shape] for c in train_cells], dtype=np.int32)
    folds = min(args.folds, len(shapes))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    row_cell = qid  # qid is the training-cell index
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=folds).split(
            np.zeros(len(train_cells)), groups=cell_group)):
        if set(cell_group[train_idx]) & set(cell_group[test_idx]):
            raise AssertionError("geometry leakage between folds")
        train_rows = np.isin(row_cell, train_idx)
        test_rows = np.isin(row_cell, test_idx)
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

    oof_curves, oof_orders = curves_for(train_cells, split_scores(train_cells, oof), max_n, spec["leaders"])
    held_curves, held_orders = curves_for(held_cells, split_scores(held_cells, held_scores), max_n, spec["leaders"])

    with open(os.path.join(args.out_dir, "curves.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "kind", "ordering", "n", "cells", "geomean", "p95", "p99", "max", "exact"])
        for subset, curves in (("oof", oof_curves), ("heldout", held_curves)):
            for (kind, name), series in sorted(curves.items()):
                for n in range(1, max_n + 1):
                    s = evaluate.summarize(series, n)
                    writer.writerow([subset, kind, name, n, s["cells"], f"{s['geomean']:.6f}", f"{s['p95']:.6f}",
                                     f"{s['p99']:.6f}", f"{s['max']:.6f}", f"{s['exact']:.4f}"])

    lines = [f"{spec['label']} RK ranker: {len(cells)} cells ({len(train_cells)} train in {len(shapes)} shapes, "
             f"{len(held_cells)} held out), {len(X) + len(Xh)} candidates, max {max_n} per cell",
             "datasets: " + "; ".join(provenance), ""]
    depth = {}
    for name in ("model", "model_leaders") if spec["leaders"] else ("model",):
        depth[name] = choose_depths(oof_curves, max_n, name, args)
        lines.append(f"{name}: Regular depth {depth[name][0]}, Quick depth {depth[name][1]} (out-of-fold)")
    regular_plain = depth["model"][0]
    regular_leaders = depth["model_leaders"][0] if spec["leaders"] else None
    use_leaders = regular_leaders is not None and (regular_plain is None or regular_leaders < regular_plain)
    chosen = "model_leaders" if use_leaders else "model"
    regular, quick = depth[chosen]
    lines += [f"chosen ordering: {chosen} (leader prefix {'kept' if use_leaders else 'dropped'}); "
              f"REGULAR_DEPTH={regular} QUICK_DEPTH={quick}"]
    ultra = None
    if spec["ultra"]:
        # 1205: VALUE and ARG cells (both kinds present since the FP32 ARG sweep): the deeper of the two depths.
        for kind in KIND_GROUPS:
            series = oof_curves.get((kind, chosen))
            if not series:
                continue
            depth_kind = evaluate.smallest_depth(series, max_n, lambda s: s["max"] <= args.ultra_max)
            if depth_kind is None:
                ultra = None
                break
            ultra = depth_kind if ultra is None else max(ultra, depth_kind)
        lines.append(f"ULTRA_DEPTH={ultra} (out-of-fold max regret <= {args.ultra_max}); set {spec['ultra']} "
                     f"from Quick / Regular / Ultra")
    lines.append("")
    for subset, curves, subset_cells, subset_orders in (("out-of-fold", oof_curves, train_cells, oof_orders),
                                                         ("held-out", held_curves, held_cells, held_orders)):
        lines.append(f"{subset} regret at depth N (geomean / p95 / p99 / max, exact-oracle fraction):")
        for (kind, name), series in sorted(curves.items()):
            lines.append(f"  {kind:11} {name}:")
            for n in sorted(set(REPORT_DEPTHS) | {d for d in (regular, quick, ultra) if d}):
                if n > max_n:
                    continue
                s = evaluate.summarize(series, n)
                mark = (" <- Regular" if n == regular else " <- Quick" if n == quick else
                        " <- Ultra" if n == ultra else "")
                lines.append(f"    N={n:3d}  {s['geomean']:.4f} / {s['p95']:.4f} / {s['p99']:.4f} / {s['max']:.4f}"
                             f"  exact {s['exact']:.3f}{mark}")
        for kind in KIND_GROUPS:
            kind_cells = [(c, o) for c, o in zip(subset_cells, subset_orders) if kind_group(c) == kind]
            for label, n in (("Regular", regular), ("Quick", quick)):
                if n and kind_cells:
                    cost = tuning_cost([c for c, _ in kind_cells], [o for _, o in kind_cells], chosen, n)
                    lines.append(f"  {kind} {label} N={n}: timing cost per autotune round {cost:.1f}x one oracle run")
        lines.append("")

    if regular:
        with open(os.path.join(args.out_dir, "misses.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subset", "kind", "outer", "R", "K", "dtype", "operation", "regret_at_regular",
                             "oracle_family", "oracle_configuration", "oracle_rank_in_order"])
            for subset, subset_cells, subset_orders in (("oof", train_cells, oof_orders),
                                                        ("heldout", held_cells, held_orders)):
                for cell, cell_orders in zip(subset_cells, subset_orders):
                    order = cell_orders[chosen]
                    regret = evaluate.regret_curve(cell, order, regular)[-1]
                    if regret > 1.0:
                        best = min(range(len(cell.candidates)), key=lambda i: cell.candidates[i].median_ms)
                        writer.writerow([subset, cell.kind, cell.outer, cell.r, cell.k, cell.dtype, cell.operation,
                                         f"{regret:.4f}", cell.candidates[best].family,
                                         cell.candidates[best].configuration, order.index(best) + 1])

    # Export the final model and check the C++ evaluation order against XGBoost's own margins.
    model_path = os.path.join(args.out_dir, spec["json"])
    final.get_booster().save_model(model_path)
    model_json, model_bytes = export.load_model_json(model_path)
    trees = export.parse_trees(model_json)
    base = str(model_json["learner"]["learner_model_param"]["base_score"]).strip("[]")
    base_score = float(base)
    all_rows = [c.features for cell in cells for c in cell.candidates]
    rng = random.Random(args.seed)
    sample = rng.sample(all_rows, min(args.fixtures, len(all_rows)))
    xgb_margin = final.get_booster().predict(xgboost.DMatrix(np.asarray(sample, dtype=np.float32)), output_margin=True)
    worst = max(abs(export.margin(trees, row) - (float(m) - base_score)) for row, m in zip(sample, xgb_margin))
    if worst > 1e-3:
        raise AssertionError(f"exported tree evaluation disagrees with XGBoost by {worst}")
    provenance_text = f"{len(cells)} cells, seed {args.seed}, {chosen}, Regular {regular}, Quick {quick}"
    if spec["ultra"]:
        provenance_text = f"{args.dtype}, {provenance_text}, Ultra {ultra}"
    sha = export.write_header(os.path.join(args.out_dir, spec["model_header"]), trees, dataset.FEATURE_COUNT,
                              model_bytes, provenance_text, namespace=spec["model_namespace"],
                              trained_flag=spec["trained_flag"])
    export.write_parity_fixtures(os.path.join(args.out_dir, spec["fixtures_header"]), trees, sample,
                                 dataset.FEATURE_COUNT, sha, namespace=spec["parity_namespace"])
    lines.append(f"exported {len(trees)} trees, model SHA256 {sha}; parity vs XGBoost max |diff| {worst:.2e} "
                 f"over {len(sample)} fixtures")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
