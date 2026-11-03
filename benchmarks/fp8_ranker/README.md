# FP8 RK ranker training (1139+)

To retrain both rankers for a GPU, end to end (sweeps, training, install, verification), follow
`benchmarks/FP8_RANKER_RETRAINING.md`.

This trains one ranker for retained FP8 RK VALUE and ARG candidates. It replaces `experiment_fp8_geometry_ranker.py`
and the frozen Thor1011 model.

1. **Dataset.** Run `thor_fp8_retained_cold_l2_sweep --training` (see `FP8_RETAINED_COLD_L2_SWEEP.md`). Every legal
   candidate in every cell is timed end to end, and the C++ ranker features are written beside its time.
2. **Train and evaluate.** Run `python3 benchmarks/fp8_ranker/train.py fp8-training.csv.gz --out-dir=fp8_ranker_out`.
   It needs numpy, scikit-learn and xgboost.
3. **Read** `fp8_ranker_out/summary.txt`. It gives the Regular and Quick depths and regret curves against the frozen
   ranker (VALUE) and against the frozen model or generator order (ARG). The curves are given out of fold and on the
   held-out shapes. `curves.csv` has every depth, and `misses.csv` lists the cells where the oracle falls outside the
   Regular shortlist.
4. **Cut over.** Not done by this script. The generated `Fp8GeometryRankerModel.h` (26 features: the 25
   `rankerFeatures` plus the aggregate kind) and `Fp8GeometryRankerParityFixtures.h` replace the frozen model in
   `Utilities/TensorOperations/Cub/Fp8RK/`. `rankerFeatures` gains the kind, and the depths become constants.

**Labels.** Each candidate's regret is its time divided by the cell oracle, graded 4/3/2/1/0 at 1%, 3%, 7% and 15%.
Candidates within 1% of the oracle count as equally good. The objective is XGBoost `rank:ndcg`, grouped by cell.

**Validation.** Cross-validation folds never split an (R, K) shape, and the held-out shapes are never trained on.

**Depth rules.**
- Regular is the smallest N with p99 regret ≤ 1.02, for VALUE and for ARG.
- Quick is the smallest N with geomean ≤ 1.01 and p95 ≤ 1.05, for both.
- Depths are chosen from the out-of-fold curves and checked on the held-out shapes.
- The family-leader prefix (each family's leader first, production's current rule) is kept only if it lowers the
  Regular depth.

**Export.** The exported trees are evaluated in the C++ order (float32, strictly less-than, leaves summed in tree
order). They are checked against XGBoost's own margins before the header is written, and the fixtures hold the exact
margins the C++ model must reproduce.
