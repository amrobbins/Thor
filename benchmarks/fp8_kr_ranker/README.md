# FP8 R / KR shape ranker (1158+)

To retrain both rankers for a GPU, end to end (sweeps, training, install, verification), follow
`benchmarks/FP8_RANKER_RETRAINING.md`.

The KR counterpart of `benchmarks/fp8_ranker` (RK). One model ranks the retained KR candidate pool for VALUE and ARG;
production autotunes the top N of that order at the stamp's tuning depth.

1. **Dataset.** Run `thor_fp8_kr_training_sweep` (any 1158+ build). Every shape of production's pool
   (`fp8KRCandidatePool`: the heuristic list, then the legal shapes around it) is timed with the RK sweep's method,
   and the C++ ranker features (`fp8KRRankerFeatures`, 24) are written beside each time (`f0..f23`). For training,
   sweep sizes from 64 KiB so small sites are ranked too, e.g.
   `./thor_fp8_kr_training_sweep --random=6000 --min-kib=64 2> kr-ranker.log | gzip -1 > kr-ranker.csv.gz`.
2. **Train and evaluate.** `python3 benchmarks/fp8_kr_ranker/train.py kr-ranker.csv.gz --out-dir=fp8_kr_ranker_out`
   (numpy, scikit-learn and xgboost). It reuses `fp8_ranker/evaluate.py` and `fp8_ranker/export.py`.
3. **Read** `fp8_kr_ranker_out/summary.txt`: the Quick, Regular and Ultra depths and the regret curves of the model
   order against the heuristic order, out of fold and on held-out row lengths, for all cells and for cells of 16 MiB
   and more. `curves.csv` has every depth; `misses.csv` lists the cells whose best shape is outside the Regular
   shortlist.
4. **Cut over** (not done by the script): copy `Fp8KRRankerModel.h` to `Utilities/TensorOperations/Cub/Fp8KR/` and
   `Fp8KRRankerParityFixtures.h` to `test/Utilities/TensorOperations/`, and set `FP8_KR_SHORTLIST_DEPTHS` in
   `Fp8KRSegments.h` to the printed depths. Until then the placeholder model has `TRAINED = false` and production
   times the heuristic list's prefix at the RK depths (6 / 12 / 24), as before.

**Labels, validation, depths, export**: as the RK ranker. Regret is time / cell oracle, graded 4/3/2/1/0 at 1%, 3%, 7%
and 15%; XGBoost `rank:ndcg` grouped by cell. Folds never split a row length R (the sweep holds out by R). Regular is
the smallest N with p99 regret <= 1.02 for VALUE and ARG; Quick the smallest N with geomean <= 1.01 and p95 <= 1.05;
Ultra is min(pool size, 2 x Regular). The exported trees are checked against XGBoost's own margins, and
`Fp8KRRankerTest` checks the installed header against the fixtures bit for bit.

**Features** (`dataset.py FEATURE_NAMES`, `Fp8KRSegments.cpp fp8KRRankerFeatures`): log2 rows, row bytes, bytes;
family (pack 0, vector 1, group 2, block 3, split 4); rows per thread; log2 lanes per row, chunk bytes, chunks, CTAs,
CTAs per SM, threads, bytes per thread, bytes per CTA, words per lane; row bytes mod 4 and mod 16; log2 rows per SM;
kind (VALUE 0, ARG index 1, ARG value and index 2); combine; transform; E5M2; log2 output bytes; the heuristic list
position (0 if unlisted); log2 SMs.
