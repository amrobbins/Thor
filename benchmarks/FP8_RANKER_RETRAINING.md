# Retraining the FP8 reduction rankers for a GPU (1172)

Thor's retained FP8 reductions choose each pass in two steps at stamp time. A trained **ranker** orders the legal
kernel shapes of the site, and a short **contest** times the top N on the GPU and keeps the fastest (N is the
reduction tuning depth: Quick, Regular or Ultra). There are two rankers:

| Ranker | Sites | Model header | Trainer | Training sweep |
| --- | --- | --- | --- | --- |
| RK geometry ranker | RK (`[outer, R, K]`, reduce R), VALUE and ARG | `Utilities/TensorOperations/Cub/Fp8RK/Fp8GeometryRankerModel.h` | `benchmarks/fp8_ranker/train.py` | `thor_fp8_retained_cold_l2_sweep --training` |
| KR shape ranker | R and KR (contiguous rows), VALUE and ARG | `Utilities/TensorOperations/Cub/Fp8KR/Fp8KRRankerModel.h` | `benchmarks/fp8_kr_ranker/train.py` | `thor_fp8_kr_training_sweep` |

Both installed models were trained on one GPU with 170 SMs: RK model `b1a7bc53…` (1165, 15,614 cells, 1–256 MiB) and
KR model `c82f0c6b…` (1159, 12,256 cells). There is no second GPU to check them on, so this document is the
portability plan. **Retrain both rankers when Thor's FP8 reductions should be tuned for a different GPU** (another
architecture, or a part with a very different SM count, L2 size or memory bandwidth).

## What a ranker trained on another GPU costs

The ranker only affects speed, never correctness. Every candidate it can return is a legal, compiled kernel, and the
contest measures the top N on the GPU that is actually running. A mis-ranked GPU therefore loses only when the fastest
shape falls outside the top N. That shows up as **regret** (the picked shape's time over the fastest shape's) in the
audits below, not as wrong results. Ultra (24 RK / 12 KR candidates) is the safety margin while a ranker is suspect.

Some ranker features describe the GPU: log2 SM count, CTAs per SM from the compiled kernels' occupancy, and rows per
SM. These let the models generalize a little across SM counts. They are no substitute for timings from the target GPU:
relative kernel speeds depend on bandwidth, L2 and the SASS the architecture produces.

**One model per build.** Each ranker is a compiled-in header, so a build carries one RK model and one KR model.
Retraining for a GPU replaces the models for every GPU that build runs on. Supporting several architectures in one
build (models selected by device) is not implemented. It would mean a model table keyed by compute capability, chosen
where the selector computes its features.

## Prerequisites

- The GPU to tune for (any architecture Thor supports, sm_80 and later). Keep it otherwise idle while sweeping: the
  samples are wall-clock GPU times.
- Python with numpy, scikit-learn and **xgboost** for the trainers. Every other analysis script needs only numpy.
- GPU memory. The sweeps' cold rings default to `--arena-gib=12` (RK), `--arena-gib=8` (KR) and
  `--output-ring-gib=3`. On a GPU with less than about 20 GB, lower them, for example `--arena-gib=6
  --output-ring-gib=1.5`. The rings must stay much larger than L2, so lower them no further than needed.
- The reuse distances in `benchmarks/ColdTiming.h` (`InputReuseBytes` = 2 GiB, `OutputReuseBytes` = 1 GiB) keep
  every launch cold for an L2 of up to about 100 MiB. For a GPU with a larger L2, raise both to at least about 10x
  the L2 size before sweeping, or inputs will be warm and the rankers will learn L2-resident speeds.

Build everything once, from the build directory:

```
cmake --build . --target thor_tests thor_fp8_retained_cold_l2_sweep thor_fp8_kr_training_sweep
```

## 1. Record the GPU's current state (optional, recommended)

Audit the installed models on the new GPU first. This tells you how much retraining gains, and the numbers are the
baseline for step 6.

```
./thor_fp8_retained_cold_l2_sweep --training --stamp --targets-mib=1,8,64,256 --max-cells=800 2> rk-stamp-before.log | gzip -1 > rk-stamp-before.csv.gz
python3 ../benchmarks/fp8_rk_stamp.py rk-stamp-before.csv.gz
./thor_fp8_kr_training_sweep --seed=1159 --random=1500 --min-kib=64 2> kr-confirm-before.log | gzip -1 > kr-confirm-before.csv.gz
python3 ../benchmarks/fp8_kr_training.py kr-confirm-before.csv.gz
```

`fp8_rk_stamp.py` section 2 gives the stamp's pick against the fastest timed candidate, by size. Section 2b says
whether the ranker's top N held the fastest shape. `fp8_kr_training.py` section 8 does the same for KR. If the picks are within 2% in at least 95% of cells and the top N
holds the oracle in at least 98%, as on the training GPU (1165: mean regret 0.0–0.3%, 96–100% within 2%; top 12
holds the oracle in 98–100%), retraining is optional.

## 2. Sweep the RK training set

```
./thor_fp8_retained_cold_l2_sweep --training --training-seed=1165 --training-random=6000 --targets-mib=1,4,16,64,256 2> rk-training.log | gzip -1 > rk-training.csv.gz
```

This is the 1165 recipe. It times every legal RK candidate (`generalShortlist`, untruncated, for VALUE;
`retainedFp8ArgCandidates` for ARG) in about 15,600 cells, and writes the C++ ranker features beside each time. Time a
slice with `--max-cells=200` first and scale `--training-random` to the time you can spend. Progress and an ETA go to
stderr every 25 cells. Keep `--training-heldout-percent` at its default (10): held-out shapes validate the model and
are never trained on. Column and method details are in `FP8_RETAINED_COLD_L2_SWEEP.md` ("Ranker training dataset").

## 3. Sweep the KR training set

```
./thor_fp8_kr_training_sweep --random=6000 --min-kib=64 2> kr-ranker.log | gzip -1 > kr-ranker.csv.gz
```

This times every shape of production's pool (`fp8KRCandidatePool`) and writes `fp8KRRankerFeatures` (24 features) per
shape. `--min-kib=64` includes small sites, so they are ranked too. Each cell's production pick is recorded as well,
which is how step 1's KR audit works.

Both sweeps time the same way (`ColdTiming.h`): cold input and output views; per candidate, one untimed launch,
then a timed probe; samples of about 1 ms of back-to-back launches. Do not change the method between the sweep and
the audits, or the comparisons stop meaning anything.

## 4. Train

```
python3 ../benchmarks/fp8_ranker/train.py rk-training.csv.gz --out-dir=fp8_ranker_out
python3 ../benchmarks/fp8_kr_ranker/train.py kr-ranker.csv.gz --out-dir=fp8_kr_ranker_out
```

Each trainer fits XGBoost `rank:ndcg`, grouped by cell. Labels grade each candidate's regret against the cell's fastest
candidate (4/3/2/1/0 at 1%, 3%, 7%, 15%). Cross-validation folds never split a shape (RK: an (R, K) shape; KR: a row
length). Each trainer writes:

- `summary.txt`: the depths and the regret curves (geomean, p95, p99 and max regret of the top N), out of fold and on
  the held-out shapes, against the installed model or heuristic order.
- `curves.csv`: every depth. `misses.csv`: the cells whose fastest shape is outside the Regular shortlist.
- The model header and parity fixtures (below), already checked against XGBoost's own margins.

**Read `summary.txt` before installing.** The new model should beat the installed order on the held-out curves. At
the production depths, worst-case regret should be close to 1.02 or better: RK N = 12, KR N = 12; on the training GPU
these were 1.017 (ARG) and 1.015 (VALUE) out of fold. If it is worse, look at `misses.csv` for a pattern (a family
or size range) before shipping.

## 5. Install (cut over)

| File | From | To |
| --- | --- | --- |
| RK model | `fp8_ranker_out/Fp8GeometryRankerModel.h` | `Utilities/TensorOperations/Cub/Fp8RK/` |
| RK parity fixtures | `fp8_ranker_out/Fp8GeometryRankerParityFixtures.h` | `test/Utilities/TensorOperations/` |
| KR model | `fp8_kr_ranker_out/Fp8KRRankerModel.h` | `Utilities/TensorOperations/Cub/Fp8KR/` |
| KR parity fixtures | `fp8_kr_ranker_out/Fp8KRRankerParityFixtures.h` | `test/Utilities/TensorOperations/` |

Then:

- **Depths.** These are user decisions, not the trainer's. The trainers print the smallest depths that meet their
  rules (Regular: p99 regret ≤ 1.02; Quick: geomean ≤ 1.01 and p95 ≤ 1.05; KR Ultra: min(pool, 2 × Regular)).
  Production keeps RK at 6 / 12 / 24 (`fp8RKTuningPlan` in `Fp8RK/Fp8RKProductionCandidates.h`) and KR at 6 / 12 / 12
  (`FP8_KR_SHORTLIST_DEPTHS` in `Fp8KR/Fp8KRSegments.h`), above the printed minimums. Lower them only if the stamp
  cost matters more than the margin.
- **Record the model.** Update the SHA256 in the first comment of `Fp8GeometryRankerTest.cpp`. The KR header carries
  its SHA in its own comment. Add a dated paragraph to `Fp8RK/README.md`: the GPU, the cell count, the size range, the
  seed and the out-of-fold and held-out regret at the production depths.
- **Features must match.** If the C++ features (`rankerFeatures`, `fp8KRRankerFeatures`) changed since the sweep, the
  sweep is stale: re-sweep. The parity tests catch a mismatch between the header and the fixtures, but not a
  dataset swept with older features.

## 6. Verify

```
cmake --build . --target thor_tests thor_fp8_retained_cold_l2_sweep thor_fp8_kr_training_sweep
./thor_tests --gtest_filter='Fp8GeometryRanker*:Fp8KRRanker*:CubReduction*:CubArgReduction*:Fp8*:DenseReductionSelectionCache*'
./thor_fp8_retained_cold_l2_sweep --training --stamp --targets-mib=1,8,64,256 --max-cells=800 2> rk-stamp-after.log | gzip -1 > rk-stamp-after.csv.gz
python3 ../benchmarks/fp8_rk_stamp.py rk-stamp-after.csv.gz
./thor_fp8_kr_training_sweep --seed=1159 --random=1500 --min-kib=64 2> kr-confirm-after.log | gzip -1 > kr-confirm-after.csv.gz
python3 ../benchmarks/fp8_kr_training.py kr-confirm-after.csv.gz --compare=kr-confirm-before.csv.gz
```

- `ModelMatchesTrainingFixturesExactly` (both rankers): the installed header reproduces the trainer's margins bit for
  bit. If it fails, the files are from different runs.
- `fp8_rk_stamp.py`: section 1 is the stamp wall time by size, section 2 the picks against the oracle, section 2b
  the top-N coverage and the sweep's own sample spread. The audit cells use a different seed from the training
  cells, so this is an out-of-sample check.
- `fp8_kr_training.py`: section 8 is the production pick (the stamp at each site) against the fastest shape per
  cell; section 3 the regret of the first N listed shapes; with `--compare`, section 9 the run-to-run agreement with
  the step 1 run.

Accept when the picks are at least as good as step 1 and the top-N coverage is at least 98% at Regular. Commit the
two headers, the two fixtures, the test comment and the README paragraph together.

## FP32 KR ranker (1182)

FP32 R / KR sites get their own ranker (one ranker per dtype, so changing one dtype's kernels never forces retraining
another's). It uses the same trainer, features and rules as the FP8 KR ranker, with `--dtype=fp32`:

| | FP32 KR ranker |
| --- | --- |
| Model header | `Utilities/TensorOperations/Cub/Fp32KR/Fp32KRRankerModel.h` |
| Parity fixtures | `test/Utilities/TensorOperations/Fp32KRRankerParityFixtures.h` |
| Depths | `FP32_KR_SHORTLIST_DEPTHS` in `Utilities/TensorOperations/Cub/Fp32KR/Fp32KRStampAutotuner.h` |
| Pool | `fp32KRCandidatePool`: the FP32 list and pool (`retainedKRCandidatePool` at element size 4), without the ARG shapes the FP32 ARG kernels cannot run |
| Test | `Fp32KRRanker*` |

Installed model (1218): `787279aa…`, 24,512 FP32 cells (64 KiB–512 MiB) on the 170-SM training GPU: 1183's 12,256
VALUE and ARG cells plus 6,128 UINT32 and 6,128 UINT64 carried-index ARG cells (1217). Out of fold, the ranked top 2
has p99 regret 1.008 (VALUE) / 1.008 (ARG) / 1.009 (carried ARG), and the top 8 holds the fastest shape in every
cell; the heuristic list needed 12. (1183's model, `35c2616f…`, trained on the VALUE and ARG cells only, had p99 1.020
at the top 2 on the carried cells.) Production runs 4 / 4 / 6 (`FP32_KR_SHORTLIST_DEPTHS`; 1184 chose 2 / 4 / 6, one step
above each printed minimum; 1219 raised Quick to Regular's 4, as four shapes are cheap to autotune). Without a trained header (a
placeholder with `TRAINED = false`) the FP32 selectors time the FP32 heuristic list's prefix at the RK depths
(6 / 12 / 24).

```
./thor_fp8_kr_training_sweep --dtypes=fp32 --random=6000 --min-kib=64 --no-production 2> fp32-kr-ranker.log | gzip -1 > fp32-kr-ranker.csv.gz
./thor_fp8_kr_training_sweep --dtypes=fp32 --ops=argmin,argmax --carried-index=uint32 --random=6000 --semantics=1 --min-kib=64 --no-production 2> fp32-kr-carried32.log | gzip -1 > fp32-kr-carried32.csv.gz
./thor_fp8_kr_training_sweep --dtypes=fp32 --ops=argmin,argmax --carried-index=uint64 --random=6000 --semantics=1 --min-kib=64 --no-production 2> fp32-kr-carried64.log | gzip -1 > fp32-kr-carried64.csv.gz
python3 ../benchmarks/fp8_kr_ranker/train.py fp32-kr-ranker.csv.gz fp32-kr-carried32.csv.gz fp32-kr-carried64.csv.gz --dtype=fp32 --out-dir=fp32_kr_ranker_out
```

- **The carried sweeps (1217).** `--carried-index` makes every ARG cell a carried-index site (FP32 values plus a cold
  UINT32 / UINT64 carried index, ranker kind 3 / 4); `train.py` reports them as the `carried_arg` group and, from the
  sweep's `frozen_margin` column, the installed model's order (`frozen`).

- **The sweep's FP32 cells.** R is in elements (row bytes 4 R) with a quarter of the grid's rows, so a cell reads
  about the same bytes as an FP8 cell. Cells alternate VALUE and ARG. `--no-production` skips the production stamp,
  which still runs FP32's pre-retained kernels until the switch.
- **The trainer keeps only the dtype's cells.** A mixed dataset (`--dtypes=fp8_e4m3,fp8_e5m2,fp32`) trains either
  ranker, and `--dtype=fp8` (the default) ignores FP32 cells.
- **Install** as for FP8: copy the two headers from `fp32_kr_ranker_out/` and set `FP32_KR_SHORTLIST_DEPTHS` (the
  same depth rules; FP8 KR runs 6 / 12 / 12, FP32 KR 4 / 4 / 6).
- **Verify:**
  - `./thor_tests --gtest_filter='Fp32KRRanker*:CubReductionFp32KR*'`;
  - a confirmation sweep with another seed (`--seed=1182 --random=1500 --min-kib=64 --dtypes=fp32`) through
    `fp8_kr_training.py`.

## FP32 RK ranker (1196)

FP32 RK sites get their own ranker too. It uses the FP8 RK trainer and rules with `--dtype=fp32`, on FP32 cells of
the RK training sweep:

| | FP32 RK ranker |
| --- | --- |
| Model header | `Utilities/TensorOperations/Cub/Fp32RK/Fp32RKRankerModel.h` |
| Parity fixtures | `test/Utilities/TensorOperations/Fp32RKRankerParityFixtures.h` |
| Depths | `FP32_RK_SHORTLIST_DEPTHS` in `Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h` |
| Pool | `fp32RKCandidatePool`: every legal, resident FP32 first stage (OneWindow, DeepRSmallK, AlignedK, RuntimeK PacketComplete and rotated Prefix), families interleaved by local rank |
| Features | `fp32RKRankerFeatures`: FP8's 26 in FP32 units (the E4M3 slot is 0) |
| Test | `Fp32RKCandidates*` (the parity check skips until a trained model is installed) |

Installed model (1230): `a37fea04…`, 21,255 FP32 cells on the 170-SM training GPU. They are 1228's sweep (14,682
VALUE and ARG cells at 1, 4, 96 and 192 MiB, after the 4-byte AlignedK packet joined the pool) and 1205's 6,573 ARG
cells (96 and 192 MiB). Out of fold, ARG / VALUE: the ranked top 3 is within 1.309 / 1.172 of the oracle in every cell
(p99 1.017 / 1.023), the top 8 within 1.031 / 1.015 and the top 13 within 1.003 / 1.008. Held out, the top 13 is the
fastest kernel in every cell. Production runs 3 / 8 / 13 (`FP32_RK_SHORTLIST_DEPTHS`, user decision 2026-11-04). The
1206 model (`aa090137…`, 19,719 cells at 96 and 192 MiB) ran 2 / 6 / 14, and the 1197 model (`8c7e2ec4…`, VALUE only)
ran 2 / 6 / 13.

The 1 and 4 MiB working sets (`--targets-mib=1,4,96,192`) carry the small, latency-bound sites. There the 4-byte
AlignedK packet wins about 37% of odd-K VALUE cells, 15% (1 MiB) and 10% (4 MiB) faster at the median than the best
other kernel. Since 1229 the sweep times every staged candidate with the continuation the retained FP32 selectors
choose (1225 / 1227). Those tunings are cached per stage, so the sweep takes longer the first time a continuation shape
appears.

```
./thor_fp8_retained_cold_l2_sweep --training --dtypes=fp32 --training-random=6000 --targets-mib=1,4,96,192 \
    | gzip -1 > fp32-rk-training.csv.gz
python3 ../benchmarks/fp8_ranker/line_rate.py fp32-rk-training.csv.gz
python3 ../benchmarks/fp8_ranker/train.py fp32-rk-training.csv.gz --dtype=fp32 --out-dir=fp32_rk_ranker_out
```

The sweep alternates VALUE and ARG cells per geometry by default, so this one file trains both. 1230 also passed
1205's ARG file, which repeats the 96 and 192 MiB ARG geometries of the same seed.

- **The sweep's FP32 cells** are VALUE and (1205) ARG. Working sets stay in bytes, so an FP32 cell has a quarter of an
  FP8 cell's elements. To add ARG to the 1197 VALUE dataset, sweep ARG alone with the same seed and train on both files
  (the trainer groups cells per file, and the same seed keeps the same held-out shapes):

  ```
  ./thor_fp8_retained_cold_l2_sweep --training --dtypes=fp32 --training-random=6000 --training-semantics=1 \
      --ops=argmin,argmax | gzip -1 > fp32-rk-arg-training.csv.gz
  python3 ../benchmarks/fp8_ranker/train.py fp32-rk-training.csv.gz fp32-rk-arg-training.csv.gz --dtype=fp32 \
      --out-dir=fp32_rk_ranker_out
  ```

  The printed Ultra depth is the deeper of the VALUE and the ARG depth.
- **The trainer** keeps only the dtype's cells and orders by the model alone: the FP32 selector has no family-leader
  prefix. Besides Quick and Regular it prints an Ultra depth, the smallest N with out-of-fold max regret at most
  `--ultra-max` (default 1.01).
- **Install:** copy `Fp32RKRankerModel.h` and `Fp32RKRankerParityFixtures.h` from `fp32_rk_ranker_out/`, and set
  `FP32_RK_SHORTLIST_DEPTHS` (Quick, Regular, Ultra) from the printed depths. Without a trained header the selector
  times the pool's prefix at 6 / 12 / 24.
- **Verify:** `./thor_tests --gtest_filter='Fp32RKCandidates*:CubReductionFp32RK*'`, then a confirmation sweep with
  another seed (`--training-seed=1196 --training-random=1500`).

## Troubleshooting

- **Large sample spreads (sweep spread over about 2% in section 2b).** The GPU was not idle, clocks were throttling,
  or the cold rings were too small. Re-run with the GPU otherwise idle, and check the reuse distances against L2.
- **The trainer's held-out curves are worse than out of fold.** The random shapes are too few for the range swept.
  Raise `--training-random` (RK) or `--random` (KR).
- **An audit pick is not among the timed configurations** (`fp8_rk_stamp.py` reports these). The stamp and the sweep
  enumerated different inventories, usually because they come from different builds. Rebuild both from one tree.
