# Permanent retained FP8 RK cold-L2 benchmark (Thor1053+)

This reinstates the fast census from Thor1012 **without reinstating deleted
CUDA experiments**. It calls `Fp8RK::launchPhysicalFirstStage` through the
current `SelectedIsolatedBenchmark` executable, which also owns every selected
FP32 continuation. It enumerates the retained legal/resident physical
candidate set up to rank 128 (the local retained inventories are smaller than
this bound). The refitted geometry-ranker order is used for `rank28`.

## Run

From `~/Thor/build` (or the configured CMake build directory):

```bash
cmake --build . --target thor_fp8_retained_cold_l2_sweep -j 8

set -o pipefail
./thor_fp8_retained_cold_l2_sweep --output=fp32 |
    gzip -1 > fp8-retained-1053-fp32.csv.gz
```

For a quick CUDA/CSV sanity check:

```bash
./thor_fp8_retained_cold_l2_sweep --max-cells=2 --arena-gib=1 \
    > fp8-retained-smoke.csv
```

To time only AlignedK, use `--family=aligned_k` (also supports
`one_window`, `deep_r_small_k`, `runtime_k`). Unrelated residency queries
and physical plan construction are skipped. To compare against **historical
same-FP8 output** benchmark measurements, use `--output=same_fp8`. Default
is the production-relevant `--output=fp32`.

## Method and interpretation

- 247 historical R/K shapes, 24/96/192 MiB input targets, two FP8 dtypes,
  12 operation/transform combinations: **17,784 cells**.
- One 12 GiB rotating FP8 input allocation, initialized once; every input
  view is 256-byte aligned (1163; cudaMalloc's alignment, 32 before) with at
  least 128 bytes of trailing padding.
- One additional 2 GiB GPU workspace allocation supplies **overlapping,
  stream-ordered output and scratch views**; 256-byte starts and 128-byte
  trailing gaps are preserved. No per-cell `cudaMalloc` is required.
- Up to 128 cells enqueued per batch, one stream synchronization at the end
  of each measurement batch; keep each candidate's output/scratch alias,
  continuation, and CUDA events alive until synchronization completes.
  Reusing workspace is safe because all launches run sequentially on one stream.
- Eight throwaway warmups on the first cell; five interleaved event passes;
  per-candidate median of five GPU event times (same metric as old census).
- A complete staged candidate includes its selected production FP32
  continuation and is timed as one unit; all output finalizers are included.
- Cold inputs rotate for **every candidate measurement**, not just every
  geometry. No per-candidate cache-flush memsets or host synchronization.
- Results are descriptive; small changes require replicate GPU sweeps.
  High-throughput regressions (baseline >=1.3 and >=1.5 TB/s) take priority.
- The output-mode and physical continuation code may differ from Thor1012.
  For meaningful performance A/B, use **the same mode and harness on both
  versions**. `--output=same_fp8` preserves the old output dtype, not a
  guarantee that GPU timing is identical across compiler/runtime revisions.
- The permanent benchmark contains **no old experimental kernel launchers**;
  the retained inventory and the production executables are authoritative.

### Safety and correctness

Performance-only mode does not numerically validate outputs. Before a
production cutover, also run `thor_fp8_rk_production_cutover_smoke` and
`thor_fp8_rk_selected_continuation_gpu_smoke` plus broader edge-value tests.

The `rank28` column is the learned shortlist's position in the current
eligible inventory, not a promise that the oracle is in that prefix. Check
whether the fastest measured candidate for each workload has rank <= 28.
For family-filtered mode ranks apply only to that local eligible inventory.

Host regression checks:

```bash
python3 -m unittest benchmarks.test_fp8_retained_cold_l2_grid -v
g++ -std=c++23 -O2 -Wall -Wextra -Werror -I. \
  benchmarks/Fp8BenchmarkArenaCursorSmoke.cpp -o /tmp/fp8_arena_smoke
/tmp/fp8_arena_smoke
g++ -std=c++23 -O2 -Wall -Wextra -Werror -I. \
  benchmarks/Fp8OracleBatchQueueSmoke.cpp -o /tmp/fp8_batch_smoke
/tmp/fp8_batch_smoke
```

## Paired comparison and 28-candidate oracle coverage

Once GPU sweeps from both builds use the **same output mode**, run:

```bash
python3 ../benchmarks/compare_fp8_retained_cold_l2.py \
    fp8-retained-before.csv.gz fp8-retained-after.csv.gz
```

It matches physical candidates, separates high-throughput tiers, and checks
whether the current ranked top 28 includes the fastest measured full plan.
The comparator rejects mismatched `output=fp32` versus `output=same_fp8`.

## ARG baseline (ARG-A0, Thor1102+)

`--ops=argmin,argmax` adds production FP8 RK ARG cells (family `arg_production`, one row per cell, rank 0). ARG
ops run only when named, so the default sweep is unchanged. Each cell is timed through the same direct
`launchArg{Min,Max}Reduction` call that `StampedCubArgReduction::runOn` makes for a single reduced run (never
`ComposedDense`), with a fresh cold input view per measurement like every other candidate. The UINT32 index is
always written; the value is written too unless `--arg-outputs=index`, as FP32 (`--output=fp32`) or the input FP8
format (`--output=same_fp8`). `configuration` names today's backend: `tiled_narrow_async` (K <= 32) or
`tiled_generic_rowlanes1` (the conservative generic fallback, K > 32). The CSV header records `arg_outputs=`.

```bash
./thor_fp8_retained_cold_l2_sweep --output=fp32 --focus=arg | gzip -1 > fp8-arg-NNNN.csv.gz
python3 ../benchmarks/arg_value_compare.py fp8-arg-NNNN.csv.gz            # ARG vs VALUE min/max
python3 ../benchmarks/arg_value_compare.py fp8-arg-1102.csv.gz --after fp8-arg-NNNN.csv.gz   # ARG A/B
```

`--focus=arg` = `--shapes=all` (the 247 grid shapes, the 148 uncovered-K shapes and 11 K = 2 shapes: 403 unique),
E4M3, 24/96/192 MiB, ops `argmin,argmax,min_identity,max_identity` (4,836 cells). VALUE min/max is the line-rate
reference: `arg_value_compare.py` pairs argmin with min_identity and argmax with max_identity on the same cell and
reports ARG time over the fastest VALUE candidate (oracle) and over the fastest learned-top-28 candidate, by backend,
K class, operation and size, plus the worst cells. Throughput there counts written bytes as well as the FP8 input
(an ARG output is 8 bytes with value and index, 4 index-only; a VALUE output is 4 bytes in FP32 mode). The A-series
target is ARG <= ~1.05x VALUE. Use `--min-mib=90` to drop the noise-bound 24 MiB cells.

## Ranker training dataset (1133+)

```
./build/thor_fp8_retained_cold_l2_sweep --training > fp8_ranker_training.csv 2> fp8_ranker_training.log
```

`--training` writes the dataset that the FP8 RK geometry ranker is trained on (one model for VALUE and ARG). Each cell
is one geometry with one sampled semantics. Every legal retained candidate is timed end to end, with five interleaved
cold-L2 samples (1164: the KR sweep's method, below):

- **VALUE cells** time every candidate of `generalShortlist` (no budget truncation). A staged candidate is timed as
  its first stage plus the production-selected FP32 continuation. Output is FP32.
- **ARG cells** time every candidate of `retainedFp8ArgCandidates`, the generator ranked ARG selection uses. It covers
  the raw twins of the VALUE MIN/MAX candidates (each carrying that candidate's local rank), the single-shard rotated
  pass and the fallback depth. Each candidate runs through `launchRetainedFp8ArgPlan`, the production launcher, with
  a UINT32 index and, for `arg_value_index`, an FP32 value.

**Grid.** `allShapes()` at each target working set, plus `--training-random=N` (default 400) log-uniform (R, K)
geometries: R from 2 to 2^16, K from 2 to 2^12, under `--training-seed` (default 1133). Targets default to 96 and
192 MiB. Each geometry gets `--training-semantics=S` cells (default 2). Even cells are VALUE, with dtype and one of the
12 VALUE ops sampled. Odd cells are ARG, with argmin/argmax and index or value+index sampled. `--training-heldout-percent`
(default 10) marks (R, K) shapes as held out, covering every working set of the shape; training must use them only
for validation. `--ops`, `--dtypes`, `--targets-mib` and `--max-cells` still filter; `--family` is rejected.

**Columns.** `kind` (`value`, `arg_index` or `arg_value_index`), `cell_id`, `heldout`, `random_geometry`, `outer`,
`R`, `K`, `target_mib`, `dtype`, `operation`, `family`, `configuration`, `local_rank`, `list_position`, `staged`,
`frozen_margin`, `f0`–`f24`, then `total_median_ms`, the five samples, `launches_per_sample` and `probe_ms` (1164).

**Timing (1164).** `--training` times as `thor_fp8_kr_training_sweep` does (`benchmarks/ColdTiming.h`, shared by
both sweeps). Every launch reads a fresh cold input view and writes the cell's next rotating output view (views
spanning at least 1 GiB of a `--output-ring-gib=3` ring), so outputs pay their write-back. Each candidate first gets an
untimed launch and a timed probe launch. Each sample then times `launches_per_sample` back-to-back launches, about
`--interval-us=1000` of GPU time and at most `--max-launches=256`, and the sample is the per-launch time. Before 1164
a sample was one launch into a shared output: CUDA events resolve about 2 us, so kernels of 5-10 us were timed on a
20-40% grid (sample spreads of 8-22% below 16 MiB in the 1162/1163 stamp audits), and outputs stayed in L2.
`--continuation-audit` and the default sweep keep one launch per sample.

- `f0`–`f24` are the exact C++ `rankerFeatures` of the candidate at its local rank. For ARG they come through
  `rankerPlanForKernel(retainedFp8ArgPhysicalIdentity)`.
- `list_position` is the frozen model order for VALUE and the generator order for ARG.
- `frozen_margin` is the current model's score, so the frozen ranker is the baseline.

**Sizing.** Progress and ETA go to stderr every 25 cells. Time a slice with `--max-cells=200` and scale
`--training-random` or `--training-semantics` to reach about 20 minutes.

**FP32 (1194).** `--training --dtypes=fp32` (FP32 alone: one ranker per dtype) times FP32 VALUE cells. The candidates
are every entry of the FP32 RK pool (`Fp32RK::fp32RKCandidatePool`, what `Fp32RK::selectCachedValueSite` shortlists),
each end to end with its production FP32 continuation. The features are `fp32RKRankerFeatures` (FP8's 26 in FP32 units),
`list_position` is the pool position, `frozen_margin` the installed FP32 model's (0 for the placeholder), and `dtype`
is `fp32`. The working set stays in bytes, so an FP32 cell has a quarter of FP8's elements at the same target.
`--stamp` runs `Fp32RK::autotuneValueSite`. The header's `input_dtypes` names the dataset's dtype.

**FP32 ARG (1205).** FP32 cells alternate VALUE and ARG per geometry as FP8's do (1194-1204 were VALUE only). An FP32
ARG cell times every `Fp32RK::fp32RKArgCandidates` pass (the ARG twins of the VALUE MIN / MAX pool and the ARG-only
rotated passes: what `Fp32RK::selectCachedArgSite` shortlists) as `launchFp32RKArgPass` runs it, staged passes with
their packed continuation, writing a UINT32 index and, for `arg_value_index`, an FP32 value. The features are
`fp32RKRankerFeatures` with the ARG aggregate; `configuration` is the pass identity's name. `--stamp` runs
`Fp32RK::autotuneArgSite`. `--ops=argmin,argmax` gives ARG cells only (both cells of a geometry), for adding ARG to an
existing VALUE dataset of the same seed: the geometries and held-out shapes are the same.

**FP32 carried-index ARG (1211).** `--carried-index=uint32|uint64` (with `--training --dtypes=fp32
--ops=argmin,argmax`) makes the ARG cells carried-index (intermediate) sites: FP32 values plus a cold carried index of
that width, both streamed from the arena, so the working set counts both (an element is 8 or 12 bytes). The candidates
are `Fp32RK::fp32RKCarriedArgCandidates` (the carried twins: AlignedK and rotated from 1211, OneWindow from 1212,
DeepRSmallK from 1213), timed as `launchFp32RKCarriedArgPass` runs them (staged passes with the wide continuation),
row stride 1. The kinds are `carried_arg_index` and `carried_arg_value_index`; the features carry the carried
aggregate (3 / 4); the header's `carried_index` names the width (`line_rate.py` adds it to the input bytes). No
`--stamp` (no carried selector yet).

## Continuation audit (1134+)

```
./build/thor_fp8_retained_cold_l2_sweep --continuation-audit | gzip -1 > fp8-continuation-audit.csv.gz
python3 benchmarks/fp8_continuation_audit.py fp8-continuation-audit.csv.gz
```

This checks that the production continuations are fast before their times become training labels. Defaults: e4m3,
`sum_identity`, `max_identity` and `argmax`, `allShapes()` at 96 and 192 MiB. `--ops`, `--dtypes`, `--targets-mib`,
`--arg-outputs` and `--max-cells` filter as usual.

Each staged candidate is timed in three variants, interleaved with the rest of its cell:

- `first`: the first stage alone.
- `full`: the first stage plus the production continuation. For VALUE that is the selected FP32 passes; for ARG it is
  the packed continuation.
- `first_copy`: the first stage, then a device-to-device copy of its (L2-warm) partials.

Complete candidates are timed once, as `complete`.

`fp8_continuation_audit.py` treats `full − first` as the continuation and `first_copy − first` as the copy reference.
The copy reads and writes every partial byte, so a bandwidth-bound continuation should not be slower than it. The
script reports three things:

1. The continuation/copy ratio and the continuation's share of the end-to-end time, by kind, by continuation kernel and
   by shard count.
2. The worst candidates.
3. The ranking impact: how often each cell's winner changes, and the regret of today's winner, if every continuation
   ran at its copy reference.

Small regret in (3) means the continuations cannot distort the training labels. Large regret means the continuation
should be made fast before the dataset is taken.
