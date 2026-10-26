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
  view is 32-byte aligned with at least 128 bytes of trailing padding.
- One additional 2 GiB GPU workspace allocation supplies **overlapping,
  stream-ordered output and scratch views**; 32-byte starts and 128-byte
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
