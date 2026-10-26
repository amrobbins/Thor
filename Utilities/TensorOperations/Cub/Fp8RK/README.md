# Retained FP8 RK VALUE reductions

This directory owns the retained original-input FP8 RK physical-kernel selector:
OneWindow, DeepRSmallK, AlignedK, and RuntimeK. The geometry-aware XGBoost
ranker and its fixed 28-candidate shortlist choose *first-stage physical*
kernels, not complete stamped reduction programs. The retained legal compiled
inventory is the authority for which candidates can be launched; unsupported
variants must not be synthesized as fallback kernels.

## Production selection and execution

`selectDensePhysicalPass()` is the ordinary production entry point for a
nominated R/KR/RK site. For eligible original-input FP8 RK VALUE it calls
`selectCachedIsolatedRKSite()` and converts its selected first-stage recipe
into `DenseRetainedFp8RKPhysicalPlan`. This is the **default** production route;
there is no environment-variable opt-in or old FP8 RK production fallback.
Original E4M3/E5M2 input supports all eight VALUE reductions (SUM, MEAN,
PRODUCT, MIN, MAX, L1Norm, L2Norm and SumSquares), with FP8, FP16, BF16,
FP32, and optionally FP64 final storage under Thor's ordinary build policy.
The typed retained selector handles the operation and input transform, and
Complete kernels perform finalization and runtime FP32 scaling before output
conversion. Staged kernels write unfinalized FP32 partials; subsequent
independently planned passes perform the finalization and scale exactly once.
Unsupported geometries still reject rather than invoke a legacy FP8
fallback. Other R/KR, ARG, view, and non-FP8 reduction paths have
independent dispatch.

The shared `planDenseReduction()` owns the **only** ordinary-dense VALUE
multi-pass planner. It selects one physical pass per considered site through
`selectDensePhysicalPass()`, compares the actual next-problem intermediate bytes
with its deterministic `smallestIntermediate` rule, advances the mathematical
problem, and repeats. A retained FP8 physical pass launches **one** kernel:

- A complete first stage writes the ordinary `[outer,1,inner]` result.
- A staged first stage writes a real **FP32** `[outer,shards,inner]` output,
  which becomes the next ordinary reduction problem with an outstanding
  `IntermediateReduction(shards)` axis. A later physical pass is selected
  independently by the global planner; it is **not** the FP32 pass chosen
  hypothetically during first-stage benchmarking.

Nonfinal mathematical outputs are FP32. Original-input transforms are applied
once; finalization, divisor and output conversion belong to the final
mathematical pass. The flat `DenseExecutablePlan` owns intermediate outputs and
per-pass physical scratch **per stamp**. `runOn()` launches those selected
passes without consulting the selector/cache or allocating/tuning workspace.
Scratch-free Thor kernels allocate no dummy scratch; CUB's two-phase APIs may
require a stamp-owned non-null minimum buffer even for a zero-byte query.

`CubReduction::queryWorkspaceSizeInBytes()` selects the same physical plan
as stamping and sums the exact nonfinal FP32 output allocations plus physical
scratch (with checked addition). The stamped executor repeats that accounting
against the buffers it owns. `getDensePhysicalPassDiagnostics()` exposes the
actual ordered pass families, input/output tensor geometries and dtypes, and
intermediate/scratch bytes; family-specific packet/CTA/shard parameters are
optional when they do not apply. There is no synthetic all-family RK geometry.


## Cache-miss measurement (not a production executor)

`Fp8RKSelectionCache` is a bounded, same-key single-flight cache of the
immutable `Fp8RKSelection` physical recipe only. It owns neither Tensors,
streams, scratch nor FP32 continuation descriptors. Cache hits do not choose
FP32 continuation passes. Cache clear/eviction cannot invalidate a stamp.

For each ranked FP8 first-stage candidate on a **cache miss**:

1. `planIsolatedRKSite()` describes an isolated benchmark of the nominated
   mathematical site, not a production planning step.
2. `makeSelectedIsolatedBenchmark()` asks `selectIsolatedFp32Continuation()`
   to choose each hypothetical FP32 pass, **sequentially** through the same
   official physical selector used by production. The candidate's temporary
   program includes all required FP32 intermediates and queried physical
   scratch within the measurement budget.
3. Only after every candidate's selection dependencies are resolved does the
   autotuner acquire the shared per-device GPU measurement mutex and compare
   complete candidate programs. The winner contributes **only its first FP8
   physical kernel** to the recipe cache; all benchmark programs die after
   measurement.

Device UUID, SM count and compute capability are queried once per process-local
GPU ordinal and reused on cache hits; a cache clear evicts recipes, not device
identity. Only a cache-miss leader allocates an internal representative Tensor.
Its allocation/fill and eventual `dropReference()` execute under the shared
per-device GPU measurement mutex so `cudaFree()` cannot overlap another tuning
contest. Dependency selection still occurs **without** that lock, and the
interleaved 28-candidate timing contest acquires it separately. A user-supplied
representative belongs to its caller and is never released by the selector.

The cache and GPU measurement locks have independent purposes: cache single-flight
coalesces the same key, while the device mutex serializes different-key GPU
measurement and internally owned representative setup/teardown. No mutex is
held while awaiting a same-key cache flight. A failed or cleared in-flight tune
cannot insert a stale result, and an existing stamped executor remains valid.


`Fp8RKSelectedContinuation{,Benchmark}` and `Fp8RKIsolatedSite` therefore stay
live **only to construct comparable autotune measurements**. They are not an
alternative production executor or a parallel FP32 physical policy.

## Validation

From the existing build directory:

```bash
cmake --build . --target Thor \
    thor_fp8_rk_isolated_site_smoke \
    thor_fp8_rk_selected_continuation_smoke \
    thor_fp8_rk_selected_continuation_gpu_smoke \
    thor_fp8_rk_stamp_autotuner_smoke \
    thor_fp8_rk_stamp_lifecycle_smoke \
    thor_fp8_rk_production_cutover_smoke \
    thor_dense_retained_fp8_rk_execution_smoke \
    thor_fp8_rk_selection_cache_smoke -j

./thor_fp8_rk_isolated_site_smoke
./thor_fp8_rk_selected_continuation_smoke
./thor_fp8_rk_selected_continuation_gpu_smoke
./thor_fp8_rk_stamp_autotuner_smoke
./thor_fp8_rk_stamp_lifecycle_smoke
./thor_fp8_rk_production_cutover_smoke
./thor_dense_retained_fp8_rk_execution_smoke
./thor_fp8_rk_selection_cache_smoke
```

`thor_fp8_rk_stamp_autotuner_smoke` exercises the real benchmark builder and
then the independent ordinary production stamp. The selected-continuation
smokes cover hypothetical zero/one/multiple FP32 dependencies. Lifecycle and
cutover smokes cover cache reuse, stamp independence, and production routing;
run them on an FP8-capable GPU. Review actual test outputs; building a target
alone does not establish GPU correctness or performance.

The isolated selected-inventory target (`thor_fp8_selected_inventory_size`)
reports only selector-reachable first-stage CUDA code, **not** total production
Cub SASS. Build `Thor`, then independently inventory the real Cub CUDA objects:

```bash
# From the existing build directory:
cmake --build . --target Thor thor_fp8_selected_inventory_size -j
python3 ../benchmarks/fp8_production_object_census.py \
    --build . --out fp8-production-census
```

Review `fp8-production-census/device-functions.csv` for additional FP8 kernel
functions. Family-name counts are **not** proof of selector reachability: the
production objects can instantiate different template specializations with the
same family name. The independent census now compares exact mangled kernel
identities against `fp8-selected-inventory/report/device-functions.csv` and
writes `fp8-production-census/selector-identity-comparison.csv`. The report
separates additional production-only identities from repeated compiled instances
and missing selected identities, per GPU architecture. After updating the
inventory analyzer, regenerate its report (the CMake target above does this).
Production-only identity does not automatically mean dead code: trace its
host dispatch and other R/KR, ARG, and specialized-view users before deletion.
The census intentionally does **not** declare such kernels obsolete.
Production object SASS and selected-inventory SASS are not the same as linked
`libThor.so` size. This read-only audit does not affect production dispatch or
emit additional kernels.

The active production code is complete for the retained FP8 VALUE cutover;
future FP32/FP16/BF16 autotuners and any broader shared cache/policy framework
are independent design decisions, not required by the present stamp lifecycle.
