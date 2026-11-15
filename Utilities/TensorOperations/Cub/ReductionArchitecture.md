# Thor Reduction Architecture

The FP8 dense reduction pattern (planner, retained kernels, rankers, contest, cache, depth), which FP16, BF16 and FP32
are to follow, is written up in `FP8_REDUCTION_PATTERN.md`. Ranker retraining for a GPU is in
`benchmarks/FP8_RANKER_RETRAINING.md`.

Thor's central GPU reduction implementation lives in `Utilities/TensorOperations/Cub`.
Dense expression reductions and ragged offset-segmented reductions use these utilities as the central implementation;
they do not stage inputs through compatibility tensors and do not call `cudnnReduceTensor`. The central utility is
allowed to use either CUB device primitives or Thor-owned CUDA kernels when tensor geometry requires a layout-aware
implementation. Expression's vector-valued segmented forward caller is migrated to the central API separately.

## Ordinary dense VALUE production lifecycle

The shared `DenseReductionPlanner.cpp::planDenseReduction()` is the single
ordinary-dense VALUE pass planner. For each current mathematical reduction site,
`selectDensePhysicalPass()` chooses one physical implementation; the planner
compares actual next-tensor bytes, selects its deterministic minimum, and
continues until all reduction axes are complete. The selected immutable
physical passes form a flat `DenseReductionPlan`; **no cached complete plan** or
family-specific production continuation exists.

```text
stamp/query: current problem -> nominate R/KR/RK sites -> official physical selector
                   |                                  |
                   |                         FP8 first-stage cache
                   |                          (physical recipe only)
                   v
             selected one-pass plan -> next problem -> ... -> completed plan
                   |
              stamp allocates each FP32 intermediate and per-pass scratch
                   |
run/runOn:   existing stamped physical passes -> GPU launches (no planning/tuning/allocation)
```

Original FP8 E4M3/E5M2 ordinary RK VALUE reductions (SUM, MEAN,
PRODUCT, MIN, MAX, L1Norm, L2Norm, SumSquares), with the enabled floating
output storage dtypes, use retained OneWindow, DeepRSmallK, AlignedK and
RuntimeK through the same selector as normal production. The geometry ranker
nominates a shortlist of the reduction's tuning depth (Quick 6, Regular 12, Ultra 24).
A staged retained kernel emits a real FP32 `[O,N,K]` output and reduces **one**
physical stage; FP32 continuation is selected independently by the global
planner. During cache-miss *measurement only*, `Fp8RKSelectedContinuation`
chooses hypothetical FP32 passes via that same official selector, before the
per-device GPU measurement lock is acquired. The cached answer holds only the
winning first FP8 physical recipe, not those hypothetical passes or buffers.
Original-input transforms occur once; all nonfinal outputs are FP32, and final
reduction finalization/divisor, runtime float scale and dtype conversion occur
only on the last mathematical pass. Unsupported physical geometries and
storage dtypes reject explicitly without a private FP8 fallback.

`CubReduction::queryWorkspaceSizeInBytes()` sums each nonfinal output's actual
bytes and each pass's physical scratch. Stamping allocates exactly those
buffers per independent executor, checks the workspace accounting, and retains
them across cache clear and subsequent `runOn()` calls on compatible streams.
Thor-owned zero-scratch kernels allocate nothing. True CUB two-phase APIs may
still need a preallocated one-byte minimum physical buffer, even when CUB's
query reports zero. Read-only `getDensePhysicalPassDiagnostics()` describes
these physical passes without returning invented generic RK parameters.

Cache single-flight is per key; a shared per-device mutex serializes competing
GPU timing contests and the setup/destruction of internally allocated tuning
representatives. FP32 hypothetical dependency selection is done before the
contest lock. Cached immutable device identity avoids CUDA property queries on
cache hits. Tuning failures are not cached; cache clearing cannot invalidate
already stamped executable sequences. No GPU tuning, cache access, extra
allocation, or kernel selection is permitted within normal `runOn()`.

## Implementation ownership and extension boundaries

| Concern | Authoritative owner |
| --- | --- |
| Mathematical runs, advancement and output bytes | `DenseReductionPlanning.{h,cpp}` |
| Global one-pass-at-a-time choices and deterministic ties | `DenseReductionPlanner.{h,cpp}` |
| One physical-pass selection for R/KR/RK | `DenseReductionPhysicalSelector.{h,cpp}` |
| Production GPU occupancy and FP32 policy context | `DenseProductionPhysicalSelectionContext.{h,cpp}` |
| Retained FP8 first-kernel identity and legal physical candidates | `Fp8RK/Fp8RKPhysicalIdentity.*`, `Fp8RK/Fp8RKBoundSelection.*` |
| Cache-miss FP8 candidate measurement and immutable recipe cache | `Fp8RK/Fp8RKStampAutotuner.*`, `Fp8KR/Fp8KRStampAutotuner.*`, `DenseReductionSelectionCache.h` (one cache for every selector, 1171) |
| Benchmark-only hypothetical FP32 completion | `Fp8RK/Fp8RKSelectedContinuation*` |
| Pass workspace, tensor ownership, and launch order | `CubReduction.{h,cpp}`, `DensePhysicalPassExecution.h` |
| Ragged (offset-segmented) kernels, selection and routing (1330: the only segmented path) | `CubRagged*.cu`, `Ragged*.{h,cuh}`, `RaggedStampAutotuner.{h,cpp}` |

The hypothetical FP32 continuation is an ephemeral **measurement program**, not
part of a production stamp. The production planner independently chooses each
subsequent physical pass from the actual next tensor geometry. The original
FP8 input has no old RK fallback; unrelated non-RK paths retain their own
selection policies. The full retained FP8 VALUE operation/output surface is
specified and exercised by `Fp8RK/README.md` and the production cutover tests.

1171 (user decision, 2026-11-02): the dense reducer has one stamp-time autotuner policy
(`ReductionAutotuning::autotune`, shared probe and contest rules) and one selection cache
(`DenseReductionSelectionCache.h`). Every selector, today FP8 RK and FP8 R / KR, keys its winners there by
(kind, device identity, tuning depth, site); the cache owns single flight, the LRU bound and the depth rule. The
FP16/BF16 and FP32 selectors are to add kinds to that cache rather than caches of their own.

## Compiled kernel inventory boundaries

`thor_fp8_selected_inventory_size` compiles an **isolated, selector-reachable**
first-stage inventory. It proves the selected specializations in that target,
not that no additional FP8 kernels are present in production CUDA objects.
Inspect production objects separately after building `Thor`:

```bash
# From an existing build directory:
cmake --build . --target Thor thor_fp8_selected_inventory_size -j
python3 ../benchmarks/fp8_production_object_census.py \
    --build . --out fp8-production-census
```

Review `fp8-production-census/device-functions.csv`, especially
`other-fp8-kernel`, against genuine R/KR, ARG, specialized-view and benchmark
ownership. This is an **audit**, not an automatic deletion list: FP8-looking
functions can belong to other legitimate reduction domains. Production object
SASS, selected-inventory SASS and final linked-library size are different
measurements; record SM target and compiler/build settings when comparing.
The read-only census produces no kernels and does not change selection.

## Numeric contract

- Input and output storage may be FP8 E4M3, FP8 E5M2, FP16, BF16 or FP32 (no FP64 since 1258 / 1259).
- Input iterators convert values to FP32 before reduction.
- Reduction state and operation-specific finalization use FP32.
- The final store converts once to the configured output storage dtype.
- Stamped operations own their output and CUB workspace. `run()` and `runOn()` do not allocate or re-plan.

## Dense paths

> **1258 (phase 10 of the FP16/BF16 plan):** the `ReducersDenseR / KR / RK` families, `DenseRKInventory` and the
> direct dense plan are deleted. `selectDensePhysicalPass()` sends every dense site to a retained per-dtype selector
> (FP8, FP16/BF16 or FP32; see `FP8_REDUCTION_PATTERN.md`) and throws for anything else; FP64 reductions are not
> supported. 1259 deleted the legacy kernels too: `CubReductionCooperativeShardedTiled.cu`, the generic FP16 /
> BF16 / FP32 KParallel grid, the 8-warp rotated and FlatRows entry points, and the FP64 code. 1261 deleted the
> interval-composition planner and ARG executor. Family-era text below (including the functions it names) is kept as
> design history.

`CubReduction` selects the fastest backend implied by dense row-major geometry:

1. Device transform-reduce when the result contains one element.
2. CUB fixed-size segmented reduction when each reduction domain is physically contiguous (`inner_size == 1`).
3. The retained dense planner's R / KR / RK passes when the reduced axes are one contiguous block with trailing values
   (`TiledFixedSegment` is that classification; the Thor tiled kernels it once named are deleted, 1393).
4. Arbitrary ordinary-dense combinations of reduced/retained runs are structurally classified as `ComposedDense`, but VALUE execution is not a composed parent/child executor. `planDenseReduction()` selects one R/KR/RK physical pass, produces a fresh intermediate problem, and replans globally until complete. The selected passes are materialized as one flat `DenseExecutablePlan`; every non-final VALUE aggregate is FP32, only the first logical pass applies the public input transform, and only the final logical pass applies the public finalizer/output scale.
5. Non-dense inputs remain outside the ordinary-dense planner. Proven compact/permutation/pitched layouts use their ordained view reducers; unsupported arbitrary views fail structural analysis rather than entering a generic logical-index fallback.

1387: no view reaches the Thor tiled view kernels any more. 1390: the reducer refuses rotated-output
permutations, VIEW-DIRECT-2B and VIEW-PITCHED-TILED (`analyzeGeometry` throws `NotImplementedException`), and a caller
runs them in two parts, a dense reduction and the dense copy (`CubReduction::splitRefusedView`; the view list further
down). The 1387 layout passes inside the reducer (`ViewLayout.h`, `CubViewLayout.cu`) are deleted. 1393 deletes the
tiled view family itself (3.57 MB of SASS, unreachable since 1390): the direct and pitched component-tiled kernels,
the 32x33 payload transpose, the wide shared-transpose kernels, their `cuda::memcpy_async` full-row pipeline and
launch policy (`CubReductionTiledPolicy`), the per-operation executors over them (`CubReductionSum.cu`,
`CubReductionSumSquares.cu`, `CubReductionMin.cu`, `CubReductionMax.cu`, `CubReductionProduct.cu`,
`CubReductionL1Norm.cu`, and the template-free `CubReductionMean.cu` / `CubReductionL2Norm.cu`), and the reducer's
generic per-op path (`stampPhysicalStage`, `queryPhysicalStageWorkspace`, `launchReduction`). Every value stamp is now
a dense executable plan, StridedR or pitched PacketComplete; anything else throws `std::logic_error`.
`CubReductionOperation.cuh` keeps the scalar transforms, finalizers, work-grid and shard helpers, and the vector-packet
loaders the retained RK kernels and the KParallel / RCooperative / rotated sharded families share.

Dense reduction rank is dynamic. Structurally `ComposedDense` VALUE reductions stamp a flat sequence of physical
R/KR/RK passes in `DenseExecutablePlan`; there is no VALUE parent executor or child-stage continuation. `run()` launches
that ordered pass list on the caller stream with no host synchronization, allocation, or runtime planning. Only the
ordinary dense planning needs no per-element logical-index metadata. There is no cuDNN-derived rank-8 limit; the only
representation bound is that axis identifiers are `uint32_t`.

Value reductions support sum, product, mean, min, max, L1 norm, L2 norm, and sum-of-squares. VALUE-BINARY-CLEANUP keeps
input transforms and reduction operators compile-time specialized while sharing additive finalization at output
granularity: Sum, Mean, and the final composed L2 stage use one `IdentityFp32 + AdditiveFinalizeFp32` kernel matrix,
while SumSquares and complete L2 use one `SquareFp32 + AdditiveFinalizeFp32` matrix. Division and optional square root
are runtime finalizer data paid once per output, so L2 does not instantiate independent copies of every reduction
geometry (1393: the per-op objects that carried those matrices for the tiled view kernels are deleted). Dense
argmin/argmax use the same geometry
classification. ARG-DIRECT-1 keeps the device-wide CUB path but narrows its candidate index to UINT32 whenever the full
reduction domain permits it. Physically contiguous fixed segments in the normal UINT32 domain use a Thor-owned
warp-per-segment backend: each warp peels a bounded scalar head to a 16-byte boundary, reads the segment bulk through
coalesced 16-byte packets, handles a bounded tail, and combines lane candidates through shared memory. CUB remains only
for the exceptional contiguous domain that genuinely needs UINT64 candidate state.

Contiguous middle-axis reductions with trailing retained values use ARG-DIRECT-1 throughout the normal UINT32
candidate domain. Narrow rows (<=32 retained components) reuse the value reducer's proven staged-memory geometry: a
physical warp peels a bounded number of complete rows until the source reaches the strongest useful 16/8/4-byte
alignment, copies the contiguous bulk into a fixed double-buffered shared-memory stage, consumes a bounded suffix
directly, and assigns otherwise-idle lanes to independent reduction rows. The row-lane count is derived from retained
width (16/8/4/2/1 lanes for 2/4/8/16/32-component capacities), and candidate exchange is a shared-memory tree rather
than a warp shuffle or CUB WarpReduce. Composed FP32+UINT32 narrow stages split the same 2 KiB-per-warp stage budget
evenly between values and carried indices, preserving the existing shared-memory/occupancy budget.

For wider rows, FP16/BF16 lanes own eight values per aligned 16-byte input packet and FP32 lanes own four. Reductions
assign 1/2/4/8 whole warps to the same component tile only when the actual number of output tiles cannot by itself expose
the tiled reducer's established target active-warp population. The warp count is then raised only as necessary so a warp
that revisits multiple rows does so at a byte stride divisible by 16. That makes its row-head alignment stable: a shifted
row uses bounded scalar edge work while the remaining complete packets are directly owned aligned 16-byte loads. Per-warp values and indices
are canonicalized in separate 32-bit shared-memory arrays before the final combine; no warp-shuffle reconstruction is
used. Composed ARG stages reuse the same backend for FP32 values plus UINT32 carried original-domain indices. UINT64
candidate domains retain the conservative global-load geometry but also use shared memory rather than WarpReduce for
candidate exchange. Arg reductions remain deterministic: NaNs propagate and the lowest original flattened reduction-
domain index wins equal-value ties.

Ordinary dense ARG reductions with disjoint reduced runs use `ComposedDense` rather than the arbitrary logical-index
fallback. ARG composition reuses the value reducer's left/right interval-elimination topology, but has its own stamp-time
stage-cost model because later passes read and write a SoA candidate `(FP32 value, original-domain index)`. The carried
index width is selected once from the complete original reduction domain and is UINT32 whenever that full domain fits,
otherwise UINT64. Each stage adds `local_run_index * domain_stride` to the carried original index, so different legal
physical pass orders return exactly the same public row-major flattened reduction-domain index and tie-break on that
original index. The production cost model accounts for candidate traffic, useful direct-kernel concurrency, L2-resident
intermediates, and the current direct-kernel execution regimes. In particular, census calibration keeps the clean aligned
tiled path distinct from the provisional awkward head/bulk/tail path and models the measured throughput troughs for
saturated very-short contiguous and short-narrow stages. Those throughput penalties are not multiplied into small
under-filled later passes, where the planner's existing active-warp term already represents the dominant cost. The ARG
census can force every legal left/right elimination order for its representative composed
cases, validates exact public indices against production before timing, and reports the selected stage strategies so the
coarse ARG weights can be recalibrated after future direct-kernel improvements without changing topology. No runtime
autotuning, allocation, host synchronization, or topology discovery is introduced by planning.

The shared `CubReduction::analyzeGeometry()` selector owns execution-family classification. An ordinary dense disjoint
reduction is reported as `ComposedDense` immediately, and DENSE-GATE-FINAL makes that ownership an implementation
invariant: canonical dense geometry must resolve to `DeviceTransformReduce`, `ContiguousFixedSegment`,
`TiledFixedSegment`, or `ComposedDense`; a disjoint mask is never an invitation to enter a catch-all view reducer
(since 1261 no structural interval plan is built: `planDenseReduction()` owns every disjoint mask).
For VALUE, `ComposedDense` is now only that structural classification: `planDenseReduction()` chooses the actual
R/KR/RK pass sequence and globally replans after each pass. ARG-P1 (Thor1106) moved original FP8 input ARG onto the
same planner: `CubArgReduction::stamp()` plans FP8 dense ARG with `planDenseReduction()` (aggregate kind `Arg`,
carried index width from the whole original domain) and executes the planned passes as direct ARG stages (FP32 value +
carried original index between passes, `local * original_arg_domain_stride` added per pass, nothing added for a
reducer-created intermediate run). The public geometry and `getPath()` stay the analyzed dense geometry. FP32 (1221)
and FP16/BF16 (1254-1257) ARG followed. 1261 deleted the interval-composition ARG executor (`stampComposedDense`,
`stampComposedDenseWithPlan`), the interval-DP planners for VALUE and ARG (`analyzeDenseCompositionPlan*`) and the
CUB dense ARG adapters (`CubArgReductionMin / Max.cu`); every pass of a planned dense ARG stamp is a retained pass.

The ordinary-dense ARG selector obeys the same ownership contract. `CubArgReduction::stamp()` requires a dense-contiguous
input, and the dense gate exhausts ranks 1-9 and every non-empty reduction mask, with singleton-heavy and disjoint-run
cases, so dense ARG execution cannot acquire arbitrary-view behavior accidentally. ARG-BINARY-CLEANUP removed the old
arbitrary-view candidate iterator, device-indexing metadata, and corresponding CUB segmented-reduction instantiations.
If arbitrary-view ARG support is intentionally added later, it must receive an explicit physical-layout-aware strategy.

The same cleanup keeps the three benchmarked normal ARG fast families intact (`alignedContiguousSegmentArgReduction`,
`alignedAsyncNarrowTiledArgReduction`, and `alignedCooperativeTiledArgReduction`) while deleting the superseded
full-row/grouped/block-sharded/alignment-safe tiled families. Exceptional wide-FP8 and true UINT64 candidate domains use
one conservative generic tiled backend with `RowLanes=1`, so those rare cases no longer instantiate a matrix of historical
fallback topologies. Composed UINT32 contiguous stages likewise have no CUB segmented fallback: the first stage uses the
normal aligned contiguous kernel and every carried-index stage is FP32 by construction. No CUB segmented ARG
instantiation remains (1373's source guard). Contiguous rows of any length, VALUE or ARG, run the retained KR families,
whose geometry and wide ARG candidates are 64-bit; 1374 removed the old `int` segment-size gate
(`usesCubFixedSegmentSize`), which had rejected direct contiguous rows longer than INT_MAX. A UINT32 index output still
needs a local domain that fits (the stamp checks).
The separate offset-segmented ARG API runs its own kernels, the ragged ARG family (see the ragged section below).

### Recursive stage-planning invariant

A staged reduction does **not** own a special second-stage or terminal reducer. Every physical pass transforms one
reduction problem into a smaller ordinary reduction problem. If a pass emits FP32 partials shaped
`[outer, shards, inner]`, the continuation is simply a new reduction of axis 1 for that FP32 tensor. The planner must
analyze that intermediate geometry from scratch and choose the best available reducer for it, exactly as it would for a
user-visible input.

This rule applies after every stage, not only after the first one. A deep reduction may therefore execute as several
independently planned passes before a direct/complete reducer finishes it. The implementation may express this as a
stamped remainder/continuation chain rather than literal recursion, but the architectural rule is recursive:

```text
current reduction problem
        |
        v
choose best reducer for current geometry
        |
        +-- complete/direct --> done
        |
        `-- staged --> FP32 partial tensor
                           |
                           `--> plan that reduction problem independently
```

Consequences:

- Do not design or name architectures as fixed `first_stage + terminal` pairs.
- Do not force a simple serial finalizer merely because a particular first stage produced the partials.
- Do not reserve ROW-SPLIT, cooperative, rotated, K-parallel, or any other reducer for a particular stage number.
- First/intermediate/final semantics control transforms, finalization, and output scaling; they do not constrain reducer
  geometry.
- Planner decisions use the current stage's `outer`, `R`, `K`, dtype, alignment, available parallelism, and intermediate
  traffic. Operation-specific semantics remain orthogonal to geometry selection unless benchmark evidence proves
  otherwise.
- Workspace accounting includes every intermediate tensor and every recursively stamped remainder.

For ordinary dense VALUE, `planDenseReduction()` implements this rule through the
canonical `DenseReductionProblem` and stores the resulting flat physical-pass list.
Special view paths retain their independent physical-stage adapters. Retained
FP8 benchmark programs use the production selector for hypothetical FP32
continuation so a census compares equivalent completed reductions rather than
artificially paired first-stage/finalizer kernels.

### Low-output / deep full-row reduction sharding

(Historical: these full-row kernels were deleted in 1393.) The direct and grouped `TiledFixedSegment` full-row kernels
derive most of their launch parallelism from independent
outer/output rows. That ownership remains the fastest strategy when many outputs are available, but it is pathological
for shapes such as `[1, 104832, 512] -> [1, 1, 512]`: one warp (or one small cooperating warp group) otherwise scans the
entire reduction domain while almost every SM is idle.

The original fix for this under-parallelism was `ROW-SPLIT-FULL-ROW`, a generic two-stage fallback. It proved that
R sharding could recover device utilization, but its dedicated first-stage/finalize pairing was a transitional design.
After the KParallel/RCooperative census closed its coverage and meaningful performance gaps, the historical row-split
kernels, selectors, benchmark registration, and stamped compatibility state were removed from the compilation surface.
Their role is now historical calibration only; ordinary dense VALUE RK was then owned by `ReducersDenseRK` (itself
retired in 1258).

Aligned staged full-row work has two useful ownership modes rather than two permanently paired reducer families.
R-cooperative stages spend multiple warps on the same retained K tile and divide R among those warps. K-parallel stages
instead give warps disjoint K packets and let each owner serially walk its R shard, eliminating inter-warp reduction,
shared-memory exchange, and synchronization for that stage. K-parallel CTA width therefore follows the amount of K work:
prefer naturally aligned 16-byte lane packets and use the smallest power-of-two warp count that covers the tile, capped
at eight warps. Because warps/CTA varies, its shard planner targets device-wide useful **warps**, not blocks; block-count
targets are valid proxies only for kernels whose warps/CTA is fixed. K-parallel is consequently a geometry/ownership
choice available to the stage planner, not a special first-stage architecture and not a reason to prescribe how its FP32
partials are reduced next.

The packet-width x R-shard-depth census confirms that packet width and shard depth must remain separate geometry fields,
but it does **not** support choosing them as two independent sequential heuristics. Changing shard depth changes the
`[outer,shards,K]` continuation geometry and can move the recursively planned continuation onto a materially different
reducer, so the measured end-to-end winner can change with the packet/shard pair. Production therefore exposes one
parameterized K-parallel stage implementation for 4/8/16-byte packets on FP16/BF16/FP32 and 2/4/8/16-byte packets on
FP8, with 32/64/128/256-thread compile-time CTA specializations. FP8 always reduces in FP32. Packet16/packet8 use a
warp-private shared-memory writeback transpose when FP32 ownership would otherwise exceed 16 bytes per lane;
packet4/packet2 staged output is already coalesced as direct 16-byte/8-byte FP32 stores and needs no transpose. Its physical topology is explicit: STAGED uses more than one R shard and emits FP32
`[outer,shards,K]` partials, while COMPLETE uses exactly one R shard and applies the current pass's finalizer, output
scale, and runtime dtype conversion directly. This topology is independent of mathematical stage role: a logically
Final pass may still be physically STAGED and defer finalization to its continuation, while a logically First pass may
be physically COMPLETE without finalizing the overall operation. The stage planner should compare the small legal
packet/shard plan set jointly, including the independently planned continuation and FP32 scratch cost. Near-tied plans
should prefer less scratch rather than encoding benchmark noise into packet- or row-specific reducer families.

For additive low-output/deep reductions, the newer alignment-rotated sharded topology supersedes the older ownership
model in four benchmarked awkward-width regimes. Above 4096 retained components it handles non-4096-block-multiple
widths; from 513 through 4095 it handles packet-misaligned row strides; from 257 through 511 it handles packet-misaligned
rows once the reduction is at least 32768 deep; and from 33 through 255 it replaces the remaining x2/x4/x8 full-row
islands for packet-misaligned rows at the same reduction-depth threshold. The first stage uses eight physical warps, exact
16-byte aligned packet loads, and FP32 accumulation.

DENSE-RK-STAGE-1 makes the rotated implementation obey the same one-pass contract as the aligned cooperative and
K-parallel staged implementations. `launchAwkwardAlignmentRotatedShardedFirstStage()` emits only FP32
`[outer,shards,K]` partials and has no terminal/output/finalizer capability. Ordinary dense VALUE RK stamping now
executes the flat planner-driven `ReducersDenseRK` sequence directly: every staged pass emits a fresh FP32
`[outer,shards,K]` problem and the global RK planner chooses its continuation. There is no rotated compatibility
terminal or legacy full-row continuation in the production VALUE path.

The narrow low-precision calibration adds a second physical access implementation beneath `RCooperative` rather than a
new reducer strategy. `launchNarrowLowPrecisionFlatRCooperativeFirstStage()` flattens eight contiguous logical rows
into aligned 16-byte packet work, reconstructs `[8,K]` ownership in FP32 shared memory, and emits only FP32
`[outer,shards,K]` partials. A complete FP16/BF16 K=1..32 sweep established an operation-independent selector boundary:
all odd K use `FlatRows`, even K<=14 use KParallel, and even K>=16 use `FlatRows`. The same sweep established the

KParallel Complete is a distinct lean production hot path: it maps each output/component tile directly to one CTA and
serially walks R without staged/shard work decoding. End-to-end low-output calibration established that an underfilled
KParallel launch should switch to a staged first pass at roughly R>=96, targeting about 16 input rows per shard, then
replan the emitted FP32 intermediate as a fresh reduction problem. Comfortable Complete launches remain eligible; no
pass owns its continuation.

FlatRows shard-depth transition used by production: K<20 uses 512 rows/shard and K>=20 uses 128 rows/shard. Rotated
remains the coverage path when a calibrated FlatRows staged pass cannot be formed, such as shallow R.

The >=513 regimes retain fixed 1024-row shards. In the 257..511 regime, production keeps rows1024 when that still exposes
enough device-wide work, otherwise it uses the benchmarked rows256 low-precision or rows512 FP32 specialization. In the
33..255 regime, shard depth follows logical-tile occupancy and available SM waves: small one-tile rows favor rows256
(or rows128 when a smaller reduction needs more CTAs), fuller one/two-tile rows favor rows512, and the next tile-count
step favors rows1024 when enough parallelism remains. The same rule also exactly reproduced the rows256/rows128
crossover measured by the sub-64 census without adding a K-specific lookup table. Every admitted plan must still launch at least
one CTA per SM, so narrowing the retained width does not recreate the under-parallelized failure mode that motivated
the kernel. Exact packet-aligned sub-4096 widths remain on the existing direct/cooperative families until separately
benchmarked.

### RK post-delete coverage gate (retired)

The `thor_cub_reduction_benchmark --rk-census-gate` census and its analyzer timed the `ReducersDenseRK` family plan
after the original inventory delete. Phase 10 of the FP16/BF16 plan (1258) removed that family: every dense VALUE and
ARG site now goes to a retained per-dtype selector (FP8, FP16/BF16, FP32), and the benchmark, analyzer and family code
were deleted. Coverage of the retained passes lives in the per-family selector tests and the dense production audit.

### View ownership after DELETE

The view ownership census (formerly `thor_cub_reduction_benchmark --view-census`, removed in 1258) settled ownership
rather than benchmarking a generic fallback. The structured view families that Thor intentionally supports are
assigned directly to ordained implementations:

- VIEW-DIRECT-1 is a one-to-one compact physical permutation with every logical axis reduced (classified
  `DeviceTransformReduce`). Visitation order is irrelevant for a full value reduction, so since 1365 it runs the
  retained dense planner on `StridedSpan::asDense` of its bytes, a dense `[input_elements]`: the dense tensor's kernels
  and bits.
- VIEW-DIRECT-2A is a compact permutation whose reduced axes form the physical suffix and whose retained physical order
  already equals dense logical output order (classified `ContiguousFixedSegment`). Since 1365 it runs the retained dense
  planner as a dense `[output_elements, reduction_size]` over axis 1.
- 1376 (phase 5) generalizes both: every compact permutation whose retained axes keep their physical order runs the
  retained dense planner on its bytes, its physical non-singleton axes collapsed into alternating retained / reduced
  runs (`viewPhysicalDenseProblem`). That adds the natural-output permutation-aware tiled views (still classified
  `TiledFixedSegment` with the candidate; the 1375 audit found the tiled kernel 1.04x to over 10,000x slower in every
  cell, worst with a few outputs of long reductions) and split physical reductions, whose reduced axes are several
  physical runs (classified `ComposedDense`; they threw before 1376).
- A rank-1 affine view (classified `DeviceTransformReduce` with `device_transform_uses_affine_stride`) runs StridedR
  (`StridedReduction.h`, since 1366): a stride-0 closed form in FP64, else one Split launch whose last CTA combines the
  partials in a fixed order (1370), on a grid that follows from n (1371).
- No view runs a CUB device reduction (1373, plans/Thor_View_Reduction_Plan_2026-11-08.md phase 4): the CUB
  `TransformReduce`, affine `Reduce` and fixed-size `DeviceSegmentedReduce` executors are deleted, and the
  `DeviceTransformReduce` / `ContiguousFixedSegment` paths are classifications only.
- VIEW-PITCHED-TILED classifies a logically contiguous `[outer..., reduction..., inner...]` view when the trailing
  payload is physically contiguous, outer and reduction groups each flatten to one constant pitch, and adjacent slabs do
  not overlap (repeated-label einsum diagonal pre-reductions such as `[2,2,3]` with strides `[18,3,1]`, slices). 1390:
  the reducer refuses it (`splitRefusedView` kind `PackedInput`); Expression copies the view into a dense tensor in its
  own dtype (`copyToDenseAsync`, a DenseCopy stage) and reduces that, as it does for any view the reducer will not
  read. That is a materialization, about 2-3x the packed bytes, chosen by the user until pitch-aware dense passes exist
  (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7); the pitched kernel it replaces ran up to 22,000x
  slower than the dense stamp of the packed rows in the 1386 audit. 1391 (phase 7 A): when the slabs have no gaps
  between them and the view's source holds the gap after its last row, `splitRefusedView` of the span returns
  `SlicedOutput`: the dense planner reduces the rows read at their pitch, the dense `[O, R, Sr]` from the view's first
  element (`StridedSpan::withSourceLayout`), into `[O, Sr]`, and a DenseCopy stage writes its `[O, I]` columns. It is
  chosen when that moves fewer bytes than the packed copy (O R Sr b + O (Sr + 2 I) c against 3 O R I b + O I c). The
  gap columns are reduced and dropped, so what the gaps hold never reaches the output. 1392 (phase 7 C): a pitched view
  with enough outputs is read in place, one RuntimeK PacketComplete launch (`fp8GeneralRParallelKTileP16CompleteKernel`,
  the retained FP32 / FP16 / BF16 / FP8 Complete kernel; rows loaded at `row_stride`, slabs at `outer_stride`) over the
  3-D span `[O, R, I]` at strides `[So, Sr, 1]`, into the dense keepdims output, with no workspace and no copy
  (`CubReduction::pitchedCompleteLaunch`; the stamp's strategy chain `pitched_complete:w<W>_t<tile>`). The rule
  (1392a, fitted on the 1392 audit's forced-W timings): K tiles balanced under the dtype's cap (124 / 248 / 496
  components) and 256 bytes or more; W the most of 1 / 2 / 4 / 8 warps per `[outer, K tile]` item whose launch fits one
  wave of resident warps (32 per SM), with at least four rows per warp; in place when that launch keeps
  `min(items x W, 32 x SMs) x tile bytes` in flight of 4 KiB per SM where the split would be sliced (it reads the view
  about once) or 2 KiB per SM where it would be packed (three passes over the bytes). The sliced test is the layout's
  and dtypes' alone (the source's reach left out), so the choice follows dims, strides, dtypes and device.
  `analyzeGeometry` still refuses the layout; `analyzeValueSpanGeometry` and `readsSpanInPlace` (each with the
  reduction's output dtype) take the span and accept it, and `splitRefusedView` of the span then has no split (with
  `reducer_reads_pitched` false it gives the split the view has otherwise). A view the rule refuses packs or slices as
  above.
- The permutation-aware tiled views with a rotated `[inner, outer]` output: the reducer refuses them (1390,
  `splitRefusedView` kind `RotatedOutput`). Expression runs two stages: the retained dense planner reduces their bytes
  as the dense `[outer, R, inner]` into a buffer `[outer, inner]` in the output dtype, and a DenseCopy stage reads the
  buffer rotated (`withLayout`) and writes `[inner, outer]`. The bits are the dense tensor's, permuted. The Thor tiled
  kernels they ran (direct with a transpose epilogue, wide shared transpose) were 6-6,000x slower in the 1386 audit.
- 1393 deleted the rest of the Thor tiled view family and its per-op objects (see the backend list above).
- 1385 deleted the packet-adaptive cooperative tiled kernel (`adaptiveCooperativeTiledFixedSegmentReductionKernel`,
  216 instantiations). Its gate refused every view candidate and permuted output, and dense geometry runs the dense
  planner before the tiled launcher, so nothing reached it.
- VIEW-DIRECT-2B classifies a compact source collapsible to `[A,reduction,B,payload]` with dense logical output
  `[B,A,payload]`. 1390: the same as the rotated views above, with the dense `[A, R, B x payload]` and a copy that
  rotates payload vectors; the 32x33 payload-transpose kernel it replaces was 5-2,447x slower in the 1386 audit.
  Rotating the reduced output costs 2 x (output dtype bytes) per output against 2 x R x (input bytes) for rotating
  the input first, so it is the cheaper order whenever R x (input bytes) > (output bytes).

DELETE removes the old arbitrary rank>1 logical-index reducer completely. There is no legacy execution-family enum,
host/device mixed-radix indexing metadata, device logical-to-physical mapper, strided FP32 iterator, legacy 32/64-bit CUB
segmented-reduction specialization surface, or benchmark-only resurrection hook. Unsupported arbitrary views therefore
throw `NotImplementedException` directly from structural analysis. The final view census retains representative unsupported
layouts—stride-2 inner payloads, zero-stride broadcasts, overlapping aliases, split physical reductions whose retained
axes are out of physical order (1376: in order, they run the dense planner), singleton-heavy gapped views, and the
former synthetic UINT64-index case—but reports them as `unsupported / not_implemented` without allocating or timing an
implementation.

Einsum keeps repeated-label diagonal operands as zero-copy views when an ordained reducer owns their stride pattern. If an
operand-local pre-reduction has no ordained owner, einsum materializes only that logical operand into dense order and then
uses the normal dense reducer. This preserves the supported einsum surface without reintroducing a generic arbitrary-view
reduction mechanism.

`CubReductionViewGate`, the dense value/ARG gates, execution tests for each ordained view strategy, and the source guard in
`ReductionSourceGuardTest` collectively enforce the post-DELETE architecture. The source guard also requires the deleted
logical-index header to stay absent and rejects the historical mapper/indexing symbols if they reappear in active CUB
reduction sources.

## Ragged (offset-segmented) path

`CubSegmentedReduction` accepts values `[N,D...]` with row offsets `[B+1]` (UINT32 or UINT64) and produces `[B,D...]`
for ragged sum, mean, min and max; `CubSegmentedArgReduction` returns global packed winner indices
`row * D + component` (UINT32 or UINT64) for argmin and argmax. Inputs are FP8 E4M3 / E5M2, FP16, BF16 and FP32,
accumulated in FP32. Empty segments give 0 (sum, mean), +inf (min), -inf (max) or the index sentinel, per component;
NaNs propagate and the lowest packed index wins ties. `stamp()` validates the offset contents; `stampRuntimeOffsets()`
(the Expression path) reads them only at run time. Rows past `offsets[B]` (reserved capacity) are never read.

Since 1330 every segmented stamp runs the merge-path ragged kernels (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md;
1263-1329 also kept the legacy one-owner-per-segment kernels, CUB `DeviceSegmentedReduce` for rank-1 values and
vector warp / block kernels for `[N,D...]`, which ran at 0.03-0.06 TB/s median at >= 16 MiB and up to 3000x slower on
skewed lengths; they are deleted, and `ReductionSourceGuardTest` keeps them deleted). A site no ragged family runs is
an error at stamp, with the reason: rows over 2 GiB, or more than 2^30 components.

- **Partition.** The work is the merge of the offsets with the rows (each row weighted by its input bytes, each
  segment end by its output bytes), split evenly over a persistent grid fixed at stamp (SMs x resident CTAs). Each CTA
  finds its start with one binary search over the offsets, so time depends on (active rows, B, D, dtype), not on how
  the lengths are distributed. Shared host / device index arithmetic: `RaggedMergePath.h`.
- **Families,** each with VALUE and ARG forms, chosen by row bytes:
  - ragged KR, rows under 16 bytes (FP8 D 1-15, FP16 / BF16 1-7, FP32 1-3): `RaggedKRKernels.cuh`,
    `RaggedKRArgKernels.cuh`, `RaggedKRWalk.h`;
  - ragged RK, rows of 16 bytes to 2 GiB (at most 2^30 components): lanes own 16-byte chunks of a K tile of at most
    512 bytes, wider rows are K tiled, `RaggedRKKernels.cuh`, `RaggedRKArgKernels.cuh`, `RaggedRKWalk.h`.
  One object per family and dtype group (`CubRagged<Family>[Arg]<Dtype>.cu`, the narrow-rows RK variants in
  `*Narrow.cu`), launchers in `CubRaggedKR[Arg].cu` / `CubRaggedRK[Arg].cu`. Any input address works (alignment to the
  element only).
- **Finishing.** Each CTA writes at most a head and a tail record for the segments it shares with its neighbours (the
  stamp's workspace); a second launch combines them in CTA order and stores the output (`CubRaggedFinish*.cu`: block,
  wide and FP32 ARG pair-record finishers). No atomics; results are bit-identical from run to run on one stamp (SUM may
  round differently between stamps with different grids).
- **Selection** (`RaggedStampAutotuner.{h,cpp}`). Three candidate shapes per family; every legal candidate is timed at
  stamp on synthetic uniform offsets over the stamped capacity (9 rounds, one trimmed at each end, no depth), and the
  first in list order within 1% of the best wins. Winners are cached in `DenseReductionSelectionCache` under the kinds
  `RaggedKR` / `RaggedRK`, keyed by op, dtypes, D, capacity, B and the input's address mod 16. Pass names:
  `ragged_kr:<shape>` or `ragged_rk:<shape>`, then `ragged_finish` / `ragged_finish_arg`. `getPath()` stays
  `OffsetSegmented`.
- **Measured (1325, RTX 5090, copy about 1.54 TB/s, moved >= 16 MiB, p50):** 1.13 / 1.22 / 1.26 TB/s (E4M3 / FP16 /
  FP32), against the legacy kernels' 0.026 / 0.039 / 0.057; the spread between the fastest and slowest length
  distribution at one shape p90 1.07-1.31x (legacy 1420-3030x). Balanced long-segment cells, where the legacy kernels
  had one owner per segment and no imbalance, are the exception: 31-55 cells per dtype were slower than legacy
  (median 0.83-0.85x, worst 0.53x), accepted with the switch (user decision, 2026-11-08).

`RaggedExpression::segment_mean()` emits the direct segmented-mean stage, so FP32 accumulation, division by row
length, empty-row handling and output conversion happen in one reduction stage without materializing row lengths or
segmented sums. Expression's segmented forward reductions and the segment min / max backward (through
`CubSegmentedArgReduction`) all stamp this path.

## Expression integration

`BuiltReduction` caches the normalized axes, result kind, operation, and geometry. A plan produces either a value or
indices; the backend is not selectable. `StampedReduction`, `StampedArgMinMax`, and
`StampedReduceMinMaxBackward` bind those plans to concrete tensors. Dense min/max backward scatters through CUB's winning local indices. Ragged segmented min/max backward uses
`CubSegmentedArgReduction` to produce global packed winners, then launches a flat device-runtime active-prefix zero over
`[0, offsets[B] * D)` followed by a winner-only scatter. Reserved ragged capacity beyond `offsets[B]` is left untouched,
and no host readback of the dynamic active extent is required.

The test `ExpressionReductionArchitecture.ActiveSourcesDoNotUseCudnnReductionApis` prevents the retired cuDNN
reduction descriptors, workspace queries, and execution API from being reintroduced anywhere in active Thor sources.
The source guard `ExpressionReductionArchitecture.GeneralReductionsAreCentralizedUnderCubReduction` scans Thor's
`Utilities`, `DeepLearning`, and `bindings` sources for direct general-purpose CUB value or arg reductions, and
`ExpressionReductionArchitecture.NoReductionCallsACubDeviceReduction` (1373) rejects any `cub::DeviceReduce::` or
`cub::DeviceSegmentedReduce::` call and the device-reduce headers everywhere, the central utility included. The obsolete
standalone `CubDeviceReduce*` and `CubDeviceSegmentedReduce*` primitive wrappers were removed after all value and
offset-segmented callers moved to the central utility. `FlatScatterAddKernel` still uses CUB ReduceByKey,
which is a keyed grouping primitive rather than a tensor-axis reduction and therefore remains separate.

Loss shaping uses central CUB sums with an explicit FP32 output scale: batch and classwise losses divide only by the
batch size, while elementwise losses sum non-batch elements without normalization. Binary accuracy uses the same
scaled-sum facility. The obsolete `BatchReduce` class has been removed.
