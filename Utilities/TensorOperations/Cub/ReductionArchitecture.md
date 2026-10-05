# Thor Reduction Architecture

Thor's central GPU reduction implementation lives in `Utilities/TensorOperations/Cub`.
Dense expression reductions and ragged offset-segmented reductions use these utilities as the central implementation;
they do not stage inputs through compatibility tensors and do not call `cudnnReduceTensor`. The central utility is
allowed to use either CUB device primitives or Thor-owned CUDA kernels when tensor geometry requires a layout-aware
implementation. Expression's vector-valued segmented forward caller is migrated to the central API separately.

## Numeric contract

- Input storage may be FP8 E4M3, FP8 E5M2, FP16, BF16, FP32, or an enabled FP64 type.
- Input iterators convert values to FP32 before reduction.
- Reduction state and operation-specific finalization use FP32.
- The final store converts once to the configured output storage dtype.
- Stamped operations own their output and CUB workspace. `run()` and `runOn()` do not allocate or re-plan.

## Dense paths

`CubReduction` selects the fastest backend implied by dense row-major geometry:

1. Device transform-reduce when the result contains one element.
2. CUB fixed-size segmented reduction when each reduction domain is physically contiguous (`inner_size == 1`).
3. A tiled row-vector CUDA/CUB-warp backend when the reduced axes are one contiguous block with trailing values.
4. Arbitrary ordinary-dense combinations of reduced/retained runs are structurally classified as `ComposedDense`, but VALUE execution is not a composed parent/child executor. `planDenseReduction()` selects one R/KR/RK physical pass, produces a fresh intermediate problem, and replans globally until complete. The selected passes are materialized as one flat `DenseExecutablePlan`; every non-final VALUE aggregate is FP32, only the first logical pass applies the public input transform, and only the final logical pass applies the public finalizer/output scale.
5. Non-dense inputs remain outside the ordinary-dense planner. Proven compact/permutation/pitched layouts use their ordained view reducers; unsupported arbitrary views fail structural analysis rather than entering a generic logical-index fallback.

The tiled path also accepts zero-copy logical permutations when stride analysis proves that the visible source is physically dense `[outer, reduction, inner]` storage. In that case the reducer traverses the physical source layout directly and writes the requested retained-axis dense order without materializing the permutation. Natural `[outer,inner]` output keeps the ordinary tuned store mapping. Production `[inner,outer]` output uses the shared-transpose retained writer described below so final global stores remain coalesced.

For production `[inner,outer]` writes, eight physical warps reduce adjacent outer rows for one contiguous retained-component tile. Each lane owns a contiguous register packet, using the same <=16 FP32-accumulator budget and CUDA vector-packet loaders as the direct full-row family; retained widths above 512 become independent 512-component tiles rather than increasing per-thread state. Reduction state never leaves registers. Only finalized FP32 retained values are staged through a padded shared-memory tile, after which threads read that tile in transposed order and write adjacent outer coordinates contiguously in dense `[inner,outer]`. The largest retained tile is 8 x 512 values (about 16 KiB), there is no global reduction/permutation intermediate, and grid-stride iteration handles arbitrarily tall outputs without treating 65,535 as a `grid.x` architectural limit.

The tiled path views the input as `[outer, reduction, inner]` and selects a layout-aware kernel by trailing width.
For `inner <= 32`, each physical warp owns the complete trailing row and a private two-stage `cuda::memcpy_async`
global-to-shared pipeline. The pipeline copies several complete consecutive rows as one contiguous slab and overlaps
the next copy stage with FP32 reduction of the current stage. For `inner <= 16`, otherwise-unused lanes split reduction
rows and CUB logical `WarpReduce` combines those row partials.

For `33 <= inner < 512`, a physical warp can still own the complete trailing vector, with each lane owning 2, 4, 8, or
16 output components. The async pipeline therefore continues to copy complete consecutive rows instead of regressing to
per-row component-strip copies. Shared-memory consumption is striped by component round (`lane + 32 * round`) so each
round is contiguous across the warp. A full row must fit in the warp's 2 KiB async stage; this holds through `inner=511`
for FP32 and narrower storage, while FP64 widths above 256 intentionally retain the direct Patch-2 component-tiled
backend. Exact widths 64, 128, 256, and 512 use a direct full-row kernel instead: each lane owns a contiguous 2/4/8/16
component packet and loads it with CUDA-native 2/4/8/16-byte vector types or a short compile-time sequence of `uint4`
loads, keeping all accumulation in registers with no synchronization or shared memory.

Sixteen FP32 accumulators per lane are the current proven-fast register tile, while Thor keeps the complete kernel near
an approximately 48-register/thread design budget. Wider rows first scale the same full-row engine horizontally instead
of increasing per-thread accumulator count: 513..1024 components use 2 warps/output, 1025..2048 use 4, and 2049..4096
use all 8 warps in the 256-thread block. For awkward widths the output group cooperatively stages complete contiguous
rows; exact 1024/2048/4096 widths use direct vector packets with no shared memory or synchronization. FP64 may select a
larger group for awkward widths because each warp contributes both 512 components of register ownership and 2 KiB of
async-stage capacity.

`inner > 4096` is not a performance-path limit. Different trailing components are independent reductions, so Thor
shards one output across independent blocks without any inter-block reduction, atomics, or second pass. Exact multiples
of 4096 retain the x16 vector-direct packet path in every block. Arbitrary widths keep the same contiguous 16-component
per-thread ownership and use an alignment-safe global-to-register loader: each logical packet is reconstructed from an
aligned 16-byte window, with the tensor's 128-byte trailing allocation padding making the final fixed-width packet safe
without scalar tail loads. The host scheduler normally preserves every complete 4096-component shard and emits one
remainder shard. For a sub-2048 remainder it borrows the final full shard and creates two ~half-block tails only when the
smaller tail still contributes at least 512 aggregate useful warps; this avoids pathological tiny-tail launches when
outer parallelism is plentiful without sacrificing proven 4096-component geometry at low outer parallelism. The
arbitrary-width large-D path uses no shared memory, asynchronous-copy bookkeeping, synchronization, or inter-block
communication. Increasing D therefore creates more independent component blocks while keeping per-thread state bounded.

Aligned 4/8/16-byte async runs remain useful up through one block per output, where complete-row staging repairs awkward
small/medium widths. Total dynamic shared allocation there stays fixed at 32 KiB/block: it is repartitioned from eight
private one-warp pipelines into four 2-warp groups, two 4-warp groups, or one 8-warp group as D grows. Once D requires
multiple independent component blocks, the direct striped backend removes shared memory from the scaling path entirely.
This mirrors the reducer geometry used elsewhere in Thor:
small widths can place multiple logical reductions in one warp, medium widths use one warp per output, large widths use
one cooperative block per output, and very large widths use multiple independent component blocks per output. All tiled
backends accumulate and finalize in FP32 and require no stamped dynamic workspace.

Dense reduction rank is dynamic. Structurally `ComposedDense` VALUE reductions stamp a flat sequence of physical
R/KR/RK passes in `DenseExecutablePlan`; there is no VALUE parent executor or child-stage continuation. `run()` launches
that ordered pass list on the caller stream with no host synchronization, allocation, or runtime planning. Only the
ordinary dense planning needs no per-element logical-index metadata. There is no cuDNN-derived rank-8 limit; the only
representation bound is that axis identifiers are `uint32_t`.

Value reductions support sum, product, mean, min, max, L1 norm, L2 norm, and sum-of-squares. VALUE-BINARY-CLEANUP keeps input transforms and reduction operators compile-time specialized while sharing additive finalization at output granularity: Sum, Mean, and the final composed L2 stage use one `IdentityFp32 + AdditiveFinalizeFp32` kernel matrix, while SumSquares and complete L2 use one `SquareFp32 + AdditiveFinalizeFp32` matrix. Division and optional square root are runtime finalizer data paid once per output, so L2 does not instantiate independent copies of every tiled/CUB reduction geometry. `CubReductionMean.cu` and `CubReductionL2Norm.cu` remain intentionally template-free. Dense argmin/argmax use the same geometry
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
`TiledFixedSegment`, or `ComposedDense`. The structural dense planner must produce a complete all-direct composition for
every disjoint mask; failure is an internal logic error rather than an invitation to enter a catch-all view reducer.
For VALUE, `ComposedDense` is now only that structural classification: `planDenseReduction()` chooses the actual
R/KR/RK pass sequence and globally replans after each pass. ARG still uses the transitional interval-composition
executor until DENSE-ARG-CUTOVER; it must ultimately move onto the same topology planner and flat pass executor.

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
normal aligned contiguous kernel and every carried-index stage is FP32 by construction. The remaining CUB segmented ARG
instantiations in the dense executor are therefore limited to genuine UINT64 *composed* contiguous stages; direct
contiguous ARG above UINT32 is unreachable because the CUB fixed-segment API's `int` segment-size gate rejects it first.
The separate offset-segmented ARG API retains its own intentionally independent segmented implementation.

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

`CubReduction::stampPhysicalStage()` already follows this invariant for cooperative sharded tiled reductions: after the
first pass writes FP32 partials, it analyzes `[outer, shards, inner]` and calls the same physical-stage planner again.
Benchmark staged candidates must preserve the same behavior so a census measures reducer chains rather than artificial
first-stage/finalizer pairings.

### Low-output / deep full-row reduction sharding

The direct and grouped `TiledFixedSegment` full-row kernels derive most of their launch parallelism from independent
outer/output rows. That ownership remains the fastest strategy when many outputs are available, but it is pathological
for shapes such as `[1, 104832, 512] -> [1, 1, 512]`: one warp (or one small cooperating warp group) otherwise scans the
entire reduction domain while almost every SM is idle.

The original fix for this under-parallelism was `ROW-SPLIT-FULL-ROW`, a generic two-stage fallback. It proved that
R sharding could recover device utilization, but its dedicated first-stage/finalize pairing was a transitional design.
After the KParallel/RCooperative census closed its coverage and meaningful performance gaps, the historical row-split
kernels, selectors, benchmark registration, and stamped compatibility state were removed from the compilation surface.
Their role is now historical calibration only; ordinary dense VALUE RK is owned entirely by `ReducersDenseRK`.

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

### RK post-delete coverage gate

`thor_cub_reduction_benchmark --rk-census-gate` is the durable post-delete gate for ordinary dense RK. Production now
stamps and executes the `ReducersDenseRK` family plan directly, so the gate times that production path only. It does not
compile or execute a second legacy/reference VALUE reducer inventory: retaining those historical CUDA/CUB kernels only
for live A/B measurement would keep their template specializations in the binary and defeat the deletion.

The gate spans the existing low-output/deep-R, retained-width, output-count, wide awkward, and sub-4096/sub-512/
sub-256/sub-64 awkward populations for FP16/BF16/FP32. SUM/MIN/MAX/PRODUCT exercise the core geometry-dependent
operation classes, and a compact set of Complete, aligned staged, K-parallel, rotated, and FlatRows sentinels additionally
covers Mean/L1/L2/SumSquares so input-transform/finalizer semantics are validated through multi-pass family execution.
Coverage gaps are reported without stopping the remaining census, while numerical disagreement remains fatal.

Capture stdout and run `benchmarks/analyze_rk_census_gate.py <output>`. Missing modern rows and explicit coverage gaps
always fail analysis. The analyzer also requires the census to exercise K-parallel, R-cooperative, rotated
R-cooperative, flat-row R-cooperative, Complete, and Staged modern plans so an accidentally dead part of the two-strategy
inventory cannot hide behind generic coverage. New post-delete CSVs are modern-only. The analyzer continues to accept
archived pre-delete paired CSVs when a historical production-vs-modern comparison is useful, without requiring those
legacy kernels to remain in the current compilation surface.

### View ownership after DELETE

`thor_cub_reduction_benchmark --view-census` is now the final ownership census rather than a benchmark of a generic
fallback. The structured view families that Thor intentionally supports are assigned directly to ordained implementations:

- VIEW-DIRECT-1 sends a one-to-one compact physical permutation with every logical axis reduced directly through
  `DeviceTransformReduce`, because visitation order is irrelevant for a full value reduction.
- VIEW-DIRECT-2A sends a compact permutation whose reduced axes form the physical suffix and whose retained physical order
  already equals dense logical output order through `ContiguousFixedSegment`.
- VIEW-PITCHED-TILED handles a logically contiguous `[outer..., reduction..., inner...]` view when the trailing payload is
  physically contiguous, outer and reduction groups each flatten to one constant pitch, and adjacent slabs do not overlap.
  Its dedicated Thor kernel addresses `outer * outer_stride + reduction * reduction_stride + inner`, so repeated-label
  einsum diagonal pre-reductions such as `[2,2,3]` with strides `[18,3,1]` remain zero-copy and coalesced.
- VIEW-DIRECT-2B handles a compact source collapsible to `[A,reduction,B,payload]` with dense logical output
  `[B,A,payload]`. Its dedicated Thor kernel reduces from compact physical storage, stages finalized FP32 values through a
  fixed `32x33` shared-memory tile, and writes dense retained order directly without a global transpose intermediate.

DELETE removes the old arbitrary rank>1 logical-index reducer completely. There is no legacy execution-family enum,
host/device mixed-radix indexing metadata, device logical-to-physical mapper, strided FP32 iterator, legacy 32/64-bit CUB
segmented-reduction specialization surface, or benchmark-only resurrection hook. Unsupported arbitrary views therefore
throw `NotImplementedException` directly from structural analysis. The final view census retains representative unsupported
layouts—stride-2 inner payloads, zero-stride broadcasts, overlapping aliases, split physical reductions, singleton-heavy
gapped views, and the former synthetic UINT64-index case—but reports them as `unsupported / not_implemented` without
allocating or timing an implementation.

Einsum keeps repeated-label diagonal operands as zero-copy views when an ordained reducer owns their stride pattern. If an
operand-local pre-reduction has no ordained owner, einsum materializes only that logical operand into dense order and then
uses the normal dense reducer. This preserves the supported einsum surface without reintroducing a generic arbitrary-view
reduction mechanism.

`CubReductionViewGate`, the dense value/ARG gates, execution tests for each ordained view strategy, and the source guard in
`ReductionSourceGuardTest` collectively enforce the post-DELETE architecture. The source guard also requires the deleted
logical-index header to stay absent and rejects the historical mapper/indexing symbols if they reappear in active CUB
reduction sources.

## Offset-segmented path

`CubSegmentedReduction` accepts values `[N,D...]` with row offsets `[B+1]` and produces `[B,D...]` for ragged sum,
mean, min, and max. Rank-1 values retain CUB `DeviceSegmentedReduce`; vector-valued rows use a zero-workspace Thor CUDA
backend where adjacent threads own adjacent trailing components and each thread walks the rows in one segment. This
keeps every row load coalesced while preserving FP32 accumulation, runtime output conversion, and the scalar empty-row
identities. Expression's scalar and vector segmented forward reductions both stamp this central implementation; the former private
Expression vector forward kernel has been removed. `CubSegmentedArgReduction` accepts the same `[N,D...]` row geometry
and returns global packed winner indices `[B,D...]`; narrow vectors split segment rows across logical warp lanes while
wider vectors assign adjacent trailing components to adjacent threads. Empty segments return the maximum index sentinel
per component, NaNs propagate, and the lowest packed index wins ties. `RaggedExpression::segment_mean()` emits the direct segmented-mean
stage, so FP32 accumulation, division by row length, empty-row handling, and output conversion happen in one reduction
stage without materializing row lengths or segmented sums. Segment offsets are row indices for both scalar and vector
values, are validated while stamping, and empty segments use the same explicit identities as
`CubReduction::getFp32EmptyReductionValue()`.

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
`Utilities`, `DeepLearning`, and `bindings` sources for direct general-purpose CUB value or arg reductions. The obsolete
standalone `CubDeviceReduce*` and `CubDeviceSegmentedReduce*` primitive wrappers were removed after all value and
offset-segmented callers moved to the central utility. `FlatScatterAddKernel` still uses CUB ReduceByKey,
which is a keyed grouping primitive rather than a tensor-axis reduction and therefore remains separate.

Loss shaping uses central CUB sums with an explicit FP32 output scale: batch and classwise losses divide only by the
batch size, while elementwise losses sum non-batch elements without normalization. Binary accuracy uses the same
scaled-sum facility. The obsolete `BatchReduce` class has been removed.
