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
4. Arbitrary ordinary-dense combinations of reduced/retained runs use a stamped `ComposedDense` plan. The planner removes one reduced run at a time through only the proven direct families above, keeps every non-final aggregate in FP32, and chooses pass order at stamp time with the dense-stage cost model. For transformed/finalized operations, only the first pass applies the public input transform and only the final pass applies the public finalizer/output scale.
5. Fixed-size segmented reduction over a logical counting/transform iterator remains only as the correctness fallback for genuinely irregular/non-dense views while that traversal is being replaced.

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

Dense reduction rank is dynamic. `ComposedDense` value reductions do not stamp logical-index metadata; all child
reductions and their First/Intermediate/Final semantics are fully planned while stamping and `run()` launches them
consecutively on the caller stream with no host synchronization, allocation, or runtime planning. Only the remaining
genuinely irregular-view fallback packs dimensions, strides, and axis lists into a rank-sized GPU metadata tensor while
stamping. There is no cuDNN-derived rank-8 limit; the only representation bound is that axis identifiers are `uint32_t`.

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
Operation-specific value and ARG planners choose only the detailed legal stage order inside `ComposedDense`; they do not
rewrite the execution family.

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

### Low-output / deep full-row reduction sharding

The direct and grouped `TiledFixedSegment` full-row kernels derive most of their launch parallelism from independent
outer/output rows. That ownership remains the fastest strategy when many outputs are available, but it is pathological
for shapes such as `[1, 104832, 512] -> [1, 1, 512]`: one warp (or one small cooperating warp group) otherwise scans the
entire reduction domain while almost every SM is idle.

Release census calibration therefore ordains a separate two-stage row-split strategy for the measured low-output,
deep-reduction regime: FP16/BF16/FP32 inputs, reduction extent at least 1024, outer extent at most 128, and retained
width 15..4096. The first stage keeps the full-row family's coalescing invariant—threads own adjacent trailing
components—but splits reduction rows across enough CTAs for approximately two waves over the target GPU's SMs. Each CTA
writes one FP32 partial vector. A second deterministic kernel combines those partial vectors in shard order, applies the
operation's finalizer exactly once, applies the runtime output scale exactly once, and performs the final storage-dtype
conversion. Sum, mean, L1/L2, sum-squares, product, min, and max therefore share the same row-split execution mechanism;
input transforms occur only in the first stage and finalization only in the second.

The row-split workspace is stamped explicitly as `outer * shards_per_output * inner * sizeof(float)`. The shard count is
chosen at stamp/query time from the target GPU's multiprocessor count and recovered from the stamped workspace at run
time, so steady-state launches perform no device-property query or allocation. Existing direct/grouped/block-sharded
kernels remain unchanged outside the calibrated gate, including many-output and short-reduction cases where the extra
partial/finalize pass is unnecessary. The dense composition planner models row-split stages as saturated rather than
retaining the old one/few-warp parallelism penalty.

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
