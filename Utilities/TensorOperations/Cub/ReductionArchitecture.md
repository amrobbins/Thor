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
4. Fixed-size segmented reduction over a logical counting/transform iterator only for genuinely disjoint reduced axes.

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

Dense reduction rank is dynamic. Only the disjoint-axis fallback packs dimensions, strides, and axis lists into a
rank-sized GPU metadata tensor while stamping. There is no cuDNN-derived rank-8 limit; the only representation bound is
that axis identifiers are `uint32_t`.

Value reductions support sum, product, mean, min, max, L1 norm, and L2 norm. Dense argmin/argmax use the same geometry
classification: device-wide and physically contiguous domains remain on CUB, genuinely disjoint axes retain the logical
index fallback, and contiguous middle-axis reductions use Thor's tiled backend. The ARG tiled backend keeps FP32 values
paired with local reduction-row indices, normally using UINT32 candidate indices even for UINT64 outputs and promoting
the hot candidate state to UINT64 only when the reduction domain itself exceeds UINT32. Narrow and awkward widths use
the coalesced component-tiled CUB-warp backend. ARG caps contiguous packet ownership at four candidate pairs per lane:
128 components use one warp, 256/512/1024 use 2/4/8-warp groups, and larger exact widths use independent
1024-component block shards. Arbitrary D > 4096 uses the same x4 alignment-safe packets with preserved 1024-component
shards plus at most one remainder shard. Arg reductions produce deterministic local flattened indices: NaNs propagate, and
the lowest logical index wins equal-value ties.

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
