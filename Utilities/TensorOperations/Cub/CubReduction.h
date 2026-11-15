#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReductionPassRole.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Cub/StridedRLaunch.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {

enum class CubReductionOp : uint8_t {
    Sum = 0,
    Min = 1,
    Max = 2,
    Product = 3,
    Mean = 4,
    L1Norm = 5,
    L2Norm = 6,
    SumSquares = 7,
};

enum class CubArgReductionOp : uint8_t {
    ArgMin = 0,
    ArgMax = 1,
};

struct CubArgReductionOutputOptions {
    bool produce_value = true;
    bool produce_index = true;
    std::optional<DataType> value_output_dtype = std::nullopt;
    DataType index_output_dtype = DataType::UINT32;
};

enum class CubReductionPath : uint8_t {
    DeviceTransformReduce = 0,
    ContiguousFixedSegment = 1,
    TiledFixedSegment = 2,
    OffsetSegmented = 4,
    ComposedDense = 5,
};

enum class CubReductionTiledRetainedOutputOrder : uint8_t {
    NaturalOuterInner = 0,
    PermutedInnerOuter = 1,
};

/** Role of one collapsed run in a dense row-major reduction traversal. */
enum class CubReductionDenseRunKind : uint8_t {
    Reduced = 0,
    Retained = 1,
};

/**
 * One collapsed run of adjacent, traversal-equivalent dimensions in a dense row-major tensor.
 *
 * extent is the product of the non-singleton dimensions represented by the run. physical_stride is the source-element
 * stride between adjacent coordinates of that collapsed run. domain_stride is the row-major multiplier of that run in
 * the flattened reduction domain (for Reduced runs) or flattened output domain (for Retained runs). Singleton
 * dimensions are omitted because they neither advance storage nor contribute a coordinate; runs on either side of an
 * omitted singleton may therefore collapse together when they have the same role.
 */
struct CubReductionDenseRun {
    CubReductionDenseRunKind kind = CubReductionDenseRunKind::Reduced;
    uint64_t extent = 1;
    uint64_t physical_stride = 1;
    uint64_t domain_stride = 1;
};

/**
 * Stamp-time traversal plan for an ordinary dense row-major tensor.
 *
 * Replacement dense-run reducers can consume this compact run sequence instead of rediscovering multidimensional
 * coordinates from every scalar's flattened logical index. Launch policy is intentionally not part of this structure;
 * block size, sharding, and scratch policy are selected separately.
 */
struct CubReductionDenseRunGeometry {
    std::vector<CubReductionDenseRun> runs;
    uint32_t reduced_run_count = 0;
    uint32_t retained_run_count = 0;
};

/**
 * Production planning metadata for a zero-copy logical permutation whose visible storage is physically dense.
 *
 * Singleton axes are intentionally omitted from the physical axis vectors and logical retained-order vector because
 * they do not affect address order.
 * The source can therefore be traversed as one dense [outer, reduction, inner] tensor even though its logical Tensor
 * view is non-dense. retained_output_order describes whether the logical dense output order matches the natural
 * [outer..., inner...] retained order or requires the flattened [inner..., outer...] retained-axis permutation.
 *
 * 1376: the retained dense planner reduces such a view as the dense physical [outer, reduction, inner]. 1390: a
 * permuted [inner..., outer...] output is refused (CubReductionViewSplit). The metadata is the classification.
 */
struct CubReductionPermutationAwareTiledGeometry {
    std::vector<uint32_t> physical_outer_axes;
    std::vector<uint32_t> physical_reduction_axes;
    std::vector<uint32_t> physical_inner_axes;
    std::vector<uint32_t> logical_non_singleton_retained_axes;

    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;

    CubReductionTiledRetainedOutputOrder retained_output_order =
        CubReductionTiledRetainedOutputOrder::NaturalOuterInner;
};

/**
 * VIEW-PITCHED-TILED metadata for a non-overlapping affine view that is logically
 * [outer..., reduction..., inner...] and whose trailing inner block is physically contiguous.
 *
 * The outer and reduction groups may contain padding between adjacent flattened entries,
 * but each group must itself be row-major flattenable to one constant element stride. The
 * resulting source address is therefore exactly
 *
 *   outer * outer_stride + reduction * reduction_stride + inner
 *
 * with no per-scalar mixed-radix coordinate reconstruction. This is intentionally narrower
 * than a general strided-view reducer: zero-stride/broadcast aliases, overlapping views,
 * retained-axis permutations, and non-contiguous inner payloads are not candidates.
 * 1390: such a view is refused; the caller reduces its dense copy (CubReductionViewSplit). The metadata is its
 * classification.
 */
struct CubReductionPitchedTiledGeometry {
    uint64_t outer_size = 1;
    uint64_t reduction_size = 1;
    uint64_t inner_size = 1;
    uint64_t outer_stride = 0;
    uint64_t reduction_stride = 1;
};

/**
 * VIEW-DIRECT-2B metadata for the compact-permutation retained-order transform
 *
 *   physical source: [A..., reduction..., B..., payload...]
 *   logical output:  [B..., A..., payload...]
 *
 * after singleton axes are removed. The source is physically dense, so all sizes below are flattened once on the
 * host. 1390: such a view is refused; the caller reduces the bytes as the dense [A, reduction, B x payload] and its
 * dense copy writes logical [B,A,payload] (CubReductionViewSplit). The metadata is the classification.
 */
struct CubReductionPayloadTransposeTiledGeometry {
    std::vector<uint32_t> physical_a_axes;
    std::vector<uint32_t> physical_reduction_axes;
    std::vector<uint32_t> physical_b_axes;
    std::vector<uint32_t> physical_payload_axes;

    uint64_t a_size = 1;
    uint64_t reduction_size = 1;
    uint64_t b_size = 1;
    uint64_t payload_size = 1;
};

struct CubReductionGeometry {
    std::vector<uint32_t> axes;
    uint32_t rank = 0;

    // For an ordinary dense tensor whose reduced axes form one contiguous logical block, the input can be viewed as
    // [outer_size, reduction_size, inner_size]. For a production permutation-aware tiled reduction these fields are
    // instead the equivalent dense physical [outer, reduction, inner] geometry recovered from the view's strides.
    // They remain zero for layouts that cannot use either tiled geometry.
    bool reduced_axes_are_contiguous = false;
    uint64_t outer_size = 0;
    uint64_t inner_size = 0;

    uint64_t input_elements = 0;
    uint64_t reduction_size = 0;
    uint64_t output_elements = 0;
    std::vector<uint64_t> output_dimensions;
    std::vector<uint64_t> squeezed_output_dimensions;
    // Populated only for ordinary dense row-major storage. This is traversal geometry, not launch policy. Production
    // composed reducers consume this metadata during value-path planning and stamping. SUM can eliminate arbitrary
    // reduced-run sequences through direct reducer stages; other value operations and arg reductions use it as their
    // migration metadata until their composed semantics are implemented.
    std::optional<CubReductionDenseRunGeometry> dense_run_geometry = std::nullopt;

    // A rank-1 non-dense view (an extracted matrix diagonal, a column, a stride-0 broadcast) is an affine sequence
    // reduced without materializing it or paying the general logical-axis index mapping. 1366: it runs StridedR
    // (StridedReduction.h), no longer CUB's DeviceReduce over an affine iterator.
    bool device_transform_uses_affine_stride = false;
    uint64_t affine_input_stride = 1;

    // Physical-layout analysis is relative to input.getMemPtr(), which already includes a Tensor view's storage
    // element offset.  Storage offsets are therefore intentionally not baked into this reusable geometry.
    bool physical_layout_is_dense_permutation = false;
    std::vector<uint32_t> physical_non_singleton_axis_order;

    // VIEW-DIRECT-2A: a compact physical permutation whose non-singleton reduced axes form the complete physical
    // trailing block can be handed directly to CUB DeviceSegmentedReduce when the physical retained-axis order already
    // matches Thor's logical dense output order. Each output is then one ordinary contiguous physical segment; no
    // logical-to-physical input mapper or output permutation is required. More general retained-order permutations are
    // intentionally left for VIEW-DIRECT-2B.
    bool permutation_aware_contiguous_segments = false;

    std::optional<CubReductionPermutationAwareTiledGeometry> permutation_aware_tiled_geometry = std::nullopt;

    // VIEW-DIRECT-2B: compact dense physical [A,reduction,B,payload] storage whose logical retained order is
    // [B,A,payload]. 1390: refused (CubReductionViewSplit).
    std::optional<CubReductionPayloadTransposeTiledGeometry> payload_transpose_tiled_geometry = std::nullopt;

    // VIEW-PITCHED-TILED, classified under TiledFixedSegment. Output order is always the natural logical
    // [outer,inner] order. 1390: refused (CubReductionViewSplit).
    std::optional<CubReductionPitchedTiledGeometry> pitched_tiled_geometry = std::nullopt;

    // TiledFixedSegment logically produces a dense [outer_size, inner_size] matrix after reducing the physically
    // contiguous middle domain. Natural output uses output_index = outer * inner_size + inner. Production
    // permutation-aware planning can instead request dense [inner_size, outer_size]. Input traversal and reduction
    // parallelism remain in the tuned tiled implementation; only the retained-output writer changes.
    uint64_t tiled_output_outer_stride = 0;
    uint64_t tiled_output_inner_stride = 0;
    bool tiled_output_permuted = false;
    // A permuted retained output is finalized through a small shared-memory tile before global storage. This keeps the
    // reduction itself in registers while turning the dense [inner,outer] destination into coalesced writes across
    // adjacent outer coordinates. No reduction/permutation intermediate is allocated in global memory.
    bool tiled_output_shared_transpose = false;

    CubReductionPath path = CubReductionPath::DeviceTransformReduce;
};
namespace CubReductionInternal {
struct DenseRetainedFp8ArgPhysicalPlan;
struct DenseRetainedFp32ArgPhysicalPlan;  // 1221

/**
 * One bound ordinary-dense physical pass.
 *
 * The family-local physical plan stays type-erased at the execution-plan level. role is explicit so semantic placement
 * never depends on reducer-owned continuation state. The pass owns both any persistent intermediate output and any
 * direct-backend temporary storage queried at stamp time, so execution needs no reducer-owned continuation state.
 */
struct DenseExecutablePass {
    std::shared_ptr<const DenseReductionPhysicalPlan> physical_plan;
    DensePassRole role = DensePassRole::Complete;
    StridedSpan input;
    mutable Tensor output;
    CubReductionGeometry geometry;

    // Persistent storage owned by this physical pass. intermediate_bytes accounts for a non-final aggregate output;
    // temp_storage_bytes/temp_storage are direct-backend scratch queried at stamp time. Modern RK kernels currently need
    // no separate launch scratch. Keeping both here makes the flat pass representation sufficient for R/KR/RK.
    size_t intermediate_bytes = 0;
    size_t temp_storage_bytes = 0;
    Tensor temp_storage;
    // output and temp_storage as the spans the launch takes, made once at stamp.
    StridedSpan output_span;
    StridedSpan temp_storage_span;

    [[nodiscard]] size_t workspaceBytes() const {
        if (intermediate_bytes > std::numeric_limits<size_t>::max() - temp_storage_bytes) {
            throw std::overflow_error("Dense executable pass workspace size overflows size_t.");
        }
        return intermediate_bytes + temp_storage_bytes;
    }
};

/** Flat ordered physical-pass representation for ordinary dense execution. */
struct DenseExecutablePlan {
    std::vector<DenseExecutablePass> passes;
    size_t workspace_size_bytes = 0;
};

}  // namespace CubReductionInternal

/** Read-only view of one already stamped dense pass (no GPU allocation or selection).
 * Fields that do not apply to the selected physical family remain nullopt;
 * there is no invented packet, CTA, or shard geometry for retained FP8.
 */
struct DensePhysicalPassDiagnostic {
    std::string family;
    CubReductionInternal::DensePassRole role;
    DataType input_dtype;
    DataType output_dtype;
    std::vector<uint64_t> input_dimensions;
    std::vector<uint64_t> output_dimensions;
    size_t intermediate_bytes = 0;
    size_t scratch_bytes = 0;
    std::optional<size_t> packet_bytes;
    std::optional<uint32_t> block_threads;
    std::optional<uint64_t> stage_blocks;
    std::optional<uint64_t> shards_per_output;
};

/**
 * 1390 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md): the two parts by which a caller runs a value reduction
 * of a view the reducer refuses. Reductions read their input and write their output in physical order; a permutation
 * is the dense copy's (copyToDenseAsync, Utilities/TensorOperations/Copy/StridedCopy.h). CubReduction::analyzeGeometry
 * throws NotImplementedException for these views, and CubReduction::splitRefusedView says how to run them.
 */
struct CubReductionViewSplit {
    enum class Kind : uint8_t {
        // F (a compact permutation [O, R, I] whose output is [I, O]) and G (VIEW-DIRECT-2B, [A, R, B, P] whose output
        // is [B, A, P]): reduce the view's bytes, the dense dense_dimensions = [a, r, b x p] from its first element
        // (input.asDense), over dense_axes into a dense buffer_dimensions = [a, b x p] of the output dtype. The output,
        // dense [b, a, p] in its logical order (keepdims extents aside), is that buffer read with copy_dimensions
        // {b, a, p} and copy_strides {p, b x p, 1}: its dense copy.
        RotatedOutput,
        // 1391 (phase 7 A), H whose rows read at their pitch cost less than the packed copy: [O, R, I] rows of I
        // contiguous elements at row pitch Sr > I, with no gap between slabs (slab pitch R x Sr, or O = 1). Reduce the
        // dense dense_dimensions = [O, R, Sr] from the view's first element (input.withSourceLayout: it reaches Sr - I
        // elements past the view's last, inside its source tensor) over dense_axes into a dense buffer_dimensions =
        // [O, Sr]; the output [O, I] is that buffer read with copy_dimensions {O, I} and copy_strides {Sr, 1}. The gap
        // columns are reduced and dropped. Only CubReduction::splitRefusedView of a span returns it.
        SlicedOutput,
        // H (VIEW-PITCHED-TILED: rows of contiguous elements with gaps between them): reduce the view's dense copy
        // (dense_dimensions, the view's own, in its dtype) over dense_axes (the view's own).
        PackedInput,
    };
    Kind kind = Kind::RotatedOutput;
    std::vector<uint64_t> dense_dimensions;
    std::vector<uint32_t> dense_axes;
    // RotatedOutput / SlicedOutput: the dense reduction's output and how its copy reads it as the output.
    std::vector<uint64_t> buffer_dimensions;
    std::vector<uint64_t> copy_dimensions;
    std::vector<uint64_t> copy_strides;
    uint64_t a = 1, b = 1, p = 1;  // RotatedOutput
    // SlicedOutput: O, R, I and Sr above.
    uint64_t rows_outer = 1, rows = 1, row_width = 1, row_pitch = 1;

    // RotatedOutput and SlicedOutput reduce the view's bytes (or its source's) and copy the output; PackedInput copies
    // the input.
    [[nodiscard]] bool copiesOutput() const { return kind != Kind::PackedInput; }
    // RotatedOutput / SlicedOutput: what the first part reduces, the dense dense_dimensions from `view`'s first
    // element, and what the second part copies, the dense reduction's output `buffer` (buffer_dimensions) read as the
    // output's dense copy_dimensions.
    [[nodiscard]] StridedSpan reducedInput(const StridedSpan& view) const {
        if (kind == Kind::RotatedOutput) return view.asDense(dense_dimensions);
        if (kind == Kind::SlicedOutput)
            return view.withSourceLayout(dense_dimensions, StridedSpan::denseStrides(dense_dimensions));
        throw std::logic_error("CubReductionViewSplit::reducedInput: a packed split reduces the view's dense copy.");
    }
    [[nodiscard]] StridedSpan copySource(const Tensor& buffer) const {
        return StridedSpan(buffer).withLayout(copy_dimensions, copy_strides);
    }
};

/**
 * 1392 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7 C): how the reducer reads a pitched view
 * (VIEW-PITCHED-TILED: [O, R, I] rows of I contiguous elements at row pitch Sr, slabs at pitch So) in place: one
 * RuntimeK PacketComplete launch at the view's pitches, warps_per_output (1 / 2 / 4 / 8) warps per [outer, K tile]
 * work item over R and K tiles of at most tile_components (FP32 124, FP16 / BF16 248, FP8 496). No workspace; the
 * output is the dense keepdims [O, I]. CubReduction::pitchedCompleteLaunch says when.
 */
struct CubReductionPitchedCompleteLaunch {
    uint32_t warps_per_output = 1;
    uint32_t tile_components = 1;
    bool operator==(const CubReductionPitchedCompleteLaunch&) const = default;
};

class StampedCubReduction;
class StampedCubArgReduction;
class StampedCubSegmentedArgReduction;
class StampedCubSegmentedReduction;
namespace CubReductionInternal {
struct RaggedRoutedStamp;  // 1325: RaggedStampAutotuner.h
}

/**
 * Describes a one-or-more-axis CUB reduction.
 *
 * CubReduction is intentionally not executable. stamp() validates the concrete tensors, selects the centralized
 * reduction backend, queries any required temporary storage, and allocates both the output and workspace.
 * StampedCubReduction::run()
 * subsequently performs no allocation or CUB planning.
 *
 * Reduction axes must be non-empty, unique, and strictly increasing. Every path converts input values to FP32 before
 * applying the operation-specific input transform and reduction operator. Accumulation therefore always occurs in
 * FP32. Operation-specific output finalization, such as mean division or the final square root for L2 norm, also
 * occurs in FP32 before the configured storage conversion. The output storage dtype defaults to the input storage
 * dtype and may be overridden explicitly. A finite runtime output scale, defaulting to one, is applied in FP32 after
 * operation-specific finalization and before the single storage conversion.
 */
class CubReduction {
   public:
    // tuning_depth (1141): stamp-time autotuning depth of retained FP8 sites. 1169: nullopt takes the ambient depth
    // (resolveCubReductionTuningDepth: this thread's ScopedCubReductionTuningDepth, else the process default).
    CubReduction(CubReductionOp op,
                 uint32_t axis,
                 std::optional<DataType> output_dtype = std::nullopt,
                 float output_scale = 1.0f,
                 std::optional<CubReductionTuningDepth> tuning_depth = std::nullopt);
    CubReduction(CubReductionOp op,
                 std::vector<uint32_t> axes,
                 std::optional<DataType> output_dtype = std::nullopt,
                 float output_scale = 1.0f,
                 std::optional<CubReductionTuningDepth> tuning_depth = std::nullopt);

    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }
    [[nodiscard]] CubReductionTuningDepth getTuningDepth() const { return tuning_depth; }
    [[nodiscard]] DataType resolveOutputDataType(DataType input_dtype) const;

    /**
     * Returns the FP32 result defined for an empty reduction domain.
     *
     * Dense Thor tensors currently reject zero-sized dimensions, so this value is primarily an explicit semantic
     * contract for reuse by future offset-segmented reductions. Mean follows Thor's empty-segment convention and
     * produces zero.
     */
    [[nodiscard]] static float getFp32EmptyReductionValue(CubReductionOp op);

    /**
     * Analyzes reduction geometry and selects an ordained execution family. DENSE-GATE-FINAL makes dense ownership a
     * hard invariant, and DELETE removes the arbitrary logical-index fallback entirely. Unsupported rank>1 views throw
     * NotImplementedException instead of being assigned a catch-all execution path.
     */
    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               uint32_t axis);
    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint32_t>& axes);
    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint64_t>& input_strides,
                                                               const std::vector<uint32_t>& axes);
    /** 1390: how a caller runs a view analyzeGeometry refuses (CubReductionViewSplit), else nothing: the reducer
     *  owns the layout, or nothing does. */
    [[nodiscard]] static std::optional<CubReductionViewSplit> splitRefusedView(
        const std::vector<uint64_t>& input_dimensions,
        const std::vector<uint64_t>& input_strides,
        const std::vector<uint32_t>& axes);
    /** 1391 (phase 7 A): the same for a view whose storage is known. A pitched view (H) is SlicedOutput when its rows
     *  can be read at their pitch inside the view's source tensor (StridedSpan::withSourceLayout) and that moves fewer
     *  bytes than the packed copy, with `output_dtype` the reduction's output; otherwise as above. 1392: nothing for a
     *  pitched view the reducer reads in place (pitchedCompleteLaunch). */
    [[nodiscard]] static std::optional<CubReductionViewSplit> splitRefusedView(const StridedSpan& input,
                                                                               const std::vector<uint32_t>& axes,
                                                                               DataType output_dtype);
    /** 1392 (phase 7 C): as above with `reducer_reads_pitched` false: the split a pitched view the reducer reads in
     *  place (pitchedCompleteLaunch) would have otherwise (benchmarks compare the two). */
    [[nodiscard]] static std::optional<CubReductionViewSplit> splitRefusedView(const StridedSpan& input,
                                                                               const std::vector<uint32_t>& axes,
                                                                               DataType output_dtype,
                                                                               bool reducer_reads_pitched);

    /**
     * 1392 (phase 7 C): the in-place launch for a pitched view (H) the reducer reads, or nothing: not a pitched view, a
     * dtype or extent PacketComplete does not take (an input base not element-aligned, more than 2^32 - 1 outer items
     * or K, more than 2^32 - 9 rows), K tiles under 256 bytes, or too little read in flight to beat the split it
     * replaces (the reducer then refuses the view and splitRefusedView says how to run it). The K tiles are balanced
     * under the dtype's cap. 1392a: warps_per_output is the most of 1 / 2 / 4 / 8 whose launch fits one wave of
     * resident warps (32 per SM), with at least four rows per warp; in place when that launch keeps 4 KiB per SM in
     * flight against a sliced split, 2 KiB per SM against a packed one (`output_dtype`, the reduction's output, decides
     * which: splitRefusedView). `warps_per_output` forces W and skips the tile-width and in-flight rules (benchmarks).
     */
    [[nodiscard]] static std::optional<CubReductionPitchedCompleteLaunch> pitchedCompleteLaunch(
        const StridedSpan& input,
        const std::vector<uint32_t>& axes,
        DataType output_dtype,
        std::optional<uint32_t> warps_per_output = std::nullopt);
    /** 1392: true when a value reduction stamps `input` as it is: analyzeGeometry accepts its layout, or it is a
     *  pitched view the reducer reads in place (pitchedCompleteLaunch, with the reduction's `output_dtype`). */
    [[nodiscard]] static bool readsSpanInPlace(const StridedSpan& input,
                                               const std::vector<uint32_t>& axes,
                                               DataType output_dtype);

    /**
     * Returns the executable geometry for a value reduction.
     *
     * analyzeGeometry() classifies structural access/shape, including the transitional ComposedDense geometry tag for
     * ordinary dense disjoint reductions. VALUE stamping now feeds every ordinary dense R/KR/RK problem through the
     * flat dense planner; special views remain on their ordained view executors. This wrapper validates the value
     * operation while preserving that structural geometry metadata (1374: no segment-size limit; every executor
     * carries 64-bit geometry).
     */
    [[nodiscard]] static CubReductionGeometry analyzeValueGeometry(CubReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint32_t>& axes);
    [[nodiscard]] static CubReductionGeometry analyzeValueGeometry(CubReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint64_t>& input_strides,
                                                                    const std::vector<uint32_t>& axes);
    /** 1392: the same for a span: a pitched view the reducer reads in place (pitchedCompleteLaunch, with the
     *  reduction's `output_dtype`) is accepted. */
    [[nodiscard]] static CubReductionGeometry analyzeValueSpanGeometry(CubReductionOp op,
                                                                        const StridedSpan& input,
                                                                        const std::vector<uint32_t>& axes,
                                                                        DataType output_dtype);

    /** Queries backend temporary storage without allocating input, output, or workspace tensors. */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const StridedSpan& input, const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const StridedSpan& input,
                                                             const Tensor& preallocated_output,
                                                             const Stream& stream) const;
    /** 1392 (benchmarks): a pitched view's in-place stamp at `warps_per_output`, whether or not the fill rule picks
     *  it (pitchedCompleteLaunch with W forced). Throws NotImplementedException when PacketComplete cannot read it. */
    [[nodiscard]] std::shared_ptr<StampedCubReduction> stampPitched(const StridedSpan& input,
                                                                    uint32_t warps_per_output,
                                                                    const Stream& stream) const;

   private:
    CubReductionOp op;
    std::vector<uint32_t> axes;
    // The classification analyzeGeometry refuses F / G / H from (1390): every view an ordained layout names.
    [[nodiscard]] static CubReductionGeometry classifyGeometry(const std::vector<uint64_t>& input_dimensions,
                                                                const std::vector<uint64_t>& input_strides,
                                                                const std::vector<uint32_t>& axes);
    [[nodiscard]] std::shared_ptr<StampedCubReduction> stampValidated(const StridedSpan& input,
                                                                      const Tensor& output,
                                                                      const CubReductionGeometry& geometry,
                                                                      const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubReduction> stampPitchedComplete(
        const StridedSpan& input,
        const Tensor& output,
        const CubReductionGeometry& geometry,
        const CubReductionPitchedCompleteLaunch& launch,
        const Stream& stream) const;

    std::optional<DataType> output_dtype;
    float output_scale;
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular;
};

/**
 * Describes an offset-segmented reduction over dense values [N,D...] -> [B,D...].
 *
 * Segment i is the half-open row range [offsets[i], offsets[i + 1]). Rank-1 input retains the scalar CUB fast path;
 * vector-valued input uses a Thor-owned coalesced CUDA backend under the same centralized API. Inputs are converted to
 * FP32 and all reduction arithmetic is performed in FP32. Empty segments use CubReduction::getFp32EmptyReductionValue().
 * Mean divides by the segment row count and returns zero for an empty segment. The ordinary stamp() API validates
 * offset contents immediately. stampRuntimeOffsets() is for graph inputs whose contents are populated only after stamping;
 * callers of that path must preserve the same zero-based, nondecreasing, in-bounds row-partition contract at execution.
 */
class CubSegmentedReduction {
   public:
    explicit CubSegmentedReduction(CubReductionOp op,
                                   std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] DataType resolveOutputDataType(DataType input_dtype) const;
    [[nodiscard]] static bool isInputDataTypeSupported(DataType dtype);
    [[nodiscard]] static bool isOffsetDataTypeSupported(DataType dtype);

    [[nodiscard]] std::shared_ptr<StampedCubSegmentedReduction> stamp(
        const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedReduction> stamp(
        const Tensor& input,
        const Tensor& preallocated_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

    // Expression/network offsets are runtime inputs and therefore do not contain
    // meaningful row-partition data while the physical graph is being stamped.
    // This path validates tensor shape/type/storage but deliberately defers content
    // validity to the runtime row-partition producer.
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedReduction> stampRuntimeOffsets(
        const Tensor& input,
        const Tensor& preallocated_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedReduction> stampValidated(
        const Tensor& input,
        const Tensor& output,
        const Tensor& segment_offsets,
        uint64_t num_segments,
        const Stream& stream) const;

    CubReductionOp op;
    std::optional<DataType> output_dtype;
};

/** Concrete, allocation-free-at-run-time offset-segmented reduction. */
class StampedCubSegmentedReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    /**
     * Benchmarks only (1263): runs this stamp on another input, offsets and output of the stamp's dtypes and element
     * counts, with the stamp's own workspace, so the ragged audit can time a stamp on rotating cold views without
     * re-stamping. The replacements are spans (the audits' cold views are spans of one arena). Production binds its
     * tensors.
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const StridedSpan& replacement_segment_offsets,
                                  const StridedSpan& replacement_output,
                                  Stream& run_stream) const;
    /** The physical passes this stamp runs, in order (diagnostic only, 1263). */
    [[nodiscard]] std::vector<std::string> getPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] CubReductionPath getPath() const { return CubReductionPath::OffsetSegmented; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] DataType getOffsetDataType() const { return segment_offsets.getDataType(); }
    [[nodiscard]] uint64_t getNumItems() const { return num_items; }
    [[nodiscard]] uint64_t getNumSegments() const { return num_segments; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    /** 1325: true when the stamp runs the retained ragged kernels (1330: always; the legacy kernels are deleted). */
    [[nodiscard]] bool usesRetainedRagged() const { return ragged != nullptr; }

   private:
    friend class CubSegmentedReduction;

    StampedCubSegmentedReduction(CubReductionOp op,
                                 const Tensor& input,
                                 const Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 size_t temp_storage_bytes,
                                 const Tensor& temp_storage,
                                 const Stream& stream);

    CubReductionOp op;
    const Tensor input;
    mutable Tensor output;
    const Tensor segment_offsets;
    const uint64_t num_items;
    const uint64_t num_segments;
    // For a direct reduction this is the backend temporary-storage size. For a ComposedDense parent it is the
    // aggregate stamped workspace footprint: all child temp storage plus all non-final value/index intermediates.
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    // The bound tensors as spans, made once at stamp so that a run converts nothing.
    const StridedSpan input_span{input};
    const StridedSpan offsets_span{segment_offsets};
    const StridedSpan output_span{output};
    const StridedSpan workspace_span{temp_storage};
    Stream stream;
    // 1325: the retained ragged plan (its workspace is temp_storage); 1330: never null.
    std::shared_ptr<const CubReductionInternal::RaggedRoutedStamp> ragged;
};

/**
 * Describes an offset-segmented argmin or argmax over packed values [N,D...].
 *
 * Row offsets [B+1] produce global packed-value winner indices [B,D...]. Empty segments produce the configured
 * UINT64_MAX/UINT32_MAX sentinel independently for every trailing component. Candidate values are compared in FP32,
 * NaNs propagate, and the lowest packed index wins ties, matching CubArgReduction.
 */
class CubSegmentedArgReduction {
   public:
    explicit CubSegmentedArgReduction(CubArgReductionOp op, DataType index_output_dtype = DataType::UINT64);

    [[nodiscard]] CubArgReductionOp getOperation() const { return op; }
    [[nodiscard]] DataType getIndexOutputDataType() const { return index_output_dtype; }
    [[nodiscard]] static bool isInputDataTypeSupported(DataType dtype);
    [[nodiscard]] static bool isOffsetDataTypeSupported(DataType dtype);

    [[nodiscard]] std::shared_ptr<StampedCubSegmentedArgReduction> stamp(
        const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedArgReduction> stamp(
        const Tensor& input,
        const Tensor& preallocated_index_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

    // Variant for graph/runtime-populated offsets; see CubSegmentedReduction::stampRuntimeOffsets.
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedArgReduction> stampRuntimeOffsets(
        const Tensor& input,
        const Tensor& preallocated_index_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedCubSegmentedArgReduction> stampValidated(
        const Tensor& input,
        const Tensor& index_output,
        const Tensor& segment_offsets,
        uint64_t num_segments,
        const Stream& stream) const;

    CubArgReductionOp op;
    DataType index_output_dtype;
};

class StampedCubSegmentedArgReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    /** Benchmarks only (1263): as StampedCubSegmentedReduction::runOnTensorsForBenchmark, with the index output. */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const StridedSpan& replacement_segment_offsets,
                                  const StridedSpan& replacement_index_output,
                                  Stream& run_stream) const;
    /** The physical passes this stamp runs, in order (diagnostic only, 1263). */
    [[nodiscard]] std::vector<std::string> getPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getIndexOutputTensor() const { return index_output; }
    [[nodiscard]] DataType getOffsetDataType() const { return segment_offsets.getDataType(); }
    [[nodiscard]] uint64_t getNumSegments() const { return num_segments; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    /** 1325: true when the stamp runs the retained ragged ARG kernels (1330: always). */
    [[nodiscard]] bool usesRetainedRagged() const { return ragged != nullptr; }

   private:
    friend class CubSegmentedArgReduction;

    StampedCubSegmentedArgReduction(CubArgReductionOp op,
                                    const Tensor& input,
                                    const Tensor& index_output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_segments,
                                    size_t temp_storage_bytes,
                                    const Tensor& temp_storage,
                                    const Stream& stream);

    CubArgReductionOp op;
    const Tensor input;
    mutable Tensor index_output;
    const Tensor segment_offsets;
    uint64_t num_segments;
    size_t temp_storage_bytes;
    Tensor temp_storage;
    // The bound tensors as spans, made once at stamp so that a run converts nothing.
    const StridedSpan input_span{input};
    const StridedSpan offsets_span{segment_offsets};
    const StridedSpan index_output_span{index_output};
    const StridedSpan workspace_span{temp_storage};
    Stream stream;
    // 1325: the retained ragged plan (its workspace is temp_storage); 1330: never null.
    std::shared_ptr<const CubReductionInternal::RaggedRoutedStamp> ragged;
};

/**
 * Describes an index-producing CUB argmin or argmax reduction.
 *
 * The candidate value is converted to FP32 before comparison. The winning index is the local flattened index within
 * the logical reduction domain, with reduced coordinates ordered row-major by the sorted reduction-axis list. NaNs
 * propagate and the lowest local index wins ties, including ties between multiple NaNs.
 */
class CubArgReduction {
   public:
    CubArgReduction(CubArgReductionOp op,
                    uint32_t axis,
                    CubArgReductionOutputOptions outputs = {},
                    std::optional<CubReductionTuningDepth> tuning_depth = std::nullopt);
    CubArgReduction(CubArgReductionOp op,
                    std::vector<uint32_t> axes,
                    CubArgReductionOutputOptions outputs = {},
                    std::optional<CubReductionTuningDepth> tuning_depth = std::nullopt);

    [[nodiscard]] CubArgReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] const CubArgReductionOutputOptions& getOutputOptions() const { return outputs; }
    [[nodiscard]] CubReductionTuningDepth getTuningDepth() const { return tuning_depth; }
    [[nodiscard]] DataType resolveValueOutputDataType(DataType input_dtype) const;
    [[nodiscard]] static float getFp32EmptyReductionValue(CubArgReductionOp op);
    [[nodiscard]] static uint64_t getEmptyReductionIndex() { return std::numeric_limits<uint64_t>::max(); }

    /**
     * Returns the executable path for an ordinary dense ARG reduction. analyzeGeometry() already selects an ordained
     * dense family, including ComposedDense for disjoint reduced runs; this wrapper reasserts that DENSE-GATE-FINAL
     * invariant without changing the path. Index-width limits are the stamp's (a UINT32 index output needs a local
     * domain that fits); 1374 removed the old CUB `int` segment-size limit.
     */
    [[nodiscard]] static CubReductionGeometry analyzeDenseGeometry(
        const std::vector<uint64_t>& input_dimensions, const std::vector<uint32_t>& axes);

    /** Adds one reduced run's local winner coordinate to an already-composed original flattened arg index. */
    [[nodiscard]] static uint64_t composeOriginalArgIndex(uint64_t previous_index,
                                                          uint64_t local_run_index,
                                                          uint64_t domain_stride);

    /**
     * Queries the complete stamped workspace footprint without allocating input, output, intermediate, or temporary
     * tensors: every non-final FP32 value / carried-index intermediate of the planned pass list plus each retained
     * pass's scratch, matching StampedCubArgReduction::getWorkspaceSizeInBytes().
     */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(const StridedSpan& input, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(
        const StridedSpan& input,
        const std::optional<Tensor>& preallocated_value_output,
        const std::optional<Tensor>& preallocated_index_output,
        const Stream& stream) const;

   private:
    // ARG-P1: original FP8 input. planDenseReduction() pass list executed by direct ARG stages.
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampPlannedDenseValidated(
        const StridedSpan& input,
        std::optional<Tensor> value_output,
        std::optional<Tensor> index_output,
        const CubReductionGeometry& geometry,
        const Stream& stream) const;
    CubArgReductionOp op;
    std::vector<uint32_t> axes;
    CubArgReductionOutputOptions outputs;
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular;
};

/**
 * Concrete argmin/argmax reduction with no run-time allocation. Every planned pass's scratch and all value / index
 * intermediates are allocated while stamping and retained by this object for subsequent runs.
 */
class StampedCubArgReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    /**
     * Benchmarks only: runs a planned FP8 ARG stamp (ARG-P1) on another input of the same dtype and element count, so
     * cold-input sweeps can time the stamped production plan without re-stamping. Production binds its input.
     */
    void runOnInputForBenchmark(const StridedSpan& replacement_input, Stream& run_stream) const;
    /**
     * Benchmarks only (1174): runs this planned stamp on another input and other outputs of the stamp's
     * dtypes and element counts. Intermediates and scratch stay the stamp's own. Cold-L2 audits time a production
     * stamp on rotating input and output views without re-stamping. A replacement output must be given exactly when
     * the stamp has that output. Production binds its tensors.
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const std::optional<StridedSpan>& replacement_value_output,
                                  const std::optional<StridedSpan>& replacement_index_output,
                                  Stream& run_stream) const;
    /** Executed retained passes in order (e.g. "fp8_arg_one_window:o8_w1_p16_c1", "f16_arg_rk:..."). Diagnostic
     *  only. */
    [[nodiscard]] std::vector<std::string> getPlannedPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] CubArgReductionOp getOperation() const { return op; }
    [[nodiscard]] CubReductionPath getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getValueAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const CubReductionGeometry& getGeometry() const { return geometry; }
    [[nodiscard]] std::vector<std::vector<uint32_t>> getComposedStageAxes() const;
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    [[nodiscard]] const std::optional<Tensor>& getValueOutputTensor() const { return value_output; }
    [[nodiscard]] const std::optional<Tensor>& getIndexOutputTensor() const { return index_output; }

   private:
    friend class CubArgReduction;
    // Workspace-allocation regression tests; never used by production dispatch.
    friend struct CubReductionTiledWorkspaceProbe;

    StampedCubArgReduction(CubArgReductionOp op,
                           CubReductionGeometry geometry,
                           const StridedSpan& value_input,
                           std::optional<Tensor> carried_index_input,
                           std::optional<Tensor> value_output,
                           std::optional<Tensor> index_output,
                           size_t temp_storage_bytes,
                           const Tensor& temp_storage,
                           uint64_t domain_stride,
                           DataType carried_index_dtype,
                           const Stream& stream);

    StampedCubArgReduction(CubArgReductionOp op,
                           CubReductionGeometry geometry,
                           const StridedSpan& input,
                           std::optional<Tensor> value_output,
                           std::optional<Tensor> index_output,
                           size_t workspace_size_bytes,
                           std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages,
                           const Stream& stream);

    CubArgReductionOp op;
    CubReductionGeometry geometry;
    const StridedSpan input;
    mutable std::optional<Tensor> value_output;
    mutable std::optional<Tensor> index_output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;

    // Present only on the per-pass stages of a planned ARG stamp. The first stage has no carried-index input;
    // later stages consume the previous stage's SoA index tensor. domain_stride maps this stage's local row coordinate
    // directly into the original flattened reduction domain.
    std::optional<Tensor> carried_index_input;
    // The outputs and the carried index as the spans the launches take, made once at stamp (bindLaunchSpans).
    std::optional<StridedSpan> value_output_span;
    std::optional<StridedSpan> index_output_span;
    std::optional<StridedSpan> carried_index_span;
    std::optional<DataType> composed_carried_index_dtype;
    uint64_t composed_domain_stride = 1;
    std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages;
    // ARG-A2: a planned stage that launches a retained FP8 ARG physical kernel instead of a direct ARG reducer.
    std::shared_ptr<const CubReductionInternal::DenseRetainedFp8ArgPhysicalPlan> fp8_arg_plan;
    // 1221: a planned stage that launches a retained FP32 ARG pass (the retained FP32 routing), original or carried.
    std::shared_ptr<const CubReductionInternal::DenseRetainedFp32ArgPhysicalPlan> fp32_arg_plan;

    void launchBoundStage(const StridedSpan& stage_input, Stream& run_stream) const;
    void bindLaunchSpans();
    void launchBoundStage(const StridedSpan& stage_input, const StridedSpan* value_output_ptr,
                          const StridedSpan* index_output_ptr,
                          Stream& run_stream) const;

    Stream stream;
};

/**
 * Concrete, allocation-free-at-run-time CUB reduction operation.
 *
 * A stamped operation is bound to its input/output tensors and geometry. runOn() may use another stream on the same
 * GPU, but the tensor bindings, reduction operation, axes, output dtype, selected CUB path, and workspace size are
 * fixed at stamp time.
 */
class StampedCubReduction {
   public:
    void run();
    void run(float output_scale);
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, float output_scale) const;
    /**
     * Benchmarks only (1174): runs the stamped plan on another input and output with the stamp's dtypes and element
     * counts, at the stamp's output scale. Intermediates and scratch stay the stamp's own. Cold-L2 audits time a
     * production stamp on rotating input and output views without re-stamping. Production binds its tensors.
     * 1335: a stamp of a non-dense view input takes a replacement view of the same dims and strides (the view audit,
     * plans/Thor_View_Reduction_Plan_2026-11-08.md); a dense stamp takes a dense replacement, as before. 1365: a
     * VIEW-DIRECT-1 / -2A stamp's first pass reads the replacement view as dense (StridedSpan::asDense), as it reads its
     * own; its strategy chain starts "view_physical:".
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input, const StridedSpan& replacement_output,
                                  Stream& run_stream) const;

    [[nodiscard]] uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }
    [[nodiscard]] const StridedSpan& getInput() const { return input; }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] CubReductionPath getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const CubReductionGeometry& getGeometry() const { return geometry; }
    /**
     * Returns the current-run reduction axis executed by each physical VALUE pass when the structural geometry is
     * ComposedDense. Ordinary dense VALUE always sources this from DenseExecutablePlan.
     * This is diagnostic planning metadata only; run() never consults it.
     */
    [[nodiscard]] std::vector<std::vector<uint32_t>> getComposedStageAxes() const;
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }

    /** True when ordinary dense execution is represented as a flat ordered physical-pass plan. */
    [[nodiscard]] bool usesDenseExecutablePlan() const { return !dense_executable_plan.passes.empty(); }
    [[nodiscard]] size_t getDenseExecutablePassCount() const { return dense_executable_plan.passes.size(); }
    /** Physical sequence as stamped; never selects, allocates, or retunes. */
    [[nodiscard]] std::vector<DensePhysicalPassDiagnostic> getDensePhysicalPassDiagnostics() const;

    // Backward-compatible census metadata; physical parameters belong to the
    // per-pass diagnostics above, not a misleading first-RK special case.
    [[nodiscard]] bool usesModernRKFamilyPlan() const;
    [[nodiscard]] const std::string& getModernRKStrategyChain() const { return modern_rk_strategy_chain; }
    /** 1366: a rank-1 strided view's StridedR launch (closed form, Block or Split; StridedReduction.h), else nothing.
     *  Its name ("strided_r:...") is the strategy chain. */
    [[nodiscard]] const std::optional<CubReductionInternal::StridedRLaunch>& getStridedRLaunch() const {
        return strided_r_launch;
    }
    /** 1392 (phase 7 C): a pitched view's in-place PacketComplete launch, else nothing. Its name
     *  ("pitched_complete:w<W>_t<tile>") is the strategy chain. */
    [[nodiscard]] const std::optional<CubReductionPitchedCompleteLaunch>& getPitchedCompleteLaunch() const {
        return pitched_complete_launch;
    }

   private:
    friend class CubReduction;
    // Phase-3 GPU smoke verifies that independent stamps never share their
    // mutable workspace. This is test-only friendship, not a public accessor.
    friend struct Fp8RKStampLifecycleProbe;
    // GPU integration smoke: inject a deterministic already-selected flat plan.
    friend struct DenseRetainedFp8RKExecutionProbe;
    // Workspace-allocation regression tests; never used by production dispatch.
    friend struct CubReductionTiledWorkspaceProbe;

    StampedCubReduction(CubReductionOp op,
                        CubReductionGeometry geometry,
                        const StridedSpan& input,
                        const Tensor& output,
                        size_t temp_storage_bytes,
                        const Tensor& temp_storage,
                        bool apply_input_transform,
                        bool apply_output_finalize,
                        uint64_t semantic_total_reduction_size,
                        float output_scale,
                        const Stream& stream);

    StampedCubReduction(CubReductionOp op,
                        CubReductionGeometry geometry,
                        const StridedSpan& input,
                        const Tensor& output,
                        CubReductionInternal::DenseExecutablePlan dense_executable_plan,
                        std::string modern_rk_strategy_chain,
                        bool apply_input_transform,
                        bool apply_output_finalize,
                        uint64_t semantic_total_reduction_size,
                        float output_scale,
                        const Stream& stream);

    CubReductionOp op;
    CubReductionGeometry geometry;
    const StridedSpan input;
    mutable Tensor output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    // output and temp_storage as the spans the direct launch takes, made once at stamp (initialized from them).
    const StridedSpan output_span{output};
    const StridedSpan temp_storage_span{temp_storage};
    CubReductionInternal::DenseExecutablePlan dense_executable_plan;
    std::string modern_rk_strategy_chain;
    // Physical passes need the logical operation's transform/finalizer placement, not merely the public operation.
    // This prevents L1/L2/SumSquares transforms from being re-applied to FP32 shard partials.
    bool apply_input_transform = true;
    bool apply_output_finalize = true;
    uint64_t semantic_total_reduction_size = 1;
    // 1366: case C (a rank-1 strided view) runs StridedR; its Split partials are temp_storage.
    std::optional<CubReductionInternal::StridedRLaunch> strided_r_launch;
    // 1392: a pitched view read in place (one PacketComplete launch; no workspace).
    std::optional<CubReductionPitchedCompleteLaunch> pitched_complete_launch;

    const float output_scale;
    Stream stream;
};

}  // namespace ThorImplementation
