#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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

// Host/device-independent launch-policy constants shared by the tuned TiledFixedSegment implementation, the dense
// composition cost planner, and reduction benchmarks. Keeping these values in the common reduction header prevents
// the planner from silently estimating a different full-row ownership model than the kernels actually launch.
namespace CubReductionTiledPolicy {
inline constexpr uint64_t WARP_THREADS = 32;
inline constexpr uint64_t WARPS_PER_BLOCK = 8;
inline constexpr uint64_t TARGET_ACTIVE_WARPS = 1024;
inline constexpr uint64_t FULL_ROW_MAX_COMPONENTS_PER_LANE = 16;
inline constexpr uint64_t FULL_ROW_COMPONENTS_PER_WARP = WARP_THREADS * FULL_ROW_MAX_COMPONENTS_PER_LANE;
inline constexpr uint64_t FULL_ROW_MAX_WARPS_PER_OUTPUT = WARPS_PER_BLOCK;
inline constexpr uint64_t FULL_ROW_GROUP_MAX_INNER_SIZE =
    FULL_ROW_COMPONENTS_PER_WARP * FULL_ROW_MAX_WARPS_PER_OUTPUT;
inline constexpr uint64_t FULL_ROW_COMPONENTS_PER_BLOCK = FULL_ROW_GROUP_MAX_INNER_SIZE;
inline constexpr uint64_t ASYNC_STAGE_BYTES_PER_WARP = 2048;

// ROW-SPLIT-FULL-ROW: the direct/vectorized full-row kernels deliberately derive their parallelism from independent
// output rows.  That is ideal for ordinary dense reductions, but release benchmarking on SM120 shows a distinct
// regime where there are only a few output rows and an enormous reduction depth: one/few CTAs otherwise serialize a
// very large memory stream while most of the GPU is idle.  In that regime production splits each output's reduction
// rows across enough independent CTAs for roughly two SM waves, writes FP32 partial vectors, then performs one tiny
// deterministic final reduction.  Keep the gate intentionally conservative: every measured FP16/BF16/FP32 case with
// reduction >= 1024 and outer <= 128 won decisively, while the existing kernels remain ordained outside this regime.
inline constexpr uint64_t ROW_SPLIT_MIN_REDUCTION_SIZE = 1024;
inline constexpr uint64_t ROW_SPLIT_MAX_OUTER_SIZE = 128;
inline constexpr uint64_t ROW_SPLIT_MIN_INNER_SIZE = 15;
inline constexpr uint64_t ROW_SPLIT_MAX_INNER_SIZE = FULL_ROW_GROUP_MAX_INNER_SIZE;
inline constexpr uint64_t ROW_SPLIT_TARGET_SM_WAVES = 2;
}  // namespace CubReductionTiledPolicy

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

/** Execution position of one direct stage inside a dense reduction composition. */
enum class CubReductionDenseCompositionStageRole : uint8_t {
    Complete = 0,
    First = 1,
    Intermediate = 2,
    Final = 3,
};

/**
 * Host-side description of one direct stage selected by the dense composition planner.
 *
 * reduced_run_ordinal is the ordinal among the original tensor's non-singleton reduced runs. dense_run_index is the
 * corresponding index in CubReductionDenseRunGeometry::runs when such a physical run exists. It is nullopt only for
 * the degenerate all-reduced-axes-singleton case, where one extent-1 direct stage is retained to perform conversion
 * and finalization. reduction_extent and domain_stride always describe the original reduction domain, while
 * reduction_axes describe the contiguous logical interval executed by this stage; retained singleton axes may be
 * absorbed into that interval because they have no physical coordinate.
 */
struct CubReductionDenseCompositionStage {
    uint32_t reduced_run_ordinal = 0;
    std::optional<uint32_t> dense_run_index = std::nullopt;
    uint64_t reduction_extent = 1;
    uint64_t domain_stride = 1;
    CubReductionDenseCompositionStageRole role = CubReductionDenseCompositionStageRole::Complete;

    std::vector<uint32_t> reduction_axes;
    std::vector<uint64_t> input_dimensions;
    std::vector<uint64_t> output_dimensions;

    CubReductionPath expected_path = CubReductionPath::TiledFixedSegment;
    uint64_t input_elements = 1;
    uint64_t output_elements = 1;
    uint64_t stage_reduction_size = 1;
    uint64_t outer_size = 1;
    uint64_t inner_size = 1;
};

/** Ordered direct-stage topology selected at stamp/build time for a dense reduction. */
struct CubReductionDenseCompositionPlan {
    std::vector<CubReductionDenseCompositionStage> stages;
};

/**
 * Stamp-time ARG dense-composition metadata consumed by the pair-carrying executor. carried_index_dtype is chosen
 * from the complete original flattened reduction domain, never from an individual direct stage, so later composed
 * stages cannot silently overflow a locally narrow index representation.
 */
struct CubArgReductionDenseCompositionPlan {
    CubReductionDenseCompositionPlan topology;
    DataType carried_index_dtype = DataType::UINT32;
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
 * When this metadata is present, the value-reduction planner can execute the view through the normal optimized
 * TiledFixedSegment family without materializing the logical permutation. The tuned kernels consume the source pointer
 * as dense physical [outer, reduction, inner] storage and use the retained-output write strides below to produce the
 * requested logical dense output order directly.
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
 * host and the kernel performs no logical-coordinate reconstruction. The dedicated 32x33 shared-memory strategy
 * keeps payload reads coalesced, stages finalized values in [A,payload] tiles, and emits logical [B,A,payload]
 * directly without a global transpose intermediate.
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

    // DeviceTransformReduce normally consumes a dense pointer directly.  A rank-1
    // non-dense view (for example, an extracted matrix diagonal) is still an
    // affine sequence and can use the same single-output CUB reduction without
    // materializing or paying the fully general logical-axis index mapping.
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
    // [B,A,payload]. This is intentionally a separate Thor-owned 32x33 shared-memory strategy rather than a
    // general retained-axis permutation engine.
    std::optional<CubReductionPayloadTransposeTiledGeometry> payload_transpose_tiled_geometry = std::nullopt;

    // VIEW-PITCHED-TILED is a separate Thor-owned input strategy under the TiledFixedSegment execution family.
    // It never changes or reuses the ordained dense tiled kernels: launch dispatch selects a dedicated pitched
    // kernel whenever this metadata is present. Output order is always the natural logical [outer,inner] order.
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

class StampedCubReduction;
class StampedCubArgReduction;
class StampedCubSegmentedArgReduction;
class StampedCubSegmentedReduction;

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
    CubReduction(CubReductionOp op,
                 uint32_t axis,
                 std::optional<DataType> output_dtype = std::nullopt,
                 float output_scale = 1.0f);
    CubReduction(CubReductionOp op,
                 std::vector<uint32_t> axes,
                 std::optional<DataType> output_dtype = std::nullopt,
                 float output_scale = 1.0f);

    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }
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

    /**
     * Returns the executable geometry for a value reduction.
     *
     * analyzeGeometry() already selects the final operation-independent execution family, including ComposedDense for
     * ordinary dense disjoint reductions. This wrapper validates the value operation and fixed-segment limits so
     * callers that cache a value-reduction plan observe the same path that stamp() will execute. Unsupported views are
     * rejected directly by analyzeGeometry(); there is no arbitrary logical-index execution family after DELETE.
     */
    [[nodiscard]] static CubReductionGeometry analyzeValueGeometry(CubReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint32_t>& axes);
    [[nodiscard]] static CubReductionGeometry analyzeValueGeometry(CubReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint64_t>& input_strides,
                                                                    const std::vector<uint32_t>& axes);

    /**
     * Returns the operation-independent dense direct-stage topology for the supplied ordinary dense reduction.
     * This is host-only planning/introspection. It performs no allocation, synchronization, or runtime autotuning.
     */
    [[nodiscard]] static std::optional<CubReductionDenseCompositionPlan> analyzeDenseCompositionPlan(
        const std::vector<uint64_t>& input_dimensions, const std::vector<uint32_t>& axes);

    /** Queries backend temporary storage without allocating input, output, or workspace tensors. */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const Tensor& input, const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const Tensor& input,
                                                             const Tensor& preallocated_output,
                                                             const Stream& stream) const;

   private:
    CubReductionOp op;
    std::vector<uint32_t> axes;
    [[nodiscard]] std::shared_ptr<StampedCubReduction> stampValidated(const Tensor& input,
                                                                      const Tensor& output,
                                                                      const CubReductionGeometry& geometry,
                                                                      const Stream& stream) const;

    std::optional<DataType> output_dtype;
    float output_scale;
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
    Stream stream;
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

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getIndexOutputTensor() const { return index_output; }
    [[nodiscard]] DataType getOffsetDataType() const { return segment_offsets.getDataType(); }
    [[nodiscard]] uint64_t getNumSegments() const { return num_segments; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }

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
    Stream stream;
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
                    CubArgReductionOutputOptions outputs = {});
    CubArgReduction(CubArgReductionOp op,
                    std::vector<uint32_t> axes,
                    CubArgReductionOutputOptions outputs = {});

    [[nodiscard]] CubArgReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] const CubArgReductionOutputOptions& getOutputOptions() const { return outputs; }
    [[nodiscard]] DataType resolveValueOutputDataType(DataType input_dtype) const;
    [[nodiscard]] static float getFp32EmptyReductionValue(CubArgReductionOp op);
    [[nodiscard]] static uint64_t getEmptyReductionIndex() { return std::numeric_limits<uint64_t>::max(); }

    /**
     * Returns the executable path for an ordinary dense ARG reduction. analyzeGeometry() already selects an ordained
     * dense family, including ComposedDense for disjoint reduced runs; this wrapper reasserts that DENSE-GATE-FINAL
     * invariant and performs ARG-side executable-limit validation without changing the path.
     */
    [[nodiscard]] static CubReductionGeometry analyzeDenseGeometry(
        const std::vector<uint64_t>& input_dimensions, const std::vector<uint32_t>& axes);

    /**
     * Builds the host-only structural ARG dense-composition plan. This overload intentionally carries no GPU cost
     * calibration and remains useful for topology/indexing tests. The carried index width is selected from the complete
     * original reduction domain.
     */
    [[nodiscard]] static std::optional<CubArgReductionDenseCompositionPlan> analyzeDenseCompositionPlan(
        const std::vector<uint64_t>& input_dimensions, const std::vector<uint32_t>& axes);

    /**
     * Builds the production ARG dense-composition plan with the current coarse measured stage-cost model. Planning is
     * stamp-time only: this performs no allocation, synchronization, autotuning, or trial execution.
     */
    [[nodiscard]] static std::optional<CubArgReductionDenseCompositionPlan> analyzeDenseCompositionPlanForExecution(
        const std::vector<uint64_t>& input_dimensions,
        const std::vector<uint32_t>& axes,
        DataType input_dtype,
        const CubArgReductionOutputOptions& outputs,
        const Stream& stream);

    /**
     * Builds one explicit legal left/right end-elimination order for benchmark/test calibration. run_order contains
     * reduced-run ordinals in execution order and is rejected unless every step removes a current interval end.
     */
    [[nodiscard]] static std::optional<CubArgReductionDenseCompositionPlan> analyzeDenseCompositionPlanForRunOrder(
        const std::vector<uint64_t>& input_dimensions,
        const std::vector<uint32_t>& axes,
        const std::vector<uint32_t>& run_order);

    /** Adds one reduced run's local winner coordinate to an already-composed original flattened arg index. */
    [[nodiscard]] static uint64_t composeOriginalArgIndex(uint64_t previous_index,
                                                          uint64_t local_run_index,
                                                          uint64_t domain_stride);

    /**
     * Queries the complete stamped workspace footprint without allocating input, output, intermediate, or temporary
     * tensors. For ComposedDense this includes every non-final FP32 value/carried-index intermediate plus each direct
     * stage's backend temporary storage, matching StampedCubArgReduction::getWorkspaceSizeInBytes().
     */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(const Tensor& input, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(
        const Tensor& input,
        const std::optional<Tensor>& preallocated_value_output,
        const std::optional<Tensor>& preallocated_index_output,
        const Stream& stream) const;

    /**
     * Explicitly stamps the dense pair-composition executor. After ARG-PLAN-3 this remains a useful forced-composition
     * test/benchmark hook, while normal production stamp() selects ComposedDense automatically for dense disjoint runs.
     */
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampComposedDense(const Tensor& input,
                                                                             const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampComposedDense(
        const Tensor& input,
        const std::optional<Tensor>& preallocated_value_output,
        const std::optional<Tensor>& preallocated_index_output,
        const Stream& stream) const;

    /** Benchmark/test-only execution hook for an already validated explicit composition plan. */
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampComposedDenseWithPlan(
        const Tensor& input, const CubArgReductionDenseCompositionPlan& plan, const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampValidated(
        const Tensor& input,
        std::optional<Tensor> value_output,
        std::optional<Tensor> index_output,
        const CubReductionGeometry& geometry,
        const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampComposedDenseValidated(
        const Tensor& input,
        std::optional<Tensor> value_output,
        std::optional<Tensor> index_output,
        const CubReductionGeometry& geometry,
        const CubArgReductionDenseCompositionPlan& plan,
        const Stream& stream) const;

    CubArgReductionOp op;
    std::vector<uint32_t> axes;
    CubArgReductionOutputOptions outputs;
};

/**
 * Concrete argmin/argmax reduction with no run-time allocation. Direct temporary storage and all ComposedDense
 * value/index intermediates are allocated while stamping and retained by this object for subsequent runs.
 */
class StampedCubArgReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;

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

    StampedCubArgReduction(CubArgReductionOp op,
                           CubReductionGeometry geometry,
                           const Tensor& input,
                           std::optional<Tensor> value_output,
                           std::optional<Tensor> index_output,
                           size_t temp_storage_bytes,
                           const Tensor& temp_storage,
                           const Stream& stream);

    StampedCubArgReduction(CubArgReductionOp op,
                           CubReductionGeometry geometry,
                           const Tensor& value_input,
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
                           const Tensor& input,
                           std::optional<Tensor> value_output,
                           std::optional<Tensor> index_output,
                           size_t workspace_size_bytes,
                           std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages,
                           const Stream& stream);

    CubArgReductionOp op;
    CubReductionGeometry geometry;
    const Tensor input;
    mutable std::optional<Tensor> value_output;
    mutable std::optional<Tensor> index_output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;

    // Present only on direct stages owned by a ComposedDense ARG executor. The first stage has no carried-index input;
    // later stages consume the previous stage's SoA index tensor. domain_stride maps this stage's local row coordinate
    // directly into the original flattened reduction domain.
    std::optional<Tensor> carried_index_input;
    std::optional<DataType> composed_carried_index_dtype;
    uint64_t composed_domain_stride = 1;
    std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages;

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

    [[nodiscard]] uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getInputTensor() const { return input; }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] CubReductionPath getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const CubReductionGeometry& getGeometry() const { return geometry; }
    /**
     * Returns the public reduction axes executed by each direct pass of a ComposedDense reduction, in execution order.
     * Non-composed reductions return an empty vector. This is planning metadata only; run() never consults it.
     */
    [[nodiscard]] std::vector<std::vector<uint32_t>> getComposedStageAxes() const;
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }

   private:
    friend class CubReduction;

    StampedCubReduction(CubReductionOp op,
                        CubReductionGeometry geometry,
                        const Tensor& input,
                        const Tensor& output,
                        size_t temp_storage_bytes,
                        const Tensor& temp_storage,
                        float output_scale,
                        const Stream& stream);

    StampedCubReduction(CubReductionOp op,
                        CubReductionGeometry geometry,
                        const Tensor& input,
                        const Tensor& output,
                        size_t workspace_size_bytes,
                        std::vector<std::shared_ptr<StampedCubReduction>> composed_stages,
                        float output_scale,
                        const Stream& stream);

    CubReductionOp op;
    CubReductionGeometry geometry;
    const Tensor input;
    mutable Tensor output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    std::vector<std::shared_ptr<StampedCubReduction>> composed_stages;
    const float output_scale;
    Stream stream;
};

}  // namespace ThorImplementation
