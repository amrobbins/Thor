#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
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
    StridedFixedSegment = 3,
    OffsetSegmented = 4,
};

/**
 * Host-side arbitrary-axis indexing metadata.
 *
 * Rank is intentionally dynamic. Output coordinates are decoded in retained-axis order and reduction coordinates are
 * decoded in reduced-axis order. Both orders are row-major and therefore match the flattened ordering of
 * keep-dimension and squeezed outputs.
 */
struct CubReductionIndexing {
    std::vector<uint32_t> reduced_axes;
    std::vector<uint32_t> retained_axes;
    std::vector<uint64_t> input_strides;
    std::vector<uint64_t> reduced_dimensions;
    std::vector<uint64_t> retained_dimensions;
};

/** Trivially-copyable device view over stamped, rank-sized arbitrary-axis metadata. */
struct CubReductionDeviceIndexing {
    uint32_t reduced_axis_count = 0;
    uint32_t retained_axis_count = 0;
    const uint64_t* reduced_axes = nullptr;
    const uint64_t* retained_axes = nullptr;
    const uint64_t* input_strides = nullptr;
    const uint64_t* reduced_dimensions = nullptr;
    const uint64_t* retained_dimensions = nullptr;
};

static_assert(std::is_trivially_copyable_v<CubReductionDeviceIndexing>);

enum class CubReductionTiledRetainedOutputOrder : uint8_t {
    NaturalOuterInner = 0,
    PermutedInnerOuter = 1,
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
    CubReductionIndexing indexing;
    CubReductionDeviceIndexing device_indexing;

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
    std::optional<CubReductionPermutationAwareTiledGeometry> permutation_aware_tiled_geometry = std::nullopt;

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

    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               uint32_t axis);
    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint32_t>& axes);
    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint64_t>& input_strides,
                                                               const std::vector<uint32_t>& axes);

    /** Maps one logical (output, reduction) coordinate pair to the source tensor's physical element offset. */
    [[nodiscard]] static uint64_t mapLogicalReductionIndexToPhysicalIndex(const CubReductionGeometry& geometry,
                                                                          uint64_t output_index,
                                                                          uint64_t reduction_index);

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
 * Mean divides by the segment row count and returns zero for an empty segment. Offset contents are validated during
 * stamping; callers that update a stamped offsets tensor must preserve the same zero-based, nondecreasing, in-bounds
 * row-partition contract.
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

    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(const Tensor& input, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stamp(
        const Tensor& input,
        const std::optional<Tensor>& preallocated_value_output,
        const std::optional<Tensor>& preallocated_index_output,
        const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedCubArgReduction> stampValidated(
        const Tensor& input,
        std::optional<Tensor> value_output,
        std::optional<Tensor> index_output,
        const CubReductionGeometry& geometry,
        const Stream& stream) const;

    CubArgReductionOp op;
    std::vector<uint32_t> axes;
    CubArgReductionOutputOptions outputs;
};

/** Concrete, allocation-free-at-run-time argmin/argmax reduction. */
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
                           std::optional<Tensor> indexing_metadata,
                           const Stream& stream);

    CubArgReductionOp op;
    CubReductionGeometry geometry;
    const Tensor input;
    mutable std::optional<Tensor> value_output;
    mutable std::optional<Tensor> index_output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    std::optional<Tensor> indexing_metadata;
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
                        std::optional<Tensor> indexing_metadata,
                        float output_scale,
                        const Stream& stream);

    CubReductionOp op;
    CubReductionGeometry geometry;
    const Tensor input;
    mutable Tensor output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    std::optional<Tensor> indexing_metadata;
    const float output_scale;
    Stream stream;
};

}  // namespace ThorImplementation
