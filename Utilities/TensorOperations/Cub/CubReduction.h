#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
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
};

enum class CubReductionPath : uint8_t {
    DeviceTransformReduce = 0,
    ContiguousFixedSegment = 1,
    StridedFixedSegment = 2,
};

inline constexpr uint32_t CUB_REDUCTION_MAX_RANK = 8;

/**
 * Fixed-size, trivially-copyable metadata used by the arbitrary-axis logical input iterator.
 *
 * Output coordinates are decoded in retained-axis order and reduction coordinates are decoded in reduced-axis order.
 * Both orders are row-major and therefore match the flattened ordering of keep-dimension and squeezed outputs.
 */
struct CubReductionIndexing {
    uint32_t reduced_axis_count = 0;
    uint32_t retained_axis_count = 0;
    uint32_t reduced_axes[CUB_REDUCTION_MAX_RANK] = {};
    uint32_t retained_axes[CUB_REDUCTION_MAX_RANK] = {};
    uint64_t input_strides[CUB_REDUCTION_MAX_RANK] = {};
    uint64_t reduced_dimensions[CUB_REDUCTION_MAX_RANK] = {};
    uint64_t retained_dimensions[CUB_REDUCTION_MAX_RANK] = {};
};

struct CubReductionGeometry {
    std::vector<uint32_t> axes;
    uint32_t rank = 0;

    // Retained for source compatibility and diagnostics for the single-axis case. They are zero for multi-axis plans.
    uint32_t axis = 0;
    uint64_t outer_size = 0;
    uint64_t inner_size = 0;

    uint64_t input_elements = 0;
    uint64_t reduction_size = 0;
    uint64_t output_elements = 0;
    std::vector<uint64_t> output_dimensions;
    std::vector<uint64_t> squeezed_output_dimensions;
    CubReductionIndexing indexing;
    CubReductionPath path = CubReductionPath::DeviceTransformReduce;
};

class StampedCubReduction;

/**
 * Describes a one-or-more-axis CUB reduction.
 *
 * CubReduction is intentionally not executable. stamp() validates the concrete tensors, selects the CUB path,
 * queries the required temporary storage, and allocates both the output and workspace. StampedCubReduction::run()
 * subsequently performs no allocation or CUB planning.
 *
 * Reduction axes must be non-empty, unique, and strictly increasing. Every path converts input values to FP32 before
 * applying the operation-specific input transform and reduction operator. Accumulation therefore always occurs in
 * FP32. Operation-specific output finalization, such as mean division or the final square root for L2 norm, also
 * occurs in FP32 before the configured storage conversion. The output storage dtype defaults to the input storage
 * dtype and may be overridden explicitly.
 */
class CubReduction {
   public:
    CubReduction(CubReductionOp op, uint32_t axis, std::optional<DataType> output_dtype = std::nullopt);
    CubReduction(CubReductionOp op,
                 std::vector<uint32_t> axes,
                 std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
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

    /** Maps one logical (output, reduction) coordinate pair to the source tensor's row-major physical index. */
    [[nodiscard]] static uint64_t mapLogicalReductionIndexToPhysicalIndex(const CubReductionGeometry& geometry,
                                                                          uint64_t output_index,
                                                                          uint64_t reduction_index);

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
    void runOn(Stream& run_stream) const;

    [[nodiscard]] uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] CubReductionPath getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const CubReductionGeometry& getGeometry() const { return geometry; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }

   private:
    friend class CubReduction;

    StampedCubReduction(CubReductionOp op,
                        CubReductionGeometry geometry,
                        const Tensor& input,
                        const Tensor& output,
                        size_t temp_storage_bytes,
                        const Tensor& temp_storage,
                        const Stream& stream);

    CubReductionOp op;
    CubReductionGeometry geometry;
    const Tensor input;
    mutable Tensor output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    Stream stream;
};

}  // namespace ThorImplementation
