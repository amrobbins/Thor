#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ThorImplementation {

enum class CubReductionOp : uint8_t { Sum = 0, Min = 1, Max = 2 };

enum class CubReductionPath : uint8_t {
    DeviceTransformReduce = 0,
    ContiguousFixedSegment = 1,
    StridedFixedSegment = 2,
};

struct CubReductionGeometry {
    uint32_t axis = 0;
    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;
    uint64_t output_elements = 0;
    std::vector<uint64_t> output_dimensions;
    CubReductionPath path = CubReductionPath::DeviceTransformReduce;
};

class StampedCubReduction;

/**
 * Describes a single-axis CUB reduction.
 *
 * CubReduction is intentionally not executable. stamp() validates the concrete tensors, selects the CUB path,
 * queries the required temporary storage, and allocates both the output and workspace. StampedCubReduction::run()
 * subsequently performs no allocation or CUB planning.
 *
 * Every path converts input values to FP32 before applying the reduction operator and therefore always accumulates
 * in FP32. The output storage dtype defaults to the input storage dtype and may be overridden explicitly.
 */
class CubReduction {
   public:
    CubReduction(CubReductionOp op, uint32_t axis, std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] CubReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axis; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] DataType resolveOutputDataType(DataType input_dtype) const;

    [[nodiscard]] static CubReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions, uint32_t axis);

    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const Tensor& input, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedCubReduction> stamp(const Tensor& input,
                                                             const Tensor& preallocated_output,
                                                             const Stream& stream) const;

   private:
    CubReductionOp op;
    uint32_t axis;
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
 * GPU, but the tensor bindings, reduction operation, axis, output dtype, selected CUB path, and workspace size are
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
