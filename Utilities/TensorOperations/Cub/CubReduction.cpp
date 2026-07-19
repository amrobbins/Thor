#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {
namespace {

using namespace CubDevicePrimitiveSupport;

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
#endif
            return true;
        default:
            return false;
    }
}

void requireSupportedFloatingStorageDType(DataType dtype, const char* role) {
    if (!isSupportedFloatingStorageDType(dtype)) {
        throw std::invalid_argument(std::string("CUB tensor reduction does not support ") + role + " dtype " + dtypeName(dtype) +
                                    ". Supported storage dtypes follow Thor's THOR_CUB_ENABLE_FP8_TYPES and "
                                    "THOR_CUB_ENABLE_64BIT_TYPES build policy; TF32 is not a storage dtype.");
    }
}

void requireCompatibleStream(const Tensor& input, const Stream& stream) {
    if (!stream.isInitialized()) {
        throw std::invalid_argument("CUB tensor reduction requires an initialized stream.");
    }
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("CUB tensor reduction stream must belong to the input tensor's GPU.");
    }
}

void requireExpectedOutput(const Tensor& input,
                           const Tensor& output,
                           DataType output_dtype,
                           const CubReductionGeometry& geometry) {
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("CUB tensor reduction preallocated output dtype does not match the configured output dtype.");
    }
    const std::vector<uint64_t>& output_dimensions = output.getDimensions();
    if (output_dimensions != geometry.output_dimensions && output_dimensions != geometry.squeezed_output_dimensions) {
        throw std::invalid_argument(
            "CUB tensor reduction preallocated output dimensions must match either the keep-dimension or squeezed reduction shape.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument("CUB tensor reduction preallocated output element count does not match the reduction geometry.");
    }

    const uintptr_t input_begin = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    const uintptr_t output_begin = reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    const uintptr_t input_end = input_begin + input.getArraySizeInBytes();
    const uintptr_t output_end = output_begin + output.getArraySizeInBytes();
    if (input_begin < output_end && output_begin < input_end) {
        throw std::invalid_argument("CUB tensor reduction input and output storage must not overlap.");
    }
}

void requireSupportedOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Min:
        case CubReductionOp::Max:
        case CubReductionOp::Product:
        case CubReductionOp::Mean:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
            return;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

size_t queryReductionBytes(CubReductionOp op,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           const Stream& stream) {
    switch (op) {
        case CubReductionOp::Sum:
            return CubReductionInternal::querySumReductionBytes(input, output, geometry, stream);
        case CubReductionOp::Product:
            return CubReductionInternal::queryProductReductionBytes(input, output, geometry, stream);
        case CubReductionOp::Mean:
            return CubReductionInternal::queryMeanReductionBytes(input, output, geometry, stream);
        case CubReductionOp::L1Norm:
            return CubReductionInternal::queryL1NormReductionBytes(input, output, geometry, stream);
        case CubReductionOp::L2Norm:
            return CubReductionInternal::queryL2NormReductionBytes(input, output, geometry, stream);
        case CubReductionOp::Min:
            return CubReductionInternal::queryMinReductionBytes(input, output, geometry, stream);
        case CubReductionOp::Max:
            return CubReductionInternal::queryMaxReductionBytes(input, output, geometry, stream);
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

void launchReduction(CubReductionOp op,
                     const Tensor& temp_storage,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     const CubReductionGeometry& geometry,
                     Stream& stream) {
    switch (op) {
        case CubReductionOp::Sum:
            CubReductionInternal::launchSumReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::Product:
            CubReductionInternal::launchProductReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::Mean:
            CubReductionInternal::launchMeanReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::L1Norm:
            CubReductionInternal::launchL1NormReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::L2Norm:
            CubReductionInternal::launchL2NormReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::Min:
            CubReductionInternal::launchMinReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
        case CubReductionOp::Max:
            CubReductionInternal::launchMaxReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, stream);
            return;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

}  // namespace

CubReduction::CubReduction(CubReductionOp op, uint32_t axis, std::optional<DataType> output_dtype)
    : CubReduction(op, std::vector<uint32_t>{axis}, output_dtype) {}

CubReduction::CubReduction(CubReductionOp op,
                           std::vector<uint32_t> axes,
                           std::optional<DataType> output_dtype)
    : op(op), axes(std::move(axes)), output_dtype(output_dtype) {
    requireSupportedOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("CUB tensor reduction axes must be unique and strictly increasing.");
        }
    }
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "output");
    }
}

DataType CubReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "output");
    return resolved;
}

float CubReduction::getFp32EmptyReductionValue(CubReductionOp op) {
    requireSupportedOperation(op);
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
            return 0.0f;
        case CubReductionOp::Min:
            return std::numeric_limits<float>::infinity();
        case CubReductionOp::Max:
            return -std::numeric_limits<float>::infinity();
        case CubReductionOp::Product:
            return 1.0f;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions, uint32_t axis) {
    return analyzeGeometry(input_dimensions, std::vector<uint32_t>{axis});
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint32_t>& axes) {
    if (input_dimensions.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one input dimension.");
    }
    if (input_dimensions.size() > CUB_REDUCTION_MAX_RANK) {
        throw std::invalid_argument("CUB tensor reduction currently supports rank <= 8.");
    }
    if (axes.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one reduction axis.");
    }
    if (axes.size() > input_dimensions.size()) {
        throw std::invalid_argument("CUB tensor reduction has more reduction axes than input dimensions.");
    }
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= input_dimensions.size()) {
            throw std::invalid_argument("CUB tensor reduction axis is outside the input rank.");
        }
        if (i > 0 && axes[i] <= axes[i - 1]) {
            throw std::invalid_argument("CUB tensor reduction axes must be unique and strictly increasing.");
        }
    }

    auto checkedMultiply = [](uint64_t a, uint64_t b, const char* quantity) {
        if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
            throw std::invalid_argument(std::string("CUB tensor reduction ") + quantity + " overflows uint64_t.");
        }
        return a * b;
    };

    for (uint64_t dimension : input_dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument("CUB tensor reduction does not support zero-sized dense tensor dimensions.");
        }
    }

    CubReductionGeometry geometry;
    geometry.axes = axes;
    geometry.rank = static_cast<uint32_t>(input_dimensions.size());
    geometry.axis = axes.front();
    geometry.input_elements = 1;
    geometry.reduction_size = 1;
    geometry.output_elements = 1;
    geometry.output_dimensions = input_dimensions;
    geometry.squeezed_output_dimensions.reserve(input_dimensions.size() - axes.size());
    geometry.indexing.reduced_axis_count = static_cast<uint32_t>(axes.size());

    uint64_t running_stride = 1;
    for (int32_t dimension = static_cast<int32_t>(input_dimensions.size()) - 1; dimension >= 0; --dimension) {
        geometry.indexing.input_strides[dimension] = running_stride;
        running_stride = checkedMultiply(running_stride, input_dimensions[dimension], "input element count");
    }
    geometry.input_elements = running_stride;

    size_t reduced_cursor = 0;
    uint32_t retained_cursor = 0;
    for (uint32_t dimension = 0; dimension < input_dimensions.size(); ++dimension) {
        const bool reduced = reduced_cursor < axes.size() && axes[reduced_cursor] == dimension;
        if (reduced) {
            geometry.output_dimensions[dimension] = 1;
            geometry.reduction_size = checkedMultiply(
                geometry.reduction_size, input_dimensions[dimension], "reduction element count");
            geometry.indexing.reduced_axes[reduced_cursor] = dimension;
            geometry.indexing.reduced_dimensions[reduced_cursor] = input_dimensions[dimension];
            ++reduced_cursor;
        } else {
            geometry.output_elements = checkedMultiply(
                geometry.output_elements, input_dimensions[dimension], "output element count");
            geometry.squeezed_output_dimensions.push_back(input_dimensions[dimension]);
            geometry.indexing.retained_axes[retained_cursor] = dimension;
            geometry.indexing.retained_dimensions[retained_cursor] = input_dimensions[dimension];
            ++retained_cursor;
        }
    }
    geometry.indexing.retained_axis_count = retained_cursor;
    if (geometry.squeezed_output_dimensions.empty()) {
        geometry.squeezed_output_dimensions.push_back(1);
    }

    if (geometry.input_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction input element count exceeds CUB's int64 item-count limit.");
    }
    if (geometry.output_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction output element count exceeds CUB's int64 segment-count limit.");
    }

    const bool reduces_to_single_output = geometry.output_elements == 1;
    bool reduces_contiguous_suffix = !reduces_to_single_output;
    if (reduces_contiguous_suffix) {
        const uint32_t suffix_begin = static_cast<uint32_t>(input_dimensions.size() - axes.size());
        for (uint32_t i = 0; i < axes.size(); ++i) {
            if (axes[i] != suffix_begin + i) {
                reduces_contiguous_suffix = false;
                break;
            }
        }
    }

    if (reduces_to_single_output) {
        geometry.path = CubReductionPath::DeviceTransformReduce;
    } else if (reduces_contiguous_suffix) {
        geometry.path = CubReductionPath::ContiguousFixedSegment;
    } else {
        geometry.path = CubReductionPath::StridedFixedSegment;
    }

    if (geometry.path != CubReductionPath::DeviceTransformReduce
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB tensor reduction domain exceeds CUB's fixed segment-size int limit.");
    }

    if (axes.size() == 1) {
        geometry.outer_size = 1;
        geometry.inner_size = 1;
        for (uint32_t dimension = 0; dimension < axes.front(); ++dimension) {
            geometry.outer_size = checkedMultiply(geometry.outer_size, input_dimensions[dimension], "outer size");
        }
        for (uint32_t dimension = axes.front() + 1; dimension < input_dimensions.size(); ++dimension) {
            geometry.inner_size = checkedMultiply(geometry.inner_size, input_dimensions[dimension], "inner size");
        }
    }

    return geometry;
}

uint64_t CubReduction::mapLogicalReductionIndexToPhysicalIndex(const CubReductionGeometry& geometry,
                                                               uint64_t output_index,
                                                               uint64_t reduction_index) {
    if (output_index >= geometry.output_elements) {
        throw std::out_of_range("CUB tensor reduction output index is outside the geometry.");
    }
    if (reduction_index >= geometry.reduction_size) {
        throw std::out_of_range("CUB tensor reduction reduction index is outside the geometry.");
    }
    return CubReductionInternal::mapLogicalReductionIndex(geometry.indexing, output_index, reduction_index);
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry = analyzeGeometry(input.getDimensions(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    Tensor output(input.getPlacement(), TensorDescriptor(resolved_output_dtype, geometry.output_dimensions));
    return stampValidated(input, output, geometry, stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stampValidated(const Tensor& input,
                                                                  const Tensor& output,
                                                                  const CubReductionGeometry& geometry,
                                                                  const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    ScopedGpu scoped_gpu(stream.getGpuNum());
    Tensor mutable_output = output;
    const size_t temp_storage_bytes = queryReductionBytes(op, input, mutable_output, geometry, stream);
    Tensor temp_storage(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

    return std::shared_ptr<StampedCubReduction>(
        new StampedCubReduction(op, geometry, input, output, temp_storage_bytes, temp_storage, stream));
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input,
                                                         const Tensor& preallocated_output,
                                                         const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry = analyzeGeometry(input.getDimensions(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    requireExpectedOutput(input, preallocated_output, resolved_output_dtype, geometry);
    return stampValidated(input, preallocated_output, geometry, stream);
}

StampedCubReduction::StampedCubReduction(CubReductionOp op,
                                         CubReductionGeometry geometry,
                                         const Tensor& input,
                                         const Tensor& output,
                                         size_t temp_storage_bytes,
                                         const Tensor& temp_storage,
                                         const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      output(output),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubReduction::run() { runOn(stream); }

void StampedCubReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    launchReduction(op, temp_storage, temp_storage_bytes, input, output, geometry, run_stream);
}

}  // namespace ThorImplementation
