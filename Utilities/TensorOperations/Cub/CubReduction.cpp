#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

[[nodiscard]] bool isSupportedArgIndexDType(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

void requireSupportedArgIndexDType(DataType dtype) {
    if (!isSupportedArgIndexDType(dtype)) {
        throw std::invalid_argument("CUB arg reduction index output dtype must be UINT32 or UINT64.");
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


[[nodiscard]] std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> stripped;
    stripped.reserve(dimensions.size());
    for (uint64_t dimension : dimensions) {
        if (dimension != 1) {
            stripped.push_back(dimension);
        }
    }
    if (stripped.empty()) {
        stripped.push_back(1);
    }
    return stripped;
}

std::optional<Tensor> stampDeviceIndexingMetadata(CubReductionGeometry& geometry,
                                                      const TensorPlacement& placement,
                                                      const Stream& stream) {
    if (geometry.path != CubReductionPath::StridedFixedSegment) {
        geometry.device_indexing = {};
        return std::nullopt;
    }

    const CubReductionIndexing& indexing = geometry.indexing;
    const size_t rank = indexing.input_strides.size();
    const size_t reduced_count = indexing.reduced_axes.size();
    const size_t retained_count = indexing.retained_axes.size();
    if (reduced_count + retained_count != rank
        || indexing.reduced_dimensions.size() != reduced_count
        || indexing.retained_dimensions.size() != retained_count) {
        throw std::logic_error("CUB tensor reduction indexing metadata is internally inconsistent.");
    }

    std::vector<uint64_t> packed;
    packed.reserve(rank + 2 * reduced_count + 2 * retained_count);
    packed.insert(packed.end(), indexing.input_strides.begin(), indexing.input_strides.end());
    for (uint32_t axis : indexing.reduced_axes) {
        packed.push_back(axis);
    }
    for (uint32_t axis : indexing.retained_axes) {
        packed.push_back(axis);
    }
    packed.insert(packed.end(), indexing.reduced_dimensions.begin(), indexing.reduced_dimensions.end());
    packed.insert(packed.end(), indexing.retained_dimensions.begin(), indexing.retained_dimensions.end());

    Tensor host_metadata(TensorPlacement(TensorPlacement::MemDevices::CPU),
                         TensorDescriptor(DataType::UINT64, {static_cast<uint64_t>(packed.size())}));
    std::memcpy(host_metadata.getMemPtr<uint64_t>(), packed.data(), packed.size() * sizeof(uint64_t));

    Tensor device_metadata(placement, TensorDescriptor(DataType::UINT64, {static_cast<uint64_t>(packed.size())}));
    device_metadata.copyFromAsync(host_metadata, stream);
    stream.synchronize();

    const uint64_t* base = device_metadata.getMemPtr<uint64_t>();
    geometry.device_indexing.reduced_axis_count = static_cast<uint32_t>(reduced_count);
    geometry.device_indexing.retained_axis_count = static_cast<uint32_t>(retained_count);
    geometry.device_indexing.input_strides = base;
    geometry.device_indexing.reduced_axes = base + rank;
    geometry.device_indexing.retained_axes = geometry.device_indexing.reduced_axes + reduced_count;
    geometry.device_indexing.reduced_dimensions = geometry.device_indexing.retained_axes + retained_count;
    geometry.device_indexing.retained_dimensions = geometry.device_indexing.reduced_dimensions + reduced_count;
    return device_metadata;
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
    if (stripSingletonDimensions(output_dimensions) != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(
            "CUB tensor reduction preallocated output dimensions must preserve the reduction output's non-singleton layout.");
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

[[nodiscard]] bool tensorStorageOverlaps(const Tensor& lhs, const Tensor& rhs) {
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs.getMemPtr<void>());
    const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs.getMemPtr<void>());
    const uintptr_t lhs_end = lhs_begin + lhs.getArraySizeInBytes();
    const uintptr_t rhs_end = rhs_begin + rhs.getArraySizeInBytes();
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

void requireExpectedArgOutput(const Tensor& input,
                              const Tensor& output,
                              DataType output_dtype,
                              const CubReductionGeometry& geometry,
                              const char* role) {
    requireDenseContiguousGpuTensor(output, role);
    requireSameGpuPlacement(input, output, "input", role);
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " dtype does not match the configured dtype.");
    }
    if (stripSingletonDimensions(output.getDimensions())
        != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " dimensions must preserve the reduction output's non-singleton layout.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " element count does not match the reduction geometry.");
    }
    if (tensorStorageOverlaps(input, output)) {
        throw std::invalid_argument(std::string("CUB arg reduction input and ") + role + " storage must not overlap.");
    }
}

void requireArgOutputsDoNotOverlap(const std::optional<Tensor>& value_output,
                                   const std::optional<Tensor>& index_output) {
    if (value_output.has_value() && index_output.has_value()
        && tensorStorageOverlaps(value_output.value(), index_output.value())) {
        throw std::invalid_argument("CUB arg reduction value and index output storage must not overlap.");
    }
}

void requireSupportedSegmentedOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
        case CubReductionOp::Min:
        case CubReductionOp::Max:
            return;
        case CubReductionOp::Product:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
            break;
    }
    throw std::invalid_argument("Offset-segmented CUB reduction supports sum, mean, min, and max.");
}

void requireSegmentOffsets(const Tensor& input, const Tensor& segment_offsets, const Stream& stream) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment offsets");
    requireSameGpuPlacement(input, segment_offsets, "input", "segment offsets");
    requireCompatibleStream(input, stream);
    if (!CubSegmentedReduction::isOffsetDataTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("CUB segmented-reduction offsets must use UINT32 or an enabled UINT64 dtype.");
    }
    const std::vector<uint64_t>& dimensions = segment_offsets.getDimensions();
    if (dimensions.size() != 1 || dimensions[0] < 2) {
        throw std::invalid_argument(
            "CUB segmented-reduction offsets must be rank 1 with shape [num_segments + 1] and at least one segment.");
    }
    if (tensorStorageOverlaps(input, segment_offsets)) {
        throw std::invalid_argument("CUB segmented-reduction input and offsets storage must not overlap.");
    }
}

void validateSegmentOffsetContentsAtStamp(const Tensor& input,
                                          const Tensor& segment_offsets,
                                          const Stream& stream) {
    Tensor cpu_offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), segment_offsets.getDescriptor());
    cpu_offsets.copyFromAsync(segment_offsets, stream);
    stream.synchronize();

    const uint64_t offset_count = segment_offsets.getTotalNumElements();
    auto validate = [&](auto* offsets) {
        if (static_cast<uint64_t>(offsets[0]) != 0) {
            throw std::invalid_argument("CUB segmented-reduction offsets must begin at zero.");
        }
        uint64_t previous = 0;
        for (uint64_t i = 1; i < offset_count; ++i) {
            const uint64_t current = static_cast<uint64_t>(offsets[i]);
            if (current < previous) {
                throw std::invalid_argument("CUB segmented-reduction offsets must be nondecreasing.");
            }
            if (current > input.getTotalNumElements()) {
                throw std::invalid_argument("CUB segmented-reduction offset exceeds the values tensor capacity.");
            }
            previous = current;
        }
    };

    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            validate(cpu_offsets.getMemPtr<uint32_t>());
            return;
        case DataType::UINT64:
            validate(cpu_offsets.getMemPtr<uint64_t>());
            return;
        default:
            throw std::invalid_argument("CUB segmented-reduction offsets must use UINT32 or UINT64.");
    }
}

void requireExpectedSegmentedOutput(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    DataType output_dtype,
                                    uint64_t num_segments) {
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(
            "CUB segmented-reduction preallocated output dtype does not match the configured output dtype.");
    }
    if (output.getTotalNumElements() != num_segments
        || stripSingletonDimensions(output.getDimensions())
               != stripSingletonDimensions(std::vector<uint64_t>{num_segments})) {
        throw std::invalid_argument(
            "CUB segmented-reduction output must preserve the [num_segments] non-singleton layout.");
    }
    if (tensorStorageOverlaps(input, output) || tensorStorageOverlaps(segment_offsets, output)) {
        throw std::invalid_argument(
            "CUB segmented-reduction output storage must not overlap input or offsets storage.");
    }
}

void requireSupportedArgOperation(CubArgReductionOp op) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
        case CubArgReductionOp::ArgMax:
            return;
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
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
                           DataType input_dtype,
                           const void* input,
                           uint64_t input_elements,
                           DataType output_dtype,
                           void* output,
                           const CubReductionGeometry& geometry,
                           float output_scale,
                           const Stream& stream) {
    switch (op) {
        case CubReductionOp::Sum:
            return CubReductionInternal::querySumReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::Product:
            return CubReductionInternal::queryProductReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::Mean:
            return CubReductionInternal::queryMeanReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::L1Norm:
            return CubReductionInternal::queryL1NormReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::L2Norm:
            return CubReductionInternal::queryL2NormReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::Min:
            return CubReductionInternal::queryMinReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        case CubReductionOp::Max:
            return CubReductionInternal::queryMaxReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

void launchReduction(CubReductionOp op,
                     const Tensor& temp_storage,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     const CubReductionGeometry& geometry,
                     float output_scale,
                     Stream& stream) {
    switch (op) {
        case CubReductionOp::Sum:
            CubReductionInternal::launchSumReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::Product:
            CubReductionInternal::launchProductReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::Mean:
            CubReductionInternal::launchMeanReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::L1Norm:
            CubReductionInternal::launchL1NormReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::L2Norm:
            CubReductionInternal::launchL2NormReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::Min:
            CubReductionInternal::launchMinReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        case CubReductionOp::Max:
            CubReductionInternal::launchMaxReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

size_t queryArgReductionBytes(CubArgReductionOp op,
                              const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              const Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return CubReductionInternal::queryArgMinReductionBytes(
                input, value_output, index_output, geometry, stream);
        case CubArgReductionOp::ArgMax:
            return CubReductionInternal::queryArgMaxReductionBytes(
                input, value_output, index_output, geometry, stream);
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

void launchArgReduction(CubArgReductionOp op,
                        const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor* value_output,
                        Tensor* index_output,
                        const CubReductionGeometry& geometry,
                        Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            CubReductionInternal::launchArgMinReduction(
                temp_storage, temp_storage_bytes, input, value_output, index_output, geometry, stream);
            return;
        case CubArgReductionOp::ArgMax:
            CubReductionInternal::launchArgMaxReduction(
                temp_storage, temp_storage_bytes, input, value_output, index_output, geometry, stream);
            return;
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

}  // namespace

CubReduction::CubReduction(CubReductionOp op,
                           uint32_t axis,
                           std::optional<DataType> output_dtype,
                           float output_scale)
    : CubReduction(op, std::vector<uint32_t>{axis}, output_dtype, output_scale) {}

CubReduction::CubReduction(CubReductionOp op,
                           std::vector<uint32_t> axes,
                           std::optional<DataType> output_dtype,
                           float output_scale)
    : op(op), axes(std::move(axes)), output_dtype(output_dtype), output_scale(output_scale) {
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
    if (!std::isfinite(output_scale)) {
        throw std::invalid_argument("CUB tensor reduction output scale must be finite.");
    }
}

DataType CubReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "output");
    return resolved;
}

size_t CubReduction::queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                               const Stream& stream) const {
    requireSupportedFloatingStorageDType(input_descriptor.getDataType(), "input");
    const CubReductionGeometry geometry = analyzeGeometry(input_descriptor.getDimensions(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input_descriptor.getDataType());
    ScopedGpu scoped_gpu(stream.getGpuNum());
    return queryReductionBytes(op,
                               input_descriptor.getDataType(),
                               nullptr,
                               input_descriptor.getTotalNumElements(),
                               resolved_output_dtype,
                               nullptr,
                               geometry,
                               output_scale,
                               stream);
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
    if (input_dimensions.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction rank exceeds the uint32 axis representation limit.");
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
    geometry.indexing.input_strides.resize(input_dimensions.size());
    geometry.indexing.reduced_axes.reserve(axes.size());
    geometry.indexing.reduced_dimensions.reserve(axes.size());
    geometry.indexing.retained_axes.reserve(input_dimensions.size() - axes.size());
    geometry.indexing.retained_dimensions.reserve(input_dimensions.size() - axes.size());

    uint64_t running_stride = 1;
    for (size_t dimension = input_dimensions.size(); dimension-- > 0;) {
        geometry.indexing.input_strides[dimension] = running_stride;
        running_stride = checkedMultiply(running_stride, input_dimensions[dimension], "input element count");
    }
    geometry.input_elements = running_stride;

    size_t reduced_cursor = 0;
    for (uint32_t dimension = 0; dimension < input_dimensions.size(); ++dimension) {
        const bool reduced = reduced_cursor < axes.size() && axes[reduced_cursor] == dimension;
        if (reduced) {
            geometry.output_dimensions[dimension] = 1;
            geometry.reduction_size = checkedMultiply(
                geometry.reduction_size, input_dimensions[dimension], "reduction element count");
            geometry.indexing.reduced_axes.push_back(dimension);
            geometry.indexing.reduced_dimensions.push_back(input_dimensions[dimension]);
            ++reduced_cursor;
        } else {
            geometry.output_elements = checkedMultiply(
                geometry.output_elements, input_dimensions[dimension], "output element count");
            geometry.squeezed_output_dimensions.push_back(input_dimensions[dimension]);
            geometry.indexing.retained_axes.push_back(dimension);
            geometry.indexing.retained_dimensions.push_back(input_dimensions[dimension]);
        }
    }
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
    CubReductionGeometry stamped_geometry = geometry;
    std::optional<Tensor> indexing_metadata =
        stampDeviceIndexingMetadata(stamped_geometry, input.getPlacement(), stream);
    Tensor mutable_output = output;
    const size_t temp_storage_bytes = queryReductionBytes(op,
                                                          input.getDataType(),
                                                          input.getMemPtr<void>(),
                                                          input.getTotalNumElements(),
                                                          mutable_output.getDataType(),
                                                          mutable_output.getMemPtr<void>(),
                                                          stamped_geometry,
                                                          output_scale,
                                                          stream);
    Tensor temp_storage(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

    return std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
                                                                        std::move(stamped_geometry),
                                                                        input,
                                                                        output,
                                                                        temp_storage_bytes,
                                                                        temp_storage,
                                                                        std::move(indexing_metadata),
                                                                        output_scale,
                                                                        stream));
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

CubSegmentedReduction::CubSegmentedReduction(CubReductionOp op,
                                                       std::optional<DataType> output_dtype)
    : op(op), output_dtype(output_dtype) {
    requireSupportedSegmentedOperation(op);
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "segmented output");
    }
}

bool CubSegmentedReduction::isInputDataTypeSupported(DataType dtype) {
    return isSupportedFloatingStorageDType(dtype);
}

bool CubSegmentedReduction::isOffsetDataTypeSupported(DataType dtype) {
    if (dtype == DataType::UINT32) {
        return true;
    }
#if THOR_CUB_ENABLE_64BIT_TYPES
    if (dtype == DataType::UINT64) {
        return true;
    }
#endif
    return false;
}

DataType CubSegmentedReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "segmented input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "segmented output");
    return resolved;
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stamp(
    const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    if (input.getDimensions().size() != 1) {
        throw std::invalid_argument("CUB segmented reduction currently requires a rank-1 values tensor.");
    }
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    Tensor output(input.getPlacement(),
                  TensorDescriptor(resolveOutputDataType(input.getDataType()), {num_segments}));
    return stampValidated(input, output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stamp(
    const Tensor& input,
    const Tensor& preallocated_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    if (input.getDimensions().size() != 1) {
        throw std::invalid_argument("CUB segmented reduction currently requires a rank-1 values tensor.");
    }
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedOutput(input,
                                   preallocated_output,
                                   segment_offsets,
                                   resolveOutputDataType(input.getDataType()),
                                   num_segments);
    return stampValidated(input, preallocated_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stampValidated(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_segments,
    const Stream& stream) const {
    const uint64_t num_items = input.getTotalNumElements();
    if (num_items > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
        || num_segments > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB segmented-reduction item or segment count exceeds int64 limits.");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    Tensor mutable_output = output;
    const size_t temp_storage_bytes = CubReductionInternal::queryOffsetSegmentedReductionBytes(
        op, input, mutable_output, segment_offsets, num_items, num_segments, stream);
    Tensor temp_storage(
        input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));
    return std::shared_ptr<StampedCubSegmentedReduction>(new StampedCubSegmentedReduction(op,
                                                                                         input,
                                                                                         output,
                                                                                         segment_offsets,
                                                                                         num_items,
                                                                                         num_segments,
                                                                                         temp_storage_bytes,
                                                                                         temp_storage,
                                                                                         stream));
}

StampedCubSegmentedReduction::StampedCubSegmentedReduction(CubReductionOp op,
                                                           const Tensor& input,
                                                           const Tensor& output,
                                                           const Tensor& segment_offsets,
                                                           uint64_t num_items,
                                                           uint64_t num_segments,
                                                           size_t temp_storage_bytes,
                                                           const Tensor& temp_storage,
                                                           const Stream& stream)
    : op(op),
      input(input),
      output(output),
      segment_offsets(segment_offsets),
      num_items(num_items),
      num_segments(num_segments),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubSegmentedReduction::run() { runOn(stream); }

void StampedCubSegmentedReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    CubReductionInternal::launchOffsetSegmentedReduction(op,
                                                         temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         segment_offsets,
                                                         num_items,
                                                         num_segments,
                                                         run_stream);
}

CubArgReduction::CubArgReduction(CubArgReductionOp op,
                                 uint32_t axis,
                                 CubArgReductionOutputOptions outputs)
    : CubArgReduction(op, std::vector<uint32_t>{axis}, std::move(outputs)) {}

CubArgReduction::CubArgReduction(CubArgReductionOp op,
                                 std::vector<uint32_t> axes,
                                 CubArgReductionOutputOptions outputs)
    : op(op), axes(std::move(axes)), outputs(std::move(outputs)) {
    requireSupportedArgOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("CUB arg reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("CUB arg reduction axes must be unique and strictly increasing.");
        }
    }
    if (!this->outputs.produce_value && !this->outputs.produce_index) {
        throw std::invalid_argument("CUB arg reduction must produce a value, an index, or both.");
    }
    if (!this->outputs.produce_value && this->outputs.value_output_dtype.has_value()) {
        throw std::invalid_argument("CUB arg reduction cannot configure a value dtype when value output is disabled.");
    }
    if (this->outputs.value_output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(this->outputs.value_output_dtype.value(), "value output");
    }
    if (this->outputs.produce_index) {
        requireSupportedArgIndexDType(this->outputs.index_output_dtype);
    }
}

DataType CubArgReduction::resolveValueOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = outputs.value_output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "value output");
    return resolved;
}

float CubArgReduction::getFp32EmptyReductionValue(CubArgReductionOp op) {
    requireSupportedArgOperation(op);
    return op == CubArgReductionOp::ArgMin ? std::numeric_limits<float>::infinity()
                                           : -std::numeric_limits<float>::infinity();
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(const Tensor& input,
                                                               const Stream& stream) const {
    return stamp(input, std::nullopt, std::nullopt, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(
    const Tensor& input,
    const std::optional<Tensor>& preallocated_value_output,
    const std::optional<Tensor>& preallocated_index_output,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    requireSupportedFloatingStorageDType(input.getDataType(), "input");

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), axes);
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB arg reduction domain does not fit in a UINT32 local index.");
    }

    std::optional<Tensor> value_output = std::nullopt;
    if (outputs.produce_value) {
        const DataType value_dtype = resolveValueOutputDataType(input.getDataType());
        if (preallocated_value_output.has_value()) {
            requireExpectedArgOutput(input, preallocated_value_output.value(), value_dtype, geometry, "value output");
            value_output = preallocated_value_output;
        } else {
            value_output = Tensor(input.getPlacement(), TensorDescriptor(value_dtype, geometry.output_dimensions));
        }
    } else if (preallocated_value_output.has_value()) {
        throw std::invalid_argument("CUB arg reduction received a value output while value production is disabled.");
    }

    std::optional<Tensor> index_output = std::nullopt;
    if (outputs.produce_index) {
        if (preallocated_index_output.has_value()) {
            requireExpectedArgOutput(
                input, preallocated_index_output.value(), outputs.index_output_dtype, geometry, "index output");
            index_output = preallocated_index_output;
        } else {
            index_output = Tensor(
                input.getPlacement(), TensorDescriptor(outputs.index_output_dtype, geometry.output_dimensions));
        }
    } else if (preallocated_index_output.has_value()) {
        throw std::invalid_argument("CUB arg reduction received an index output while index production is disabled.");
    }

    requireArgOutputsDoNotOverlap(value_output, index_output);
    return stampValidated(input, std::move(value_output), std::move(index_output), geometry, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampValidated(
    const Tensor& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    const CubReductionGeometry& geometry,
    const Stream& stream) const {
    ScopedGpu scoped_gpu(stream.getGpuNum());
    CubReductionGeometry stamped_geometry = geometry;
    std::optional<Tensor> indexing_metadata =
        stampDeviceIndexingMetadata(stamped_geometry, input.getPlacement(), stream);
    Tensor* value_output_ptr = value_output.has_value() ? &value_output.value() : nullptr;
    Tensor* index_output_ptr = index_output.has_value() ? &index_output.value() : nullptr;
    const size_t temp_storage_bytes =
        queryArgReductionBytes(op, input, value_output_ptr, index_output_ptr, stamped_geometry, stream);
    Tensor temp_storage(
        input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

    return std::shared_ptr<StampedCubArgReduction>(new StampedCubArgReduction(op,
                                                                             std::move(stamped_geometry),
                                                                             input,
                                                                             std::move(value_output),
                                                                             std::move(index_output),
                                                                             temp_storage_bytes,
                                                                             temp_storage,
                                                                             std::move(indexing_metadata),
                                                                             stream));
}

StampedCubArgReduction::StampedCubArgReduction(CubArgReductionOp op,
                                               CubReductionGeometry geometry,
                                               const Tensor& input,
                                               std::optional<Tensor> value_output,
                                               std::optional<Tensor> index_output,
                                               size_t temp_storage_bytes,
                                               const Tensor& temp_storage,
                                               std::optional<Tensor> indexing_metadata,
                                               const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      value_output(std::move(value_output)),
      index_output(std::move(index_output)),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      indexing_metadata(std::move(indexing_metadata)),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped CUB arg reduction requires at least one output.");
    }
}

void StampedCubArgReduction::run() { runOn(stream); }

void StampedCubArgReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    Tensor* value_output_ptr = value_output.has_value() ? &value_output.value() : nullptr;
    Tensor* index_output_ptr = index_output.has_value() ? &index_output.value() : nullptr;
    launchArgReduction(
        op, temp_storage, temp_storage_bytes, input, value_output_ptr, index_output_ptr, geometry, run_stream);
}

StampedCubReduction::StampedCubReduction(CubReductionOp op,
                                         CubReductionGeometry geometry,
                                         const Tensor& input,
                                         const Tensor& output,
                                         size_t temp_storage_bytes,
                                         const Tensor& temp_storage,
                                         std::optional<Tensor> indexing_metadata,
                                         float output_scale,
                                         const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      output(output),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      indexing_metadata(std::move(indexing_metadata)),
      output_scale(output_scale),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubReduction::run() { runOn(stream); }

void StampedCubReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    launchReduction(op, temp_storage, temp_storage_bytes, input, output, geometry, output_scale, run_stream);
}

}  // namespace ThorImplementation
