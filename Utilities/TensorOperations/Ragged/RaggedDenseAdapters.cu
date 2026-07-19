#include "Utilities/TensorOperations/Ragged/RaggedDenseAdapters.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

uint64_t checkedMul(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t.");
    }
    return a * b;
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char* label) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t.");
    }
    return a + b;
}

uint64_t elementSizeBytes(DataType dtype) {
    const float bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t rounded = static_cast<uint64_t>(bytes);
    if (static_cast<float>(rounded) != bytes || rounded == 0) {
        throw std::invalid_argument("ragged dense adapter received an unsupported non-integral element size.");
    }
    return rounded;
}

uint64_t trailingElementsFromDense(const Tensor& dense) {
    const std::vector<uint64_t> dims = dense.getDimensions();
    if (dims.size() < 2) {
        throw std::invalid_argument("ragged dense adapter dense tensor must have shape [B, max_length, ...].");
    }
    uint64_t elements = 1;
    for (size_t i = 2; i < dims.size(); ++i) {
        elements = checkedMul(elements, dims[i], "ragged dense adapter elements_per_value");
    }
    return elements;
}

std::vector<uint64_t> expectedValuesDimensionsForDense(const Tensor& dense, uint64_t max_total_values) {
    const std::vector<uint64_t> dense_dims = dense.getDimensions();
    if (dense_dims.size() < 2) {
        throw std::invalid_argument("ragged dense adapter dense tensor must have shape [B, max_length, ...].");
    }
    std::vector<uint64_t> values_dims;
    values_dims.reserve(dense_dims.size() - 1);
    values_dims.push_back(max_total_values);
    values_dims.insert(values_dims.end(), dense_dims.begin() + 2, dense_dims.end());
    return values_dims;
}

void requireRank1GpuVector(const Tensor& tensor, const char* name) {
    requireDenseContiguousGpuTensor(tensor, name);
    if (tensor.getNumDimensions() != 1) {
        throw std::invalid_argument(std::string(name) + " must be rank 1.");
    }
}

void validateDenseAndValues(const Tensor& dense, const Tensor& values) {
    requireDenseContiguousGpuTensor(dense, "ragged dense adapter dense input");
    requireDenseContiguousGpuTensor(values, "ragged dense adapter ragged values output");
    requireSameGpuPlacement(dense, values, "ragged dense adapter dense input", "ragged dense adapter ragged values output");
    if (dense.getDataType() != values.getDataType()) {
        throw std::invalid_argument("ragged dense adapter dense input and ragged values output must have the same dtype.");
    }
    if (dense.getNumDimensions() < 2) {
        throw std::invalid_argument("ragged dense adapter dense input must have shape [B, max_length, ...].");
    }
    if (values.getNumDimensions() + 1 != dense.getNumDimensions()) {
        throw std::invalid_argument("ragged dense adapter ragged values rank must equal dense rank - 1.");
    }
    const std::vector<uint64_t> dense_dims = dense.getDimensions();
    const std::vector<uint64_t> values_dims = values.getDimensions();
    if (dense_dims[0] == 0 || dense_dims[1] == 0 || values_dims[0] == 0) {
        throw std::invalid_argument("ragged dense adapter dimensions must be non-zero.");
    }
    if (expectedValuesDimensionsForDense(dense, values_dims[0]) != values_dims) {
        throw std::invalid_argument("ragged dense adapter ragged values dimensions must be [max_total_values] + dense trailing dimensions.");
    }
}

void validateOffsetsForDense(const Tensor& dense, const Tensor& offsets, const Tensor& values) {
    validateDenseAndValues(dense, values);
    requireRank1GpuVector(offsets, "ragged dense adapter offsets");
    requireSameGpuPlacement(dense, offsets, "ragged dense adapter dense input", "ragged dense adapter offsets");
    if (!isRowPartitionOffsetDTypeSupported(offsets.getDataType())) {
        throw std::invalid_argument("ragged dense adapter offsets dtype must be UINT32 or UINT64.");
    }
    if (!canonicalRowPartitionOffsetCanRepresent(offsets.getDataType(), values.getDimensions()[0])) {
        throw std::invalid_argument("ragged dense adapter offsets dtype cannot represent max_total_values.");
    }
    const uint64_t batch_size = dense.getDimensions()[0];
    if (offsets.getDimensions()[0] != checkedAdd(batch_size, 1, "ragged dense adapter offsets element count")) {
        throw std::invalid_argument("ragged dense adapter offsets must have shape [B + 1].");
    }
}

void validateLengthsForDense(const Tensor& dense, const Tensor& lengths, const Tensor& values, const Tensor& offsets) {
    validateOffsetsForDense(dense, offsets, values);
    requireRank1GpuVector(lengths, "ragged dense adapter lengths");
    requireSameGpuPlacement(dense, lengths, "ragged dense adapter dense input", "ragged dense adapter lengths");
    if (!isRowPartitionOffsetDTypeSupported(lengths.getDataType())) {
        throw std::invalid_argument("ragged dense adapter lengths dtype must be UINT32 or UINT64.");
    }
    if (lengths.getDataType() != offsets.getDataType()) {
        throw std::invalid_argument("ragged dense adapter lengths and offsets must have the same dtype.");
    }
    const uint64_t batch_size = dense.getDimensions()[0];
    if (lengths.getDimensions()[0] != batch_size) {
        throw std::invalid_argument("ragged dense adapter lengths must have shape [B].");
    }
}

void validateRaggedToDense(const RaggedTensor& ragged, const Tensor& dense) {
    if (!ragged.isInitialized()) {
        throw std::invalid_argument("raggedToDense requires an initialized RaggedTensor.");
    }
    requireDenseContiguousGpuTensor(dense, "raggedToDense dense output");
    const Tensor values = ragged.getValues();
    const Tensor offsets = ragged.getOffsets();
    requireSameGpuPlacement(values, dense, "raggedToDense ragged values", "raggedToDense dense output");
    requireSameGpuPlacement(offsets, dense, "raggedToDense ragged offsets", "raggedToDense dense output");
    if (dense.getDataType() != values.getDataType()) {
        throw std::invalid_argument("raggedToDense dense output dtype must match ragged values dtype.");
    }
    if (dense.getNumDimensions() != values.getNumDimensions() + 1) {
        throw std::invalid_argument("raggedToDense dense output rank must equal ragged values rank + 1.");
    }
    const std::vector<uint64_t> dense_dims = dense.getDimensions();
    if (dense_dims[0] != ragged.getBatchSize()) {
        throw std::invalid_argument("raggedToDense dense output batch dimension must match ragged batch size.");
    }
    if (dense_dims[1] == 0) {
        throw std::invalid_argument("raggedToDense dense output max_length must be non-zero.");
    }
    if (expectedValuesDimensionsForDense(dense, ragged.getMaxTotalValues()) != values.getDimensions()) {
        throw std::invalid_argument("raggedToDense dense output trailing dimensions must match ragged values trailing dimensions.");
    }
}

void validateValidationErrorBits(const Tensor& validation_error_bits, const TensorPlacement& placement) {
    requireDenseContiguousGpuTensor(validation_error_bits, "ragged dense adapter validation_error_bits");
    if (validation_error_bits.getPlacement() != placement) {
        throw std::invalid_argument("ragged dense adapter validation_error_bits placement must match adapter tensors.");
    }
    if (validation_error_bits.getDataType() != DataType::UINT32 || validation_error_bits.getTotalNumElements() != 1) {
        throw std::invalid_argument("ragged dense adapter validation_error_bits must be a UINT32 scalar tensor.");
    }
}

template <typename OffsetT>
__global__ void validateDenseAdapterOffsetsKernel(const OffsetT* offsets,
                                                  uint32_t* validation_error_bits,
                                                  uint64_t batch_size,
                                                  uint64_t max_length,
                                                  uint64_t max_total_values) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row == 0 && static_cast<uint64_t>(offsets[0]) != 0) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO));
    }
    if (row >= batch_size) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
    if (end < begin) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC));
        return;
    }
    if (begin > max_total_values || end > max_total_values) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_EXCEED_CAPACITY));
        return;
    }
    if (end - begin > max_length) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_ROW_LENGTH_EXCEEDS_MAX));
    }
}

template <typename OffsetT>
void launchValidateDenseAdapterOffsets(const Tensor& offsets,
                                       Tensor& validation_error_bits,
                                       uint64_t batch_size,
                                       uint64_t max_length,
                                       uint64_t max_total_values,
                                       Stream& stream) {
    validation_error_bits.fill(0.0, stream);
    constexpr uint32_t threads = 256;
    const uint64_t blocks64 = std::max<uint64_t>(1, (batch_size + threads - 1) / threads);
    if (blocks64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("ragged dense adapter validation grid exceeds CUDA gridDim.x limit.");
    }
    validateDenseAdapterOffsetsKernel<OffsetT><<<static_cast<uint32_t>(blocks64), threads, 0, stream.getStream()>>>(
        offsets.getMemPtr<OffsetT>(),
        validation_error_bits.getMemPtr<uint32_t>(),
        batch_size,
        max_length,
        max_total_values);
    CUDA_CHECK(cudaGetLastError());
}


template <typename OffsetT>
__global__ void denseToRaggedValuesKernel(const uint8_t* dense,
                                          uint8_t* values,
                                          const OffsetT* offsets,
                                          uint64_t batch_size,
                                          uint64_t max_length,
                                          uint64_t bytes_per_value,
                                          const uint32_t* validation_error_bits) {
    if (*validation_error_bits != ROW_PARTITION_VALID) {
        return;
    }
    const uint64_t row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t raw_end = static_cast<uint64_t>(offsets[row + 1]);
    if (raw_end <= begin) {
        return;
    }

    const uint64_t row_values = raw_end - begin;
    const uint64_t row_bytes = row_values * bytes_per_value;
    const uint64_t dense_row_byte_offset = row * max_length * bytes_per_value;
    const uint64_t values_byte_offset = begin * bytes_per_value;

    for (uint64_t byte_idx = threadIdx.x; byte_idx < row_bytes; byte_idx += blockDim.x) {
        values[values_byte_offset + byte_idx] = dense[dense_row_byte_offset + byte_idx];
    }
}

template <typename OffsetT>
__global__ void raggedToDenseValuesKernel(const uint8_t* values,
                                          uint8_t* dense,
                                          const OffsetT* offsets,
                                          uint64_t batch_size,
                                          uint64_t max_length,
                                          uint64_t bytes_per_value,
                                          const uint32_t* validation_error_bits) {
    if (*validation_error_bits != ROW_PARTITION_VALID) {
        return;
    }
    const uint64_t row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t raw_end = static_cast<uint64_t>(offsets[row + 1]);
    if (raw_end <= begin) {
        return;
    }

    const uint64_t row_values = raw_end - begin;
    const uint64_t row_bytes = row_values * bytes_per_value;
    const uint64_t dense_row_byte_offset = row * max_length * bytes_per_value;
    const uint64_t values_byte_offset = begin * bytes_per_value;

    for (uint64_t byte_idx = threadIdx.x; byte_idx < row_bytes; byte_idx += blockDim.x) {
        dense[dense_row_byte_offset + byte_idx] = values[values_byte_offset + byte_idx];
    }
}

template <typename OffsetT>
void launchDenseToRaggedValues(const Tensor& dense,
                               Tensor& values,
                               const Tensor& offsets,
                               uint64_t batch_size,
                               uint64_t max_length,
                               uint64_t bytes_per_value,
                               Tensor& validation_error_bits,
                               Stream& stream) {
    if (batch_size == 0) {
        return;
    }
    if (batch_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("raggedFromDense batch size exceeds CUDA gridDim.x limit.");
    }
    constexpr uint32_t threads = 256;
    denseToRaggedValuesKernel<OffsetT><<<static_cast<uint32_t>(batch_size), threads, 0, stream.getStream()>>>(
        static_cast<const uint8_t*>(dense.getMemPtr<void>()),
        static_cast<uint8_t*>(values.getMemPtr<void>()),
        offsets.getMemPtr<OffsetT>(),
        batch_size,
        max_length,
        bytes_per_value,
        validation_error_bits.getMemPtr<uint32_t>());
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchRaggedToDenseValues(const RaggedTensor& ragged,
                               Tensor& dense,
                               uint64_t max_length,
                               uint64_t bytes_per_value,
                               Tensor& validation_error_bits,
                               Stream& stream) {
    const uint64_t batch_size = ragged.getBatchSize();
    if (batch_size == 0) {
        return;
    }
    if (batch_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("raggedToDense batch size exceeds CUDA gridDim.x limit.");
    }
    constexpr uint32_t threads = 256;
    Tensor values = ragged.getValues();
    Tensor offsets = ragged.getOffsets();
    raggedToDenseValuesKernel<OffsetT><<<static_cast<uint32_t>(batch_size), threads, 0, stream.getStream()>>>(
        static_cast<const uint8_t*>(values.getMemPtr<void>()),
        static_cast<uint8_t*>(dense.getMemPtr<void>()),
        offsets.getMemPtr<OffsetT>(),
        batch_size,
        max_length,
        bytes_per_value,
        validation_error_bits.getMemPtr<uint32_t>());
    CUDA_CHECK(cudaGetLastError());
}

uint64_t bytesPerValue(const Tensor& dense) {
    return checkedMul(trailingElementsFromDense(dense), elementSizeBytes(dense.getDataType()), "ragged dense adapter bytes_per_value");
}

}  // namespace

RaggedFromDenseWithLengthsPlan prepareRaggedFromDenseWithLengths(const Tensor& dense,
                                                                const Tensor& lengths,
                                                                const Tensor& values,
                                                                const Tensor& offsets) {
    validateLengthsForDense(dense, lengths, values, offsets);

    RaggedFromDenseWithLengthsPlan plan;
    plan.placement = dense.getPlacement();
    plan.valuesDataType = dense.getDataType();
    plan.offsetsDataType = offsets.getDataType();
    plan.batchSize = dense.getDimensions()[0];
    plan.maxLength = dense.getDimensions()[1];
    plan.maxTotalValues = values.getDimensions()[0];
    plan.elementsPerValue = trailingElementsFromDense(dense);
    plan.valueElementSizeBytes = elementSizeBytes(dense.getDataType());
    plan.tempStorageBytes = rowPartitionLengthsToOffsetsTempBytes(lengths, offsets, plan.batchSize);
    return plan;
}

size_t raggedFromDenseWithLengthsTempBytes(const Tensor& dense,
                                           const Tensor& lengths,
                                           const Tensor& values,
                                           const Tensor& offsets) {
    return prepareRaggedFromDenseWithLengths(dense, lengths, values, offsets).tempStorageBytes;
}

RaggedTensor raggedFromDense(const RaggedFromDenseWithLengthsPlan& plan,
                             const Tensor& temp_storage,
                             const Tensor& dense,
                             const Tensor& lengths,
                             Tensor& values,
                             Tensor& offsets,
                             Tensor& validation_error_bits,
                             Stream& stream) {
    validateLengthsForDense(dense, lengths, values, offsets);
    if (dense.getPlacement() != plan.placement || dense.getDataType() != plan.valuesDataType ||
        offsets.getDataType() != plan.offsetsDataType || dense.getDimensions()[0] != plan.batchSize ||
        dense.getDimensions()[1] != plan.maxLength || values.getDimensions()[0] != plan.maxTotalValues ||
        trailingElementsFromDense(dense) != plan.elementsPerValue || elementSizeBytes(dense.getDataType()) != plan.valueElementSizeBytes) {
        throw std::invalid_argument("raggedFromDense tensors do not match prepared plan.");
    }
    validateValidationErrorBits(validation_error_bits, plan.placement);
    if (plan.tempStorageBytes > 0) {
        requireTempStorage(temp_storage, plan.placement, plan.tempStorageBytes);
    }

    RowPartitionLengthsToOffsetsPlan offsets_plan = prepareRowPartitionLengthsToOffsets(lengths, offsets, plan.batchSize);
    if (offsets_plan.temp_storage_bytes > plan.tempStorageBytes) {
        throw std::invalid_argument("raggedFromDense prepared temp storage is too small for lengths-to-offsets.");
    }
    offsets_plan.temp_storage_bytes = plan.tempStorageBytes;
    rowPartitionLengthsToOffsets(offsets_plan, temp_storage, lengths, offsets, stream);

    switch (offsets.getDataType()) {
        case DataType::UINT32:
            launchValidateDenseAdapterOffsets<uint32_t>(offsets, validation_error_bits, plan.batchSize, plan.maxLength, plan.maxTotalValues, stream);
            break;
        case DataType::UINT64:
            launchValidateDenseAdapterOffsets<uint64_t>(offsets, validation_error_bits, plan.batchSize, plan.maxLength, plan.maxTotalValues, stream);
            break;
        default:
            throw std::invalid_argument("raggedFromDense offsets dtype must be UINT32 or UINT64.");
    }

    const uint64_t bytes_per_value = checkedMul(plan.elementsPerValue, plan.valueElementSizeBytes, "raggedFromDense bytes_per_value");
    switch (offsets.getDataType()) {
        case DataType::UINT32:
            launchDenseToRaggedValues<uint32_t>(dense, values, offsets, plan.batchSize, plan.maxLength, bytes_per_value, validation_error_bits, stream);
            break;
        case DataType::UINT64:
            launchDenseToRaggedValues<uint64_t>(dense, values, offsets, plan.batchSize, plan.maxLength, bytes_per_value, validation_error_bits, stream);
            break;
        default:
            throw std::invalid_argument("raggedFromDense offsets dtype must be UINT32 or UINT64.");
    }
    return RaggedTensor(values, offsets);
}

RaggedTensor raggedFromDense(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& dense,
                             const Tensor& lengths,
                             Tensor& values,
                             Tensor& offsets,
                             Tensor& validation_error_bits,
                             Stream& stream) {
    RaggedFromDenseWithLengthsPlan plan = prepareRaggedFromDenseWithLengths(dense, lengths, values, offsets);
    if (temp_storage_bytes < plan.tempStorageBytes) {
        throw std::invalid_argument("raggedFromDense temp_storage_bytes is smaller than the prepared requirement.");
    }
    plan.tempStorageBytes = temp_storage_bytes;
    return raggedFromDense(plan, temp_storage, dense, lengths, values, offsets, validation_error_bits, stream);
}

RaggedTensor raggedFromDense(const Tensor& dense,
                             const Tensor& offsets,
                             Tensor& values,
                             Tensor& validation_error_bits,
                             Stream& stream) {
    validateOffsetsForDense(dense, offsets, values);
    const uint64_t batch_size = dense.getDimensions()[0];
    const uint64_t max_length = dense.getDimensions()[1];
    const uint64_t max_total_values = values.getDimensions()[0];
    const uint64_t bytes_per_value = bytesPerValue(dense);
    validateValidationErrorBits(validation_error_bits, dense.getPlacement());

    switch (offsets.getDataType()) {
        case DataType::UINT32:
            launchValidateDenseAdapterOffsets<uint32_t>(offsets, validation_error_bits, batch_size, max_length, max_total_values, stream);
            launchDenseToRaggedValues<uint32_t>(dense, values, offsets, batch_size, max_length, bytes_per_value, validation_error_bits, stream);
            break;
        case DataType::UINT64:
            launchValidateDenseAdapterOffsets<uint64_t>(offsets, validation_error_bits, batch_size, max_length, max_total_values, stream);
            launchDenseToRaggedValues<uint64_t>(dense, values, offsets, batch_size, max_length, bytes_per_value, validation_error_bits, stream);
            break;
        default:
            throw std::invalid_argument("raggedFromDense offsets dtype must be UINT32 or UINT64.");
    }
    return RaggedTensor(values, offsets);
}

void raggedToDense(const RaggedTensor& ragged,
                   Tensor& dense,
                   double padding_value,
                   Tensor& validation_error_bits,
                   Stream& stream) {
    validateRaggedToDense(ragged, dense);
    validateValidationErrorBits(validation_error_bits, dense.getPlacement());
    dense.fill(padding_value, stream);

    const uint64_t max_length = dense.getDimensions()[1];
    const uint64_t bytes_per_value = checkedMul(trailingElementsFromDense(dense), elementSizeBytes(dense.getDataType()), "raggedToDense bytes_per_value");

    switch (ragged.getOffsetsDataType()) {
        case DataType::UINT32:
            launchValidateDenseAdapterOffsets<uint32_t>(ragged.getOffsets(), validation_error_bits, ragged.getBatchSize(), max_length, ragged.getMaxTotalValues(), stream);
            launchRaggedToDenseValues<uint32_t>(ragged, dense, max_length, bytes_per_value, validation_error_bits, stream);
            break;
        case DataType::UINT64:
            launchValidateDenseAdapterOffsets<uint64_t>(ragged.getOffsets(), validation_error_bits, ragged.getBatchSize(), max_length, ragged.getMaxTotalValues(), stream);
            launchRaggedToDenseValues<uint64_t>(ragged, dense, max_length, bytes_per_value, validation_error_bits, stream);
            break;
        default:
            throw std::invalid_argument("raggedToDense offsets dtype must be UINT32 or UINT64.");
    }
}

}  // namespace ThorImplementation
