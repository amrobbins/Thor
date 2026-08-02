#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

__device__ uint64_t readUnalignedUint64(const uint8_t *bytes) {
    uint64_t value = 0;
#pragma unroll
    for (uint64_t i = 0; i < sizeof(uint64_t); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

template <typename OffsetT>
__global__ void buildRaggedOffsetsKernel(
    const uint8_t *__restrict__ records,
    const uint64_t *__restrict__ rowIndices,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t logicalRows,
    uint64_t batchSize,
    OffsetT *__restrict__ offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    offsets[0] = static_cast<OffsetT>(0);
    uint64_t active = 0;
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        const uint8_t *reference = records + sourceRow * recordSizeBytes + referenceOffsetBytes;
        active += readUnalignedUint64(reference + sizeof(uint64_t));
        offsets[row + 1] = static_cast<OffsetT>(active);
    }
    for (uint64_t row = logicalRows + 1; row <= batchSize; ++row) {
        offsets[row] = static_cast<OffsetT>(active);
    }
}

template <typename OffsetT>
__global__ void gatherRaggedValuesKernel(
    const uint8_t *__restrict__ records,
    const uint8_t *__restrict__ packedValues,
    const uint64_t *__restrict__ rowIndices,
    const OffsetT *__restrict__ offsets,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    uint64_t logicalRows,
    uint8_t *__restrict__ destination) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x);
         row < logicalRows;
         row += static_cast<uint64_t>(gridDim.x)) {
        const uint64_t sourceRow = rowIndices[row];
        const uint8_t *reference = records + sourceRow * recordSizeBytes + referenceOffsetBytes;
        const uint64_t start = readUnalignedUint64(reference);
        const uint64_t count = readUnalignedUint64(reference + sizeof(uint64_t));
        const uint64_t sourceByte = start * valueBytes;
        const uint64_t destinationByte = static_cast<uint64_t>(offsets[row]) * valueBytes;
        const uint64_t bytes = count * valueBytes;
        for (uint64_t byte = static_cast<uint64_t>(threadIdx.x);
             byte < bytes;
             byte += static_cast<uint64_t>(blockDim.x)) {
            destination[destinationByte + byte] = packedValues[sourceByte + byte];
        }
    }
}

template <typename OffsetT>
void launchTyped(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();

    buildRaggedOffsetsKernel<OffsetT><<<1, 1, 0, stream.getStream()>>>(
        recordStorage.getMemPtr<uint8_t>(),
        rowIndicesDevice.getMemPtr<uint64_t>(),
        recordSizeBytes,
        referenceOffsetBytes,
        logicalRows,
        batchSize,
        destinationOffsets.getMemPtr<OffsetT>());
    CUDA_CHECK(cudaGetLastError());

    constexpr int threadsPerBlock = 256;
    const int blocks = static_cast<int>(std::min<uint64_t>(logicalRows, 65535));
    gatherRaggedValuesKernel<OffsetT><<<blocks, threadsPerBlock, 0, stream.getStream()>>>(
        recordStorage.getMemPtr<uint8_t>(),
        packedValuesStorage.isInitialized()
            ? static_cast<const uint8_t *>(packedValuesStorage.getMemPtr<void>())
            : nullptr,
        rowIndicesDevice.getMemPtr<uint64_t>(),
        destinationOffsets.getMemPtr<OffsetT>(),
        recordSizeBytes,
        referenceOffsetBytes,
        valueBytes,
        logicalRows,
        static_cast<uint8_t *>(destinationValues.getMemPtr()));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchDeviceResidentRaggedMaterializationKernel(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(packedValuesStorage.isInitialized() || storedValueCount == 0);
    THOR_THROW_IF_FALSE(destinationValues.isInitialized());
    THOR_THROW_IF_FALSE(destinationOffsets.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!packedValuesStorage.isInitialized() ||
                        packedValuesStorage.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationValues.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationOffsets.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destinationValues.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes >= 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(referenceOffsetBytes <= recordSizeBytes - 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(valueBytes > 0);
    THOR_THROW_IF_FALSE(
        storedValueCount <= std::numeric_limits<uint64_t>::max() / valueBytes);
    THOR_THROW_IF_FALSE(
        (packedValuesStorage.isInitialized() ? packedValuesStorage.getArraySizeInBytes() : 0) ==
        storedValueCount * valueBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() == numExamples * recordSizeBytes);

    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= batchSize);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().front() == batchSize + 1);
    THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() > 0);

    switch (destinationOffsets.getDataType()) {
        case DataType::UINT32:
            THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() <=
                                std::numeric_limits<uint32_t>::max());
            launchTyped<uint32_t>(
                recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
                valueBytes, logicalRows, destinationValues, destinationOffsets,
                rowIndicesDevice, stream);
            return;
        case DataType::UINT64:
            launchTyped<uint64_t>(
                recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
                valueBytes, logicalRows, destinationValues, destinationOffsets,
                rowIndicesDevice, stream);
            return;
        default:
            break;
    }
    throw std::runtime_error(
        "Device resident ragged offsets must use canonical UINT32 or UINT64 dtype.");
}
