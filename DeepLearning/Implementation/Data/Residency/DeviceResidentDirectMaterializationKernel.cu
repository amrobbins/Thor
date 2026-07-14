#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

__global__ void materializeDirectFieldKernel(
    const uint8_t *__restrict__ records,
    const uint64_t *__restrict__ rowIndices,
    uint8_t *__restrict__ destination,
    uint64_t batchSize,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t totalBytes) {
    const uint64_t linear =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t outputOffset = linear;
         outputOffset < totalBytes;
         outputOffset += stride) {
        const uint64_t batchRow = outputOffset / fieldBytes;
        const uint64_t fieldByte = outputOffset - batchRow * fieldBytes;
        if (batchRow >= batchSize) {
            return;
        }
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow < numExamples) {
            destination[outputOffset] =
                records[sourceRow * recordSizeBytes + fieldOffsetBytes + fieldByte];
        }
    }
}

}  // namespace

void launchDeviceResidentDirectMaterializationKernel(
    const Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    Tensor &destination,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destination.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes > 0);
    THOR_THROW_IF_FALSE(fieldBytes > 0);
    THOR_THROW_IF_FALSE(fieldOffsetBytes <= recordSizeBytes);
    THOR_THROW_IF_FALSE(fieldBytes <= recordSizeBytes - fieldOffsetBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() % recordSizeBytes == 0);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() / recordSizeBytes == numExamples);

    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(destination.getDimensions().front() == batchSize);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() % fieldBytes == 0);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() / fieldBytes == batchSize);
    const uint64_t totalBytes = destination.getArraySizeInBytes();

    constexpr int threadsPerBlock = 256;
    uint64_t blocks64 = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;
    blocks64 = std::max<uint64_t>(1, std::min<uint64_t>(blocks64, 65535));
    const int blocks = static_cast<int>(blocks64);

    materializeDirectFieldKernel<<<blocks, threadsPerBlock, 0, stream.getStream()>>>(
        recordStorage.getMemPtr<uint8_t>(),
        rowIndicesDevice.getMemPtr<uint64_t>(),
        static_cast<uint8_t *>(destination.getMemPtr()),
        batchSize,
        numExamples,
        recordSizeBytes,
        fieldOffsetBytes,
        fieldBytes,
        totalBytes);
    CUDA_CHECK(cudaGetLastError());
}
