#include "Utilities/TensorOperations/Masking/BatchValidity.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace ThorImplementation {
namespace {

uint64_t requireBatchCapacity(const Tensor& tensor, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(tensor.isInitialized());
    THOR_THROW_IF_FALSE(tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!tensor.hasCustomStrides());
    THOR_THROW_IF_FALSE(tensor.isDenseContiguous());
    const std::vector<uint64_t> dimensions = tensor.getDimensions();
    THOR_THROW_IF_FALSE(!dimensions.empty());
    THOR_THROW_IF_FALSE(dimensions.front() >= 1);
    THOR_THROW_IF_FALSE(dimensions.front() <= std::numeric_limits<uint32_t>::max());
    THOR_THROW_IF_FALSE(validExampleCount >= 1);
    THOR_THROW_IF_FALSE(validExampleCount <= dimensions.front());
    return dimensions.front();
}

__global__ void writeBatchValidityMaskKernel(float* values,
                                             uint64_t elementsPerExample,
                                             uint32_t batchCapacity,
                                             uint32_t validExampleCount) {
    const uint64_t totalElements = elementsPerExample * batchCapacity;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < totalElements;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint32_t example = static_cast<uint32_t>(index / elementsPerExample);
        values[index] = example < validExampleCount ? 1.0f : 0.0f;
    }
}

}  // namespace

void zeroInvalidBatchTail(Tensor& tensor, uint32_t validExampleCount, Stream stream) {
    const uint64_t batchCapacity = requireBatchCapacity(tensor, validExampleCount);
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(stream.getGpuNum() == tensor.getPlacement().getDeviceNum());
    if (validExampleCount == batchCapacity)
        return;

    const uint64_t totalBytes = tensor.getArraySizeInBytes();
    THOR_THROW_IF_FALSE(totalBytes % batchCapacity == 0);
    const uint64_t bytesPerExample = totalBytes / batchCapacity;
    const uint64_t byteOffset = bytesPerExample * validExampleCount;
    const uint64_t bytesToZero = totalBytes - byteOffset;

    ScopedGpu scopedGpu(tensor.getPlacement().getDeviceNum());
    uint8_t* firstInvalidByte = static_cast<uint8_t*>(tensor.getMemPtr<void>()) + byteOffset;
    CUDA_CHECK(cudaMemsetAsync(firstInvalidByte, 0, bytesToZero, stream.getStream()));
}

void writeBatchValidityMask(Tensor& tensor, uint32_t validExampleCount, Stream stream) {
    const uint64_t batchCapacity64 = requireBatchCapacity(tensor, validExampleCount);
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(stream.getGpuNum() == tensor.getPlacement().getDeviceNum());
    THOR_THROW_IF_FALSE(tensor.getDataType() == DataType::FP32);
    const uint32_t batchCapacity = static_cast<uint32_t>(batchCapacity64);
    THOR_THROW_IF_FALSE(tensor.getTotalNumElements() % batchCapacity == 0);
    const uint64_t elementsPerExample = tensor.getTotalNumElements() / batchCapacity;

    constexpr uint32_t blockSize = 256;
    const uint64_t totalElements = tensor.getTotalNumElements();
    const uint32_t gridSize = static_cast<uint32_t>(std::min<uint64_t>((totalElements + blockSize - 1) / blockSize, 65535));

    ScopedGpu scopedGpu(tensor.getPlacement().getDeviceNum());
    writeBatchValidityMaskKernel<<<gridSize, blockSize, 0, stream.getStream()>>>(
        tensor.getMemPtr<float>(), elementsPerExample, batchCapacity, validExampleCount);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
