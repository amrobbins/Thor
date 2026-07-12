#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

__global__ void gatherRowsByByteKernel(const uint8_t *__restrict__ source,
                                       uint8_t *__restrict__ destination,
                                       const uint64_t *__restrict__ rowIndices,
                                       uint64_t batchSize,
                                       uint64_t rowBytes,
                                       uint64_t sourceRows,
                                       uint64_t totalBytes) {
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                            static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t offset = linear; offset < totalBytes; offset += stride) {
        const uint64_t batchRow = offset / rowBytes;
        const uint64_t rowByteOffset = offset - batchRow * rowBytes;
        if (batchRow >= batchSize) {
            return;
        }
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow < sourceRows) {
            destination[offset] = source[sourceRow * rowBytes + rowByteOffset];
        }
    }
}

void validateGatherTensorShapes(const Tensor &source, const Tensor &destination, const Tensor &rowIndicesDevice) {
    THOR_THROW_IF_FALSE(source.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(source.getPlacement() == destination.getPlacement());
    THOR_THROW_IF_FALSE(source.getPlacement() == rowIndicesDevice.getPlacement());
    THOR_THROW_IF_FALSE(source.getDataType() == destination.getDataType());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == ThorImplementation::DataType::UINT64);

    const std::vector<uint64_t> sourceDims = source.getDimensions();
    const std::vector<uint64_t> destDims = destination.getDimensions();
    const std::vector<uint64_t> indexDims = rowIndicesDevice.getDimensions();
    THOR_THROW_IF_FALSE(!sourceDims.empty());
    THOR_THROW_IF_FALSE(!destDims.empty());
    THOR_THROW_IF_FALSE(sourceDims.size() == destDims.size());
    THOR_THROW_IF_FALSE(indexDims.size() == 1);
    THOR_THROW_IF_FALSE(indexDims.at(0) == destDims.at(0));
    THOR_THROW_IF_FALSE(sourceDims.at(0) > 0);
    THOR_THROW_IF_FALSE(destDims.at(0) > 0);
    for (uint64_t i = 1; i < sourceDims.size(); ++i) {
        THOR_THROW_IF_FALSE(sourceDims.at(i) == destDims.at(i));
    }
}

}  // namespace

void launchDeviceResidentNamedGatherKernel(const Tensor &source, Tensor &destination, const Tensor &rowIndicesDevice, Stream &stream) {
    validateGatherTensorShapes(source, destination, rowIndicesDevice);

    const uint64_t batchSize = destination.getDimensions().at(0);
    const uint64_t sourceRows = source.getDimensions().at(0);
    const uint64_t rowBytes = destination.getArraySizeInBytes() / batchSize;
    THOR_THROW_IF_FALSE(rowBytes > 0);
    const uint64_t totalBytes = destination.getArraySizeInBytes();
    THOR_THROW_IF_FALSE(totalBytes == batchSize * rowBytes);

    constexpr int threadsPerBlock = 256;
    uint64_t blocks64 = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;
    blocks64 = std::max<uint64_t>(1, std::min<uint64_t>(blocks64, 65535));
    const int blocks = static_cast<int>(blocks64);

    gatherRowsByByteKernel<<<blocks, threadsPerBlock, 0, stream.getStream()>>>(static_cast<const uint8_t *>(source.getMemPtr()),
                                                                              static_cast<uint8_t *>(destination.getMemPtr()),
                                                                              rowIndicesDevice.getMemPtr<uint64_t>(),
                                                                              batchSize,
                                                                              rowBytes,
                                                                              sourceRows,
                                                                              totalBytes);
    CUDA_CHECK(cudaGetLastError());
}
