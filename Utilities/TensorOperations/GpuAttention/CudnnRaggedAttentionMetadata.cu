#include "Utilities/TensorOperations/GpuAttention/CudnnRaggedAttentionMetadata.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {

template <typename OffsetT>
__global__ void canonicalRowPartitionToCudnnAttentionKernel(const OffsetT* offsets,
                                                            int32_t* sequenceLengths,
                                                            int32_t* firstElementOffsets,
                                                            int32_t* secondElementOffsets,
                                                            uint64_t batchSize,
                                                            uint64_t firstElementsPerToken,
                                                            uint64_t secondElementsPerToken) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i > batchSize) {
        return;
    }

    const uint64_t tokenOffset = static_cast<uint64_t>(offsets[i]);
    firstElementOffsets[i] = static_cast<int32_t>(tokenOffset * firstElementsPerToken);
    secondElementOffsets[i] = static_cast<int32_t>(tokenOffset * secondElementsPerToken);
    if (i < batchSize) {
        sequenceLengths[i] = static_cast<int32_t>(static_cast<uint64_t>(offsets[i + 1]) - tokenOffset);
    }
}

void requireGpuTensor(const Tensor& tensor, const char* name, int gpuNum) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("cuDNN ragged attention metadata tensor '") + name + "' is uninitialized.");
    }
    const TensorPlacement placement = tensor.getPlacement();
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU || placement.getDeviceNum() != gpuNum) {
        throw std::invalid_argument(std::string("cuDNN ragged attention metadata tensor '") + name +
                                    "' must be on the attention stream GPU.");
    }
}

void requireInt32Tensor(const Tensor& tensor, const std::vector<uint64_t>& dimensions, const char* name, int gpuNum) {
    requireGpuTensor(tensor, name, gpuNum);
    if (tensor.getDataType() != DataType::INT32 || tensor.getDimensions() != dimensions) {
        throw std::invalid_argument(std::string("cuDNN ragged attention metadata tensor '") + name +
                                    "' must be INT32 with the expected dimensions.");
    }
}

void requireElementCapacityFitsInt32(uint64_t maxTotalValues, uint64_t elementsPerToken, const char* label) {
    if (elementsPerToken == 0) {
        throw std::invalid_argument(std::string(label) + " elements-per-token must be positive.");
    }
    if (maxTotalValues > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / elementsPerToken) {
        throw std::invalid_argument(std::string(label) + " exceeds cuDNN INT32 ragged element-offset capacity.");
    }
}

}  // namespace

void convertCanonicalRowPartitionForCudnnAttention(const Tensor& canonicalOffsets,
                                                    uint64_t batchSize,
                                                    uint64_t maxTotalValues,
                                                    uint64_t firstElementsPerToken,
                                                    uint64_t secondElementsPerToken,
                                                    Tensor sequenceLengths,
                                                    Tensor firstElementOffsets,
                                                    Tensor secondElementOffsets,
                                                    Stream stream) {
    if (batchSize == 0 || batchSize > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("cuDNN ragged attention batch size must fit INT32 and be non-zero.");
    }
    if (maxTotalValues == 0 || maxTotalValues > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("cuDNN ragged attention token capacity must fit INT32 and be non-zero.");
    }
    requireElementCapacityFitsInt32(maxTotalValues, firstElementsPerToken, "First cuDNN ragged offset");
    requireElementCapacityFitsInt32(maxTotalValues, secondElementsPerToken, "Second cuDNN ragged offset");

    const int gpuNum = stream.getGpuNum();
    requireGpuTensor(canonicalOffsets, "canonical_offsets", gpuNum);
    if (!isCanonicalRowPartitionOffsetDataType(canonicalOffsets.getDataType())) {
        throw std::invalid_argument("cuDNN ragged attention canonical offsets must use UINT32 or UINT64.");
    }
    if (canonicalOffsets.getDimensions() != std::vector<uint64_t>{batchSize + 1}) {
        throw std::invalid_argument("cuDNN ragged attention canonical offsets must have shape [batch+1].");
    }
    requireInt32Tensor(sequenceLengths, {batchSize}, "sequence_lengths", gpuNum);
    requireInt32Tensor(firstElementOffsets, {batchSize + 1}, "first_element_offsets", gpuNum);
    requireInt32Tensor(secondElementOffsets, {batchSize + 1}, "second_element_offsets", gpuNum);

    ScopedGpu scopedGpu(gpuNum);
    constexpr uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>((batchSize + 1 + threads - 1) / threads);
    if (canonicalOffsets.getDataType() == DataType::UINT32) {
        canonicalRowPartitionToCudnnAttentionKernel<<<blocks, threads, 0, stream.getStream()>>>(
            canonicalOffsets.getMemPtr<uint32_t>(),
            sequenceLengths.getMemPtr<int32_t>(),
            firstElementOffsets.getMemPtr<int32_t>(),
            secondElementOffsets.getMemPtr<int32_t>(),
            batchSize,
            firstElementsPerToken,
            secondElementsPerToken);
    } else {
        canonicalRowPartitionToCudnnAttentionKernel<<<blocks, threads, 0, stream.getStream()>>>(
            canonicalOffsets.getMemPtr<uint64_t>(),
            sequenceLengths.getMemPtr<int32_t>(),
            firstElementOffsets.getMemPtr<int32_t>(),
            secondElementOffsets.getMemPtr<int32_t>(),
            batchSize,
            firstElementsPerToken,
            secondElementsPerToken);
    }
    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to launch cuDNN ragged attention metadata conversion: ") +
                                 cudaGetErrorString(launchStatus));
    }
}

}  // namespace ThorImplementation
