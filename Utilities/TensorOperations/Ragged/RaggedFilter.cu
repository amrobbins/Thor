#include "Utilities/TensorOperations/Ragged/RaggedFilter.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerBlock = kThreads / kWarpSize;
constexpr uint32_t kMaxPortableBlocks = 65535;

static_assert(kThreads % kWarpSize == 0, "RaggedFilter block size must contain complete warps.");

template <typename OffsetT>
__global__ void filterRowLengthsKernel(const unsigned char* maskValues,
                                       const OffsetT* inputOffsets,
                                       OffsetT* outputLengths,
                                       uint64_t batchSize) {
    __shared__ unsigned long long selectedCount;
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        if (threadIdx.x == 0) selectedCount = 0;
        __syncthreads();

        const uint64_t rowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t rowEnd = static_cast<uint64_t>(inputOffsets[row + 1]);
        unsigned long long localCount = 0;
        for (uint64_t token = rowBegin + threadIdx.x; token < rowEnd; token += blockDim.x) {
            localCount += maskValues[token] != 0 ? 1ULL : 0ULL;
        }
        if (localCount != 0) atomicAdd(&selectedCount, localCount);
        __syncthreads();

        if (threadIdx.x == 0) outputLengths[row] = static_cast<OffsetT>(selectedCount);
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t laneId() { return threadIdx.x & (kWarpSize - 1U); }
__device__ __forceinline__ uint32_t warpId() { return threadIdx.x / kWarpSize; }

// Compute the stable compacted rank of each selected lane in one block-sized
// tile. All threads in the block must call this helper. The returned rank is
// relative to the current tile; tileSelectedCount is valid after the final
// synchronization.
__device__ __forceinline__ uint32_t stableTileRank(bool selected,
                                                   uint32_t* warpCounts,
                                                   uint32_t* warpPrefixes,
                                                   uint32_t& tileSelectedCount) {
    const uint32_t lane = laneId();
    const uint32_t warp = warpId();
    const unsigned int ballot = __ballot_sync(0xffffffffU, selected);
    const unsigned int lowerMask = lane == 0 ? 0U : ((1U << lane) - 1U);
    const uint32_t laneRank = static_cast<uint32_t>(__popc(ballot & lowerMask));
    if (lane == 0) warpCounts[warp] = static_cast<uint32_t>(__popc(ballot));
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t prefix = 0;
        for (uint32_t w = 0; w < kWarpsPerBlock; ++w) {
            warpPrefixes[w] = prefix;
            prefix += warpCounts[w];
        }
        tileSelectedCount = prefix;
    }
    __syncthreads();
    return warpPrefixes[warp] + laneRank;
}

template <typename OffsetT>
__global__ void filterValuesKernel(const unsigned char* inputValues,
                                   const unsigned char* maskValues,
                                   const OffsetT* inputOffsets,
                                   const OffsetT* outputOffsets,
                                   unsigned char* outputValues,
                                   unsigned long valueElementSizeBytes,
                                   uint64_t elementsPerValue,
                                   uint64_t batchSize) {
    __shared__ uint32_t warpCounts[kWarpsPerBlock];
    __shared__ uint32_t warpPrefixes[kWarpsPerBlock];
    __shared__ uint32_t tileSelectedCount;
    __shared__ uint64_t selectedBeforeTile;

    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t inputRowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t inputRowEnd = static_cast<uint64_t>(inputOffsets[row + 1]);
        const uint64_t outputRowBegin = static_cast<uint64_t>(outputOffsets[row]);
        if (threadIdx.x == 0) selectedBeforeTile = 0;
        __syncthreads();

        for (uint64_t tileBegin = inputRowBegin; tileBegin < inputRowEnd; tileBegin += blockDim.x) {
            const uint64_t inputToken = tileBegin + threadIdx.x;
            const bool selected = inputToken < inputRowEnd && maskValues[inputToken] != 0;
            const uint32_t rankInTile =
                stableTileRank(selected, warpCounts, warpPrefixes, tileSelectedCount);

            if (selected) {
                const uint64_t outputToken = outputRowBegin + selectedBeforeTile + rankInTile;
                const uint64_t inputScalarBegin = inputToken * elementsPerValue;
                const uint64_t outputScalarBegin = outputToken * elementsPerValue;
                for (uint64_t scalar = 0; scalar < elementsPerValue; ++scalar) {
                    const unsigned char* source =
                        inputValues + (inputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                    unsigned char* destination =
                        outputValues + (outputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                    for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) selectedBeforeTile += tileSelectedCount;
            __syncthreads();
        }
    }
}

template <typename OffsetT>
__global__ void zeroActiveInputGradientKernel(const OffsetT* inputOffsets,
                                              unsigned char* inputGradient,
                                              unsigned long valueElementSizeBytes,
                                              uint64_t elementsPerValue,
                                              uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t rowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t rowEnd = static_cast<uint64_t>(inputOffsets[row + 1]);
        const uint64_t scalarBegin = rowBegin * elementsPerValue;
        const uint64_t scalarElements = (rowEnd - rowBegin) * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarElements; scalar += blockDim.x) {
            unsigned char* destination =
                inputGradient + (scalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = 0;
        }
    }
}

template <typename OffsetT>
__global__ void scatterFilteredGradientKernel(const unsigned char* maskValues,
                                              const OffsetT* inputOffsets,
                                              const OffsetT* outputOffsets,
                                              const unsigned char* outputGradient,
                                              unsigned char* inputGradient,
                                              unsigned long valueElementSizeBytes,
                                              uint64_t elementsPerValue,
                                              uint64_t batchSize) {
    __shared__ uint32_t warpCounts[kWarpsPerBlock];
    __shared__ uint32_t warpPrefixes[kWarpsPerBlock];
    __shared__ uint32_t tileSelectedCount;
    __shared__ uint64_t selectedBeforeTile;

    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t inputRowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t inputRowEnd = static_cast<uint64_t>(inputOffsets[row + 1]);
        const uint64_t outputRowBegin = static_cast<uint64_t>(outputOffsets[row]);
        if (threadIdx.x == 0) selectedBeforeTile = 0;
        __syncthreads();

        for (uint64_t tileBegin = inputRowBegin; tileBegin < inputRowEnd; tileBegin += blockDim.x) {
            const uint64_t inputToken = tileBegin + threadIdx.x;
            const bool selected = inputToken < inputRowEnd && maskValues[inputToken] != 0;
            const uint32_t rankInTile =
                stableTileRank(selected, warpCounts, warpPrefixes, tileSelectedCount);

            if (selected) {
                const uint64_t outputToken = outputRowBegin + selectedBeforeTile + rankInTile;
                const uint64_t inputScalarBegin = inputToken * elementsPerValue;
                const uint64_t outputScalarBegin = outputToken * elementsPerValue;
                for (uint64_t scalar = 0; scalar < elementsPerValue; ++scalar) {
                    const unsigned char* source =
                        outputGradient + (outputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                    unsigned char* destination =
                        inputGradient + (inputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                    for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) selectedBeforeTile += tileSelectedCount;
            __syncthreads();
        }
    }
}

uint32_t blocksForRows(uint64_t batchSize) {
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(batchSize, 1), kMaxPortableBlocks));
}

void requireGpu(const Tensor& tensor, const char* name) {
    if (!tensor.isInitialized() || tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string("RaggedFilter ") + name + " must be an initialized GPU tensor.");
    }
}

void requireSamePlacement(const Tensor& a, const Tensor& b, const char* what) {
    if (a.getPlacement() != b.getPlacement()) {
        throw std::invalid_argument(std::string("RaggedFilter ") + what + " must share one GPU placement.");
    }
}

void validateOffsets(const Tensor& offsets, uint64_t batchSize, const char* name) {
    requireGpu(offsets, name);
    if (!isCanonicalRowPartitionOffsetDataType(offsets.getDataType())) {
        throw std::invalid_argument(std::string("RaggedFilter ") + name + " must use UINT32 or UINT64 offsets.");
    }
    if (offsets.getDimensions() != std::vector<uint64_t>{batchSize + 1}) {
        throw std::invalid_argument(std::string("RaggedFilter ") + name + " must have shape [batch_size + 1].");
    }
}

uint64_t elementsPerValue(const Tensor& values) {
    const std::vector<uint64_t> dimensions = values.getDimensions();
    if (dimensions.empty()) throw std::invalid_argument("RaggedFilter values must have a packed leading dimension.");
    uint64_t elements = 1;
    for (uint64_t d = 1; d < dimensions.size(); ++d) {
        if (elements > std::numeric_limits<uint64_t>::max() / dimensions[d]) {
            throw std::overflow_error("RaggedFilter trailing value element count overflow.");
        }
        elements *= dimensions[d];
    }
    return elements;
}

void validateMask(const Tensor& maskValues, const Tensor& inputOffsets) {
    requireGpu(maskValues, "mask values");
    requireSamePlacement(maskValues, inputOffsets, "mask/offsets");
    if (maskValues.getDataType() != DataType::BOOLEAN) {
        throw std::invalid_argument("RaggedFilter mask values must use BOOLEAN dtype.");
    }
    const std::vector<uint64_t> dimensions = maskValues.getDimensions();
    if (dimensions.size() != 1) {
        throw std::invalid_argument("RaggedFilter mask values must have shape [max_total_values].");
    }
}

void validateValuesPair(const Tensor& inputValues, const Tensor& outputValues) {
    requireGpu(inputValues, "input values");
    requireGpu(outputValues, "output values");
    requireSamePlacement(inputValues, outputValues, "input/output values");
    const std::vector<uint64_t> inputDimensions = inputValues.getDimensions();
    const std::vector<uint64_t> outputDimensions = outputValues.getDimensions();
    if (inputValues.getDataType() != outputValues.getDataType() || inputDimensions.size() != outputDimensions.size() ||
        inputDimensions.empty()) {
        throw std::invalid_argument("RaggedFilter input/output values must share dtype and trailing rank.");
    }
    if (inputDimensions[0] != outputDimensions[0]) {
        throw std::invalid_argument("RaggedFilter input/output packed capacities must match.");
    }
    if (!std::equal(inputDimensions.begin() + 1, inputDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedFilter input/output values must share identical trailing dimensions.");
    }
}

void validateMaskCapacity(const Tensor& maskValues, const Tensor& inputValues) {
    const std::vector<uint64_t> maskDimensions = maskValues.getDimensions();
    const std::vector<uint64_t> inputDimensions = inputValues.getDimensions();
    if (maskDimensions.empty() || inputDimensions.empty() || maskDimensions[0] != inputDimensions[0]) {
        throw std::invalid_argument("RaggedFilter mask packed capacity must equal the feature packed capacity.");
    }
}

template <typename OffsetT>
void launchLengthsTyped(const Tensor& maskValues,
                        const Tensor& inputOffsets,
                        Tensor& outputLengths,
                        uint64_t batchSize,
                        Stream& stream) {
    filterRowLengthsKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        static_cast<const unsigned char*>(maskValues.getMemPtr()),
        inputOffsets.getMemPtr<OffsetT>(),
        outputLengths.getMemPtr<OffsetT>(),
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchValuesTyped(const Tensor& inputValues,
                       const Tensor& maskValues,
                       const Tensor& inputOffsets,
                       const Tensor& outputOffsets,
                       Tensor& outputValues,
                       uint64_t batchSize,
                       Stream& stream) {
    filterValuesKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        static_cast<const unsigned char*>(inputValues.getMemPtr()),
        static_cast<const unsigned char*>(maskValues.getMemPtr()),
        inputOffsets.getMemPtr<OffsetT>(),
        outputOffsets.getMemPtr<OffsetT>(),
        static_cast<unsigned char*>(outputValues.getMemPtr()),
        static_cast<unsigned long>(TensorDescriptor::getElementSizeInBytes(inputValues.getDataType())),
        elementsPerValue(inputValues),
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchBackwardTyped(const Tensor& maskValues,
                         const Tensor& inputOffsets,
                         const Tensor& outputOffsets,
                         const Tensor& outputGradient,
                         Tensor& inputGradient,
                         uint64_t batchSize,
                         Stream& stream) {
    const unsigned long valueElementSizeBytes =
        static_cast<unsigned long>(TensorDescriptor::getElementSizeInBytes(inputGradient.getDataType()));
    const uint64_t trailingElements = elementsPerValue(inputGradient);
    zeroActiveInputGradientKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        inputOffsets.getMemPtr<OffsetT>(),
        static_cast<unsigned char*>(inputGradient.getMemPtr()),
        valueElementSizeBytes,
        trailingElements,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
    scatterFilteredGradientKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        static_cast<const unsigned char*>(maskValues.getMemPtr()),
        inputOffsets.getMemPtr<OffsetT>(),
        outputOffsets.getMemPtr<OffsetT>(),
        static_cast<const unsigned char*>(outputGradient.getMemPtr()),
        static_cast<unsigned char*>(inputGradient.getMemPtr()),
        valueElementSizeBytes,
        trailingElements,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchRaggedFilterRowLengths(const Tensor& mask_values,
                                  const Tensor& input_offsets,
                                  Tensor& output_lengths,
                                  uint64_t batch_size,
                                  Stream& stream) {
    validateOffsets(input_offsets, batch_size, "input offsets");
    validateMask(mask_values, input_offsets);
    requireGpu(output_lengths, "row lengths");
    requireSamePlacement(input_offsets, output_lengths, "input offsets/row lengths");
    if (output_lengths.getDataType() != input_offsets.getDataType() ||
        output_lengths.getDimensions() != std::vector<uint64_t>{batch_size}) {
        throw std::invalid_argument("RaggedFilter row lengths must match offsets dtype and have shape [batch_size].");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchLengthsTyped<uint32_t>(mask_values, input_offsets, output_lengths, batch_size, stream);
            return;
        case DataType::UINT64:
            launchLengthsTyped<uint64_t>(mask_values, input_offsets, output_lengths, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedFilter offsets must use UINT32 or UINT64 storage.");
    }
}

void launchRaggedFilterValues(const Tensor& input_values,
                              const Tensor& mask_values,
                              const Tensor& input_offsets,
                              const Tensor& output_offsets,
                              Tensor& output_values,
                              uint64_t batch_size,
                              Stream& stream) {
    validateOffsets(input_offsets, batch_size, "input offsets");
    validateOffsets(output_offsets, batch_size, "output offsets");
    requireSamePlacement(input_offsets, output_offsets, "input/output offsets");
    if (input_offsets.getDataType() != output_offsets.getDataType()) {
        throw std::invalid_argument("RaggedFilter input/output offsets must share dtype.");
    }
    validateMask(mask_values, input_offsets);
    validateValuesPair(input_values, output_values);
    requireSamePlacement(input_values, input_offsets, "values/offsets");
    validateMaskCapacity(mask_values, input_values);

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchValuesTyped<uint32_t>(
                input_values, mask_values, input_offsets, output_offsets, output_values, batch_size, stream);
            return;
        case DataType::UINT64:
            launchValuesTyped<uint64_t>(
                input_values, mask_values, input_offsets, output_offsets, output_values, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedFilter offsets must use UINT32 or UINT64 storage.");
    }
}

void launchRaggedFilterBackward(const Tensor& mask_values,
                                const Tensor& input_offsets,
                                const Tensor& output_offsets,
                                const Tensor& output_gradient,
                                Tensor& input_gradient,
                                uint64_t batch_size,
                                Stream& stream) {
    validateOffsets(input_offsets, batch_size, "input offsets");
    validateOffsets(output_offsets, batch_size, "output offsets");
    requireSamePlacement(input_offsets, output_offsets, "input/output offsets");
    if (input_offsets.getDataType() != output_offsets.getDataType()) {
        throw std::invalid_argument("RaggedFilter backward input/output offsets must share dtype.");
    }
    validateMask(mask_values, input_offsets);
    validateValuesPair(input_gradient, output_gradient);
    requireSamePlacement(input_gradient, input_offsets, "gradient/offsets");
    validateMaskCapacity(mask_values, input_gradient);

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchBackwardTyped<uint32_t>(
                mask_values, input_offsets, output_offsets, output_gradient, input_gradient, batch_size, stream);
            return;
        case DataType::UINT64:
            launchBackwardTyped<uint64_t>(
                mask_values, input_offsets, output_offsets, output_gradient, input_gradient, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedFilter backward offsets must use UINT32 or UINT64 storage.");
    }
}

}  // namespace ThorImplementation
