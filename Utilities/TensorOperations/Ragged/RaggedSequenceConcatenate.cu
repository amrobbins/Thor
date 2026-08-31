#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxPortableBlocks = 65535;

template <typename OffsetT>
__device__ __forceinline__ uint64_t offsetAt(void *const *offsets, uint32_t input, uint64_t row) {
    return static_cast<uint64_t>(reinterpret_cast<const OffsetT *>(offsets[input])[row]);
}

template <typename OffsetT>
__global__ void produceOutputOffsetsKernel(OffsetT *outputOffsets,
                                           void *const *inputOffsets,
                                           uint32_t numInputs,
                                           uint64_t batchSize) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row <= batchSize;
         row += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        uint64_t outputOffset = 0;
        for (uint32_t input = 0; input < numInputs; ++input) {
            outputOffset += offsetAt<OffsetT>(inputOffsets, input, row);
        }
        outputOffsets[row] = static_cast<OffsetT>(outputOffset);
    }
}

template <typename OffsetT>
__device__ __forceinline__ void rowPlacement(void *const *inputOffsets,
                                             uint32_t numInputs,
                                             uint32_t input,
                                             uint64_t row,
                                             uint64_t &sourceBegin,
                                             uint64_t &rowLength,
                                             uint64_t &outputBegin) {
    sourceBegin = offsetAt<OffsetT>(inputOffsets, input, row);
    const uint64_t sourceEnd = offsetAt<OffsetT>(inputOffsets, input, row + 1);
    rowLength = sourceEnd - sourceBegin;

    outputBegin = 0;
    for (uint32_t i = 0; i < numInputs; ++i) {
        outputBegin += offsetAt<OffsetT>(inputOffsets, i, row);
    }
    for (uint32_t i = 0; i < input; ++i) {
        const uint64_t begin = offsetAt<OffsetT>(inputOffsets, i, row);
        const uint64_t end = offsetAt<OffsetT>(inputOffsets, i, row + 1);
        outputBegin += end - begin;
    }
}

template <typename OffsetT>
__global__ void concatenateValuesKernel(unsigned char *outputValues,
                                        unsigned char *const *inputValues,
                                        void *const *inputOffsets,
                                        uint32_t numInputs,
                                        unsigned long valueElementSizeBytes,
                                        uint64_t elementsPerValue,
                                        uint64_t batchSize) {
    const uint64_t totalPairs = batchSize * static_cast<uint64_t>(numInputs);
    for (uint64_t pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
        const uint64_t row = pair / numInputs;
        const uint32_t input = static_cast<uint32_t>(pair - row * numInputs);
        uint64_t sourceBegin = 0;
        uint64_t rowLength = 0;
        uint64_t outputBegin = 0;
        rowPlacement<OffsetT>(inputOffsets, numInputs, input, row, sourceBegin, rowLength, outputBegin);

        const uint64_t scalarElements = rowLength * elementsPerValue;
        const uint64_t sourceScalarBegin = sourceBegin * elementsPerValue;
        const uint64_t outputScalarBegin = outputBegin * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarElements; scalar += blockDim.x) {
            const unsigned char *source = inputValues[input] + (sourceScalarBegin + scalar) * valueElementSizeBytes;
            unsigned char *destination = outputValues + (outputScalarBegin + scalar) * valueElementSizeBytes;
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
        }
    }
}

template <typename OffsetT>
__global__ void splitGradientKernel(unsigned char *const *inputGradients,
                                    const unsigned char *outputGradient,
                                    void *const *inputOffsets,
                                    uint32_t numInputs,
                                    unsigned long valueElementSizeBytes,
                                    uint64_t elementsPerValue,
                                    uint64_t batchSize) {
    const uint64_t totalPairs = batchSize * static_cast<uint64_t>(numInputs);
    for (uint64_t pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
        const uint64_t row = pair / numInputs;
        const uint32_t input = static_cast<uint32_t>(pair - row * numInputs);
        if (inputGradients[input] == nullptr) continue;

        uint64_t destinationBegin = 0;
        uint64_t rowLength = 0;
        uint64_t outputBegin = 0;
        rowPlacement<OffsetT>(inputOffsets, numInputs, input, row, destinationBegin, rowLength, outputBegin);

        const uint64_t scalarElements = rowLength * elementsPerValue;
        const uint64_t destinationScalarBegin = destinationBegin * elementsPerValue;
        const uint64_t outputScalarBegin = outputBegin * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarElements; scalar += blockDim.x) {
            const unsigned char *source = outputGradient + (outputScalarBegin + scalar) * valueElementSizeBytes;
            unsigned char *destination = inputGradients[input] + (destinationScalarBegin + scalar) * valueElementSizeBytes;
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
        }
    }
}

uint32_t blocksForItems(uint64_t items, uint32_t threads = kThreads) {
    if (items == 0) return 1;
    const uint64_t needed = (items + threads - 1U) / threads;
    return static_cast<uint32_t>(std::min<uint64_t>(needed, kMaxPortableBlocks));
}

uint32_t blocksForPairs(uint64_t batchSize, uint32_t numInputs) {
    if (numInputs == 0) throw std::invalid_argument("RaggedSequenceConcatenate requires at least one input.");
    if (batchSize > std::numeric_limits<uint64_t>::max() / numInputs) {
        throw std::invalid_argument("RaggedSequenceConcatenate batch/input pair count overflow.");
    }
    const uint64_t pairs = batchSize * static_cast<uint64_t>(numInputs);
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(pairs, 1), kMaxPortableBlocks));
}

void validateOffsetSize(std::size_t bytes) {
    if (bytes != sizeof(uint32_t) && bytes != sizeof(uint64_t)) {
        throw std::invalid_argument("RaggedSequenceConcatenate offsets must use UINT32 or UINT64 storage.");
    }
}

template <typename OffsetT>
void launchForwardTyped(void *outputValues,
                        void *outputOffsets,
                        void *inputValues[],
                        void *inputOffsets[],
                        uint32_t numInputs,
                        std::size_t valueElementSizeBytes,
                        uint64_t elementsPerValue,
                        uint64_t batchSize,
                        Stream stream) {
    produceOutputOffsetsKernel<OffsetT><<<blocksForItems(batchSize + 1), kThreads, 0, stream.getStream()>>>(
        static_cast<OffsetT *>(outputOffsets), inputOffsets, numInputs, batchSize);
    CUDA_CHECK(cudaGetLastError());

    concatenateValuesKernel<OffsetT><<<blocksForPairs(batchSize, numInputs), kThreads, 0, stream.getStream()>>>(
        static_cast<unsigned char *>(outputValues),
        reinterpret_cast<unsigned char **>(inputValues),
        inputOffsets,
        numInputs,
        static_cast<unsigned long>(valueElementSizeBytes),
        elementsPerValue,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchBackwardTyped(void *inputGradients[],
                         const void *outputGradient,
                         void *inputOffsets[],
                         uint32_t numInputs,
                         std::size_t valueElementSizeBytes,
                         uint64_t elementsPerValue,
                         uint64_t batchSize,
                         Stream stream) {
    splitGradientKernel<OffsetT><<<blocksForPairs(batchSize, numInputs), kThreads, 0, stream.getStream()>>>(
        reinterpret_cast<unsigned char **>(inputGradients),
        static_cast<const unsigned char *>(outputGradient),
        inputOffsets,
        numInputs,
        static_cast<unsigned long>(valueElementSizeBytes),
        elementsPerValue,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchRaggedSequenceConcatenate(void *output_values,
                                     void *output_offsets,
                                     void *input_values[],
                                     void *input_offsets[],
                                     uint32_t num_inputs,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     std::size_t offsets_element_size_bytes,
                                     uint64_t batch_size,
                                     Stream stream) {
    if (num_inputs < 2) throw std::invalid_argument("RaggedSequenceConcatenate requires at least two inputs.");
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchForwardTyped<uint32_t>(output_values,
                                     output_offsets,
                                     input_values,
                                     input_offsets,
                                     num_inputs,
                                     value_element_size_bytes,
                                     elements_per_value,
                                     batch_size,
                                     stream);
    } else {
        launchForwardTyped<uint64_t>(output_values,
                                     output_offsets,
                                     input_values,
                                     input_offsets,
                                     num_inputs,
                                     value_element_size_bytes,
                                     elements_per_value,
                                     batch_size,
                                     stream);
    }
}

void launchRaggedSequenceConcatenateBackward(void *input_gradients[],
                                             const void *output_gradient,
                                             void *input_offsets[],
                                             uint32_t num_inputs,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             std::size_t offsets_element_size_bytes,
                                             uint64_t batch_size,
                                             Stream stream) {
    if (num_inputs < 2) throw std::invalid_argument("RaggedSequenceConcatenate backward requires at least two inputs.");
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchBackwardTyped<uint32_t>(input_gradients,
                                      output_gradient,
                                      input_offsets,
                                      num_inputs,
                                      value_element_size_bytes,
                                      elements_per_value,
                                      batch_size,
                                      stream);
    } else {
        launchBackwardTyped<uint64_t>(input_gradients,
                                      output_gradient,
                                      input_offsets,
                                      num_inputs,
                                      value_element_size_bytes,
                                      elements_per_value,
                                      batch_size,
                                      stream);
    }
}
