#include "Utilities/TensorOperations/Ragged/RaggedSequenceSlice.h"

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
constexpr uint32_t kMaxPortableBlocks = 65535;

template <typename OffsetT>
__device__ __forceinline__ uint64_t clippedSliceLength(const OffsetT* offsets,
                                                       uint64_t row,
                                                       uint64_t start,
                                                       uint64_t length) {
    const uint64_t rowBegin = static_cast<uint64_t>(offsets[row]);
    const uint64_t rowEnd = static_cast<uint64_t>(offsets[row + 1]);
    const uint64_t rowLength = rowEnd - rowBegin;
    if (rowLength <= start) return 0;
    const uint64_t available = rowLength - start;
    return available < length ? available : length;
}

template <typename OffsetT>
__global__ void sliceRowLengthsKernel(const OffsetT* inputOffsets,
                                      OffsetT* outputLengths,
                                      uint64_t start,
                                      uint64_t length,
                                      uint64_t batchSize) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < batchSize;
         row += stride) {
        outputLengths[row] = static_cast<OffsetT>(clippedSliceLength(inputOffsets, row, start, length));
    }
}

template <typename OffsetT>
__global__ void sliceValuesKernel(const unsigned char* inputValues,
                                  const OffsetT* inputOffsets,
                                  const OffsetT* outputOffsets,
                                  unsigned char* outputValues,
                                  unsigned long valueElementSizeBytes,
                                  uint64_t elementsPerValue,
                                  uint64_t start,
                                  uint64_t length,
                                  uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t sourceRowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t sliceLength = clippedSliceLength(inputOffsets, row, start, length);
        const uint64_t destinationRowBegin = static_cast<uint64_t>(outputOffsets[row]);
        if (sliceLength == 0) continue;
        const uint64_t sourceValueBegin = sourceRowBegin + start;
        const uint64_t scalarElements = sliceLength * elementsPerValue;
        const uint64_t sourceScalarBegin = sourceValueBegin * elementsPerValue;
        const uint64_t destinationScalarBegin = destinationRowBegin * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarElements; scalar += blockDim.x) {
            const unsigned char* source =
                inputValues + (sourceScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            unsigned char* destination =
                outputValues + (destinationScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
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
__global__ void scatterSliceGradientKernel(const OffsetT* inputOffsets,
                                           const OffsetT* outputOffsets,
                                           const unsigned char* outputGradient,
                                           unsigned char* inputGradient,
                                           unsigned long valueElementSizeBytes,
                                           uint64_t elementsPerValue,
                                           uint64_t start,
                                           uint64_t length,
                                           uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t inputRowBegin = static_cast<uint64_t>(inputOffsets[row]);
        const uint64_t sliceLength = clippedSliceLength(inputOffsets, row, start, length);
        const uint64_t outputRowBegin = static_cast<uint64_t>(outputOffsets[row]);
        if (sliceLength == 0) continue;
        const uint64_t inputSliceBegin = inputRowBegin + start;
        const uint64_t scalarElements = sliceLength * elementsPerValue;
        const uint64_t inputScalarBegin = inputSliceBegin * elementsPerValue;
        const uint64_t outputScalarBegin = outputRowBegin * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarElements; scalar += blockDim.x) {
            const unsigned char* source =
                outputGradient + (outputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            unsigned char* destination =
                inputGradient + (inputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
        }
    }
}

uint32_t blocksForItems(uint64_t items) {
    if (items == 0) return 1;
    const uint64_t needed = (items + kThreads - 1U) / kThreads;
    return static_cast<uint32_t>(std::min<uint64_t>(needed, kMaxPortableBlocks));
}

uint32_t blocksForRows(uint64_t batchSize) {
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(batchSize, 1), kMaxPortableBlocks));
}

void requireGpu(const Tensor& tensor, const char* name) {
    if (!tensor.isInitialized() || tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string("RaggedSequenceSlice ") + name + " must be an initialized GPU tensor.");
    }
}

void requireSamePlacement(const Tensor& a, const Tensor& b, const char* what) {
    if (a.getPlacement() != b.getPlacement()) {
        throw std::invalid_argument(std::string("RaggedSequenceSlice ") + what + " must share one GPU placement.");
    }
}

void validateOffsets(const Tensor& offsets, uint64_t batchSize, const char* name) {
    requireGpu(offsets, name);
    if (!isCanonicalRowPartitionOffsetDataType(offsets.getDataType())) {
        throw std::invalid_argument(std::string("RaggedSequenceSlice ") + name + " must use UINT32 or UINT64 offsets.");
    }
    if (offsets.getDimensions() != std::vector<uint64_t>{batchSize + 1}) {
        throw std::invalid_argument(std::string("RaggedSequenceSlice ") + name + " must have shape [batch_size + 1].");
    }
}

uint64_t elementsPerValue(const Tensor& values) {
    const std::vector<uint64_t> dimensions = values.getDimensions();
    if (dimensions.empty()) throw std::invalid_argument("RaggedSequenceSlice values must have a packed leading dimension.");
    uint64_t elements = 1;
    for (uint64_t d = 1; d < dimensions.size(); ++d) {
        if (elements > std::numeric_limits<uint64_t>::max() / dimensions[d]) {
            throw std::overflow_error("RaggedSequenceSlice trailing value element count overflow.");
        }
        elements *= dimensions[d];
    }
    return elements;
}

void validateValuesPair(const Tensor& inputValues, const Tensor& outputValues) {
    requireGpu(inputValues, "input values");
    requireGpu(outputValues, "output values");
    requireSamePlacement(inputValues, outputValues, "input/output values");
    const std::vector<uint64_t> inputDimensions = inputValues.getDimensions();
    const std::vector<uint64_t> outputDimensions = outputValues.getDimensions();
    if (inputValues.getDataType() != outputValues.getDataType() || inputDimensions.size() != outputDimensions.size() ||
        inputDimensions.empty()) {
        throw std::invalid_argument("RaggedSequenceSlice input/output values must share dtype and trailing rank.");
    }
    if (!std::equal(inputDimensions.begin() + 1, inputDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedSequenceSlice input/output values must share identical trailing dimensions.");
    }
}

template <typename OffsetT>
void launchLengthsTyped(const Tensor& inputOffsets,
                        Tensor& outputLengths,
                        uint64_t start,
                        uint64_t length,
                        uint64_t batchSize,
                        Stream& stream) {
    sliceRowLengthsKernel<OffsetT><<<blocksForItems(batchSize), kThreads, 0, stream.getStream()>>>(
        inputOffsets.getMemPtr<OffsetT>(), outputLengths.getMemPtr<OffsetT>(), start, length, batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchValuesTyped(const Tensor& inputValues,
                       const Tensor& inputOffsets,
                       const Tensor& outputOffsets,
                       Tensor& outputValues,
                       uint64_t start,
                       uint64_t length,
                       uint64_t batchSize,
                       Stream& stream) {
    sliceValuesKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        static_cast<const unsigned char*>(inputValues.getMemPtr()),
        inputOffsets.getMemPtr<OffsetT>(),
        outputOffsets.getMemPtr<OffsetT>(),
        static_cast<unsigned char*>(outputValues.getMemPtr()),
        static_cast<unsigned long>(TensorDescriptor::getElementSizeInBytes(inputValues.getDataType())),
        elementsPerValue(inputValues),
        start,
        length,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void launchBackwardTyped(const Tensor& inputOffsets,
                         const Tensor& outputOffsets,
                         const Tensor& outputGradient,
                         Tensor& inputGradient,
                         uint64_t start,
                         uint64_t length,
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
    scatterSliceGradientKernel<OffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        inputOffsets.getMemPtr<OffsetT>(),
        outputOffsets.getMemPtr<OffsetT>(),
        static_cast<const unsigned char*>(outputGradient.getMemPtr()),
        static_cast<unsigned char*>(inputGradient.getMemPtr()),
        valueElementSizeBytes,
        trailingElements,
        start,
        length,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchRaggedSequenceSliceRowLengths(const Tensor& input_offsets,
                                         Tensor& output_lengths,
                                         uint64_t start,
                                         uint64_t length,
                                         uint64_t batch_size,
                                         Stream& stream) {
    if (length == 0) throw std::invalid_argument("RaggedSequenceSlice length must be greater than zero.");
    validateOffsets(input_offsets, batch_size, "input offsets");
    requireGpu(output_lengths, "row lengths");
    requireSamePlacement(input_offsets, output_lengths, "input offsets/row lengths");
    if (output_lengths.getDataType() != input_offsets.getDataType() ||
        output_lengths.getDimensions() != std::vector<uint64_t>{batch_size}) {
        throw std::invalid_argument("RaggedSequenceSlice row lengths must match offsets dtype and have shape [batch_size].");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchLengthsTyped<uint32_t>(input_offsets, output_lengths, start, length, batch_size, stream);
            return;
        case DataType::UINT64:
            launchLengthsTyped<uint64_t>(input_offsets, output_lengths, start, length, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedSequenceSlice offsets must use UINT32 or UINT64 storage.");
    }
}

void launchRaggedSequenceSliceValues(const Tensor& input_values,
                                     const Tensor& input_offsets,
                                     const Tensor& output_offsets,
                                     Tensor& output_values,
                                     uint64_t start,
                                     uint64_t length,
                                     uint64_t batch_size,
                                     Stream& stream) {
    if (length == 0) throw std::invalid_argument("RaggedSequenceSlice length must be greater than zero.");
    validateOffsets(input_offsets, batch_size, "input offsets");
    validateOffsets(output_offsets, batch_size, "output offsets");
    requireSamePlacement(input_offsets, output_offsets, "input/output offsets");
    if (input_offsets.getDataType() != output_offsets.getDataType()) {
        throw std::invalid_argument("RaggedSequenceSlice input/output offsets must share dtype.");
    }
    validateValuesPair(input_values, output_values);
    requireSamePlacement(input_values, input_offsets, "values/offsets");

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchValuesTyped<uint32_t>(input_values, input_offsets, output_offsets, output_values, start, length, batch_size, stream);
            return;
        case DataType::UINT64:
            launchValuesTyped<uint64_t>(input_values, input_offsets, output_offsets, output_values, start, length, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedSequenceSlice offsets must use UINT32 or UINT64 storage.");
    }
}

void launchRaggedSequenceSliceBackward(const Tensor& input_offsets,
                                       const Tensor& output_offsets,
                                       const Tensor& output_gradient,
                                       Tensor& input_gradient,
                                       uint64_t start,
                                       uint64_t length,
                                       uint64_t batch_size,
                                       Stream& stream) {
    if (length == 0) throw std::invalid_argument("RaggedSequenceSlice backward length must be greater than zero.");
    validateOffsets(input_offsets, batch_size, "input offsets");
    validateOffsets(output_offsets, batch_size, "output offsets");
    requireSamePlacement(input_offsets, output_offsets, "input/output offsets");
    if (input_offsets.getDataType() != output_offsets.getDataType()) {
        throw std::invalid_argument("RaggedSequenceSlice backward input/output offsets must share dtype.");
    }
    validateValuesPair(input_gradient, output_gradient);
    requireSamePlacement(input_gradient, input_offsets, "gradient/offsets");

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (input_offsets.getDataType()) {
        case DataType::UINT32:
            launchBackwardTyped<uint32_t>(
                input_offsets, output_offsets, output_gradient, input_gradient, start, length, batch_size, stream);
            return;
        case DataType::UINT64:
            launchBackwardTyped<uint64_t>(
                input_offsets, output_offsets, output_gradient, input_gradient, start, length, batch_size, stream);
            return;
        default:
            throw std::invalid_argument("RaggedSequenceSlice backward offsets must use UINT32 or UINT64 storage.");
    }
}

}  // namespace ThorImplementation
