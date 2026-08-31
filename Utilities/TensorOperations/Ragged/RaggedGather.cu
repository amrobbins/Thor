#include "Utilities/TensorOperations/Ragged/RaggedGather.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxPortableBlocks = 65535;

uint32_t blocksForRows(uint64_t rows) {
    if (rows == 0) return 1;
    return static_cast<uint32_t>(std::min<uint64_t>(rows, kMaxPortableBlocks));
}

void requireGpu(const Tensor& tensor, const char* name) {
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string("RaggedGather ") + name + " must use GPU placement.");
    }
}

void requireSamePlacement(const Tensor& lhs, const Tensor& rhs, const char* names) {
    if (lhs.getPlacement() != rhs.getPlacement()) {
        throw std::invalid_argument(std::string("RaggedGather ") + names + " must share one placement.");
    }
}

void validateOffsets(const Tensor& offsets, uint64_t batchSize, const char* name) {
    requireGpu(offsets, name);
    if (!isCanonicalRowPartitionOffsetDataType(offsets.getDataType())) {
        throw std::invalid_argument(std::string("RaggedGather ") + name + " must use UINT32 or UINT64 offsets.");
    }
    if (offsets.getDimensions() != std::vector<uint64_t>{batchSize + 1}) {
        throw std::invalid_argument(std::string("RaggedGather ") + name + " must have shape [batch_size + 1].");
    }
}

uint64_t elementsPerValue(const Tensor& values) {
    const std::vector<uint64_t> dimensions = values.getDimensions();
    if (dimensions.empty()) throw std::invalid_argument("RaggedGather values must have a packed leading dimension.");
    uint64_t elements = 1;
    for (uint64_t d = 1; d < dimensions.size(); ++d) {
        if (elements > std::numeric_limits<uint64_t>::max() / dimensions[d]) {
            throw std::overflow_error("RaggedGather trailing value element count overflow.");
        }
        elements *= dimensions[d];
    }
    return elements;
}

void validateForward(const Tensor& sourceValues,
                     const Tensor& sourceOffsets,
                     const Tensor& indicesValues,
                     const Tensor& indicesOffsets,
                     const Tensor& outputValues,
                     uint64_t batchSize) {
    validateOffsets(sourceOffsets, batchSize, "source offsets");
    validateOffsets(indicesOffsets, batchSize, "indices offsets");
    requireGpu(sourceValues, "source values");
    requireGpu(indicesValues, "indices values");
    requireGpu(outputValues, "output values");
    requireSamePlacement(sourceValues, sourceOffsets, "source values/offsets");
    requireSamePlacement(sourceValues, indicesValues, "source/indices values");
    requireSamePlacement(sourceValues, indicesOffsets, "source values/indices offsets");
    requireSamePlacement(sourceValues, outputValues, "source/output values");

    if (indicesValues.getDataType() != DataType::UINT32 && indicesValues.getDataType() != DataType::UINT64) {
        throw std::invalid_argument("RaggedGather indices values must use UINT32 or UINT64 dtype.");
    }
    if (indicesValues.getDimensions().size() != 1 ||
        indicesValues.getDimensions()[0] != outputValues.getDimensions()[0]) {
        throw std::invalid_argument("RaggedGather indices must be scalar packed values with the same capacity as output values.");
    }
    const std::vector<uint64_t> sourceDimensions = sourceValues.getDimensions();
    const std::vector<uint64_t> outputDimensions = outputValues.getDimensions();
    if (sourceValues.getDataType() != outputValues.getDataType() || sourceDimensions.size() != outputDimensions.size() ||
        sourceDimensions.empty()) {
        throw std::invalid_argument("RaggedGather source/output values must share dtype and trailing rank.");
    }
    if (!std::equal(sourceDimensions.begin() + 1, sourceDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedGather source/output values must share identical trailing dimensions.");
    }
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT>
__global__ void gatherKernel(const unsigned char* sourceValues,
                             const SourceOffsetT* sourceOffsets,
                             const IndexT* indicesValues,
                             const IndexOffsetT* indicesOffsets,
                             unsigned char* outputValues,
                             unsigned long valueElementSizeBytes,
                             uint64_t elementsPerValue,
                             uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t sourceBegin = static_cast<uint64_t>(sourceOffsets[row]);
        const uint64_t sourceEnd = static_cast<uint64_t>(sourceOffsets[row + 1]);
        const uint64_t sourceLength = sourceEnd - sourceBegin;
        const uint64_t indicesBegin = static_cast<uint64_t>(indicesOffsets[row]);
        const uint64_t indicesEnd = static_cast<uint64_t>(indicesOffsets[row + 1]);

        for (uint64_t outputToken = indicesBegin + threadIdx.x; outputToken < indicesEnd; outputToken += blockDim.x) {
            const uint64_t localIndex = static_cast<uint64_t>(indicesValues[outputToken]);
            const uint64_t outputScalarBegin = outputToken * elementsPerValue;
            if (localIndex >= sourceLength) {
                // Guard invalid runtime data from crossing row boundaries. Valid
                // RaggedGather inputs must never exercise this branch.
                for (uint64_t scalar = 0; scalar < elementsPerValue; ++scalar) {
                    unsigned char* destination =
                        outputValues + (outputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                    for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = 0;
                }
                continue;
            }

            const uint64_t sourceToken = sourceBegin + localIndex;
            const uint64_t sourceScalarBegin = sourceToken * elementsPerValue;
            for (uint64_t scalar = 0; scalar < elementsPerValue; ++scalar) {
                const unsigned char* source =
                    sourceValues + (sourceScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                unsigned char* destination =
                    outputValues + (outputScalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
                for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = source[byte];
            }
        }
    }
}

template <typename SourceOffsetT>
__global__ void zeroActiveSourceGradientKernel(const SourceOffsetT* sourceOffsets,
                                               unsigned char* sourceGradient,
                                               unsigned long valueElementSizeBytes,
                                               uint64_t elementsPerValue,
                                               uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t rowBegin = static_cast<uint64_t>(sourceOffsets[row]);
        const uint64_t rowEnd = static_cast<uint64_t>(sourceOffsets[row + 1]);
        const uint64_t scalarBegin = rowBegin * elementsPerValue;
        const uint64_t scalarCount = (rowEnd - rowBegin) * elementsPerValue;
        for (uint64_t scalar = threadIdx.x; scalar < scalarCount; scalar += blockDim.x) {
            unsigned char* destination =
                sourceGradient + (scalarBegin + scalar) * static_cast<uint64_t>(valueElementSizeBytes);
            for (unsigned long byte = 0; byte < valueElementSizeBytes; ++byte) destination[byte] = 0;
        }
    }
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT, typename ValueT>
__global__ void gatherBackwardKernel(const SourceOffsetT* sourceOffsets,
                                     const IndexT* indicesValues,
                                     const IndexOffsetT* indicesOffsets,
                                     const ValueT* outputGradient,
                                     ValueT* sourceGradient,
                                     uint64_t elementsPerValue,
                                     uint64_t batchSize) {
    for (uint64_t row = blockIdx.x; row < batchSize; row += gridDim.x) {
        const uint64_t sourceBegin = static_cast<uint64_t>(sourceOffsets[row]);
        const uint64_t sourceEnd = static_cast<uint64_t>(sourceOffsets[row + 1]);
        const uint64_t sourceLength = sourceEnd - sourceBegin;
        const uint64_t indicesBegin = static_cast<uint64_t>(indicesOffsets[row]);
        const uint64_t indicesEnd = static_cast<uint64_t>(indicesOffsets[row + 1]);

        for (uint64_t outputToken = indicesBegin + threadIdx.x; outputToken < indicesEnd; outputToken += blockDim.x) {
            const uint64_t localIndex = static_cast<uint64_t>(indicesValues[outputToken]);
            if (localIndex >= sourceLength) continue;
            const uint64_t sourceToken = sourceBegin + localIndex;
            const uint64_t sourceScalarBegin = sourceToken * elementsPerValue;
            const uint64_t outputScalarBegin = outputToken * elementsPerValue;
            for (uint64_t scalar = 0; scalar < elementsPerValue; ++scalar) {
                atomicAdd(&sourceGradient[sourceScalarBegin + scalar], outputGradient[outputScalarBegin + scalar]);
            }
        }
    }
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT>
void launchGatherTyped(const Tensor& sourceValues,
                       const Tensor& sourceOffsets,
                       const Tensor& indicesValues,
                       const Tensor& indicesOffsets,
                       Tensor& outputValues,
                       uint64_t batchSize,
                       Stream& stream) {
    gatherKernel<SourceOffsetT, IndexOffsetT, IndexT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        static_cast<const unsigned char*>(sourceValues.getMemPtr()),
        sourceOffsets.getMemPtr<SourceOffsetT>(),
        indicesValues.getMemPtr<IndexT>(),
        indicesOffsets.getMemPtr<IndexOffsetT>(),
        static_cast<unsigned char*>(outputValues.getMemPtr()),
        static_cast<unsigned long>(TensorDescriptor::getElementSizeInBytes(sourceValues.getDataType())),
        elementsPerValue(sourceValues),
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT, typename ValueT>
void launchBackwardTyped(const Tensor& sourceOffsets,
                         const Tensor& indicesValues,
                         const Tensor& indicesOffsets,
                         const Tensor& outputGradient,
                         Tensor& sourceGradient,
                         uint64_t batchSize,
                         Stream& stream) {
    const uint64_t trailingElements = elementsPerValue(sourceGradient);
    zeroActiveSourceGradientKernel<SourceOffsetT><<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(
        sourceOffsets.getMemPtr<SourceOffsetT>(),
        static_cast<unsigned char*>(sourceGradient.getMemPtr()),
        static_cast<unsigned long>(TensorDescriptor::getElementSizeInBytes(sourceGradient.getDataType())),
        trailingElements,
        batchSize);
    CUDA_CHECK(cudaGetLastError());
    gatherBackwardKernel<SourceOffsetT, IndexOffsetT, IndexT, ValueT>
        <<<blocksForRows(batchSize), kThreads, 0, stream.getStream()>>>(sourceOffsets.getMemPtr<SourceOffsetT>(),
                                                                       indicesValues.getMemPtr<IndexT>(),
                                                                       indicesOffsets.getMemPtr<IndexOffsetT>(),
                                                                       outputGradient.getMemPtr<ValueT>(),
                                                                       sourceGradient.getMemPtr<ValueT>(),
                                                                       trailingElements,
                                                                       batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SourceOffsetT, typename IndexOffsetT>
void dispatchForwardIndex(const Tensor& sourceValues,
                          const Tensor& sourceOffsets,
                          const Tensor& indicesValues,
                          const Tensor& indicesOffsets,
                          Tensor& outputValues,
                          uint64_t batchSize,
                          Stream& stream) {
    if (indicesValues.getDataType() == DataType::UINT32) {
        launchGatherTyped<SourceOffsetT, IndexOffsetT, uint32_t>(
            sourceValues, sourceOffsets, indicesValues, indicesOffsets, outputValues, batchSize, stream);
    } else {
        launchGatherTyped<SourceOffsetT, IndexOffsetT, uint64_t>(
            sourceValues, sourceOffsets, indicesValues, indicesOffsets, outputValues, batchSize, stream);
    }
}

template <typename SourceOffsetT, typename IndexOffsetT, typename ValueT>
void dispatchBackwardIndex(const Tensor& sourceOffsets,
                           const Tensor& indicesValues,
                           const Tensor& indicesOffsets,
                           const Tensor& outputGradient,
                           Tensor& sourceGradient,
                           uint64_t batchSize,
                           Stream& stream) {
    if (indicesValues.getDataType() == DataType::UINT32) {
        launchBackwardTyped<SourceOffsetT, IndexOffsetT, uint32_t, ValueT>(
            sourceOffsets, indicesValues, indicesOffsets, outputGradient, sourceGradient, batchSize, stream);
    } else {
        launchBackwardTyped<SourceOffsetT, IndexOffsetT, uint64_t, ValueT>(
            sourceOffsets, indicesValues, indicesOffsets, outputGradient, sourceGradient, batchSize, stream);
    }
}

template <typename SourceOffsetT>
void dispatchForwardIndexOffsets(const Tensor& sourceValues,
                                 const Tensor& sourceOffsets,
                                 const Tensor& indicesValues,
                                 const Tensor& indicesOffsets,
                                 Tensor& outputValues,
                                 uint64_t batchSize,
                                 Stream& stream) {
    if (indicesOffsets.getDataType() == DataType::UINT32) {
        dispatchForwardIndex<SourceOffsetT, uint32_t>(
            sourceValues, sourceOffsets, indicesValues, indicesOffsets, outputValues, batchSize, stream);
    } else {
        dispatchForwardIndex<SourceOffsetT, uint64_t>(
            sourceValues, sourceOffsets, indicesValues, indicesOffsets, outputValues, batchSize, stream);
    }
}

template <typename SourceOffsetT, typename ValueT>
void dispatchBackwardIndexOffsets(const Tensor& sourceOffsets,
                                  const Tensor& indicesValues,
                                  const Tensor& indicesOffsets,
                                  const Tensor& outputGradient,
                                  Tensor& sourceGradient,
                                  uint64_t batchSize,
                                  Stream& stream) {
    if (indicesOffsets.getDataType() == DataType::UINT32) {
        dispatchBackwardIndex<SourceOffsetT, uint32_t, ValueT>(
            sourceOffsets, indicesValues, indicesOffsets, outputGradient, sourceGradient, batchSize, stream);
    } else {
        dispatchBackwardIndex<SourceOffsetT, uint64_t, ValueT>(
            sourceOffsets, indicesValues, indicesOffsets, outputGradient, sourceGradient, batchSize, stream);
    }
}

}  // namespace

void launchRaggedGather(const Tensor& source_values,
                        const Tensor& source_offsets,
                        const Tensor& indices_values,
                        const Tensor& indices_offsets,
                        Tensor& output_values,
                        uint64_t batch_size,
                        Stream& stream) {
    validateForward(source_values, source_offsets, indices_values, indices_offsets, output_values, batch_size);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (source_offsets.getDataType() == DataType::UINT32) {
        dispatchForwardIndexOffsets<uint32_t>(
            source_values, source_offsets, indices_values, indices_offsets, output_values, batch_size, stream);
    } else {
        dispatchForwardIndexOffsets<uint64_t>(
            source_values, source_offsets, indices_values, indices_offsets, output_values, batch_size, stream);
    }
}

void launchRaggedGatherBackward(const Tensor& source_offsets,
                                const Tensor& indices_values,
                                const Tensor& indices_offsets,
                                const Tensor& output_gradient,
                                Tensor& source_gradient,
                                uint64_t batch_size,
                                Stream& stream) {
    validateOffsets(source_offsets, batch_size, "source offsets");
    validateOffsets(indices_offsets, batch_size, "indices offsets");
    requireGpu(indices_values, "indices values");
    requireGpu(output_gradient, "output gradient");
    requireGpu(source_gradient, "source gradient");
    requireSamePlacement(source_gradient, source_offsets, "source gradient/offsets");
    requireSamePlacement(source_gradient, indices_values, "source gradient/indices values");
    requireSamePlacement(source_gradient, indices_offsets, "source gradient/indices offsets");
    requireSamePlacement(source_gradient, output_gradient, "source/output gradients");
    if (indices_values.getDataType() != DataType::UINT32 && indices_values.getDataType() != DataType::UINT64) {
        throw std::invalid_argument("RaggedGather backward indices values must use UINT32 or UINT64 dtype.");
    }
    if (indices_values.getDimensions().size() != 1 ||
        indices_values.getDimensions()[0] != output_gradient.getDimensions()[0]) {
        throw std::invalid_argument("RaggedGather backward indices/output gradient packed capacities must match.");
    }
    const std::vector<uint64_t> sourceDimensions = source_gradient.getDimensions();
    const std::vector<uint64_t> outputDimensions = output_gradient.getDimensions();
    if (source_gradient.getDataType() != output_gradient.getDataType() || sourceDimensions.size() != outputDimensions.size() ||
        sourceDimensions.empty() ||
        !std::equal(sourceDimensions.begin() + 1, sourceDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedGather backward gradients must share dtype and trailing dimensions.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    auto launchForValueType = [&](auto typeTag) {
        using ValueT = decltype(typeTag);
        if (source_offsets.getDataType() == DataType::UINT32) {
            dispatchBackwardIndexOffsets<uint32_t, ValueT>(
                source_offsets, indices_values, indices_offsets, output_gradient, source_gradient, batch_size, stream);
        } else {
            dispatchBackwardIndexOffsets<uint64_t, ValueT>(
                source_offsets, indices_values, indices_offsets, output_gradient, source_gradient, batch_size, stream);
        }
    };

    switch (source_gradient.getDataType()) {
        case DataType::FP16:
            launchForValueType(__half{});
            return;
        case DataType::BF16:
            launchForValueType(__nv_bfloat16{});
            return;
        case DataType::FP32:
            launchForValueType(float{});
            return;
        default:
            throw std::invalid_argument("RaggedGather backward supports only FP16, BF16, and FP32 feature gradients.");
    }
}

}  // namespace ThorImplementation
