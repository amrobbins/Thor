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
#include <type_traits>
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
    const std::vector<uint64_t> sourceDimensions = sourceValues.getDimensions();
    const std::vector<uint64_t> outputDimensions = outputValues.getDimensions();
    if (sourceValues.getDataType() != outputValues.getDataType() || sourceDimensions.size() != outputDimensions.size() ||
        sourceDimensions.empty()) {
        throw std::invalid_argument("RaggedGather source/output values must share dtype and trailing rank.");
    }
    if (!std::equal(sourceDimensions.begin() + 1, sourceDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedGather source/output values must share identical trailing dimensions.");
    }
    if (indicesValues.getNumDimensions() != 1 || indicesValues.getTotalNumElements() != outputDimensions[0]) {
        throw std::invalid_argument("RaggedGather indices must be scalar packed values with the same capacity as output values.");
    }
}

uint32_t laneShiftForItems(uint64_t items) {
    if (items <= 1) return 0;
    if (items <= 2) return 1;
    if (items <= 4) return 2;
    if (items <= 8) return 3;
    if (items <= 16) return 4;
    return 5;
}

uint32_t widestAlignedCopyWidth(const Tensor& sourceValues, const Tensor& outputValues, uint64_t valueBytes) {
    const uintptr_t sourceAddress = reinterpret_cast<uintptr_t>(sourceValues.getMemPtr());
    const uintptr_t outputAddress = reinterpret_cast<uintptr_t>(outputValues.getMemPtr());
    for (uint32_t width : {16U, 8U, 4U, 2U}) {
        if (valueBytes % width == 0 && sourceAddress % width == 0 && outputAddress % width == 0) return width;
    }
    return 1;
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename CopyT,
          typename PackedIndexT,
          typename RowT,
          typename CopyIndexT>
__device__ __forceinline__ void gatherForwardRow(RowT row,
                                                 const CopyT* sourceValues,
                                                 const SourceOffsetT* sourceOffsets,
                                                 const IndexT* indicesValues,
                                                 const IndexOffsetT* indicesOffsets,
                                                 CopyT* outputValues,
                                                 CopyIndexT copyItemsPerValue,
                                                 uint32_t lanesPerToken,
                                                 uint32_t tokensPerBlock,
                                                 uint32_t tokenSlot,
                                                 uint32_t lane) {
    const SourceOffsetT sourceBegin = sourceOffsets[row];
    const SourceOffsetT sourceEnd = sourceOffsets[row + 1];
    const SourceOffsetT sourceLength = sourceEnd - sourceBegin;
    const IndexOffsetT indicesBegin = indicesOffsets[row];
    const IndexOffsetT indicesEnd = indicesOffsets[row + 1];
    const IndexOffsetT indexCount = indicesEnd - indicesBegin;
    if (static_cast<IndexOffsetT>(tokenSlot) >= indexCount) return;

    IndexOffsetT outputToken = indicesBegin + static_cast<IndexOffsetT>(tokenSlot);
    while (outputToken < indicesEnd) {
        const IndexT localIndex = indicesValues[outputToken];
        using CompareT = std::conditional_t<(sizeof(IndexT) > sizeof(SourceOffsetT)), IndexT, SourceOffsetT>;
        const bool valid = static_cast<CompareT>(localIndex) < static_cast<CompareT>(sourceLength);
        const PackedIndexT outputItemBegin =
            static_cast<PackedIndexT>(outputToken) * static_cast<PackedIndexT>(copyItemsPerValue);
        CopyT* destination = outputValues + outputItemBegin;

        if (valid) {
            const SourceOffsetT sourceToken = sourceBegin + static_cast<SourceOffsetT>(localIndex);
            const PackedIndexT sourceItemBegin =
                static_cast<PackedIndexT>(sourceToken) * static_cast<PackedIndexT>(copyItemsPerValue);
            const CopyT* source = sourceValues + sourceItemBegin;
            CopyIndexT item = static_cast<CopyIndexT>(lane);
            while (item < copyItemsPerValue) {
                destination[item] = source[item];
                const CopyIndexT remainingItems = copyItemsPerValue - item;
                if (static_cast<CopyIndexT>(lanesPerToken) >= remainingItems) break;
                item += static_cast<CopyIndexT>(lanesPerToken);
            }
        } else {
            const CopyT zero{};
            CopyIndexT item = static_cast<CopyIndexT>(lane);
            while (item < copyItemsPerValue) {
                destination[item] = zero;
                const CopyIndexT remainingItems = copyItemsPerValue - item;
                if (static_cast<CopyIndexT>(lanesPerToken) >= remainingItems) break;
                item += static_cast<CopyIndexT>(lanesPerToken);
            }
        }

        const IndexOffsetT remainingTokens = indicesEnd - outputToken;
        if (static_cast<IndexOffsetT>(tokensPerBlock) >= remainingTokens) break;
        outputToken += static_cast<IndexOffsetT>(tokensPerBlock);
    }
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename CopyT,
          typename PackedIndexT,
          typename RowT,
          typename CopyIndexT>
__device__ __forceinline__ void gatherForwardRows(const CopyT* sourceValues,
                                                  const SourceOffsetT* sourceOffsets,
                                                  const IndexT* indicesValues,
                                                  const IndexOffsetT* indicesOffsets,
                                                  CopyT* outputValues,
                                                  CopyIndexT copyItemsPerValue,
                                                  uint32_t lanesPerToken,
                                                  uint32_t tokensPerBlock,
                                                  uint32_t tokenSlot,
                                                  uint32_t lane,
                                                  RowT batchSize) {
    RowT row = static_cast<RowT>(blockIdx.x);
    const RowT rowStride = static_cast<RowT>(gridDim.x);
    while (row < batchSize) {
        gatherForwardRow<SourceOffsetT, IndexOffsetT, IndexT, CopyT, PackedIndexT, RowT, CopyIndexT>(row,
                                                                                      sourceValues,
                                                                                      sourceOffsets,
                                                                                      indicesValues,
                                                                                      indicesOffsets,
                                                                                      outputValues,
                                                                                      copyItemsPerValue,
                                                                                      lanesPerToken,
                                                                                      tokensPerBlock,
                                                                                      tokenSlot,
                                                                                      lane);
        const RowT remainingRows = batchSize - row;
        if (rowStride >= remainingRows) break;
        row += rowStride;
    }
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename CopyT,
          typename PackedIndexT,
          typename CopyIndexT,
          typename RowT>
__global__ void gatherKernel(const CopyT* sourceValues,
                             const SourceOffsetT* sourceOffsets,
                             const IndexT* indicesValues,
                             const IndexOffsetT* indicesOffsets,
                             CopyT* outputValues,
                             CopyIndexT copyItemsPerValue,
                             uint32_t laneShift,
                             RowT batchSize) {
    const uint32_t lanesPerToken = 1U << laneShift;
    const uint32_t tokensPerBlock = kThreads >> laneShift;
    const uint32_t tokenSlot = threadIdx.x >> laneShift;
    const uint32_t lane = threadIdx.x & (lanesPerToken - 1U);

    gatherForwardRows<SourceOffsetT, IndexOffsetT, IndexT, CopyT, PackedIndexT, RowT, CopyIndexT>(sourceValues,
                                                                                                 sourceOffsets,
                                                                                                 indicesValues,
                                                                                                 indicesOffsets,
                                                                                                 outputValues,
                                                                                                 copyItemsPerValue,
                                                                                                 lanesPerToken,
                                                                                                 tokensPerBlock,
                                                                                                 tokenSlot,
                                                                                                 lane,
                                                                                                 batchSize);
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename ValueT,
          typename PackedIndexT,
          typename FeatureIndexT,
          typename RowT>
__device__ __forceinline__ void gatherBackwardRows(const SourceOffsetT* sourceOffsets,
                                                   const IndexT* indicesValues,
                                                   const IndexOffsetT* indicesOffsets,
                                                   const ValueT* outputGradient,
                                                   ValueT* sourceGradient,
                                                   FeatureIndexT elementsPerValue,
                                                   uint32_t lanesPerToken,
                                                   uint32_t tokensPerBlock,
                                                   uint32_t tokenSlot,
                                                   uint32_t lane,
                                                   RowT batchSize) {
    RowT row = static_cast<RowT>(blockIdx.x);
    const RowT rowStride = static_cast<RowT>(gridDim.x);
    while (row < batchSize) {
        const SourceOffsetT sourceBegin = sourceOffsets[row];
        const SourceOffsetT sourceEnd = sourceOffsets[row + 1];
        const SourceOffsetT sourceLength = sourceEnd - sourceBegin;
        const IndexOffsetT indicesBegin = indicesOffsets[row];
        const IndexOffsetT indicesEnd = indicesOffsets[row + 1];
        const IndexOffsetT indexCount = indicesEnd - indicesBegin;

        // The packed source row is contiguous. Zero it cooperatively in this
        // CTA, then use the same CTA for the row-local scatter-add. This removes
        // the old byte-oriented zero kernel and its extra launch.
        const PackedIndexT rowScalarBegin =
            static_cast<PackedIndexT>(sourceBegin) * static_cast<PackedIndexT>(elementsPerValue);
        const PackedIndexT rowScalarCount =
            static_cast<PackedIndexT>(sourceLength) * static_cast<PackedIndexT>(elementsPerValue);
        PackedIndexT scalar = static_cast<PackedIndexT>(threadIdx.x);
        while (scalar < rowScalarCount) {
            sourceGradient[rowScalarBegin + scalar] = ValueT{};
            const PackedIndexT remainingScalars = rowScalarCount - scalar;
            if (static_cast<PackedIndexT>(blockDim.x) >= remainingScalars) break;
            scalar += static_cast<PackedIndexT>(blockDim.x);
        }
        __syncthreads();

        if (static_cast<IndexOffsetT>(tokenSlot) < indexCount) {
            IndexOffsetT outputToken = indicesBegin + static_cast<IndexOffsetT>(tokenSlot);
            while (outputToken < indicesEnd) {
                const IndexT localIndex = indicesValues[outputToken];
                using CompareT = std::conditional_t<(sizeof(IndexT) > sizeof(SourceOffsetT)), IndexT, SourceOffsetT>;
                if (static_cast<CompareT>(localIndex) < static_cast<CompareT>(sourceLength)) {
                    const SourceOffsetT sourceToken = sourceBegin + static_cast<SourceOffsetT>(localIndex);
                    const PackedIndexT sourceScalarBegin =
                        static_cast<PackedIndexT>(sourceToken) * static_cast<PackedIndexT>(elementsPerValue);
                    const PackedIndexT outputScalarBegin =
                        static_cast<PackedIndexT>(outputToken) * static_cast<PackedIndexT>(elementsPerValue);
                    FeatureIndexT feature = static_cast<FeatureIndexT>(lane);
                    if (indexCount == static_cast<IndexOffsetT>(1)) {
                        // This row has only one scatter destination, so no other
                        // token can alias it and the atomic is unnecessary.
                        while (feature < elementsPerValue) {
                            sourceGradient[sourceScalarBegin + static_cast<PackedIndexT>(feature)] =
                                outputGradient[outputScalarBegin + static_cast<PackedIndexT>(feature)];
                            const FeatureIndexT remainingFeatures = elementsPerValue - feature;
                            if (static_cast<FeatureIndexT>(lanesPerToken) >= remainingFeatures) break;
                            feature += static_cast<FeatureIndexT>(lanesPerToken);
                        }
                    } else {
                        // Arbitrary row-local gather indices may repeat. This is
                        // a true scatter-add dependency: unlike a global
                        // reduction, distinct output tokens can target the same
                        // source scalar, so atomics are required unless indices
                        // are first grouped/sorted.
                        while (feature < elementsPerValue) {
                            atomicAdd(&sourceGradient[sourceScalarBegin + static_cast<PackedIndexT>(feature)],
                                      outputGradient[outputScalarBegin + static_cast<PackedIndexT>(feature)]);
                            const FeatureIndexT remainingFeatures = elementsPerValue - feature;
                            if (static_cast<FeatureIndexT>(lanesPerToken) >= remainingFeatures) break;
                            feature += static_cast<FeatureIndexT>(lanesPerToken);
                        }
                    }
                }

                const IndexOffsetT remainingTokens = indicesEnd - outputToken;
                if (static_cast<IndexOffsetT>(tokensPerBlock) >= remainingTokens) break;
                outputToken += static_cast<IndexOffsetT>(tokensPerBlock);
            }
        }

        const RowT remainingRows = batchSize - row;
        if (rowStride >= remainingRows) break;
        row += rowStride;
    }
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename ValueT,
          typename PackedIndexT,
          typename FeatureIndexT,
          typename RowT>
__global__ void gatherBackwardKernel(const SourceOffsetT* sourceOffsets,
                                     const IndexT* indicesValues,
                                     const IndexOffsetT* indicesOffsets,
                                     const ValueT* outputGradient,
                                     ValueT* sourceGradient,
                                     FeatureIndexT elementsPerValue,
                                     uint32_t laneShift,
                                     RowT batchSize) {
    const uint32_t lanesPerToken = 1U << laneShift;
    const uint32_t tokensPerBlock = kThreads >> laneShift;
    const uint32_t tokenSlot = threadIdx.x >> laneShift;
    const uint32_t lane = threadIdx.x & (lanesPerToken - 1U);

    gatherBackwardRows<SourceOffsetT, IndexOffsetT, IndexT, ValueT, PackedIndexT, FeatureIndexT, RowT>(sourceOffsets,
                                                                                                      indicesValues,
                                                                                                      indicesOffsets,
                                                                                                      outputGradient,
                                                                                                      sourceGradient,
                                                                                                      elementsPerValue,
                                                                                                      lanesPerToken,
                                                                                                      tokensPerBlock,
                                                                                                      tokenSlot,
                                                                                                      lane,
                                                                                                      batchSize);
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename CopyT,
          typename PackedIndexT,
          typename CopyIndexT>
void launchGatherCopyIndexTyped(const Tensor& sourceValues,
                                const Tensor& sourceOffsets,
                                const Tensor& indicesValues,
                                const Tensor& indicesOffsets,
                                Tensor& outputValues,
                                uint64_t copyItemsPerValue,
                                uint64_t batchSize,
                                Stream& stream) {
    const uint32_t blocks = blocksForRows(batchSize);
    const uint32_t laneShift = laneShiftForItems(copyItemsPerValue);
    if (batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        gatherKernel<SourceOffsetT, IndexOffsetT, IndexT, CopyT, PackedIndexT, CopyIndexT, uint32_t>
            <<<blocks, kThreads, 0, stream.getStream()>>>(reinterpret_cast<const CopyT*>(sourceValues.getMemPtr()),
                                                         sourceOffsets.getMemPtr<SourceOffsetT>(),
                                                         indicesValues.getMemPtr<IndexT>(),
                                                         indicesOffsets.getMemPtr<IndexOffsetT>(),
                                                         reinterpret_cast<CopyT*>(outputValues.getMemPtr()),
                                                         static_cast<CopyIndexT>(copyItemsPerValue),
                                                         laneShift,
                                                         static_cast<uint32_t>(batchSize));
    } else {
        gatherKernel<SourceOffsetT, IndexOffsetT, IndexT, CopyT, PackedIndexT, CopyIndexT, uint64_t>
            <<<blocks, kThreads, 0, stream.getStream()>>>(reinterpret_cast<const CopyT*>(sourceValues.getMemPtr()),
                                                         sourceOffsets.getMemPtr<SourceOffsetT>(),
                                                         indicesValues.getMemPtr<IndexT>(),
                                                         indicesOffsets.getMemPtr<IndexOffsetT>(),
                                                         reinterpret_cast<CopyT*>(outputValues.getMemPtr()),
                                                         static_cast<CopyIndexT>(copyItemsPerValue),
                                                         laneShift,
                                                         batchSize);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT, typename CopyT>
void launchGatherCopyTyped(const Tensor& sourceValues,
                           const Tensor& sourceOffsets,
                           const Tensor& indicesValues,
                           const Tensor& indicesOffsets,
                           Tensor& outputValues,
                           uint64_t copyItemsPerValue,
                           uint32_t copyWidth,
                           uint64_t batchSize,
                           Stream& stream) {
    const uint64_t sourcePackedItems = sourceValues.getArraySizeInBytes() / copyWidth;
    const uint64_t outputPackedItems = outputValues.getArraySizeInBytes() / copyWidth;
    const uint64_t maxPackedItems = std::max({sourcePackedItems, outputPackedItems, copyItemsPerValue});
    if (maxPackedItems <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        launchGatherCopyIndexTyped<SourceOffsetT, IndexOffsetT, IndexT, CopyT, uint32_t, uint32_t>(sourceValues,
                                                                                                  sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputValues,
                                                                                                  copyItemsPerValue,
                                                                                                  batchSize,
                                                                                                  stream);
    } else if (copyItemsPerValue <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        launchGatherCopyIndexTyped<SourceOffsetT, IndexOffsetT, IndexT, CopyT, uint64_t, uint32_t>(sourceValues,
                                                                                                  sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputValues,
                                                                                                  copyItemsPerValue,
                                                                                                  batchSize,
                                                                                                  stream);
    } else {
        launchGatherCopyIndexTyped<SourceOffsetT, IndexOffsetT, IndexT, CopyT, uint64_t, uint64_t>(sourceValues,
                                                                                                  sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputValues,
                                                                                                  copyItemsPerValue,
                                                                                                  batchSize,
                                                                                                  stream);
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
    const uint64_t trailingElements = elementsPerValue(sourceValues);
    const uint64_t elementBytes = TensorDescriptor::getElementSizeInBytes(sourceValues.getDataType());
    if (trailingElements > std::numeric_limits<uint64_t>::max() / elementBytes) {
        throw std::overflow_error("RaggedGather trailing value byte count overflow.");
    }
    const uint64_t valueBytes = trailingElements * elementBytes;
    const uint32_t copyWidth = widestAlignedCopyWidth(sourceValues, outputValues, valueBytes);
    const uint64_t copyItemsPerValue = valueBytes / copyWidth;

    switch (copyWidth) {
        case 16:
            launchGatherCopyTyped<SourceOffsetT, IndexOffsetT, IndexT, uint4>(sourceValues,
                                                                            sourceOffsets,
                                                                            indicesValues,
                                                                            indicesOffsets,
                                                                            outputValues,
                                                                            copyItemsPerValue,
                                                                            copyWidth,
                                                                            batchSize,
                                                                            stream);
            return;
        case 8:
            launchGatherCopyTyped<SourceOffsetT, IndexOffsetT, IndexT, uint64_t>(sourceValues,
                                                                                sourceOffsets,
                                                                                indicesValues,
                                                                                indicesOffsets,
                                                                                outputValues,
                                                                                copyItemsPerValue,
                                                                                copyWidth,
                                                                                batchSize,
                                                                                stream);
            return;
        case 4:
            launchGatherCopyTyped<SourceOffsetT, IndexOffsetT, IndexT, uint32_t>(sourceValues,
                                                                                sourceOffsets,
                                                                                indicesValues,
                                                                                indicesOffsets,
                                                                                outputValues,
                                                                                copyItemsPerValue,
                                                                                copyWidth,
                                                                                batchSize,
                                                                                stream);
            return;
        case 2:
            launchGatherCopyTyped<SourceOffsetT, IndexOffsetT, IndexT, uint16_t>(sourceValues,
                                                                                sourceOffsets,
                                                                                indicesValues,
                                                                                indicesOffsets,
                                                                                outputValues,
                                                                                copyItemsPerValue,
                                                                                copyWidth,
                                                                                batchSize,
                                                                                stream);
            return;
        default:
            launchGatherCopyTyped<SourceOffsetT, IndexOffsetT, IndexT, uint8_t>(sourceValues,
                                                                               sourceOffsets,
                                                                               indicesValues,
                                                                               indicesOffsets,
                                                                               outputValues,
                                                                               copyItemsPerValue,
                                                                               copyWidth,
                                                                               batchSize,
                                                                               stream);
            return;
    }
}

template <typename SourceOffsetT,
          typename IndexOffsetT,
          typename IndexT,
          typename ValueT,
          typename PackedIndexT,
          typename FeatureIndexT>
void launchBackwardPackedTyped(const Tensor& sourceOffsets,
                               const Tensor& indicesValues,
                               const Tensor& indicesOffsets,
                               const Tensor& outputGradient,
                               Tensor& sourceGradient,
                               uint64_t trailingElements,
                               uint64_t batchSize,
                               Stream& stream) {
    const uint32_t laneShift = laneShiftForItems(trailingElements);
    const uint32_t blocks = blocksForRows(batchSize);
    if (batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        gatherBackwardKernel<SourceOffsetT, IndexOffsetT, IndexT, ValueT, PackedIndexT, FeatureIndexT, uint32_t>
            <<<blocks, kThreads, 0, stream.getStream()>>>(sourceOffsets.getMemPtr<SourceOffsetT>(),
                                                         indicesValues.getMemPtr<IndexT>(),
                                                         indicesOffsets.getMemPtr<IndexOffsetT>(),
                                                         outputGradient.getMemPtr<ValueT>(),
                                                         sourceGradient.getMemPtr<ValueT>(),
                                                         static_cast<FeatureIndexT>(trailingElements),
                                                         laneShift,
                                                         static_cast<uint32_t>(batchSize));
    } else {
        gatherBackwardKernel<SourceOffsetT, IndexOffsetT, IndexT, ValueT, PackedIndexT, FeatureIndexT, uint64_t>
            <<<blocks, kThreads, 0, stream.getStream()>>>(sourceOffsets.getMemPtr<SourceOffsetT>(),
                                                         indicesValues.getMemPtr<IndexT>(),
                                                         indicesOffsets.getMemPtr<IndexOffsetT>(),
                                                         outputGradient.getMemPtr<ValueT>(),
                                                         sourceGradient.getMemPtr<ValueT>(),
                                                         static_cast<FeatureIndexT>(trailingElements),
                                                         laneShift,
                                                         batchSize);
    }
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
    const uint64_t maxPackedScalars = std::max(sourceGradient.getTotalNumElements(), outputGradient.getTotalNumElements());
    const uint64_t packedIndexRequirement = std::max(maxPackedScalars, trailingElements);
    if (packedIndexRequirement <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        launchBackwardPackedTyped<SourceOffsetT, IndexOffsetT, IndexT, ValueT, uint32_t, uint32_t>(sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputGradient,
                                                                                                  sourceGradient,
                                                                                                  trailingElements,
                                                                                                  batchSize,
                                                                                                  stream);
    } else if (trailingElements <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        launchBackwardPackedTyped<SourceOffsetT, IndexOffsetT, IndexT, ValueT, uint64_t, uint32_t>(sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputGradient,
                                                                                                  sourceGradient,
                                                                                                  trailingElements,
                                                                                                  batchSize,
                                                                                                  stream);
    } else {
        launchBackwardPackedTyped<SourceOffsetT, IndexOffsetT, IndexT, ValueT, uint64_t, uint64_t>(sourceOffsets,
                                                                                                  indicesValues,
                                                                                                  indicesOffsets,
                                                                                                  outputGradient,
                                                                                                  sourceGradient,
                                                                                                  trailingElements,
                                                                                                  batchSize,
                                                                                                  stream);
    }
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
    const std::vector<uint64_t> sourceDimensions = source_gradient.getDimensions();
    const std::vector<uint64_t> outputDimensions = output_gradient.getDimensions();
    if (source_gradient.getDataType() != output_gradient.getDataType() || sourceDimensions.size() != outputDimensions.size() ||
        sourceDimensions.empty() ||
        !std::equal(sourceDimensions.begin() + 1, sourceDimensions.end(), outputDimensions.begin() + 1)) {
        throw std::invalid_argument("RaggedGather backward gradients must share dtype and trailing dimensions.");
    }
    if (indices_values.getNumDimensions() != 1 || indices_values.getTotalNumElements() != outputDimensions[0]) {
        throw std::invalid_argument("RaggedGather backward indices/output gradient packed capacities must match.");
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
