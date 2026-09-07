#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxSpansPerBlock = 8;
constexpr uint32_t kMaxPortableBlocks = 65535;
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFULL;

struct SpanCopyMetadata {
    const unsigned char *source;
    unsigned char *destination;
    uint64_t byteCount;
    uint32_t copyWidthBytes;
};

template <typename ActiveCountT, typename SpanIndexT>
__device__ __forceinline__ bool rowIsActive(SpanIndexT row, ActiveCountT activeRows) {
    if constexpr (sizeof(ActiveCountT) <= sizeof(SpanIndexT)) {
        return row < static_cast<SpanIndexT>(activeRows);
    } else {
        return static_cast<ActiveCountT>(row) < activeRows;
    }
}

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const unsigned char *source,
                                                            const unsigned char *destination,
                                                            uint64_t byteCount) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination) |
                               static_cast<uintptr_t>(byteCount);
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerSpan>
__device__ __forceinline__ void copyAligned(const unsigned char *source,
                                            unsigned char *destination,
                                            ItemIndexT byteCount,
                                            uint32_t lane) {
    const auto *typedSource = reinterpret_cast<const CopyT *>(source);
    auto *typedDestination = reinterpret_cast<CopyT *>(destination);
    const ItemIndexT items = byteCount / sizeof(CopyT);
    for (ItemIndexT item = static_cast<ItemIndexT>(lane); item < items; item += LanesPerSpan) {
        typedDestination[item] = typedSource[item];
    }
}

template <typename ItemIndexT, uint32_t LanesPerSpan>
__device__ __forceinline__ void copySpanIndexed(const SpanCopyMetadata &metadata, uint32_t lane) {
    const ItemIndexT byteCount = static_cast<ItemIndexT>(metadata.byteCount);
    switch (metadata.copyWidthBytes) {
        case 16:
            copyAligned<uint4, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 8:
            copyAligned<uint64_t, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 4:
            copyAligned<uint32_t, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 2:
            copyAligned<uint16_t, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
        default:
            copyAligned<uint8_t, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
    }
}

template <uint32_t LanesPerSpan>
__device__ __forceinline__ void copySpan(const SpanCopyMetadata &metadata, uint32_t lane) {
    // Keep the ordinary copy loop 32-bit. Only a genuinely enormous contiguous
    // slice needs 64-bit iteration; byte-address arithmetic remains 64-bit so
    // buffers larger than 4 GiB are still addressable.
    if (metadata.byteCount <= kMaxUint32 - LanesPerSpan) {
        copySpanIndexed<uint32_t, LanesPerSpan>(metadata, lane);
    } else {
        copySpanIndexed<uint64_t, LanesPerSpan>(metadata, lane);
    }
}

template <typename ActiveCountT, typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void raggedConcatenateSpans(unsigned char *destination,
                                       unsigned char *const *sources,
                                       uint32_t elementSizeBytes,
                                       SpanIndexT capacityRows,
                                       SpanIndexT elementsPerOutputValue,
                                       SpanIndexT outerSlicesPerValue,
                                       SpanIndexT innerElements,
                                       uint32_t numSources,
                                       const uint64_t *axisOffsets,
                                       const void *activeCount) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock,
                  "RaggedConcatenate spans per CTA are out of range.");
    static_assert(kThreads % SpansPerBlock == 0,
                  "RaggedConcatenate span groups must tile a CTA.");
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    __shared__ SpanCopyMetadata spanMetadata[SpansPerBlock];

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const SpanIndexT spansPerRow = outerSlicesPerValue * static_cast<SpanIndexT>(numSources);
    const SpanIndexT totalSpans = capacityRows * spansPerRow;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;

    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;
    while (spanBase < totalSpans) {
        if (lane == 0) {
            SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
            if (spanSlot < totalSpans - spanBase) {
                const SpanIndexT span = spanBase + static_cast<SpanIndexT>(spanSlot);
                const SpanIndexT row = span / spansPerRow;
                const ActiveCountT activeRows = reinterpret_cast<const ActiveCountT *>(activeCount)[0];
                if (rowIsActive(row, activeRows)) {
                    const SpanIndexT rowRemainder = span - row * spansPerRow;
                    const SpanIndexT outerSlice = rowRemainder / static_cast<SpanIndexT>(numSources);
                    const uint32_t sourceIndex = static_cast<uint32_t>(
                        rowRemainder - outerSlice * static_cast<SpanIndexT>(numSources));

                    const SpanIndexT axisBegin = static_cast<SpanIndexT>(axisOffsets[sourceIndex]);
                    const SpanIndexT axisEnd = static_cast<SpanIndexT>(axisOffsets[sourceIndex + 1]);
                    const SpanIndexT sourceSliceElements = (axisEnd - axisBegin) * innerElements;
                    if (sourceSliceElements != 0) {
                        const SpanIndexT sourceValueElements = outerSlicesPerValue * sourceSliceElements;
                        const SpanIndexT sourceBegin =
                            row * sourceValueElements + outerSlice * sourceSliceElements;
                        const SpanIndexT outputSliceElements = elementsPerOutputValue / outerSlicesPerValue;
                        const SpanIndexT destinationBegin =
                            row * elementsPerOutputValue + outerSlice * outputSliceElements + axisBegin * innerElements;

                        metadata.source = sources[sourceIndex] +
                                          static_cast<uint64_t>(sourceBegin) * elementSizeBytes;
                        metadata.destination = destination +
                                               static_cast<uint64_t>(destinationBegin) * elementSizeBytes;
                        metadata.byteCount = static_cast<uint64_t>(sourceSliceElements) * elementSizeBytes;
                        metadata.copyWidthBytes =
                            widestAlignedCopyWidth(metadata.source, metadata.destination, metadata.byteCount);
                    }
                }
            }
            spanMetadata[spanSlot] = metadata;
        }

        if constexpr (kLanesPerSpan == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        const SpanCopyMetadata metadata = spanMetadata[spanSlot];

        const bool hasNextSpanBase = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
        if (hasNextSpanBase) {
            if constexpr (kLanesPerSpan == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }

        if (metadata.byteCount != 0) copySpan<kLanesPerSpan>(metadata, lane);
        if (!hasNextSpanBase) break;
        spanBase += static_cast<SpanIndexT>(spanStride);
    }
}

template <typename ActiveCountT, typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void raggedSplitSpans(unsigned char *const *destinations,
                                 const unsigned char *source,
                                 uint32_t elementSizeBytes,
                                 SpanIndexT capacityRows,
                                 SpanIndexT elementsPerSourceValue,
                                 SpanIndexT outerSlicesPerValue,
                                 SpanIndexT innerElements,
                                 uint32_t numDestinations,
                                 const uint64_t *axisOffsets,
                                 const void *activeCount) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock,
                  "RaggedConcatenate split spans per CTA are out of range.");
    static_assert(kThreads % SpansPerBlock == 0,
                  "RaggedConcatenate split span groups must tile a CTA.");
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    __shared__ SpanCopyMetadata spanMetadata[SpansPerBlock];

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const SpanIndexT spansPerRow = outerSlicesPerValue * static_cast<SpanIndexT>(numDestinations);
    const SpanIndexT totalSpans = capacityRows * spansPerRow;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;

    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;
    while (spanBase < totalSpans) {
        if (lane == 0) {
            SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
            if (spanSlot < totalSpans - spanBase) {
                const SpanIndexT span = spanBase + static_cast<SpanIndexT>(spanSlot);
                const SpanIndexT row = span / spansPerRow;
                const ActiveCountT activeRows = reinterpret_cast<const ActiveCountT *>(activeCount)[0];
                if (rowIsActive(row, activeRows)) {
                    const SpanIndexT rowRemainder = span - row * spansPerRow;
                    const SpanIndexT outerSlice = rowRemainder / static_cast<SpanIndexT>(numDestinations);
                    const uint32_t destinationIndex = static_cast<uint32_t>(
                        rowRemainder - outerSlice * static_cast<SpanIndexT>(numDestinations));
                    unsigned char *destination = destinations[destinationIndex];
                    if (destination != nullptr) {
                        const SpanIndexT axisBegin = static_cast<SpanIndexT>(axisOffsets[destinationIndex]);
                        const SpanIndexT axisEnd = static_cast<SpanIndexT>(axisOffsets[destinationIndex + 1]);
                        const SpanIndexT destinationSliceElements = (axisEnd - axisBegin) * innerElements;
                        if (destinationSliceElements != 0) {
                            const SpanIndexT destinationValueElements =
                                outerSlicesPerValue * destinationSliceElements;
                            const SpanIndexT destinationBegin =
                                row * destinationValueElements + outerSlice * destinationSliceElements;
                            const SpanIndexT sourceSliceElements = elementsPerSourceValue / outerSlicesPerValue;
                            const SpanIndexT sourceBegin =
                                row * elementsPerSourceValue + outerSlice * sourceSliceElements + axisBegin * innerElements;

                            metadata.source = source + static_cast<uint64_t>(sourceBegin) * elementSizeBytes;
                            metadata.destination = destination +
                                                   static_cast<uint64_t>(destinationBegin) * elementSizeBytes;
                            metadata.byteCount =
                                static_cast<uint64_t>(destinationSliceElements) * elementSizeBytes;
                            metadata.copyWidthBytes =
                                widestAlignedCopyWidth(metadata.source, metadata.destination, metadata.byteCount);
                        }
                    }
                }
            }
            spanMetadata[spanSlot] = metadata;
        }

        if constexpr (kLanesPerSpan == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        const SpanCopyMetadata metadata = spanMetadata[spanSlot];

        const bool hasNextSpanBase = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
        if (hasNextSpanBase) {
            if constexpr (kLanesPerSpan == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }

        if (metadata.byteCount != 0) copySpan<kLanesPerSpan>(metadata, lane);
        if (!hasNextSpanBase) break;
        spanBase += static_cast<SpanIndexT>(spanStride);
    }
}

void validateActiveCountElementSize(std::size_t activeCountElementSizeBytes) {
    if (activeCountElementSizeBytes != sizeof(uint32_t) && activeCountElementSizeBytes != sizeof(uint64_t)) {
        throw std::invalid_argument("Ragged concatenate requires UINT32 or UINT64 active-count storage.");
    }
}

uint64_t spanCount(uint64_t capacityRows, uint64_t outerSlicesPerValue, uint32_t numArrays) {
    if (capacityRows == 0 || outerSlicesPerValue == 0) {
        throw std::invalid_argument("Ragged concatenate requires non-zero packed geometry.");
    }
    if (numArrays < 2) throw std::invalid_argument("Ragged concatenate requires at least two arrays.");
    if (outerSlicesPerValue > std::numeric_limits<uint64_t>::max() / numArrays) {
        throw std::invalid_argument("Ragged concatenate span geometry overflow.");
    }
    const uint64_t spansPerRow = outerSlicesPerValue * static_cast<uint64_t>(numArrays);
    if (capacityRows > std::numeric_limits<uint64_t>::max() / spansPerRow) {
        throw std::invalid_argument("Ragged concatenate span count overflow.");
    }
    return capacityRows * spansPerRow;
}

void validateGeometry(std::size_t elementSizeBytes,
                      uint64_t elementsPerValue,
                      uint64_t outerSlicesPerValue,
                      uint64_t innerElements) {
    if (elementSizeBytes == 0 || elementsPerValue == 0 || outerSlicesPerValue == 0 || innerElements == 0) {
        throw std::invalid_argument("Ragged concatenate requires non-zero element geometry.");
    }
    if (elementSizeBytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Ragged concatenate element size exceeds the CUDA 32-bit fast-path contract.");
    }
    if (elementsPerValue % outerSlicesPerValue != 0) {
        throw std::invalid_argument("Ragged concatenate outer-slice geometry does not divide the packed value.");
    }
    const uint64_t elementsPerOuterSlice = elementsPerValue / outerSlicesPerValue;
    if (elementsPerOuterSlice % innerElements != 0) {
        throw std::invalid_argument("Ragged concatenate inner geometry does not divide an outer slice.");
    }
}

template <uint32_t SpansPerBlock>
uint32_t blocksForSpans(uint64_t spans) {
    const uint64_t blocks = spans / SpansPerBlock + (spans % SpansPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <typename ActiveCountT, typename SpanIndexT, uint32_t SpansPerBlock>
void launchForwardGrouped(void *dest,
                          void *source[],
                          std::size_t elementSizeBytes,
                          SpanIndexT capacityRows,
                          SpanIndexT elementsPerOutputValue,
                          SpanIndexT outerSlicesPerValue,
                          SpanIndexT innerElements,
                          uint32_t numSourceArrays,
                          const uint64_t axisOffsets[],
                          const void *activeCount,
                          uint64_t spans,
                          Stream stream) {
    raggedConcatenateSpans<ActiveCountT, SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(spans), kThreads, 0, stream.getStream()>>>(
            static_cast<unsigned char *>(dest),
            reinterpret_cast<unsigned char **>(source),
            static_cast<uint32_t>(elementSizeBytes),
            capacityRows,
            elementsPerOutputValue,
            outerSlicesPerValue,
            innerElements,
            numSourceArrays,
            axisOffsets,
            activeCount);
    CUDA_CHECK(cudaGetLastError());
}

template <typename ActiveCountT, typename SpanIndexT>
void launchForwardIndexed(void *dest,
                          void *source[],
                          std::size_t elementSizeBytes,
                          SpanIndexT capacityRows,
                          SpanIndexT elementsPerOutputValue,
                          SpanIndexT outerSlicesPerValue,
                          SpanIndexT innerElements,
                          uint32_t numSourceArrays,
                          const uint64_t axisOffsets[],
                          const void *activeCount,
                          uint64_t spans,
                          Stream stream) {
    if (spans < 128) {
        launchForwardGrouped<ActiveCountT, SpanIndexT, 1>(dest, source, elementSizeBytes, capacityRows,
            elementsPerOutputValue, outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    } else if (spans < 256) {
        launchForwardGrouped<ActiveCountT, SpanIndexT, 2>(dest, source, elementSizeBytes, capacityRows,
            elementsPerOutputValue, outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    } else if (spans < 512) {
        launchForwardGrouped<ActiveCountT, SpanIndexT, 4>(dest, source, elementSizeBytes, capacityRows,
            elementsPerOutputValue, outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchForwardGrouped<ActiveCountT, SpanIndexT, 8>(dest, source, elementSizeBytes, capacityRows,
            elementsPerOutputValue, outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    }
}

template <typename ActiveCountT>
void launchForwardTyped(void *dest,
                        void *source[],
                        std::size_t elementSizeBytes,
                        uint64_t capacityRows,
                        uint64_t elementsPerOutputValue,
                        uint64_t outerSlicesPerValue,
                        uint64_t innerElements,
                        uint32_t numSourceArrays,
                        const uint64_t axisOffsets[],
                        const void *activeCount,
                        uint64_t spans,
                        Stream stream) {
    const bool elementsFitUint32 =
        elementsPerOutputValue <= kMaxUint32 && capacityRows <= kMaxUint32 / elementsPerOutputValue;
    if (spans <= kMaxUint32 && capacityRows <= kMaxUint32 && elementsFitUint32 &&
        outerSlicesPerValue <= kMaxUint32 && innerElements <= kMaxUint32) {
        launchForwardIndexed<ActiveCountT, uint32_t>(dest, source, elementSizeBytes,
            static_cast<uint32_t>(capacityRows), static_cast<uint32_t>(elementsPerOutputValue),
            static_cast<uint32_t>(outerSlicesPerValue), static_cast<uint32_t>(innerElements),
            numSourceArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchForwardIndexed<ActiveCountT, uint64_t>(dest, source, elementSizeBytes,
            capacityRows, elementsPerOutputValue, outerSlicesPerValue, innerElements,
            numSourceArrays, axisOffsets, activeCount, spans, stream);
    }
}

template <typename ActiveCountT, typename SpanIndexT, uint32_t SpansPerBlock>
void launchBackwardGrouped(void *dest[],
                           void *source,
                           std::size_t elementSizeBytes,
                           SpanIndexT capacityRows,
                           SpanIndexT elementsPerSourceValue,
                           SpanIndexT outerSlicesPerValue,
                           SpanIndexT innerElements,
                           uint32_t numDestArrays,
                           const uint64_t axisOffsets[],
                           const void *activeCount,
                           uint64_t spans,
                           Stream stream) {
    raggedSplitSpans<ActiveCountT, SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(spans), kThreads, 0, stream.getStream()>>>(
            reinterpret_cast<unsigned char **>(dest),
            static_cast<const unsigned char *>(source),
            static_cast<uint32_t>(elementSizeBytes),
            capacityRows,
            elementsPerSourceValue,
            outerSlicesPerValue,
            innerElements,
            numDestArrays,
            axisOffsets,
            activeCount);
    CUDA_CHECK(cudaGetLastError());
}

template <typename ActiveCountT, typename SpanIndexT>
void launchBackwardIndexed(void *dest[],
                           void *source,
                           std::size_t elementSizeBytes,
                           SpanIndexT capacityRows,
                           SpanIndexT elementsPerSourceValue,
                           SpanIndexT outerSlicesPerValue,
                           SpanIndexT innerElements,
                           uint32_t numDestArrays,
                           const uint64_t axisOffsets[],
                           const void *activeCount,
                           uint64_t spans,
                           Stream stream) {
    if (spans < 128) {
        launchBackwardGrouped<ActiveCountT, SpanIndexT, 1>(dest, source, elementSizeBytes, capacityRows,
            elementsPerSourceValue, outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    } else if (spans < 256) {
        launchBackwardGrouped<ActiveCountT, SpanIndexT, 2>(dest, source, elementSizeBytes, capacityRows,
            elementsPerSourceValue, outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    } else if (spans < 512) {
        launchBackwardGrouped<ActiveCountT, SpanIndexT, 4>(dest, source, elementSizeBytes, capacityRows,
            elementsPerSourceValue, outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchBackwardGrouped<ActiveCountT, SpanIndexT, 8>(dest, source, elementSizeBytes, capacityRows,
            elementsPerSourceValue, outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    }
}

template <typename ActiveCountT>
void launchBackwardTyped(void *dest[],
                         void *source,
                         std::size_t elementSizeBytes,
                         uint64_t capacityRows,
                         uint64_t elementsPerSourceValue,
                         uint64_t outerSlicesPerValue,
                         uint64_t innerElements,
                         uint32_t numDestArrays,
                         const uint64_t axisOffsets[],
                         const void *activeCount,
                         uint64_t spans,
                         Stream stream) {
    const bool elementsFitUint32 =
        elementsPerSourceValue <= kMaxUint32 && capacityRows <= kMaxUint32 / elementsPerSourceValue;
    if (spans <= kMaxUint32 && capacityRows <= kMaxUint32 && elementsFitUint32 &&
        outerSlicesPerValue <= kMaxUint32 && innerElements <= kMaxUint32) {
        launchBackwardIndexed<ActiveCountT, uint32_t>(dest, source, elementSizeBytes,
            static_cast<uint32_t>(capacityRows), static_cast<uint32_t>(elementsPerSourceValue),
            static_cast<uint32_t>(outerSlicesPerValue), static_cast<uint32_t>(innerElements),
            numDestArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchBackwardIndexed<ActiveCountT, uint64_t>(dest, source, elementSizeBytes,
            capacityRows, elementsPerSourceValue, outerSlicesPerValue, innerElements,
            numDestArrays, axisOffsets, activeCount, spans, stream);
    }
}

}  // namespace

void launchRaggedConcatenate(void *dest,
                             void *source[],
                             std::size_t elementSizeBytes,
                             uint64_t capacityRows,
                             uint64_t elementsPerOutputValue,
                             uint64_t outerSlicesPerValue,
                             uint64_t innerElements,
                             uint32_t numSourceArrays,
                             const uint64_t axisOffsets[],
                             const void *activeCount,
                             std::size_t activeCountElementSizeBytes,
                             Stream stream) {
    validateActiveCountElementSize(activeCountElementSizeBytes);
    validateGeometry(elementSizeBytes, elementsPerOutputValue, outerSlicesPerValue, innerElements);
    const uint64_t spans = spanCount(capacityRows, outerSlicesPerValue, numSourceArrays);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (activeCountElementSizeBytes == sizeof(uint32_t)) {
        launchForwardTyped<uint32_t>(dest, source, elementSizeBytes, capacityRows, elementsPerOutputValue,
            outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchForwardTyped<uint64_t>(dest, source, elementSizeBytes, capacityRows, elementsPerOutputValue,
            outerSlicesPerValue, innerElements, numSourceArrays, axisOffsets, activeCount, spans, stream);
    }
}

void launchRaggedSplit(void *dest[],
                       void *source,
                       std::size_t elementSizeBytes,
                       uint64_t capacityRows,
                       uint64_t elementsPerSourceValue,
                       uint64_t outerSlicesPerValue,
                       uint64_t innerElements,
                       uint32_t numDestArrays,
                       const uint64_t axisOffsets[],
                       const void *activeCount,
                       std::size_t activeCountElementSizeBytes,
                       Stream stream) {
    validateActiveCountElementSize(activeCountElementSizeBytes);
    validateGeometry(elementSizeBytes, elementsPerSourceValue, outerSlicesPerValue, innerElements);
    const uint64_t spans = spanCount(capacityRows, outerSlicesPerValue, numDestArrays);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (activeCountElementSizeBytes == sizeof(uint32_t)) {
        launchBackwardTyped<uint32_t>(dest, source, elementSizeBytes, capacityRows, elementsPerSourceValue,
            outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    } else {
        launchBackwardTyped<uint64_t>(dest, source, elementSizeBytes, capacityRows, elementsPerSourceValue,
            outerSlicesPerValue, innerElements, numDestArrays, axisOffsets, activeCount, spans, stream);
    }
}
