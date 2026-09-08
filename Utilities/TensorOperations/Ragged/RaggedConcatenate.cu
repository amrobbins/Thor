#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxSpansPerBlock = kThreads;
constexpr uint64_t kTargetBytesPerLane = 32;
constexpr uint32_t kMaxPortableBlocks = 65535;
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFULL;

static_assert(sizeof(ulonglong4_32a) == 32);
static_assert(alignof(ulonglong4_32a) == 32);

struct SpanCopyMetadata {
    const unsigned char *source;
    unsigned char *destination;
    uint64_t byteCount;
    uint32_t copyWidthBytes;
};

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const unsigned char *source,
                                                            const unsigned char *destination) {
    // Logical span length does not constrain bulk width. A final source packet
    // may extend into the next source span or Thor's 128-byte tensor padding;
    // the destination remainder is always stored at its exact logical size.
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination);
    if ((combined & 31U) == 0) return 32;
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

template <uint32_t NumBytes>
struct ExactTailBytes {
    uint8_t bytes[NumBytes];
};

template <uint32_t NumBytes>
struct RawPacketBytes {
    uint8_t bytes[NumBytes];
};

template <uint32_t TailBytes, typename CopyT>
__device__ __forceinline__ void storeExactTailFromPacket(CopyT packet, CopyT *destination) {
    static_assert(TailBytes > 0);
    static_assert(TailBytes < sizeof(CopyT));
    static_assert(sizeof(RawPacketBytes<sizeof(CopyT)>) == sizeof(CopyT));
    static_assert(sizeof(ExactTailBytes<TailBytes>) == TailBytes);

    const RawPacketBytes<sizeof(CopyT)> packetBytes =
        cuda::std::bit_cast<RawPacketBytes<sizeof(CopyT)>>(packet);
    ExactTailBytes<TailBytes> tail;
#pragma unroll
    for (uint32_t byte = 0; byte < TailBytes; ++byte) tail.bytes[byte] = packetBytes.bytes[byte];
    *reinterpret_cast<ExactTailBytes<TailBytes> *>(destination) = tail;
}

template <typename CopyT>
__device__ __attribute__((noinline)) void copyExactTailPacket(const CopyT *source,
                                                              CopyT *destination,
                                                              uint32_t tailBytes) {
    static_assert(sizeof(CopyT) == 2 || sizeof(CopyT) == 4 || sizeof(CopyT) == 8 ||
                  sizeof(CopyT) == 16 || sizeof(CopyT) == 32);
    if (tailBytes == 0 || tailBytes >= sizeof(CopyT)) return;

    const CopyT packet = *source;
#define THOR_RAGGED_CONCAT_TAIL_CASE(N)                      \
    case N:                                                   \
        if constexpr (N < sizeof(CopyT)) {                    \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                                     \
        return
    switch (tailBytes) {
        THOR_RAGGED_CONCAT_TAIL_CASE(1);
        THOR_RAGGED_CONCAT_TAIL_CASE(2);
        THOR_RAGGED_CONCAT_TAIL_CASE(3);
        THOR_RAGGED_CONCAT_TAIL_CASE(4);
        THOR_RAGGED_CONCAT_TAIL_CASE(5);
        THOR_RAGGED_CONCAT_TAIL_CASE(6);
        THOR_RAGGED_CONCAT_TAIL_CASE(7);
        THOR_RAGGED_CONCAT_TAIL_CASE(8);
        THOR_RAGGED_CONCAT_TAIL_CASE(9);
        THOR_RAGGED_CONCAT_TAIL_CASE(10);
        THOR_RAGGED_CONCAT_TAIL_CASE(11);
        THOR_RAGGED_CONCAT_TAIL_CASE(12);
        THOR_RAGGED_CONCAT_TAIL_CASE(13);
        THOR_RAGGED_CONCAT_TAIL_CASE(14);
        THOR_RAGGED_CONCAT_TAIL_CASE(15);
        THOR_RAGGED_CONCAT_TAIL_CASE(16);
        THOR_RAGGED_CONCAT_TAIL_CASE(17);
        THOR_RAGGED_CONCAT_TAIL_CASE(18);
        THOR_RAGGED_CONCAT_TAIL_CASE(19);
        THOR_RAGGED_CONCAT_TAIL_CASE(20);
        THOR_RAGGED_CONCAT_TAIL_CASE(21);
        THOR_RAGGED_CONCAT_TAIL_CASE(22);
        THOR_RAGGED_CONCAT_TAIL_CASE(23);
        THOR_RAGGED_CONCAT_TAIL_CASE(24);
        THOR_RAGGED_CONCAT_TAIL_CASE(25);
        THOR_RAGGED_CONCAT_TAIL_CASE(26);
        THOR_RAGGED_CONCAT_TAIL_CASE(27);
        THOR_RAGGED_CONCAT_TAIL_CASE(28);
        THOR_RAGGED_CONCAT_TAIL_CASE(29);
        THOR_RAGGED_CONCAT_TAIL_CASE(30);
        THOR_RAGGED_CONCAT_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_RAGGED_CONCAT_TAIL_CASE
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerSpan>
__device__ __forceinline__ void copyAligned(const unsigned char *sourceBytes,
                                            unsigned char *destinationBytes,
                                            ItemIndexT byteCount,
                                            uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);
    const ItemIndexT items = byteCount / static_cast<ItemIndexT>(sizeof(CopyT));

    ItemIndexT item = static_cast<ItemIndexT>(lane);
    while (item < items) {
        destination[item] = source[item];
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerSpan);
        if (items - item <= laneStride) break;
        item += laneStride;
    }

    if constexpr (sizeof(CopyT) > 1) {
        if (lane == 0) {
            const uint32_t tailBytes = static_cast<uint32_t>(
                byteCount % static_cast<ItemIndexT>(sizeof(CopyT)));
            if (tailBytes != 0) copyExactTailPacket(source + items, destination + items, tailBytes);
        }
    }
}

template <typename ItemIndexT, uint32_t LanesPerSpan>
__device__ __forceinline__ void copySpanIndexed(const SpanCopyMetadata &metadata, uint32_t lane) {
    const ItemIndexT byteCount = static_cast<ItemIndexT>(metadata.byteCount);
    switch (metadata.copyWidthBytes) {
        case 32:
            copyAligned<ulonglong4_32a, ItemIndexT, LanesPerSpan>(metadata.source, metadata.destination, byteCount, lane);
            break;
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
    if (metadata.byteCount <= kMaxUint32 - LanesPerSpan) {
        copySpanIndexed<uint32_t, LanesPerSpan>(metadata, lane);
    } else {
        copySpanIndexed<uint64_t, LanesPerSpan>(metadata, lane);
    }
}

template <typename SpanIndexT>
__device__ __forceinline__ SpanCopyMetadata resolveForwardSpan(
    unsigned char *destination,
    unsigned char *const *sources,
    SpanIndexT span,
    SpanIndexT outerSlicesPerValue,
    uint32_t numSources,
    uint64_t outputValueBytes,
    uint64_t outputSliceBytes,
    const RaggedConcatenateSpanGeometry *geometry) {
    SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
    const SpanIndexT spansPerRow = outerSlicesPerValue * static_cast<SpanIndexT>(numSources);
    const SpanIndexT row = span / spansPerRow;
    const SpanIndexT rowRemainder = span - row * spansPerRow;
    const SpanIndexT outerSlice = rowRemainder / static_cast<SpanIndexT>(numSources);
    const uint32_t sourceIndex = static_cast<uint32_t>(
        rowRemainder - outerSlice * static_cast<SpanIndexT>(numSources));
    const RaggedConcatenateSpanGeometry sourceGeometry = geometry[sourceIndex];
    if (sourceGeometry.spanBytes == 0) return metadata;

    metadata.source = sources[sourceIndex] +
                      static_cast<uint64_t>(row) * sourceGeometry.valueBytes +
                      static_cast<uint64_t>(outerSlice) * sourceGeometry.spanBytes;
    metadata.destination = destination +
                           static_cast<uint64_t>(row) * outputValueBytes +
                           static_cast<uint64_t>(outerSlice) * outputSliceBytes +
                           sourceGeometry.outputOffsetBytes;
    metadata.byteCount = sourceGeometry.spanBytes;
    metadata.copyWidthBytes = widestAlignedCopyWidth(metadata.source, metadata.destination);
    return metadata;
}

template <typename SpanIndexT>
__device__ __forceinline__ SpanCopyMetadata resolveBackwardSpan(
    unsigned char *const *destinations,
    const unsigned char *source,
    SpanIndexT span,
    SpanIndexT outerSlicesPerValue,
    uint32_t numDestinations,
    uint64_t sourceValueBytes,
    uint64_t sourceSliceBytes,
    const RaggedConcatenateSpanGeometry *geometry) {
    SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
    const SpanIndexT spansPerRow = outerSlicesPerValue * static_cast<SpanIndexT>(numDestinations);
    const SpanIndexT row = span / spansPerRow;
    const SpanIndexT rowRemainder = span - row * spansPerRow;
    const SpanIndexT outerSlice = rowRemainder / static_cast<SpanIndexT>(numDestinations);
    const uint32_t destinationIndex = static_cast<uint32_t>(
        rowRemainder - outerSlice * static_cast<SpanIndexT>(numDestinations));
    unsigned char *destination = destinations[destinationIndex];
    if (destination == nullptr) return metadata;
    const RaggedConcatenateSpanGeometry destinationGeometry = geometry[destinationIndex];
    if (destinationGeometry.spanBytes == 0) return metadata;

    metadata.source = source +
                      static_cast<uint64_t>(row) * sourceValueBytes +
                      static_cast<uint64_t>(outerSlice) * sourceSliceBytes +
                      destinationGeometry.outputOffsetBytes;
    metadata.destination = destination +
                           static_cast<uint64_t>(row) * destinationGeometry.valueBytes +
                           static_cast<uint64_t>(outerSlice) * destinationGeometry.spanBytes;
    metadata.byteCount = destinationGeometry.spanBytes;
    metadata.copyWidthBytes = widestAlignedCopyWidth(metadata.source, metadata.destination);
    return metadata;
}

template <typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void raggedConcatenateSpans(unsigned char *destination,
                                       unsigned char *const *sources,
                                       SpanIndexT totalSpans,
                                       SpanIndexT outerSlicesPerValue,
                                       uint32_t numSources,
                                       uint64_t outputValueBytes,
                                       uint64_t outputSliceBytes,
                                       const RaggedConcatenateSpanGeometry *geometry) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock);
    static_assert(kThreads % SpansPerBlock == 0);
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;
    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;

    if constexpr (kLanesPerSpan == 1) {
        while (spanBase < totalSpans) {
            if (spanSlot < totalSpans - spanBase) {
                const SpanCopyMetadata metadata = resolveForwardSpan(
                    destination, sources, spanBase + static_cast<SpanIndexT>(spanSlot),
                    outerSlicesPerValue, numSources, outputValueBytes, outputSliceBytes, geometry);
                if (metadata.byteCount != 0) copySpan<1>(metadata, 0);
            }
            if (static_cast<SpanIndexT>(spanStride) >= totalSpans - spanBase) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    } else {
        __shared__ SpanCopyMetadata spanMetadata[SpansPerBlock];
        while (spanBase < totalSpans) {
            if (lane == 0) {
                SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
                if (spanSlot < totalSpans - spanBase) {
                    metadata = resolveForwardSpan(
                        destination, sources, spanBase + static_cast<SpanIndexT>(spanSlot),
                        outerSlicesPerValue, numSources, outputValueBytes, outputSliceBytes, geometry);
                }
                spanMetadata[spanSlot] = metadata;
            }

            if constexpr (kLanesPerSpan <= 32) __syncwarp();
            else __syncthreads();
            const SpanCopyMetadata metadata = spanMetadata[spanSlot];

            const bool hasNext = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
            if (hasNext) {
                if constexpr (kLanesPerSpan <= 32) __syncwarp();
                else __syncthreads();
            }
            if (metadata.byteCount != 0) copySpan<kLanesPerSpan>(metadata, lane);
            if (!hasNext) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    }
}

template <typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void raggedSplitSpans(unsigned char *const *destinations,
                                 const unsigned char *source,
                                 SpanIndexT totalSpans,
                                 SpanIndexT outerSlicesPerValue,
                                 uint32_t numDestinations,
                                 uint64_t sourceValueBytes,
                                 uint64_t sourceSliceBytes,
                                 const RaggedConcatenateSpanGeometry *geometry) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock);
    static_assert(kThreads % SpansPerBlock == 0);
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;
    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;

    if constexpr (kLanesPerSpan == 1) {
        while (spanBase < totalSpans) {
            if (spanSlot < totalSpans - spanBase) {
                const SpanCopyMetadata metadata = resolveBackwardSpan(
                    destinations, source, spanBase + static_cast<SpanIndexT>(spanSlot),
                    outerSlicesPerValue, numDestinations, sourceValueBytes, sourceSliceBytes, geometry);
                if (metadata.byteCount != 0) copySpan<1>(metadata, 0);
            }
            if (static_cast<SpanIndexT>(spanStride) >= totalSpans - spanBase) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    } else {
        __shared__ SpanCopyMetadata spanMetadata[SpansPerBlock];
        while (spanBase < totalSpans) {
            if (lane == 0) {
                SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
                if (spanSlot < totalSpans - spanBase) {
                    metadata = resolveBackwardSpan(
                        destinations, source, spanBase + static_cast<SpanIndexT>(spanSlot),
                        outerSlicesPerValue, numDestinations, sourceValueBytes, sourceSliceBytes, geometry);
                }
                spanMetadata[spanSlot] = metadata;
            }

            if constexpr (kLanesPerSpan <= 32) __syncwarp();
            else __syncthreads();
            const SpanCopyMetadata metadata = spanMetadata[spanSlot];

            const bool hasNext = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
            if (hasNext) {
                if constexpr (kLanesPerSpan <= 32) __syncwarp();
                else __syncthreads();
            }
            if (metadata.byteCount != 0) copySpan<kLanesPerSpan>(metadata, lane);
            if (!hasNext) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    }
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        throw std::invalid_argument(what);
    return lhs * rhs;
}

uint64_t spanCount(uint64_t activeRows, uint64_t outerSlicesPerValue, uint32_t numArrays) {
    if (outerSlicesPerValue == 0) throw std::invalid_argument("Ragged concatenate requires non-zero outer-slice geometry.");
    if (numArrays < 2) throw std::invalid_argument("Ragged concatenate requires at least two arrays.");
    return checkedMultiply(activeRows,
                           checkedMultiply(outerSlicesPerValue, static_cast<uint64_t>(numArrays),
                                           "Ragged concatenate spans-per-row overflow."),
                           "Ragged concatenate active span count overflow.");
}

void validateLaunchGeometry(uint64_t capacityRows,
                            uint64_t valueBytes,
                            uint64_t outerSlicesPerValue,
                            uint32_t numArrays,
                            const RaggedConcatenateSpanGeometry *geometry,
                            uint64_t activeRows) {
    if (capacityRows == 0 || valueBytes == 0 || outerSlicesPerValue == 0)
        throw std::invalid_argument("Ragged concatenate requires non-zero packed geometry.");
    if (numArrays < 2) throw std::invalid_argument("Ragged concatenate requires at least two arrays.");
    if (geometry == nullptr) throw std::invalid_argument("Ragged concatenate span geometry is null.");
    if (activeRows > capacityRows) throw std::invalid_argument("Ragged concatenate active prefix exceeds packed capacity.");
    if (valueBytes % outerSlicesPerValue != 0)
        throw std::invalid_argument("Ragged concatenate outer-slice geometry does not divide the packed value.");
}

uint32_t spansPerBlockFor(uint64_t expectedSpanBytes, uint64_t spans) {
    uint32_t spansByPayload = 1;
    if (expectedSpanBytes <= kTargetBytesPerLane) spansByPayload = 256;
    else if (expectedSpanBytes <= 2 * kTargetBytesPerLane) spansByPayload = 128;
    else if (expectedSpanBytes <= 4 * kTargetBytesPerLane) spansByPayload = 64;
    else if (expectedSpanBytes <= 8 * kTargetBytesPerLane) spansByPayload = 32;
    else if (expectedSpanBytes <= 16 * kTargetBytesPerLane) spansByPayload = 16;
    else if (expectedSpanBytes <= 32 * kTargetBytesPerLane) spansByPayload = 8;
    else if (expectedSpanBytes <= 64 * kTargetBytesPerLane) spansByPayload = 4;
    else if (expectedSpanBytes <= 128 * kTargetBytesPerLane) spansByPayload = 2;

    uint32_t spansByParallelism = 1;
    if (spans >= 16384) spansByParallelism = 256;
    else if (spans >= 8192) spansByParallelism = 128;
    else if (spans >= 4096) spansByParallelism = 64;
    else if (spans >= 2048) spansByParallelism = 32;
    else if (spans >= 1024) spansByParallelism = 16;
    else if (spans >= 512) spansByParallelism = 8;
    else if (spans >= 256) spansByParallelism = 4;
    else if (spans >= 128) spansByParallelism = 2;
    return std::min(spansByPayload, spansByParallelism);
}

template <uint32_t SpansPerBlock>
uint32_t blocksForSpans(uint64_t spans) {
    const uint64_t blocks = spans / SpansPerBlock + (spans % SpansPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <typename SpanIndexT, uint32_t SpansPerBlock>
void launchForwardGrouped(void *dest,
                          void *source[],
                          SpanIndexT spans,
                          SpanIndexT outerSlicesPerValue,
                          uint32_t numSourceArrays,
                          uint64_t outputValueBytes,
                          uint64_t outputSliceBytes,
                          const RaggedConcatenateSpanGeometry *geometry,
                          Stream stream) {
    raggedConcatenateSpans<SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(static_cast<uint64_t>(spans)), kThreads, 0, stream.getStream()>>>(
            static_cast<unsigned char *>(dest), reinterpret_cast<unsigned char **>(source), spans,
            outerSlicesPerValue, numSourceArrays, outputValueBytes, outputSliceBytes, geometry);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SpanIndexT, uint32_t SpansPerBlock>
void launchBackwardGrouped(void *dest[],
                           void *source,
                           SpanIndexT spans,
                           SpanIndexT outerSlicesPerValue,
                           uint32_t numDestArrays,
                           uint64_t sourceValueBytes,
                           uint64_t sourceSliceBytes,
                           const RaggedConcatenateSpanGeometry *geometry,
                           Stream stream) {
    raggedSplitSpans<SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(static_cast<uint64_t>(spans)), kThreads, 0, stream.getStream()>>>(
            reinterpret_cast<unsigned char **>(dest), static_cast<const unsigned char *>(source), spans,
            outerSlicesPerValue, numDestArrays, sourceValueBytes, sourceSliceBytes, geometry);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SpanIndexT>
void launchForwardIndexed(void *dest,
                          void *source[],
                          SpanIndexT spans,
                          SpanIndexT outerSlicesPerValue,
                          uint32_t numSourceArrays,
                          uint64_t outputValueBytes,
                          uint64_t outputSliceBytes,
                          const RaggedConcatenateSpanGeometry *geometry,
                          uint32_t spansPerBlock,
                          Stream stream) {
#define THOR_FORWARD_GROUP(N) case N: launchForwardGrouped<SpanIndexT, N>(dest, source, spans, outerSlicesPerValue, numSourceArrays, outputValueBytes, outputSliceBytes, geometry, stream); return
    switch (spansPerBlock) {
        THOR_FORWARD_GROUP(1); THOR_FORWARD_GROUP(2); THOR_FORWARD_GROUP(4); THOR_FORWARD_GROUP(8);
        THOR_FORWARD_GROUP(16); THOR_FORWARD_GROUP(32); THOR_FORWARD_GROUP(64); THOR_FORWARD_GROUP(128); THOR_FORWARD_GROUP(256);
        default: break;
    }
#undef THOR_FORWARD_GROUP
    throw std::logic_error("Invalid RaggedConcatenate spans-per-CTA selection.");
}

template <typename SpanIndexT>
void launchBackwardIndexed(void *dest[],
                           void *source,
                           SpanIndexT spans,
                           SpanIndexT outerSlicesPerValue,
                           uint32_t numDestArrays,
                           uint64_t sourceValueBytes,
                           uint64_t sourceSliceBytes,
                           const RaggedConcatenateSpanGeometry *geometry,
                           uint32_t spansPerBlock,
                           Stream stream) {
#define THOR_BACKWARD_GROUP(N) case N: launchBackwardGrouped<SpanIndexT, N>(dest, source, spans, outerSlicesPerValue, numDestArrays, sourceValueBytes, sourceSliceBytes, geometry, stream); return
    switch (spansPerBlock) {
        THOR_BACKWARD_GROUP(1); THOR_BACKWARD_GROUP(2); THOR_BACKWARD_GROUP(4); THOR_BACKWARD_GROUP(8);
        THOR_BACKWARD_GROUP(16); THOR_BACKWARD_GROUP(32); THOR_BACKWARD_GROUP(64); THOR_BACKWARD_GROUP(128); THOR_BACKWARD_GROUP(256);
        default: break;
    }
#undef THOR_BACKWARD_GROUP
    throw std::logic_error("Invalid RaggedSplit spans-per-CTA selection.");
}

}  // namespace

std::vector<RaggedConcatenateSpanGeometry> buildRaggedConcatenateSpanGeometry(
    std::size_t elementSizeBytes,
    uint64_t outerSlicesPerValue,
    uint64_t innerElements,
    const std::vector<uint64_t>& axisOffsets) {
    if (elementSizeBytes == 0 || outerSlicesPerValue == 0 || innerElements == 0)
        throw std::invalid_argument("Ragged concatenate requires non-zero static span geometry.");
    if (axisOffsets.size() < 3 || axisOffsets.front() != 0)
        throw std::invalid_argument("Ragged concatenate requires at least two monotonic axis spans starting at zero.");

    std::vector<RaggedConcatenateSpanGeometry> geometry(axisOffsets.size() - 1);
    for (size_t i = 0; i < geometry.size(); ++i) {
        if (axisOffsets[i + 1] < axisOffsets[i])
            throw std::invalid_argument("Ragged concatenate axis offsets must be monotonic.");
        const uint64_t axisElements = axisOffsets[i + 1] - axisOffsets[i];
        const uint64_t spanElements = checkedMultiply(axisElements, innerElements,
                                                      "Ragged concatenate span element count overflow.");
        const uint64_t spanBytes = checkedMultiply(spanElements, static_cast<uint64_t>(elementSizeBytes),
                                                   "Ragged concatenate span byte count overflow.");
        geometry[i] = RaggedConcatenateSpanGeometry{
            spanBytes,
            checkedMultiply(spanBytes, outerSlicesPerValue,
                            "Ragged concatenate value byte count overflow."),
            checkedMultiply(checkedMultiply(axisOffsets[i], innerElements,
                                            "Ragged concatenate output offset element overflow."),
                            static_cast<uint64_t>(elementSizeBytes),
                            "Ragged concatenate output offset byte overflow.")};
    }
    return geometry;
}

void launchRaggedConcatenate(void *dest,
                             void *source[],
                             uint64_t capacityRows,
                             uint64_t outputValueBytes,
                             uint64_t outerSlicesPerValue,
                             uint32_t numSourceArrays,
                             const RaggedConcatenateSpanGeometry spanGeometry[],
                             uint64_t activeRows,
                             Stream stream) {
    validateLaunchGeometry(capacityRows, outputValueBytes, outerSlicesPerValue,
                           numSourceArrays, spanGeometry, activeRows);
    if (activeRows == 0) return;
    const uint64_t spans = spanCount(activeRows, outerSlicesPerValue, numSourceArrays);
    const uint64_t outputSliceBytes = outputValueBytes / outerSlicesPerValue;
    const uint64_t expectedSpanBytes = outputSliceBytes / numSourceArrays +
                                       (outputSliceBytes % numSourceArrays != 0 ? 1 : 0);
    const uint32_t spansPerBlock = spansPerBlockFor(expectedSpanBytes, spans);

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (spans <= kMaxUint32 && outerSlicesPerValue <= kMaxUint32) {
        launchForwardIndexed<uint32_t>(dest, source, static_cast<uint32_t>(spans),
            static_cast<uint32_t>(outerSlicesPerValue), numSourceArrays,
            outputValueBytes, outputSliceBytes, spanGeometry, spansPerBlock, stream);
    } else {
        launchForwardIndexed<uint64_t>(dest, source, spans, outerSlicesPerValue, numSourceArrays,
            outputValueBytes, outputSliceBytes, spanGeometry, spansPerBlock, stream);
    }
}

void launchRaggedSplit(void *dest[],
                       void *source,
                       uint64_t capacityRows,
                       uint64_t sourceValueBytes,
                       uint64_t outerSlicesPerValue,
                       uint32_t numDestArrays,
                       const RaggedConcatenateSpanGeometry spanGeometry[],
                       uint64_t activeRows,
                       Stream stream) {
    validateLaunchGeometry(capacityRows, sourceValueBytes, outerSlicesPerValue,
                           numDestArrays, spanGeometry, activeRows);
    if (activeRows == 0) return;
    const uint64_t spans = spanCount(activeRows, outerSlicesPerValue, numDestArrays);
    const uint64_t sourceSliceBytes = sourceValueBytes / outerSlicesPerValue;
    const uint64_t expectedSpanBytes = sourceSliceBytes / numDestArrays +
                                       (sourceSliceBytes % numDestArrays != 0 ? 1 : 0);
    const uint32_t spansPerBlock = spansPerBlockFor(expectedSpanBytes, spans);

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (spans <= kMaxUint32 && outerSlicesPerValue <= kMaxUint32) {
        launchBackwardIndexed<uint32_t>(dest, source, static_cast<uint32_t>(spans),
            static_cast<uint32_t>(outerSlicesPerValue), numDestArrays,
            sourceValueBytes, sourceSliceBytes, spanGeometry, spansPerBlock, stream);
    } else {
        launchBackwardIndexed<uint64_t>(dest, source, spans, outerSlicesPerValue, numDestArrays,
            sourceValueBytes, sourceSliceBytes, spanGeometry, spansPerBlock, stream);
    }
}
