#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

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

struct PairCopyMetadata {
    const unsigned char *source;
    unsigned char *destination;
    uint64_t byteCount;
    uint32_t copyWidthBytes;
};

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const unsigned char *source,
                                                            const unsigned char *destination) {
    // Logical row length does not constrain bulk width. A final source packet
    // may extend into the next packed row or Thor's 128-byte tensor padding;
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

    // Thor tensors provide 128 bytes of physical padding beyond logical storage,
    // so this full-width source packet is valid even when the logical row ends at
    // the tensor boundary. Only the exact TailBytes are written to destination.
    const CopyT packet = *source;
#define THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(N)              \
    case N:                                                    \
        if constexpr (N < sizeof(CopyT)) {                    \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                                      \
        return
    switch (tailBytes) {
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(1);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(2);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(3);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(4);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(5);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(6);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(7);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(8);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(9);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(10);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(11);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(12);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(13);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(14);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(15);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(16);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(17);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(18);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(19);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(20);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(21);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(22);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(23);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(24);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(25);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(26);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(27);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(28);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(29);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(30);
        THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_RAGGED_SEQUENCE_CONCAT_TAIL_CASE
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerPair>
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
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerPair);
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

template <typename ItemIndexT, uint32_t LanesPerPair>
__device__ __forceinline__ void copyPairIndexed(const PairCopyMetadata &metadata, uint32_t lane) {
    const ItemIndexT byteCount = static_cast<ItemIndexT>(metadata.byteCount);
    switch (metadata.copyWidthBytes) {
        case 32:
            copyAligned<ulonglong4_32a, ItemIndexT, LanesPerPair>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        case 16:
            copyAligned<uint4, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 8:
            copyAligned<uint64_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 4:
            copyAligned<uint32_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 2:
            copyAligned<uint16_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        default:
            copyAligned<uint8_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
    }
}

template <uint32_t LanesPerPair>
__device__ __forceinline__ void copyPair(const PairCopyMetadata &metadata, uint32_t lane) {
    // Keep the ordinary copy loop entirely 32-bit. Leave enough headroom for
    // item += LanesPerPair so an extreme near-4-GiB row cannot wrap the loop
    // counter; only genuinely huge rows take the 64-bit fallback.
    if (metadata.byteCount <= kMaxUint32 - LanesPerPair) {
        copyPairIndexed<uint32_t, LanesPerPair>(metadata, lane);
    } else {
        copyPairIndexed<uint64_t, LanesPerPair>(metadata, lane);
    }
}


template <typename SpanT, typename SpanIndexT>
__device__ __forceinline__ PairCopyMetadata resolveForwardSpan(
    unsigned char *outputValues,
    unsigned char *const *inputValues,
    const SpanT *copySpans,
    uint64_t bytesPerValue,
    SpanIndexT spanIndex) {
    const SpanT span = copySpans[spanIndex];
    PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
    if (span.valueCount == 0) return metadata;
    metadata.source = inputValues[span.inputIndex] + static_cast<uint64_t>(span.sourceBegin) * bytesPerValue;
    metadata.destination = outputValues + static_cast<uint64_t>(span.destinationBegin) * bytesPerValue;
    metadata.byteCount = static_cast<uint64_t>(span.valueCount) * bytesPerValue;
    metadata.copyWidthBytes = widestAlignedCopyWidth(metadata.source, metadata.destination);
    return metadata;
}

template <typename SpanT, typename SpanIndexT>
__device__ __forceinline__ PairCopyMetadata resolveBackwardSpan(
    unsigned char *const *inputGradients,
    const unsigned char *outputGradient,
    const SpanT *copySpans,
    uint64_t bytesPerValue,
    SpanIndexT spanIndex) {
    const SpanT span = copySpans[spanIndex];
    PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
    unsigned char *inputGradient = inputGradients[span.inputIndex];
    if (inputGradient == nullptr || span.valueCount == 0) return metadata;
    metadata.source = outputGradient + static_cast<uint64_t>(span.destinationBegin) * bytesPerValue;
    metadata.destination = inputGradient + static_cast<uint64_t>(span.sourceBegin) * bytesPerValue;
    metadata.byteCount = static_cast<uint64_t>(span.valueCount) * bytesPerValue;
    metadata.copyWidthBytes = widestAlignedCopyWidth(metadata.source, metadata.destination);
    return metadata;
}

template <typename SpanT, typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void concatenateValuesKernel(unsigned char *outputValues,
                                        unsigned char *const *inputValues,
                                        const SpanT *copySpans,
                                        uint64_t bytesPerValue,
                                        SpanIndexT totalSpans) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock,
                  "RaggedSequenceConcatenate spans per CTA are out of range.");
    static_assert(kThreads % SpansPerBlock == 0,
                  "RaggedSequenceConcatenate span groups must tile a CTA.");
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;
    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;

    if constexpr (kLanesPerSpan == 1) {
        while (spanBase < totalSpans) {
            if (spanSlot < totalSpans - spanBase) {
                const PairCopyMetadata metadata = resolveForwardSpan<SpanT>(
                    outputValues, inputValues, copySpans, bytesPerValue,
                    spanBase + static_cast<SpanIndexT>(spanSlot));
                if (metadata.byteCount != 0) copyPair<1>(metadata, 0);
            }
            if (static_cast<SpanIndexT>(spanStride) >= totalSpans - spanBase) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    } else {
        __shared__ PairCopyMetadata spanMetadata[SpansPerBlock];
        while (spanBase < totalSpans) {
            if (lane == 0) {
                PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
                if (spanSlot < totalSpans - spanBase) {
                    metadata = resolveForwardSpan<SpanT>(
                        outputValues, inputValues, copySpans, bytesPerValue,
                        spanBase + static_cast<SpanIndexT>(spanSlot));
                }
                spanMetadata[spanSlot] = metadata;
            }

            if constexpr (kLanesPerSpan <= 32) __syncwarp();
            else __syncthreads();
            const PairCopyMetadata metadata = spanMetadata[spanSlot];

            const bool hasNext = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
            if (hasNext) {
                if constexpr (kLanesPerSpan <= 32) __syncwarp();
                else __syncthreads();
            }

            if (metadata.byteCount != 0) copyPair<kLanesPerSpan>(metadata, lane);
            if (!hasNext) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    }
}

template <typename SpanT, typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void splitGradientKernel(unsigned char *const *inputGradients,
                                    const unsigned char *outputGradient,
                                    const SpanT *copySpans,
                                    uint64_t bytesPerValue,
                                    SpanIndexT totalSpans) {
    static_assert(SpansPerBlock >= 1 && SpansPerBlock <= kMaxSpansPerBlock,
                  "RaggedSequenceConcatenate spans per CTA are out of range.");
    static_assert(kThreads % SpansPerBlock == 0,
                  "RaggedSequenceConcatenate span groups must tile a CTA.");
    constexpr uint32_t kLanesPerSpan = kThreads / SpansPerBlock;

    const uint32_t spanSlot = threadIdx.x / kLanesPerSpan;
    const uint32_t lane = threadIdx.x - spanSlot * kLanesPerSpan;
    const uint32_t spanStride = gridDim.x * SpansPerBlock;
    SpanIndexT spanBase = static_cast<SpanIndexT>(blockIdx.x) * SpansPerBlock;

    if constexpr (kLanesPerSpan == 1) {
        while (spanBase < totalSpans) {
            if (spanSlot < totalSpans - spanBase) {
                const PairCopyMetadata metadata = resolveBackwardSpan<SpanT>(
                    inputGradients, outputGradient, copySpans, bytesPerValue,
                    spanBase + static_cast<SpanIndexT>(spanSlot));
                if (metadata.byteCount != 0) copyPair<1>(metadata, 0);
            }
            if (static_cast<SpanIndexT>(spanStride) >= totalSpans - spanBase) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    } else {
        __shared__ PairCopyMetadata spanMetadata[SpansPerBlock];
        while (spanBase < totalSpans) {
            if (lane == 0) {
                PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
                if (spanSlot < totalSpans - spanBase) {
                    metadata = resolveBackwardSpan<SpanT>(
                        inputGradients, outputGradient, copySpans, bytesPerValue,
                        spanBase + static_cast<SpanIndexT>(spanSlot));
                }
                spanMetadata[spanSlot] = metadata;
            }

            if constexpr (kLanesPerSpan <= 32) __syncwarp();
            else __syncthreads();
            const PairCopyMetadata metadata = spanMetadata[spanSlot];

            const bool hasNext = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
            if (hasNext) {
                if constexpr (kLanesPerSpan <= 32) __syncwarp();
                else __syncthreads();
            }

            if (metadata.byteCount != 0) copyPair<kLanesPerSpan>(metadata, lane);
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

uint64_t bytesPerValue(std::size_t valueElementSizeBytes, uint64_t elementsPerValue) {
    return checkedMultiply(static_cast<uint64_t>(valueElementSizeBytes), elementsPerValue,
                           "RaggedSequenceConcatenate value byte width overflow.");
}

uint64_t expectedBytesPerSpan(uint64_t activeOutputValues, uint64_t valueBytes, uint64_t spans) {
    if (spans == 0 || activeOutputValues == 0) return 0;
    const uint64_t activeBytes = checkedMultiply(
        activeOutputValues, valueBytes, "RaggedSequenceConcatenate active byte count overflow.");
    return activeBytes / spans + (activeBytes % spans != 0 ? 1 : 0);
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

    // Preserve roughly 64 CTAs of available work whenever span count allows it.
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

void validateOffsetSize(std::size_t bytes) {
    if (bytes != sizeof(uint32_t) && bytes != sizeof(uint64_t)) {
        throw std::invalid_argument("RaggedSequenceConcatenate offsets must use UINT32 or UINT64 storage.");
    }
}

template <typename SpanT, typename SpanIndexT, uint32_t SpansPerBlock>
void launchForwardGrouped(void *outputValues,
                          void *inputValues[],
                          const void *copySpans,
                          uint64_t valueBytes,
                          SpanIndexT spans,
                          Stream stream) {
    concatenateValuesKernel<SpanT, SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(static_cast<uint64_t>(spans)), kThreads, 0, stream.getStream()>>>(
            static_cast<unsigned char *>(outputValues),
            reinterpret_cast<unsigned char **>(inputValues),
            static_cast<const SpanT *>(copySpans),
            valueBytes,
            spans);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SpanT, typename SpanIndexT>
void launchForwardIndexed(void *outputValues,
                          void *inputValues[],
                          const void *copySpans,
                          uint64_t valueBytes,
                          SpanIndexT spans,
                          uint32_t spansPerBlock,
                          Stream stream) {
#define THOR_RAGGED_SEQUENCE_FORWARD_GROUP(N) \
    case N:                                   \
        launchForwardGrouped<SpanT, SpanIndexT, N>( \
            outputValues, inputValues, copySpans, valueBytes, spans, stream); \
        return
    switch (spansPerBlock) {
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(1);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(2);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(4);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(8);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(16);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(32);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(64);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(128);
        THOR_RAGGED_SEQUENCE_FORWARD_GROUP(256);
        default:
            break;
    }
#undef THOR_RAGGED_SEQUENCE_FORWARD_GROUP
    throw std::logic_error("Invalid RaggedSequenceConcatenate spans-per-CTA selection.");
}

template <typename SpanT>
void launchForwardTyped(void *outputValues,
                        void *inputValues[],
                        const void *copySpans,
                        uint64_t spanCount,
                        std::size_t valueElementSizeBytes,
                        uint64_t elementsPerValue,
                        uint64_t activeOutputValues,
                        Stream stream) {
    if (activeOutputValues == 0 || spanCount == 0) return;
    const uint64_t valueBytes = bytesPerValue(valueElementSizeBytes, elementsPerValue);
    const uint32_t spansPerBlock = spansPerBlockFor(
        expectedBytesPerSpan(activeOutputValues, valueBytes, spanCount), spanCount);

    if (spanCount <= kMaxUint32) {
        launchForwardIndexed<SpanT, uint32_t>(
            outputValues, inputValues, copySpans, valueBytes,
            static_cast<uint32_t>(spanCount), spansPerBlock, stream);
    } else {
        launchForwardIndexed<SpanT, uint64_t>(
            outputValues, inputValues, copySpans, valueBytes,
            spanCount, spansPerBlock, stream);
    }
}

template <typename SpanT, typename SpanIndexT, uint32_t SpansPerBlock>
void launchBackwardGrouped(void *inputGradients[],
                           const void *outputGradient,
                           const void *copySpans,
                           uint64_t valueBytes,
                           SpanIndexT spans,
                           Stream stream) {
    splitGradientKernel<SpanT, SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(static_cast<uint64_t>(spans)), kThreads, 0, stream.getStream()>>>(
            reinterpret_cast<unsigned char **>(inputGradients),
            static_cast<const unsigned char *>(outputGradient),
            static_cast<const SpanT *>(copySpans),
            valueBytes,
            spans);
    CUDA_CHECK(cudaGetLastError());
}

template <typename SpanT, typename SpanIndexT>
void launchBackwardIndexed(void *inputGradients[],
                           const void *outputGradient,
                           const void *copySpans,
                           uint64_t valueBytes,
                           SpanIndexT spans,
                           uint32_t spansPerBlock,
                           Stream stream) {
#define THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(N) \
    case N:                                    \
        launchBackwardGrouped<SpanT, SpanIndexT, N>( \
            inputGradients, outputGradient, copySpans, valueBytes, spans, stream); \
        return
    switch (spansPerBlock) {
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(1);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(2);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(4);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(8);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(16);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(32);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(64);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(128);
        THOR_RAGGED_SEQUENCE_BACKWARD_GROUP(256);
        default:
            break;
    }
#undef THOR_RAGGED_SEQUENCE_BACKWARD_GROUP
    throw std::logic_error("Invalid RaggedSequenceConcatenate backward spans-per-CTA selection.");
}

template <typename SpanT>
void launchBackwardTyped(void *inputGradients[],
                         const void *outputGradient,
                         const void *copySpans,
                         uint64_t spanCount,
                         std::size_t valueElementSizeBytes,
                         uint64_t elementsPerValue,
                         uint64_t activeOutputValues,
                         Stream stream) {
    if (activeOutputValues == 0 || spanCount == 0) return;
    const uint64_t valueBytes = bytesPerValue(valueElementSizeBytes, elementsPerValue);
    const uint32_t spansPerBlock = spansPerBlockFor(
        expectedBytesPerSpan(activeOutputValues, valueBytes, spanCount), spanCount);

    if (spanCount <= kMaxUint32) {
        launchBackwardIndexed<SpanT, uint32_t>(
            inputGradients, outputGradient, copySpans, valueBytes,
            static_cast<uint32_t>(spanCount), spansPerBlock, stream);
    } else {
        launchBackwardIndexed<SpanT, uint64_t>(
            inputGradients, outputGradient, copySpans, valueBytes,
            spanCount, spansPerBlock, stream);
    }
}

}  // namespace

void launchRaggedSequenceConcatenate(void *output_values,
                                     void *input_values[],
                                     const void *copy_spans,
                                     uint64_t span_count,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     std::size_t offsets_element_size_bytes,
                                     uint64_t active_output_values,
                                     Stream stream) {
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    if (active_output_values == 0) {
        if (span_count != 0)
            throw std::invalid_argument("RaggedSequenceConcatenate zero active prefix requires an empty copy plan.");
        return;
    }
    if (span_count == 0 || copy_spans == nullptr) {
        throw std::invalid_argument("RaggedSequenceConcatenate non-empty active prefix requires copy spans.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchForwardTyped<ThorImplementation::RaggedSequenceCopySpan32>(
            output_values,
            input_values,
            copy_spans,
            span_count,
            value_element_size_bytes,
            elements_per_value,
            active_output_values,
            stream);
    } else {
        launchForwardTyped<ThorImplementation::RaggedSequenceCopySpan64>(
            output_values,
            input_values,
            copy_spans,
            span_count,
            value_element_size_bytes,
            elements_per_value,
            active_output_values,
            stream);
    }
}

void launchRaggedSequenceConcatenateBackward(void *input_gradients[],
                                             const void *output_gradient,
                                             const void *copy_spans,
                                             uint64_t span_count,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             std::size_t offsets_element_size_bytes,
                                             uint64_t active_output_values,
                                             Stream stream) {
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    if (active_output_values == 0) {
        if (span_count != 0)
            throw std::invalid_argument("RaggedSequenceConcatenate backward zero active prefix requires an empty copy plan.");
        return;
    }
    if (span_count == 0 || copy_spans == nullptr) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward non-empty active prefix requires copy spans.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchBackwardTyped<ThorImplementation::RaggedSequenceCopySpan32>(
            input_gradients,
            output_gradient,
            copy_spans,
            span_count,
            value_element_size_bytes,
            elements_per_value,
            active_output_values,
            stream);
    } else {
        launchBackwardTyped<ThorImplementation::RaggedSequenceCopySpan64>(
            input_gradients,
            output_gradient,
            copy_spans,
            span_count,
            value_element_size_bytes,
            elements_per_value,
            active_output_values,
            stream);
    }
}

void launchRaggedSequenceConcatenateWithSpansPerCtaForBenchmark(
    void *output_values,
    void *input_values[],
    const void *copy_spans,
    uint64_t span_count,
    std::size_t value_element_size_bytes,
    uint64_t elements_per_value,
    std::size_t offsets_element_size_bytes,
    uint64_t active_output_values,
    uint32_t spans_per_cta,
    Stream stream) {
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    if (active_output_values == 0) {
        if (span_count != 0)
            throw std::invalid_argument("RaggedSequenceConcatenate zero active prefix requires an empty copy plan.");
        return;
    }
    if (span_count == 0 || copy_spans == nullptr) {
        throw std::invalid_argument("RaggedSequenceConcatenate non-empty active prefix requires copy spans.");
    }
    if (spans_per_cta == 0 || spans_per_cta > kMaxSpansPerBlock ||
        (spans_per_cta & (spans_per_cta - 1U)) != 0U) {
        throw std::invalid_argument("RaggedSequenceConcatenate benchmark spans-per-CTA must be a power of two in [1,256].");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    const uint64_t valueBytes = bytesPerValue(value_element_size_bytes, elements_per_value);
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        if (span_count <= kMaxUint32) {
            launchForwardIndexed<ThorImplementation::RaggedSequenceCopySpan32, uint32_t>(
                output_values, input_values, copy_spans, valueBytes, static_cast<uint32_t>(span_count), spans_per_cta, stream);
        } else {
            launchForwardIndexed<ThorImplementation::RaggedSequenceCopySpan32, uint64_t>(
                output_values, input_values, copy_spans, valueBytes, span_count, spans_per_cta, stream);
        }
    } else {
        if (span_count <= kMaxUint32) {
            launchForwardIndexed<ThorImplementation::RaggedSequenceCopySpan64, uint32_t>(
                output_values, input_values, copy_spans, valueBytes, static_cast<uint32_t>(span_count), spans_per_cta, stream);
        } else {
            launchForwardIndexed<ThorImplementation::RaggedSequenceCopySpan64, uint64_t>(
                output_values, input_values, copy_spans, valueBytes, span_count, spans_per_cta, stream);
        }
    }
}

void launchRaggedSequenceConcatenateBackwardWithSpansPerCtaForBenchmark(
    void *input_gradients[],
    const void *output_gradient,
    const void *copy_spans,
    uint64_t span_count,
    std::size_t value_element_size_bytes,
    uint64_t elements_per_value,
    std::size_t offsets_element_size_bytes,
    uint64_t active_output_values,
    uint32_t spans_per_cta,
    Stream stream) {
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    if (active_output_values == 0) {
        if (span_count != 0)
            throw std::invalid_argument("RaggedSequenceConcatenate backward zero active prefix requires an empty copy plan.");
        return;
    }
    if (span_count == 0 || copy_spans == nullptr) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward non-empty active prefix requires copy spans.");
    }
    if (spans_per_cta == 0 || spans_per_cta > kMaxSpansPerBlock ||
        (spans_per_cta & (spans_per_cta - 1U)) != 0U) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward benchmark spans-per-CTA must be a power of two in [1,256].");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    const uint64_t valueBytes = bytesPerValue(value_element_size_bytes, elements_per_value);
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        if (span_count <= kMaxUint32) {
            launchBackwardIndexed<ThorImplementation::RaggedSequenceCopySpan32, uint32_t>(
                input_gradients, output_gradient, copy_spans, valueBytes, static_cast<uint32_t>(span_count), spans_per_cta, stream);
        } else {
            launchBackwardIndexed<ThorImplementation::RaggedSequenceCopySpan32, uint64_t>(
                input_gradients, output_gradient, copy_spans, valueBytes, span_count, spans_per_cta, stream);
        }
    } else {
        if (span_count <= kMaxUint32) {
            launchBackwardIndexed<ThorImplementation::RaggedSequenceCopySpan64, uint32_t>(
                input_gradients, output_gradient, copy_spans, valueBytes, static_cast<uint32_t>(span_count), spans_per_cta, stream);
        } else {
            launchBackwardIndexed<ThorImplementation::RaggedSequenceCopySpan64, uint64_t>(
                input_gradients, output_gradient, copy_spans, valueBytes, span_count, spans_per_cta, stream);
        }
    }
}
