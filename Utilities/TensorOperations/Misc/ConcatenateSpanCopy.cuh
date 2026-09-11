#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/ConcatenateSpanGrouping.h"

#include <cuda_runtime.h>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ThorConcatenateSpanCopy {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxSpansPerBlock = kThreads;
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
    // Span length does not constrain bulk width. A final source packet may
    // legally extend into the following logical span or Thor's 128-byte tensor
    // padding, but the destination remainder is always stored exactly.
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
#define THOR_CONCAT_TAIL_CASE(N)                              \
    case N:                                                   \
        if constexpr (N < sizeof(CopyT)) {                    \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                                     \
        return
    switch (tailBytes) {
        THOR_CONCAT_TAIL_CASE(1);
        THOR_CONCAT_TAIL_CASE(2);
        THOR_CONCAT_TAIL_CASE(3);
        THOR_CONCAT_TAIL_CASE(4);
        THOR_CONCAT_TAIL_CASE(5);
        THOR_CONCAT_TAIL_CASE(6);
        THOR_CONCAT_TAIL_CASE(7);
        THOR_CONCAT_TAIL_CASE(8);
        THOR_CONCAT_TAIL_CASE(9);
        THOR_CONCAT_TAIL_CASE(10);
        THOR_CONCAT_TAIL_CASE(11);
        THOR_CONCAT_TAIL_CASE(12);
        THOR_CONCAT_TAIL_CASE(13);
        THOR_CONCAT_TAIL_CASE(14);
        THOR_CONCAT_TAIL_CASE(15);
        THOR_CONCAT_TAIL_CASE(16);
        THOR_CONCAT_TAIL_CASE(17);
        THOR_CONCAT_TAIL_CASE(18);
        THOR_CONCAT_TAIL_CASE(19);
        THOR_CONCAT_TAIL_CASE(20);
        THOR_CONCAT_TAIL_CASE(21);
        THOR_CONCAT_TAIL_CASE(22);
        THOR_CONCAT_TAIL_CASE(23);
        THOR_CONCAT_TAIL_CASE(24);
        THOR_CONCAT_TAIL_CASE(25);
        THOR_CONCAT_TAIL_CASE(26);
        THOR_CONCAT_TAIL_CASE(27);
        THOR_CONCAT_TAIL_CASE(28);
        THOR_CONCAT_TAIL_CASE(29);
        THOR_CONCAT_TAIL_CASE(30);
        THOR_CONCAT_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_CONCAT_TAIL_CASE
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

template <bool PackedIsDestination, typename SpanIndexT>
__device__ __forceinline__ SpanCopyMetadata resolveSpan(
    unsigned char *packed,
    unsigned char *const *splitArrays,
    SpanIndexT span,
    uint32_t numArrays,
    uint64_t packedSliceBytes,
    const ConcatenateSpanGeometry *geometry) {
    SpanCopyMetadata metadata{nullptr, nullptr, 0, 1};
    const SpanIndexT outerSlice = span / static_cast<SpanIndexT>(numArrays);
    const uint32_t arrayIndex = static_cast<uint32_t>(
        span - outerSlice * static_cast<SpanIndexT>(numArrays));
    const ConcatenateSpanGeometry arrayGeometry = geometry[arrayIndex];
    if (arrayGeometry.spanBytes == 0) return metadata;

    unsigned char *split = splitArrays[arrayIndex] +
                           static_cast<uint64_t>(outerSlice) * arrayGeometry.spanBytes;
    unsigned char *packedSpan = packed +
                                static_cast<uint64_t>(outerSlice) * packedSliceBytes +
                                arrayGeometry.packedOffsetBytes;
    if constexpr (PackedIsDestination) {
        metadata.source = split;
        metadata.destination = packedSpan;
    } else {
        metadata.source = packedSpan;
        metadata.destination = split;
    }
    metadata.byteCount = arrayGeometry.spanBytes;
    metadata.copyWidthBytes = widestAlignedCopyWidth(metadata.source, metadata.destination);
    return metadata;
}

template <bool PackedIsDestination, typename SpanIndexT, uint32_t SpansPerBlock>
__global__ void copyDenseConcatenateSpans(unsigned char *packed,
                                          unsigned char *const *splitArrays,
                                          SpanIndexT totalSpans,
                                          uint32_t numArrays,
                                          uint64_t packedSliceBytes,
                                          const ConcatenateSpanGeometry *geometry) {
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
                const SpanCopyMetadata metadata = resolveSpan<PackedIsDestination>(
                    packed, splitArrays, spanBase + static_cast<SpanIndexT>(spanSlot),
                    numArrays, packedSliceBytes, geometry);
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
                    metadata = resolveSpan<PackedIsDestination>(
                        packed, splitArrays, spanBase + static_cast<SpanIndexT>(spanSlot),
                        numArrays, packedSliceBytes, geometry);
                }
                spanMetadata[spanSlot] = metadata;
            }

            if constexpr (kLanesPerSpan <= 32) __syncwarp();
            else __syncthreads();
            const SpanCopyMetadata metadata = spanMetadata[spanSlot];

            const bool hasNext = static_cast<SpanIndexT>(spanStride) < totalSpans - spanBase;
            if (hasNext) {
                // Every lane has copied the old descriptor into registers before
                // its producer is allowed to reuse the shared slot.
                if constexpr (kLanesPerSpan <= 32) __syncwarp();
                else __syncthreads();
            }
            if (metadata.byteCount != 0) copySpan<kLanesPerSpan>(metadata, lane);
            if (!hasNext) break;
            spanBase += static_cast<SpanIndexT>(spanStride);
        }
    }
}

inline uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        throw std::invalid_argument(what);
    return lhs * rhs;
}

inline uint32_t spansPerBlockFor(uint64_t expectedSpanBytes, uint64_t spans) {
    return ThorConcatenateSpanGrouping::selectSpansPerCta(spans, expectedSpanBytes);
}

template <uint32_t SpansPerBlock>
inline uint32_t blocksForSpans(uint64_t spans) {
    const uint64_t blocks = spans / SpansPerBlock + (spans % SpansPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <bool PackedIsDestination, typename SpanIndexT, uint32_t SpansPerBlock>
inline void launchGrouped(void *packed,
                          void *splitArrays[],
                          SpanIndexT spans,
                          uint32_t numArrays,
                          uint64_t packedSliceBytes,
                          const ConcatenateSpanGeometry *geometry,
                          Stream stream) {
    copyDenseConcatenateSpans<PackedIsDestination, SpanIndexT, SpansPerBlock>
        <<<blocksForSpans<SpansPerBlock>(static_cast<uint64_t>(spans)), kThreads, 0, stream.getStream()>>>(
            static_cast<unsigned char *>(packed), reinterpret_cast<unsigned char **>(splitArrays), spans,
            numArrays, packedSliceBytes, geometry);
    CUDA_CHECK(cudaGetLastError());
}

template <bool PackedIsDestination, typename SpanIndexT>
inline void launchIndexed(void *packed,
                          void *splitArrays[],
                          SpanIndexT spans,
                          uint32_t numArrays,
                          uint64_t packedSliceBytes,
                          const ConcatenateSpanGeometry *geometry,
                          uint32_t spansPerBlock,
                          Stream stream) {
#define THOR_DENSE_CONCAT_GROUP(N)                                                                        \
    case N:                                                                                                \
        launchGrouped<PackedIsDestination, SpanIndexT, N>(packed, splitArrays, spans, numArrays,           \
                                                          packedSliceBytes, geometry, stream);              \
        return
    switch (spansPerBlock) {
        THOR_DENSE_CONCAT_GROUP(1);
        THOR_DENSE_CONCAT_GROUP(2);
        THOR_DENSE_CONCAT_GROUP(4);
        THOR_DENSE_CONCAT_GROUP(8);
        THOR_DENSE_CONCAT_GROUP(16);
        THOR_DENSE_CONCAT_GROUP(32);
        THOR_DENSE_CONCAT_GROUP(64);
        THOR_DENSE_CONCAT_GROUP(128);
        THOR_DENSE_CONCAT_GROUP(256);
        default:
            break;
    }
#undef THOR_DENSE_CONCAT_GROUP
    throw std::logic_error("Invalid dense concatenate spans-per-CTA selection.");
}

template <bool PackedIsDestination>
inline void launch(void *packed,
                   void *splitArrays[],
                   uint64_t outerSlices,
                   uint32_t numArrays,
                   uint64_t packedSliceBytes,
                   const ConcatenateSpanGeometry *geometry,
                   Stream stream) {
    if (outerSlices == 0) return;
    if (numArrays == 0) throw std::invalid_argument("Dense concatenate requires at least one array.");
    if (packedSliceBytes == 0) throw std::invalid_argument("Dense concatenate requires a non-zero packed slice size.");
    if (geometry == nullptr) throw std::invalid_argument("Dense concatenate span geometry is null.");

    const uint64_t spans = checkedMultiply(outerSlices, static_cast<uint64_t>(numArrays),
                                           "Dense concatenate span count overflow.");
    const uint64_t expectedSpanBytes = packedSliceBytes / numArrays +
                                       (packedSliceBytes % numArrays != 0 ? 1 : 0);
    const uint32_t spansPerBlock = spansPerBlockFor(expectedSpanBytes, spans);

    if (spans <= kMaxUint32) {
        launchIndexed<PackedIsDestination, uint32_t>(packed, splitArrays, static_cast<uint32_t>(spans),
                                                     numArrays, packedSliceBytes, geometry, spansPerBlock, stream);
    } else {
        launchIndexed<PackedIsDestination, uint64_t>(packed, splitArrays, spans,
                                                     numArrays, packedSliceBytes, geometry, spansPerBlock, stream);
    }
}

template <bool PackedIsDestination>
inline void launchWithSpansPerCtaForBenchmark(void *packed,
                                               void *splitArrays[],
                                               uint64_t outerSlices,
                                               uint32_t numArrays,
                                               uint64_t packedSliceBytes,
                                               const ConcatenateSpanGeometry *geometry,
                                               uint32_t spansPerBlock,
                                               Stream stream) {
    if (outerSlices == 0) return;
    if (numArrays == 0) throw std::invalid_argument("Dense concatenate requires at least one array.");
    if (packedSliceBytes == 0) throw std::invalid_argument("Dense concatenate requires a non-zero packed slice size.");
    if (geometry == nullptr) throw std::invalid_argument("Dense concatenate span geometry is null.");
    if (spansPerBlock == 0 || spansPerBlock > kMaxSpansPerBlock ||
        (spansPerBlock & (spansPerBlock - 1U)) != 0U) {
        throw std::invalid_argument("Dense concatenate benchmark spans-per-CTA must be a power of two in [1,256].");
    }

    const uint64_t spans = checkedMultiply(outerSlices, static_cast<uint64_t>(numArrays),
                                           "Dense concatenate span count overflow.");
    if (spans <= kMaxUint32) {
        launchIndexed<PackedIsDestination, uint32_t>(
            packed, splitArrays, static_cast<uint32_t>(spans), numArrays, packedSliceBytes,
            geometry, spansPerBlock, stream);
    } else {
        launchIndexed<PackedIsDestination, uint64_t>(
            packed, splitArrays, spans, numArrays, packedSliceBytes, geometry, spansPerBlock, stream);
    }
}

}  // namespace ThorConcatenateSpanCopy
