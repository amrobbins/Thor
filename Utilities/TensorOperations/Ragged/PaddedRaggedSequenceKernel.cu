#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequenceKernel.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr uint32_t kPaddedPackTileDim = 32;
constexpr uint32_t kPaddedPackBlockRows = 8;
constexpr uint32_t kPaddedPackMaxGridDimension = 65535;
static_assert(kPaddedPackTileDim % kPaddedPackBlockRows == 0);

constexpr uint32_t kPaddedDirectCopyThreadsPerBlock = 256;
constexpr uint32_t kPaddedDirectCopyMaxRowsPerBlock = kPaddedDirectCopyThreadsPerBlock;
constexpr uint64_t kPaddedDirectCopyTargetBytesPerLane = 32;
constexpr uint32_t kPaddedDirectCopyMaxGridDimension = 65535;

constexpr uint32_t kPaddedTailZeroMaxThreads = 256;
constexpr uint32_t kPaddedTailZeroWarpThreads = 32;
constexpr uint64_t kPaddedTailZeroTargetBytesPerLane = 32;
constexpr uint32_t kPaddedTailZeroBulkStoreBytes = 32;
// grid.y is limited to 65535 on CUDA devices supported by Thor. Keep grid.x
// at the same conservative bound and stride either logical dimension when a
// representation is larger; this preserves direct row/channel coordinates.
constexpr uint32_t kPaddedTailZeroMaxGridDimension = 65535;

static_assert(sizeof(ulonglong4_32a) == kPaddedTailZeroBulkStoreBytes);
static_assert(alignof(ulonglong4_32a) == kPaddedTailZeroBulkStoreBytes);

template <uint32_t Value>
struct PaddedPowerOfTwoShift {
    static_assert(Value != 0 && (Value & (Value - 1U)) == 0U);
    static constexpr uint32_t value = 1U + PaddedPowerOfTwoShift<(Value >> 1U)>::value;
};

template <>
struct PaddedPowerOfTwoShift<1U> {
    static constexpr uint32_t value = 0U;
};

uint32_t paddedElementShift(uint32_t elementBytes) {
    switch (elementBytes) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        default: throw std::invalid_argument("Padded ragged element byte width must be a power of two <= 8.");
    }
}

uint64_t checkedTailZeroMul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument("Padded ragged tail-zero geometry overflows uint64_t.");
    }
    return a * b;
}

uint32_t cappedTailZeroGridDimension(uint64_t extent) {
    return static_cast<uint32_t>(
        std::min<uint64_t>(extent, static_cast<uint64_t>(kPaddedTailZeroMaxGridDimension)));
}

uint32_t paddedTailZeroElementBytes(DataType dtype) {
    const float elementBytesFloat = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t elementBytes = static_cast<uint64_t>(elementBytesFloat);
    if (static_cast<float>(elementBytes) != elementBytesFloat ||
        (elementBytes != 1 && elementBytes != 2 && elementBytes != 4 && elementBytes != 8)) {
        throw std::invalid_argument(
            "Padded ragged tail zeroing supports whole-byte value element sizes of 1, 2, 4, or 8 bytes.");
    }
    return static_cast<uint32_t>(elementBytes);
}

uint32_t paddedTailZeroLanesPerChannel(uint64_t maxTailBytes) {
    if (maxTailBytes == 0) return 0;
    const uint64_t requested =
        1 + (maxTailBytes - 1) / kPaddedTailZeroTargetBytesPerLane;
    uint32_t lanes = 1;
    while (lanes < requested && lanes < kPaddedTailZeroMaxThreads) lanes <<= 1;
    return lanes;
}

uint32_t paddedTailZeroLaneShift(uint32_t lanesPerChannel) {
    uint32_t shift = 0;
    while ((1U << shift) < lanesPerChannel) ++shift;
    return shift;
}

uint32_t roundTailZeroThreadsToWarp(uint32_t activeThreads) {
    if (activeThreads == 0 || activeThreads > kPaddedTailZeroMaxThreads) {
        throw std::invalid_argument("Padded ragged tail-zero active thread geometry is invalid.");
    }
    return std::min<uint32_t>(
        kPaddedTailZeroMaxThreads,
        ((activeThreads + kPaddedTailZeroWarpThreads - 1) / kPaddedTailZeroWarpThreads) *
            kPaddedTailZeroWarpThreads);
}

bool isPowerOfTwo(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

__device__ __forceinline__ uint32_t paddedDirectCopyWidth(const unsigned char* source,
                                                           const unsigned char* destination) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination);
    if ((combined & 31U) == 0) return 32;
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

template <typename CopyT>
__device__ __forceinline__ void paddedDirectCopyOne(const unsigned char* source,
                                                    unsigned char* destination) {
    *reinterpret_cast<CopyT*>(destination) = *reinterpret_cast<const CopyT*>(source);
}

__device__ __forceinline__ void paddedDirectCopyExactTail(const unsigned char* source,
                                                          unsigned char* destination,
                                                          uint32_t byteCount) {
    // Never over-read the undefined padded tail on unpack, and never over-write
    // the inactive padded tail on pack. The bulk loop has already aligned both
    // pointers to at least the selected transaction width; greedily finish the
    // final <32 bytes with exact naturally aligned loads/stores.
    while (byteCount != 0) {
        const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                                   reinterpret_cast<uintptr_t>(destination);
        if (byteCount >= 16 && (combined & 15U) == 0) {
            paddedDirectCopyOne<uint4>(source, destination);
            source += 16;
            destination += 16;
            byteCount -= 16;
        } else if (byteCount >= 8 && (combined & 7U) == 0) {
            paddedDirectCopyOne<uint64_t>(source, destination);
            source += 8;
            destination += 8;
            byteCount -= 8;
        } else if (byteCount >= 4 && (combined & 3U) == 0) {
            paddedDirectCopyOne<uint32_t>(source, destination);
            source += 4;
            destination += 4;
            byteCount -= 4;
        } else if (byteCount >= 2 && (combined & 1U) == 0) {
            paddedDirectCopyOne<uint16_t>(source, destination);
            source += 2;
            destination += 2;
            byteCount -= 2;
        } else {
            paddedDirectCopyOne<uint8_t>(source, destination);
            ++source;
            ++destination;
            --byteCount;
        }
    }
}

template <typename CopyT, typename SpanIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void paddedDirectCopyAligned(const unsigned char* sourceBytes,
                                                        unsigned char* destinationBytes,
                                                        SpanIndexT byteCount,
                                                        uint32_t lane) {
    constexpr uint32_t kCopyShift = PaddedPowerOfTwoShift<sizeof(CopyT)>::value;
    const CopyT* source = reinterpret_cast<const CopyT*>(sourceBytes);
    CopyT* destination = reinterpret_cast<CopyT*>(destinationBytes);
    const SpanIndexT itemCount = byteCount >> kCopyShift;

    SpanIndexT item = static_cast<SpanIndexT>(lane);
    while (item < itemCount) {
        destination[item] = source[item];
        const SpanIndexT laneStride = static_cast<SpanIndexT>(LanesPerRow);
        if (laneStride >= itemCount - item) break;
        item += laneStride;
    }

    if constexpr (sizeof(CopyT) > 1) {
        if (lane == 0) {
            const SpanIndexT bulkBytes = itemCount << kCopyShift;
            const uint32_t tailBytes = static_cast<uint32_t>(byteCount - bulkBytes);
            if (tailBytes != 0) {
                paddedDirectCopyExactTail(sourceBytes + static_cast<size_t>(bulkBytes),
                                          destinationBytes + static_cast<size_t>(bulkBytes),
                                          tailBytes);
            }
        }
    }
}

template <typename SpanIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void paddedDirectCopySpan(const unsigned char* source,
                                                     unsigned char* destination,
                                                     SpanIndexT byteCount,
                                                     uint32_t lane) {
    switch (paddedDirectCopyWidth(source, destination)) {
        case 32:
            paddedDirectCopyAligned<ulonglong4_32a, SpanIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            return;
        case 16:
            paddedDirectCopyAligned<uint4, SpanIndexT, LanesPerRow>(source, destination, byteCount, lane);
            return;
        case 8:
            paddedDirectCopyAligned<uint64_t, SpanIndexT, LanesPerRow>(source, destination, byteCount, lane);
            return;
        case 4:
            paddedDirectCopyAligned<uint32_t, SpanIndexT, LanesPerRow>(source, destination, byteCount, lane);
            return;
        case 2:
            paddedDirectCopyAligned<uint16_t, SpanIndexT, LanesPerRow>(source, destination, byteCount, lane);
            return;
        default:
            paddedDirectCopyAligned<uint8_t, SpanIndexT, LanesPerRow>(source, destination, byteCount, lane);
            return;
    }
}

template <typename SpanIndexT>
struct PaddedDirectCopyRowMetadata {
    const unsigned char* source;
    unsigned char* destination;
    SpanIndexT byteCount;
};

template <typename OffsetT,
          typename RowIndexT,
          typename ValueIndexT,
          typename SpanIndexT,
          uint32_t RowsPerBlock,
          bool PackedToPadded>
__global__ void channelOnePaddedRaggedDirectCopyKernel(const unsigned char* sourceValues,
                                                       const OffsetT* rowOffsets,
                                                       unsigned char* destinationValues,
                                                       RowIndexT batchSize,
                                                       ValueIndexT maxTotalValues,
                                                       ValueIndexT widthCapacity,
                                                       uint32_t elementShift) {
    static_assert(RowsPerBlock >= 1 && RowsPerBlock <= kPaddedDirectCopyMaxRowsPerBlock);
    static_assert((RowsPerBlock & (RowsPerBlock - 1U)) == 0U);
    constexpr uint32_t kRowsShift = PaddedPowerOfTwoShift<RowsPerBlock>::value;
    constexpr uint32_t kLanesPerRow = kPaddedDirectCopyThreadsPerBlock >> kRowsShift;
    constexpr uint32_t kLanesShift = PaddedPowerOfTwoShift<kLanesPerRow>::value;

    __shared__ PaddedDirectCopyRowMetadata<SpanIndexT> rowMetadata[kLanesPerRow == 1 ? 1 : RowsPerBlock];

    const uint32_t rowSlot = threadIdx.x >> kLanesShift;
    const uint32_t lane = threadIdx.x & (kLanesPerRow - 1U);
    const uint32_t rowStride = gridDim.x * RowsPerBlock;
    RowIndexT rowBase = static_cast<RowIndexT>(blockIdx.x) * RowsPerBlock;

    while (rowBase < batchSize) {
        if constexpr (kLanesPerRow == 1) {
            if (static_cast<RowIndexT>(rowSlot) < batchSize - rowBase) {
                const RowIndexT row = rowBase + static_cast<RowIndexT>(rowSlot);
                const OffsetT beginRaw = rowOffsets[row];
                const OffsetT endRaw = rowOffsets[row + static_cast<RowIndexT>(1)];
                if (endRaw >= beginRaw && endRaw <= static_cast<OffsetT>(maxTotalValues)) {
                    const OffsetT rowLengthRaw = endRaw - beginRaw;
                    if (static_cast<ValueIndexT>(rowLengthRaw) <= widthCapacity) {
                        const ValueIndexT begin = static_cast<ValueIndexT>(beginRaw);
                        const SpanIndexT byteCount = static_cast<SpanIndexT>(rowLengthRaw) << elementShift;
                        if (byteCount != 0) {
                            const ValueIndexT packedElementOffset = begin;
                            const ValueIndexT paddedElementOffset =
                                static_cast<ValueIndexT>(row) * widthCapacity;
                            const ValueIndexT sourceElementOffset =
                                PackedToPadded ? packedElementOffset : paddedElementOffset;
                            const ValueIndexT destinationElementOffset =
                                PackedToPadded ? paddedElementOffset : packedElementOffset;
                            paddedDirectCopySpan<SpanIndexT, 1>(
                                sourceValues + (static_cast<size_t>(sourceElementOffset) << elementShift),
                                destinationValues + (static_cast<size_t>(destinationElementOffset) << elementShift),
                                byteCount,
                                0);
                        }
                    }
                }
            }
        } else {
            if (lane == 0) {
                PaddedDirectCopyRowMetadata<SpanIndexT> metadata{nullptr, nullptr, 0};
                if (static_cast<RowIndexT>(rowSlot) < batchSize - rowBase) {
                    const RowIndexT row = rowBase + static_cast<RowIndexT>(rowSlot);
                    const OffsetT beginRaw = rowOffsets[row];
                    const OffsetT endRaw = rowOffsets[row + static_cast<RowIndexT>(1)];
                    if (endRaw >= beginRaw && endRaw <= static_cast<OffsetT>(maxTotalValues)) {
                        const OffsetT rowLengthRaw = endRaw - beginRaw;
                        if (static_cast<ValueIndexT>(rowLengthRaw) <= widthCapacity) {
                            const ValueIndexT begin = static_cast<ValueIndexT>(beginRaw);
                                const SpanIndexT byteCount = static_cast<SpanIndexT>(rowLengthRaw) << elementShift;
                            if (byteCount != 0) {
                                const ValueIndexT packedElementOffset = begin;
                                const ValueIndexT paddedElementOffset =
                                    static_cast<ValueIndexT>(row) * widthCapacity;
                                const ValueIndexT sourceElementOffset =
                                    PackedToPadded ? packedElementOffset : paddedElementOffset;
                                const ValueIndexT destinationElementOffset =
                                    PackedToPadded ? paddedElementOffset : packedElementOffset;
                                metadata.source =
                                    sourceValues + (static_cast<size_t>(sourceElementOffset) << elementShift);
                                metadata.destination =
                                    destinationValues + (static_cast<size_t>(destinationElementOffset) << elementShift);
                                metadata.byteCount = byteCount;
                            }
                        }
                    }
                }
                rowMetadata[rowSlot] = metadata;
            }

            if constexpr (kLanesPerRow <= 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
            const PaddedDirectCopyRowMetadata<SpanIndexT> metadata = rowMetadata[rowSlot];
            if (metadata.byteCount != 0) {
                paddedDirectCopySpan<SpanIndexT, kLanesPerRow>(
                    metadata.source,
                    metadata.destination,
                    metadata.byteCount,
                    lane);
            }
        }

        if (static_cast<RowIndexT>(rowStride) >= batchSize - rowBase) break;
        rowBase += static_cast<RowIndexT>(rowStride);
    }
}

template <typename ElementT, typename OffsetT, typename CoordIndexT, typename ValueIndexT>
__global__ void packedToPaddedRaggedSequenceTiledKernel(const ElementT* packedValues,
                                                         const OffsetT* rowOffsets,
                                                         ElementT* paddedValues,
                                                         CoordIndexT batchSize,
                                                         CoordIndexT maxTotalValues,
                                                         CoordIndexT channels,
                                                         CoordIndexT widthCapacity) {
    // Match Thor's mature dense transpose geometry: warps read contiguous
    // packed channels, then write contiguous padded timesteps after the shared
    // transpose. The +1 removes the classic 32-way bank conflict for 32-bit
    // storage while remaining correct for the other raw storage widths.
    __shared__ ElementT tile[kPaddedPackTileDim][kPaddedPackTileDim + 1];
    __shared__ CoordIndexT rowBeginShared;
    __shared__ CoordIndexT rowLengthShared;
    __shared__ uint32_t rowValidShared;

    const uint32_t lane = threadIdx.x;
    const uint32_t warpRow = threadIdx.y;
    CoordIndexT row = static_cast<CoordIndexT>(blockIdx.z);
    const CoordIndexT rowStride = static_cast<CoordIndexT>(gridDim.z);

    while (row < batchSize) {
        if (lane == 0 && warpRow == 0) {
            const OffsetT beginOffset = rowOffsets[row];
            const OffsetT endOffset = rowOffsets[row + static_cast<CoordIndexT>(1)];
            const bool valid = endOffset >= beginOffset &&
                               endOffset <= static_cast<OffsetT>(maxTotalValues) &&
                               static_cast<CoordIndexT>(endOffset - beginOffset) <= widthCapacity;
            rowValidShared = valid ? 1U : 0U;
            if (valid) {
                rowBeginShared = static_cast<CoordIndexT>(beginOffset);
                rowLengthShared = static_cast<CoordIndexT>(endOffset - beginOffset);
            }
        }
        __syncthreads();

        if (rowValidShared != 0U) {
            const CoordIndexT rowBegin = rowBeginShared;
            const CoordIndexT rowLength = rowLengthShared;
            const CoordIndexT channelTileStride =
                static_cast<CoordIndexT>(gridDim.x) * static_cast<CoordIndexT>(kPaddedPackTileDim);
            const CoordIndexT timestepTileStride =
                static_cast<CoordIndexT>(gridDim.y) * static_cast<CoordIndexT>(kPaddedPackTileDim);

            CoordIndexT channelTileStart =
                static_cast<CoordIndexT>(blockIdx.x) * static_cast<CoordIndexT>(kPaddedPackTileDim);
            while (channelTileStart < channels) {
                CoordIndexT timestepTileStart =
                    static_cast<CoordIndexT>(blockIdx.y) * static_cast<CoordIndexT>(kPaddedPackTileDim);
                while (timestepTileStart < rowLength) {
#pragma unroll
                    for (uint32_t offset = 0; offset < kPaddedPackTileDim; offset += kPaddedPackBlockRows) {
                        const uint32_t tileRow = warpRow + offset;
                        const CoordIndexT timestep = timestepTileStart + static_cast<CoordIndexT>(tileRow);
                        const CoordIndexT channel = channelTileStart + static_cast<CoordIndexT>(lane);
                        if (timestep < rowLength && channel < channels) {
                            const ValueIndexT packedIndex =
                                (static_cast<ValueIndexT>(rowBegin) + static_cast<ValueIndexT>(timestep)) *
                                    static_cast<ValueIndexT>(channels) +
                                static_cast<ValueIndexT>(channel);
                            tile[tileRow][lane] = packedValues[packedIndex];
                        }
                    }
                    __syncthreads();

#pragma unroll
                    for (uint32_t offset = 0; offset < kPaddedPackTileDim; offset += kPaddedPackBlockRows) {
                        const uint32_t tileColumn = warpRow + offset;
                        const CoordIndexT channel = channelTileStart + static_cast<CoordIndexT>(tileColumn);
                        const CoordIndexT timestep = timestepTileStart + static_cast<CoordIndexT>(lane);
                        if (channel < channels && timestep < rowLength) {
                            const ValueIndexT paddedIndex =
                                (static_cast<ValueIndexT>(row) * static_cast<ValueIndexT>(channels) +
                                 static_cast<ValueIndexT>(channel)) *
                                    static_cast<ValueIndexT>(widthCapacity) +
                                static_cast<ValueIndexT>(timestep);
                            paddedValues[paddedIndex] = tile[lane][tileColumn];
                        }
                    }
                    // The next logical tile reuses shared storage.
                    __syncthreads();

                    if (timestepTileStride >= rowLength - timestepTileStart) break;
                    timestepTileStart += timestepTileStride;
                }

                if (channelTileStride >= channels - channelTileStart) break;
                channelTileStart += channelTileStride;
            }
        }

        // Protect the shared row metadata before the next strided row is
        // published by thread 0. This is also required for rows with no work.
        __syncthreads();
        if (rowStride >= batchSize - row) break;
        row += rowStride;
    }
}

template <typename T>
__device__ __forceinline__ void storePaddedTailZero(unsigned char* destination) {
    *reinterpret_cast<T*>(destination) = T{};
}

__device__ __forceinline__ void zeroPaddedTailExactSmallSpan(unsigned char* destination, uint32_t byteCount) {
    // This handles only the <32-byte prefix/suffix around the aligned bulk.
    // Greedily use the widest exact store allowed by the current address; no
    // store is allowed to cross the logical tail boundary.
    while (byteCount != 0) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(destination);
        if (byteCount >= 16 && (address & 15U) == 0) {
            storePaddedTailZero<uint4>(destination);
            destination += 16;
            byteCount -= 16;
        } else if (byteCount >= 8 && (address & 7U) == 0) {
            storePaddedTailZero<uint64_t>(destination);
            destination += 8;
            byteCount -= 8;
        } else if (byteCount >= 4 && (address & 3U) == 0) {
            storePaddedTailZero<uint32_t>(destination);
            destination += 4;
            byteCount -= 4;
        } else if (byteCount >= 2 && (address & 1U) == 0) {
            storePaddedTailZero<uint16_t>(destination);
            destination += 2;
            byteCount -= 2;
        } else {
            storePaddedTailZero<uint8_t>(destination);
            ++destination;
            --byteCount;
        }
    }
}

template <typename SpanIndexT>
__device__ __forceinline__ void zeroPaddedTailSpan(unsigned char* destination,
                                                   SpanIndexT byteCount,
                                                   uint32_t lane,
                                                   uint32_t lanesPerChannel) {
    if (byteCount == 0) return;

    const uintptr_t address = reinterpret_cast<uintptr_t>(destination);
    uint32_t prefixBytes = static_cast<uint32_t>(
        (kPaddedTailZeroBulkStoreBytes - (address & (kPaddedTailZeroBulkStoreBytes - 1))) &
        (kPaddedTailZeroBulkStoreBytes - 1));
    if (static_cast<SpanIndexT>(prefixBytes) > byteCount) {
        prefixBytes = static_cast<uint32_t>(byteCount);
    }
    if (lane == 0 && prefixBytes != 0) {
        zeroPaddedTailExactSmallSpan(destination, prefixBytes);
    }

    unsigned char* bulk = destination + prefixBytes;
    const SpanIndexT remaining = byteCount - static_cast<SpanIndexT>(prefixBytes);
    constexpr uint32_t kBulkStoreShift = PaddedPowerOfTwoShift<kPaddedTailZeroBulkStoreBytes>::value;
    const SpanIndexT packetCount = remaining >> kBulkStoreShift;
    const ulonglong4_32a zeroPacket{};
    SpanIndexT packet = static_cast<SpanIndexT>(lane);
    while (packet < packetCount) {
        *reinterpret_cast<ulonglong4_32a*>(
            bulk + static_cast<size_t>(packet << kBulkStoreShift)) = zeroPacket;
        const SpanIndexT laneStride = static_cast<SpanIndexT>(lanesPerChannel);
        if (laneStride >= packetCount - packet) break;
        packet += laneStride;
    }

    if (lane == 0) {
        const SpanIndexT bulkBytes = packetCount << kBulkStoreShift;
        const uint32_t suffixBytes = static_cast<uint32_t>(remaining - bulkBytes);
        if (suffixBytes != 0) {
            zeroPaddedTailExactSmallSpan(bulk + static_cast<size_t>(bulkBytes), suffixBytes);
        }
    }
}

__device__ __forceinline__ void syncPaddedTailZeroBlock() {
    if (blockDim.x <= kPaddedTailZeroWarpThreads) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}

template <typename OffsetT, typename CoordIndexT, typename ValueIndexT, typename SpanIndexT>
__global__ void zeroPaddedRaggedSequenceTailKernel(unsigned char* paddedValues,
                                                   const OffsetT* rowOffsets,
                                                   CoordIndexT batchSize,
                                                   CoordIndexT channels,
                                                   CoordIndexT widthCapacity,
                                                   uint32_t elementShift,
                                                   uint32_t laneShift,
                                                   uint32_t channelsPerBlock) {
    __shared__ CoordIndexT rowLengthShared;

    const uint32_t lanesPerChannel = 1U << laneShift;
    const uint32_t channelSlot = threadIdx.x >> laneShift;
    const uint32_t lane = threadIdx.x & (lanesPerChannel - 1U);
    CoordIndexT row = static_cast<CoordIndexT>(blockIdx.y);
    const CoordIndexT rowStride = static_cast<CoordIndexT>(gridDim.y);

    while (row < batchSize) {
        if (threadIdx.x == 0) {
            const OffsetT begin = rowOffsets[row];
            const OffsetT end = rowOffsets[row + static_cast<CoordIndexT>(1)];
            if (end < begin) {
                rowLengthShared = 0;
            } else {
                const OffsetT rowLength = end - begin;
                if constexpr (sizeof(OffsetT) <= sizeof(CoordIndexT)) {
                    const CoordIndexT rowLengthCoord = static_cast<CoordIndexT>(rowLength);
                    rowLengthShared = rowLengthCoord >= widthCapacity ? widthCapacity : rowLengthCoord;
                } else {
                    rowLengthShared = rowLength >= static_cast<OffsetT>(widthCapacity)
                                          ? widthCapacity
                                          : static_cast<CoordIndexT>(rowLength);
                }
            }
        }
        syncPaddedTailZeroBlock();

        const CoordIndexT rowLength = rowLengthShared;
        if (rowLength < widthCapacity) {
            const CoordIndexT channelGroupStride =
                static_cast<CoordIndexT>(gridDim.x) * static_cast<CoordIndexT>(channelsPerBlock);
            CoordIndexT channelBase =
                static_cast<CoordIndexT>(blockIdx.x) * static_cast<CoordIndexT>(channelsPerBlock);
            while (channelBase < channels) {
                if (channelSlot < channelsPerBlock) {
                    const CoordIndexT channel = channelBase + static_cast<CoordIndexT>(channelSlot);
                    if (channel < channels) {
                        const ValueIndexT rowChannel =
                            static_cast<ValueIndexT>(row) * static_cast<ValueIndexT>(channels) +
                            static_cast<ValueIndexT>(channel);
                        const ValueIndexT tailElementBegin =
                            rowChannel * static_cast<ValueIndexT>(widthCapacity) +
                            static_cast<ValueIndexT>(rowLength);
                        const CoordIndexT tailElements = widthCapacity - rowLength;
                        const SpanIndexT tailBytes = static_cast<SpanIndexT>(tailElements) << elementShift;
                        unsigned char* tail =
                            paddedValues + (static_cast<size_t>(tailElementBegin) << elementShift);
                        zeroPaddedTailSpan<SpanIndexT>(tail, tailBytes, lane, lanesPerChannel);
                    }
                }

                if (channelGroupStride >= channels - channelBase) break;
                channelBase += channelGroupStride;
            }
        }

        // Protect rowLengthShared before thread 0 publishes the next row.
        syncPaddedTailZeroBlock();
        if (rowStride >= batchSize - row) break;
        row += rowStride;
    }
}

template <typename ElementT, typename OffsetT, typename CoordIndexT, typename ValueIndexT>
__global__ void paddedToPackedRaggedSequenceTiledKernel(const ElementT* paddedValues,
                                                         const OffsetT* rowOffsets,
                                                         ElementT* packedValues,
                                                         CoordIndexT batchSize,
                                                         CoordIndexT maxTotalValues,
                                                         CoordIndexT channels,
                                                         CoordIndexT widthCapacity) {
    // Exact inverse of packedToPaddedRaggedSequenceTiledKernel: warps read
    // contiguous padded timesteps, transpose through shared memory, then write
    // contiguous packed channels. Raw storage words are moved without dtype
    // conversion, so FP16/BF16/FP32 payload bits are preserved exactly.
    __shared__ ElementT tile[kPaddedPackTileDim][kPaddedPackTileDim + 1];
    __shared__ CoordIndexT rowBeginShared;
    __shared__ CoordIndexT rowLengthShared;
    __shared__ uint32_t rowValidShared;

    const uint32_t lane = threadIdx.x;
    const uint32_t warpRow = threadIdx.y;
    CoordIndexT row = static_cast<CoordIndexT>(blockIdx.z);
    const CoordIndexT rowStride = static_cast<CoordIndexT>(gridDim.z);

    while (row < batchSize) {
        if (lane == 0 && warpRow == 0) {
            const OffsetT beginOffset = rowOffsets[row];
            const OffsetT endOffset = rowOffsets[row + static_cast<CoordIndexT>(1)];
            const bool valid = endOffset >= beginOffset &&
                               endOffset <= static_cast<OffsetT>(maxTotalValues) &&
                               static_cast<CoordIndexT>(endOffset - beginOffset) <= widthCapacity;
            rowValidShared = valid ? 1U : 0U;
            if (valid) {
                rowBeginShared = static_cast<CoordIndexT>(beginOffset);
                rowLengthShared = static_cast<CoordIndexT>(endOffset - beginOffset);
            }
        }
        __syncthreads();

        if (rowValidShared != 0U) {
            const CoordIndexT rowBegin = rowBeginShared;
            const CoordIndexT rowLength = rowLengthShared;
            const CoordIndexT channelTileStride =
                static_cast<CoordIndexT>(gridDim.x) * static_cast<CoordIndexT>(kPaddedPackTileDim);
            const CoordIndexT timestepTileStride =
                static_cast<CoordIndexT>(gridDim.y) * static_cast<CoordIndexT>(kPaddedPackTileDim);

            CoordIndexT channelTileStart =
                static_cast<CoordIndexT>(blockIdx.x) * static_cast<CoordIndexT>(kPaddedPackTileDim);
            while (channelTileStart < channels) {
                CoordIndexT timestepTileStart =
                    static_cast<CoordIndexT>(blockIdx.y) * static_cast<CoordIndexT>(kPaddedPackTileDim);
                while (timestepTileStart < rowLength) {
#pragma unroll
                    for (uint32_t offset = 0; offset < kPaddedPackTileDim; offset += kPaddedPackBlockRows) {
                        const uint32_t tileColumn = warpRow + offset;
                        const CoordIndexT channel = channelTileStart + static_cast<CoordIndexT>(tileColumn);
                        const CoordIndexT timestep = timestepTileStart + static_cast<CoordIndexT>(lane);
                        if (channel < channels && timestep < rowLength) {
                            const ValueIndexT paddedIndex =
                                (static_cast<ValueIndexT>(row) * static_cast<ValueIndexT>(channels) +
                                 static_cast<ValueIndexT>(channel)) *
                                    static_cast<ValueIndexT>(widthCapacity) +
                                static_cast<ValueIndexT>(timestep);
                            tile[lane][tileColumn] = paddedValues[paddedIndex];
                        }
                    }
                    __syncthreads();

#pragma unroll
                    for (uint32_t offset = 0; offset < kPaddedPackTileDim; offset += kPaddedPackBlockRows) {
                        const uint32_t tileRow = warpRow + offset;
                        const CoordIndexT timestep = timestepTileStart + static_cast<CoordIndexT>(tileRow);
                        const CoordIndexT channel = channelTileStart + static_cast<CoordIndexT>(lane);
                        if (timestep < rowLength && channel < channels) {
                            const ValueIndexT packedIndex =
                                (static_cast<ValueIndexT>(rowBegin) + static_cast<ValueIndexT>(timestep)) *
                                    static_cast<ValueIndexT>(channels) +
                                static_cast<ValueIndexT>(channel);
                            packedValues[packedIndex] = tile[tileRow][lane];
                        }
                    }
                    __syncthreads();

                    if (timestepTileStride >= rowLength - timestepTileStart) break;
                    timestepTileStart += timestepTileStride;
                }

                if (channelTileStride >= channels - channelTileStart) break;
                channelTileStart += channelTileStride;
            }
        }

        __syncthreads();
        if (rowStride >= batchSize - row) break;
        row += rowStride;
    }
}


uint32_t paddedDirectCopyRowsPerBlock(uint64_t maxRowBytes, uint64_t batchSize) {
    uint32_t rowsByPayload = 1;
    if (maxRowBytes <= kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 256;
    } else if (maxRowBytes <= 2 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 128;
    } else if (maxRowBytes <= 4 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 64;
    } else if (maxRowBytes <= 8 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 32;
    } else if (maxRowBytes <= 16 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 16;
    } else if (maxRowBytes <= 32 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 8;
    } else if (maxRowBytes <= 64 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 4;
    } else if (maxRowBytes <= 128 * kPaddedDirectCopyTargetBytesPerLane) {
        rowsByPayload = 2;
    }

    // Match the optimized materialization kernels' ~64-CTA parallelism floor.
    uint32_t rowsByParallelism = 1;
    if (batchSize >= 16384) {
        rowsByParallelism = 256;
    } else if (batchSize >= 8192) {
        rowsByParallelism = 128;
    } else if (batchSize >= 4096) {
        rowsByParallelism = 64;
    } else if (batchSize >= 2048) {
        rowsByParallelism = 32;
    } else if (batchSize >= 1024) {
        rowsByParallelism = 16;
    } else if (batchSize >= 512) {
        rowsByParallelism = 8;
    } else if (batchSize >= 256) {
        rowsByParallelism = 4;
    } else if (batchSize >= 128) {
        rowsByParallelism = 2;
    }
    return std::min(rowsByPayload, rowsByParallelism);
}

uint32_t paddedDirectCopyBlocksForRows(uint64_t batchSize, uint32_t rowsPerBlock) {
    const uint64_t blocks = batchSize / rowsPerBlock + (batchSize % rowsPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::max<uint64_t>(blocks, 1), static_cast<uint64_t>(kPaddedDirectCopyMaxGridDimension)));
}

uint32_t cappedPaddedPackGridDimension(uint64_t extent) {
    return static_cast<uint32_t>(
        std::min<uint64_t>(extent, static_cast<uint64_t>(kPaddedPackMaxGridDimension)));
}

uint64_t checkedPaddedPackMul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument("Padded ragged pack geometry overflows uint64_t.");
    }
    return a * b;
}

uint64_t paddedPackTileCount(uint64_t extent) {
    if (extent == 0) return 0;
    return 1 + (extent - 1) / kPaddedPackTileDim;
}

uint32_t paddedPackElementBytes(DataType dtype) {
    const float elementBytesFloat = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t elementBytes = static_cast<uint64_t>(elementBytesFloat);
    if (static_cast<float>(elementBytes) != elementBytesFloat ||
        (elementBytes != 1 && elementBytes != 2 && elementBytes != 4 && elementBytes != 8)) {
        throw std::invalid_argument(
            "Padded ragged pack supports whole-byte value element sizes of 1, 2, 4, or 8 bytes.");
    }
    return static_cast<uint32_t>(elementBytes);
}

void validatePackedToPaddedTensors(const Tensor& packedValues,
                                   const Tensor& rowOffsets,
                                   const Tensor& paddedValues,
                                   const PaddedRaggedPackLaunchPlan& plan,
                                   Stream& stream) {
    if (plan.batchSize == 0 || plan.channels == 0 || plan.widthCapacity == 0 ||
        plan.useChannelOneDirectCopy != (plan.channels == 1)) {
        throw std::invalid_argument("Padded ragged pack launch plan geometry is invalid.");
    }
    if (plan.useChannelOneDirectCopy) {
        const uint32_t expectedElementBytes = paddedPackElementBytes(plan.valuesDataType);
        const uint64_t expectedMaxRowBytes = checkedPaddedPackMul(plan.widthCapacity, expectedElementBytes);
        if (plan.channelTileBlocks != 0 || plan.timestepTileBlocks != 0 || plan.rowBlocks != 0 ||
            plan.directCopyElementBytes != expectedElementBytes ||
            plan.directCopyElementShift != paddedElementShift(expectedElementBytes) ||
            plan.directCopyMaxRowBytes != expectedMaxRowBytes || plan.directCopyRowsPerBlock == 0 ||
            plan.directCopyRowsPerBlock > kPaddedDirectCopyMaxRowsPerBlock ||
            !isPowerOfTwo(plan.directCopyRowsPerBlock) ||
            plan.directCopyRowBlocks == 0 || plan.directCopyRowBlocks > kPaddedDirectCopyMaxGridDimension ||
            plan.directCopyUse32BitRowIndexing !=
                (plan.batchSize <= std::numeric_limits<uint32_t>::max()) ||
            plan.directCopyUse32BitSpanIndexing !=
                (expectedMaxRowBytes <= std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("Padded ragged C==1 direct-copy launch plan geometry is invalid.");
        }
    } else if (plan.channelTileBlocks == 0 || plan.timestepTileBlocks == 0 || plan.rowBlocks == 0 ||
               plan.directCopyElementBytes != 0 || plan.directCopyElementShift != 0 ||
               plan.directCopyMaxRowBytes != 0 ||
               plan.directCopyRowsPerBlock != 0 || plan.directCopyRowBlocks != 0 ||
               plan.directCopyUse32BitRowIndexing || plan.directCopyUse32BitSpanIndexing) {
        throw std::invalid_argument("Padded ragged tiled launch plan geometry is invalid.");
    }
    const bool expected32BitCoordinates = plan.batchSize <= std::numeric_limits<uint32_t>::max() &&
                                          plan.maxTotalValues <= std::numeric_limits<uint32_t>::max() &&
                                          plan.channels <= std::numeric_limits<uint32_t>::max() &&
                                          plan.widthCapacity <= std::numeric_limits<uint32_t>::max();
    const bool expected32BitValueIndexing =
        expected32BitCoordinates && plan.selectedValueElements <= std::numeric_limits<uint32_t>::max() &&
        plan.packedValueElements <= std::numeric_limits<uint32_t>::max();
    if (plan.use32BitCoordinateIndexing != expected32BitCoordinates ||
        plan.use32BitIndexing != expected32BitValueIndexing ||
        plan.selectedValueBytes > paddedValues.getArraySizeInBytes() ||
        plan.packedValueBytes != packedValues.getArraySizeInBytes()) {
        throw std::invalid_argument("Padded ragged pack launch plan index/byte extent policy is inconsistent.");
    }
    // All immutable geometry/index-width policy was validated when the plan was
    // created. Runtime validation deliberately does not reconstruct tile counts
    // or rediscover the 32/64-bit dispatch policy.
    if (packedValues.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        packedValues.getPlacement().getDeviceNum() != stream.getGpuNum() ||
        rowOffsets.getPlacement() != packedValues.getPlacement() ||
        paddedValues.getPlacement() != packedValues.getPlacement()) {
        throw std::invalid_argument("Padded ragged pack tensors must share the execution GPU.");
    }
    if (!packedValues.isDenseContiguous() || !rowOffsets.isDenseContiguous() || !paddedValues.isDenseContiguous()) {
        throw std::invalid_argument("Padded ragged pack requires dense contiguous backing tensors.");
    }
    if (packedValues.getDataType() != plan.valuesDataType || paddedValues.getDataType() != plan.valuesDataType ||
        rowOffsets.getDataType() != plan.offsetsDataType ||
        !isCanonicalRowPartitionOffsetDataType(plan.offsetsDataType)) {
        throw std::invalid_argument("Padded ragged pack tensor dtypes do not match the launch plan.");
    }
    if (plan.batchSize == std::numeric_limits<uint64_t>::max() ||
        rowOffsets.getDimensions() != std::vector<uint64_t>({plan.batchSize + 1}) ||
        packedValues.getDimensions() != std::vector<uint64_t>({plan.maxTotalValues, plan.channels}) ||
        paddedValues.getNumDimensions() != 1 ||
        paddedValues.getTotalNumElements() < plan.selectedValueElements) {
        throw std::invalid_argument("Padded ragged pack tensor shapes do not match the launch plan.");
    }
}

template <typename OffsetT,
          typename RowIndexT,
          typename ValueIndexT,
          typename SpanIndexT,
          bool PackedToPadded,
          uint32_t RowsPerBlock>
void launchChannelOneDirectCopyGrouped(const Tensor& sourceValues,
                                       Tensor& destinationValues,
                                       const Tensor& rowOffsets,
                                       const PaddedRaggedPackLaunchPlan& plan,
                                       Stream& stream) {
    const unsigned char* source = reinterpret_cast<const unsigned char*>(sourceValues.getMemPtr<void>());
    unsigned char* destination = reinterpret_cast<unsigned char*>(destinationValues.getMemPtr<void>());
    channelOnePaddedRaggedDirectCopyKernel<
        OffsetT, RowIndexT, ValueIndexT, SpanIndexT, RowsPerBlock, PackedToPadded>
        <<<plan.directCopyRowBlocks, kPaddedDirectCopyThreadsPerBlock, 0, stream.getStream()>>>(
            source,
            rowOffsets.getMemPtr<OffsetT>(),
            destination,
            static_cast<RowIndexT>(plan.batchSize),
            static_cast<ValueIndexT>(plan.maxTotalValues),
            static_cast<ValueIndexT>(plan.widthCapacity),
            plan.directCopyElementShift);
}

template <typename OffsetT,
          typename RowIndexT,
          typename ValueIndexT,
          typename SpanIndexT,
          bool PackedToPadded>
void launchChannelOneDirectCopyRows(const Tensor& sourceValues,
                                    Tensor& destinationValues,
                                    const Tensor& rowOffsets,
                                    const PaddedRaggedPackLaunchPlan& plan,
                                    Stream& stream) {
#define THOR_PADDED_DIRECT_ROWS_CASE(N) \
    case N: \
        launchChannelOneDirectCopyGrouped<OffsetT, RowIndexT, ValueIndexT, SpanIndexT, PackedToPadded, N>( \
            sourceValues, destinationValues, rowOffsets, plan, stream); \
        return
    switch (plan.directCopyRowsPerBlock) {
        THOR_PADDED_DIRECT_ROWS_CASE(1);
        THOR_PADDED_DIRECT_ROWS_CASE(2);
        THOR_PADDED_DIRECT_ROWS_CASE(4);
        THOR_PADDED_DIRECT_ROWS_CASE(8);
        THOR_PADDED_DIRECT_ROWS_CASE(16);
        THOR_PADDED_DIRECT_ROWS_CASE(32);
        THOR_PADDED_DIRECT_ROWS_CASE(64);
        THOR_PADDED_DIRECT_ROWS_CASE(128);
        THOR_PADDED_DIRECT_ROWS_CASE(256);
        default:
            throw std::logic_error("Invalid padded ragged C==1 rows-per-CTA selection.");
    }
#undef THOR_PADDED_DIRECT_ROWS_CASE
}

template <typename OffsetT, typename ValueIndexT, typename SpanIndexT, bool PackedToPadded>
void dispatchChannelOneDirectCopyRowIndex(const Tensor& sourceValues,
                                          Tensor& destinationValues,
                                          const Tensor& rowOffsets,
                                          const PaddedRaggedPackLaunchPlan& plan,
                                          Stream& stream) {
    if (plan.directCopyUse32BitRowIndexing) {
        launchChannelOneDirectCopyRows<OffsetT, uint32_t, ValueIndexT, SpanIndexT, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    } else {
        launchChannelOneDirectCopyRows<OffsetT, uint64_t, ValueIndexT, SpanIndexT, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    }
}

template <typename OffsetT, typename ValueIndexT, bool PackedToPadded>
void dispatchChannelOneDirectCopySpanIndex(const Tensor& sourceValues,
                                           Tensor& destinationValues,
                                           const Tensor& rowOffsets,
                                           const PaddedRaggedPackLaunchPlan& plan,
                                           Stream& stream) {
    if (plan.directCopyUse32BitSpanIndexing) {
        dispatchChannelOneDirectCopyRowIndex<OffsetT, ValueIndexT, uint32_t, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    } else {
        dispatchChannelOneDirectCopyRowIndex<OffsetT, ValueIndexT, uint64_t, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    }
}

template <typename OffsetT, bool PackedToPadded>
void dispatchChannelOneDirectCopy(const Tensor& sourceValues,
                                  Tensor& destinationValues,
                                  const Tensor& rowOffsets,
                                  const PaddedRaggedPackLaunchPlan& plan,
                                  Stream& stream) {
    // Linear element addressing and byte-span traversal are independent. A
    // >4 GiB row therefore widens only the byte counter when its element
    // offsets still fit UINT32; a large batch can widen element addresses while
    // retaining 32-bit span traversal for short rows.
    if (plan.use32BitIndexing) {
        dispatchChannelOneDirectCopySpanIndex<OffsetT, uint32_t, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    } else {
        dispatchChannelOneDirectCopySpanIndex<OffsetT, uint64_t, PackedToPadded>(
            sourceValues, destinationValues, rowOffsets, plan, stream);
    }
}

template <typename ElementT, typename OffsetT, typename CoordIndexT, typename ValueIndexT>
void launchTypedPackedToPadded(const Tensor& packedValues,
                               Tensor& paddedValues,
                               const Tensor& rowOffsets,
                               const PaddedRaggedPackLaunchPlan& plan,
                               Stream& stream) {
    const ElementT* packed = reinterpret_cast<const ElementT*>(packedValues.getMemPtr<void>());
    ElementT* padded = reinterpret_cast<ElementT*>(paddedValues.getMemPtr<void>());
    const dim3 grid(plan.channelTileBlocks, plan.timestepTileBlocks, plan.rowBlocks);
    const dim3 block(kPaddedPackTileDim, kPaddedPackBlockRows);
    packedToPaddedRaggedSequenceTiledKernel<ElementT, OffsetT, CoordIndexT, ValueIndexT>
        <<<grid, block, 0, stream.getStream()>>>(packed,
                                                rowOffsets.getMemPtr<OffsetT>(),
                                                padded,
                                                static_cast<CoordIndexT>(plan.batchSize),
                                                static_cast<CoordIndexT>(plan.maxTotalValues),
                                                static_cast<CoordIndexT>(plan.channels),
                                                static_cast<CoordIndexT>(plan.widthCapacity));
}

template <typename OffsetT, typename CoordIndexT, typename ValueIndexT>
void dispatchPackedToPaddedElementSize(const Tensor& packedValues,
                                       Tensor& paddedValues,
                                       const Tensor& rowOffsets,
                                       const PaddedRaggedPackLaunchPlan& plan,
                                       Stream& stream) {
    switch (paddedPackElementBytes(plan.valuesDataType)) {
        case 1:
            launchTypedPackedToPadded<uint8_t, OffsetT, CoordIndexT, ValueIndexT>(
                packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        case 2:
            launchTypedPackedToPadded<uint16_t, OffsetT, CoordIndexT, ValueIndexT>(
                packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        case 4:
            launchTypedPackedToPadded<uint32_t, OffsetT, CoordIndexT, ValueIndexT>(
                packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        case 8:
            launchTypedPackedToPadded<uint64_t, OffsetT, CoordIndexT, ValueIndexT>(
                packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged pack value element size is invalid.");
    }
}

template <typename OffsetT>
void dispatchPackedToPaddedIndex(const Tensor& packedValues,
                                 Tensor& paddedValues,
                                 const Tensor& rowOffsets,
                                 const PaddedRaggedPackLaunchPlan& plan,
                                 Stream& stream) {
    if (plan.useChannelOneDirectCopy) {
        dispatchChannelOneDirectCopy<OffsetT, true>(packedValues, paddedValues, rowOffsets, plan, stream);
        return;
    }
    if (plan.use32BitCoordinateIndexing) {
        if (plan.use32BitIndexing) {
            dispatchPackedToPaddedElementSize<OffsetT, uint32_t, uint32_t>(
                packedValues, paddedValues, rowOffsets, plan, stream);
        } else {
            dispatchPackedToPaddedElementSize<OffsetT, uint32_t, uint64_t>(
                packedValues, paddedValues, rowOffsets, plan, stream);
        }
    } else {
        dispatchPackedToPaddedElementSize<OffsetT, uint64_t, uint64_t>(
            packedValues, paddedValues, rowOffsets, plan, stream);
    }
}

template <typename ElementT, typename OffsetT, typename CoordIndexT, typename ValueIndexT>
void launchTypedPaddedToPacked(const Tensor& paddedValues,
                               Tensor& packedValues,
                               const Tensor& rowOffsets,
                               const PaddedRaggedUnpackLaunchPlan& plan,
                               Stream& stream) {
    const ElementT* padded = reinterpret_cast<const ElementT*>(paddedValues.getMemPtr<void>());
    ElementT* packed = reinterpret_cast<ElementT*>(packedValues.getMemPtr<void>());
    const dim3 grid(plan.channelTileBlocks, plan.timestepTileBlocks, plan.rowBlocks);
    const dim3 block(kPaddedPackTileDim, kPaddedPackBlockRows);
    paddedToPackedRaggedSequenceTiledKernel<ElementT, OffsetT, CoordIndexT, ValueIndexT>
        <<<grid, block, 0, stream.getStream()>>>(padded,
                                                rowOffsets.getMemPtr<OffsetT>(),
                                                packed,
                                                static_cast<CoordIndexT>(plan.batchSize),
                                                static_cast<CoordIndexT>(plan.maxTotalValues),
                                                static_cast<CoordIndexT>(plan.channels),
                                                static_cast<CoordIndexT>(plan.widthCapacity));
}

template <typename OffsetT, typename CoordIndexT, typename ValueIndexT>
void dispatchPaddedToPackedElementSize(const Tensor& paddedValues,
                                       Tensor& packedValues,
                                       const Tensor& rowOffsets,
                                       const PaddedRaggedUnpackLaunchPlan& plan,
                                       Stream& stream) {
    switch (paddedPackElementBytes(plan.valuesDataType)) {
        case 1:
            launchTypedPaddedToPacked<uint8_t, OffsetT, CoordIndexT, ValueIndexT>(
                paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        case 2:
            launchTypedPaddedToPacked<uint16_t, OffsetT, CoordIndexT, ValueIndexT>(
                paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        case 4:
            launchTypedPaddedToPacked<uint32_t, OffsetT, CoordIndexT, ValueIndexT>(
                paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        case 8:
            launchTypedPaddedToPacked<uint64_t, OffsetT, CoordIndexT, ValueIndexT>(
                paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged unpack value element size is invalid.");
    }
}

template <typename OffsetT>
void dispatchPaddedToPackedIndex(const Tensor& paddedValues,
                                 Tensor& packedValues,
                                 const Tensor& rowOffsets,
                                 const PaddedRaggedUnpackLaunchPlan& plan,
                                 Stream& stream) {
    if (plan.useChannelOneDirectCopy) {
        dispatchChannelOneDirectCopy<OffsetT, false>(paddedValues, packedValues, rowOffsets, plan, stream);
        return;
    }
    if (plan.use32BitCoordinateIndexing) {
        if (plan.use32BitIndexing) {
            dispatchPaddedToPackedElementSize<OffsetT, uint32_t, uint32_t>(
                paddedValues, packedValues, rowOffsets, plan, stream);
        } else {
            dispatchPaddedToPackedElementSize<OffsetT, uint32_t, uint64_t>(
                paddedValues, packedValues, rowOffsets, plan, stream);
        }
    } else {
        dispatchPaddedToPackedElementSize<OffsetT, uint64_t, uint64_t>(
            paddedValues, packedValues, rowOffsets, plan, stream);
    }
}

void validatePaddedTailZeroTensors(const Tensor& paddedValues,
                                   const Tensor& rowOffsets,
                                   const PaddedRaggedTailZeroLaunchPlan& plan,
                                   Stream& stream) {
    if (plan.batchSize == 0 || plan.channels == 0 || plan.widthCapacity == 0 || plan.channelBlocks == 0 ||
        plan.rowBlocks == 0 || plan.threadsPerBlock == 0 || plan.channelsPerBlock == 0 ||
        !isPowerOfTwo(plan.lanesPerChannel) || plan.lanesPerChannel > kPaddedTailZeroMaxThreads ||
        plan.laneShift >= 32 || (1U << plan.laneShift) != plan.lanesPerChannel ||
        plan.channelsPerBlock > kPaddedTailZeroMaxThreads / plan.lanesPerChannel ||
        plan.channelBlocks > kPaddedTailZeroMaxGridDimension ||
        plan.rowBlocks > kPaddedTailZeroMaxGridDimension ||
        plan.threadsPerBlock > kPaddedTailZeroMaxThreads ||
        plan.threadsPerBlock < plan.channelsPerBlock * plan.lanesPerChannel ||
        plan.threadsPerBlock % kPaddedTailZeroWarpThreads != 0) {
        throw std::invalid_argument("Padded ragged tail-zero launch plan geometry is invalid.");
    }
    const uint64_t expectedElements = checkedTailZeroMul(
        checkedTailZeroMul(plan.batchSize, plan.channels), plan.widthCapacity);
    const uint64_t expectedBytes = checkedTailZeroMul(expectedElements, plan.elementBytes);
    const uint64_t expectedMaxTailBytes = checkedTailZeroMul(plan.widthCapacity, plan.elementBytes);
    const bool expected32BitCoordinates = plan.batchSize <= std::numeric_limits<uint32_t>::max() &&
                                          plan.channels <= std::numeric_limits<uint32_t>::max() &&
                                          plan.widthCapacity <= std::numeric_limits<uint32_t>::max();
    if (plan.selectedValueElements != expectedElements || plan.selectedValueBytes != expectedBytes ||
        plan.maxTailBytes != expectedMaxTailBytes ||
        plan.use32BitCoordinateIndexing != expected32BitCoordinates ||
        plan.use32BitIndexing !=
            (expected32BitCoordinates && expectedElements <= std::numeric_limits<uint32_t>::max()) ||
        plan.use32BitSpanIndexing != (expectedMaxTailBytes <= std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Padded ragged tail-zero launch plan selected extent is inconsistent.");
    }
    if (paddedValues.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        paddedValues.getPlacement().getDeviceNum() != stream.getGpuNum() ||
        rowOffsets.getPlacement() != paddedValues.getPlacement()) {
        throw std::invalid_argument("Padded ragged tail-zero tensors must share the execution GPU.");
    }
    if (!paddedValues.isDenseContiguous() || !rowOffsets.isDenseContiguous()) {
        throw std::invalid_argument("Padded ragged tail zeroing requires dense contiguous storage.");
    }
    if (paddedValues.getDataType() != plan.valuesDataType || rowOffsets.getDataType() != plan.offsetsDataType ||
        !isCanonicalRowPartitionOffsetDataType(rowOffsets.getDataType())) {
        throw std::invalid_argument("Padded ragged tail-zero tensor dtypes do not match the launch plan.");
    }
    if (plan.batchSize == std::numeric_limits<uint64_t>::max() ||
        rowOffsets.getDimensions() != std::vector<uint64_t>({plan.batchSize + 1})) {
        throw std::invalid_argument("Padded ragged tail-zero offsets do not match the launch plan.");
    }
    if (paddedValues.getNumDimensions() != 1 || paddedValues.getTotalNumElements() < plan.selectedValueElements) {
        throw std::invalid_argument("Padded ragged tail-zero storage is smaller than the selected dense prefix.");
    }
    if ((plan.elementBytes != 1 && plan.elementBytes != 2 && plan.elementBytes != 4 && plan.elementBytes != 8) ||
        plan.elementShift != paddedElementShift(plan.elementBytes)) {
        throw std::invalid_argument("Padded ragged tail-zero launch plan element size is invalid.");
    }
}

template <typename OffsetT, typename CoordIndexT, typename ValueIndexT, typename SpanIndexT>
void launchPaddedTailZeroKernel(Tensor& paddedValues,
                                const Tensor& rowOffsets,
                                const PaddedRaggedTailZeroLaunchPlan& plan,
                                Stream& stream) {
    unsigned char* padded = reinterpret_cast<unsigned char*>(paddedValues.getMemPtr<void>());
    const dim3 grid(plan.channelBlocks, plan.rowBlocks);
    const dim3 block(plan.threadsPerBlock);
    zeroPaddedRaggedSequenceTailKernel<OffsetT, CoordIndexT, ValueIndexT, SpanIndexT>
        <<<grid, block, 0, stream.getStream()>>>(padded,
                                                rowOffsets.getMemPtr<OffsetT>(),
                                                static_cast<CoordIndexT>(plan.batchSize),
                                                static_cast<CoordIndexT>(plan.channels),
                                                static_cast<CoordIndexT>(plan.widthCapacity),
                                                plan.elementShift,
                                                plan.laneShift,
                                                plan.channelsPerBlock);
}

template <typename OffsetT, typename CoordIndexT, typename ValueIndexT>
void dispatchPaddedTailZeroSpanIndex(Tensor& paddedValues,
                                     const Tensor& rowOffsets,
                                     const PaddedRaggedTailZeroLaunchPlan& plan,
                                     Stream& stream) {
    if (plan.use32BitSpanIndexing) {
        launchPaddedTailZeroKernel<OffsetT, CoordIndexT, ValueIndexT, uint32_t>(
            paddedValues, rowOffsets, plan, stream);
    } else {
        launchPaddedTailZeroKernel<OffsetT, CoordIndexT, ValueIndexT, uint64_t>(
            paddedValues, rowOffsets, plan, stream);
    }
}

template <typename OffsetT, typename CoordIndexT>
void dispatchPaddedTailZeroValueIndex(Tensor& paddedValues,
                                      const Tensor& rowOffsets,
                                      const PaddedRaggedTailZeroLaunchPlan& plan,
                                      Stream& stream) {
    if (plan.use32BitIndexing) {
        dispatchPaddedTailZeroSpanIndex<OffsetT, CoordIndexT, uint32_t>(
            paddedValues, rowOffsets, plan, stream);
    } else {
        dispatchPaddedTailZeroSpanIndex<OffsetT, CoordIndexT, uint64_t>(
            paddedValues, rowOffsets, plan, stream);
    }
}

template <typename OffsetT>
void dispatchPaddedTailZeroIndex(Tensor& paddedValues,
                                 const Tensor& rowOffsets,
                                 const PaddedRaggedTailZeroLaunchPlan& plan,
                                 Stream& stream) {
    if (plan.use32BitCoordinateIndexing) {
        dispatchPaddedTailZeroValueIndex<OffsetT, uint32_t>(paddedValues, rowOffsets, plan, stream);
    } else {
        dispatchPaddedTailZeroValueIndex<OffsetT, uint64_t>(paddedValues, rowOffsets, plan, stream);
    }
}

}  // namespace

PaddedRaggedPackLaunchPlan preparePaddedRaggedPackLaunchPlan(uint64_t batchSize,
                                                              uint64_t maxTotalValues,
                                                              uint64_t channels,
                                                              uint64_t widthCapacity,
                                                              DataType valuesDataType,
                                                              DataType offsetsDataType) {
    if (batchSize == 0 || channels == 0 || batchSize == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Padded ragged pack planning requires nonzero batch and channel extents.");
    }
    if (!isCanonicalRowPartitionOffsetDataType(offsetsDataType)) {
        throw std::invalid_argument("Padded ragged pack offsets must use UINT32 or UINT64.");
    }
    if (!canonicalRowPartitionOffsetCanRepresent(offsetsDataType, maxTotalValues)) {
        throw std::invalid_argument("Padded ragged pack offset dtype cannot represent max_total_values.");
    }
    const uint32_t elementBytes = paddedPackElementBytes(valuesDataType);

    PaddedRaggedPackLaunchPlan plan;
    plan.valuesDataType = valuesDataType;
    plan.offsetsDataType = offsetsDataType;
    plan.batchSize = batchSize;
    plan.maxTotalValues = maxTotalValues;
    plan.channels = channels;
    plan.widthCapacity = widthCapacity;
    plan.selectedValueElements =
        checkedPaddedPackMul(checkedPaddedPackMul(batchSize, channels), widthCapacity);
    plan.packedValueElements = checkedPaddedPackMul(maxTotalValues, channels);
    plan.selectedValueBytes = checkedPaddedPackMul(plan.selectedValueElements, elementBytes);
    plan.packedValueBytes = checkedPaddedPackMul(plan.packedValueElements, elementBytes);

    if (widthCapacity != 0) {
        plan.use32BitCoordinateIndexing = batchSize <= std::numeric_limits<uint32_t>::max() &&
                                          maxTotalValues <= std::numeric_limits<uint32_t>::max() &&
                                          channels <= std::numeric_limits<uint32_t>::max() &&
                                          widthCapacity <= std::numeric_limits<uint32_t>::max();
        plan.use32BitIndexing = plan.use32BitCoordinateIndexing &&
                                plan.selectedValueElements <= std::numeric_limits<uint32_t>::max() &&
                                plan.packedValueElements <= std::numeric_limits<uint32_t>::max();
        if (channels == 1) {
            plan.useChannelOneDirectCopy = true;
            plan.directCopyElementBytes = elementBytes;
            plan.directCopyElementShift = paddedElementShift(elementBytes);
            plan.directCopyMaxRowBytes = checkedPaddedPackMul(widthCapacity, elementBytes);
            plan.directCopyRowsPerBlock = paddedDirectCopyRowsPerBlock(plan.directCopyMaxRowBytes, batchSize);
            plan.directCopyRowBlocks = paddedDirectCopyBlocksForRows(batchSize, plan.directCopyRowsPerBlock);
            plan.directCopyUse32BitRowIndexing = batchSize <= std::numeric_limits<uint32_t>::max();
            plan.directCopyUse32BitSpanIndexing =
                plan.directCopyMaxRowBytes <= std::numeric_limits<uint32_t>::max();
        } else {
            const uint64_t channelTiles = paddedPackTileCount(channels);
            const uint64_t timestepTiles = paddedPackTileCount(widthCapacity);
            plan.channelTileBlocks = cappedPaddedPackGridDimension(channelTiles);
            plan.timestepTileBlocks = cappedPaddedPackGridDimension(timestepTiles);
            plan.rowBlocks = cappedPaddedPackGridDimension(batchSize);
        }
    }
    return plan;
}

PaddedRaggedUnpackLaunchPlan preparePaddedRaggedUnpackLaunchPlan(uint64_t batchSize,
                                                                  uint64_t maxTotalValues,
                                                                  uint64_t channels,
                                                                  uint64_t widthCapacity,
                                                                  DataType valuesDataType,
                                                                  DataType offsetsDataType) {
    return preparePaddedRaggedPackLaunchPlan(
        batchSize, maxTotalValues, channels, widthCapacity, valuesDataType, offsetsDataType);
}

PaddedRaggedTailZeroLaunchPlan preparePaddedRaggedTailZeroLaunchPlan(uint64_t batchSize,
                                                                    uint64_t channels,
                                                                    uint64_t widthCapacity,
                                                                    DataType valuesDataType,
                                                                    DataType offsetsDataType) {
    if (batchSize == 0 || channels == 0 || batchSize == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Padded ragged tail-zero planning requires nonzero batch and channel extents.");
    }
    if (!isCanonicalRowPartitionOffsetDataType(offsetsDataType)) {
        throw std::invalid_argument("Padded ragged tail-zero offsets must use UINT32 or UINT64.");
    }
    const uint32_t elementBytes = paddedTailZeroElementBytes(valuesDataType);

    PaddedRaggedTailZeroLaunchPlan plan;
    plan.valuesDataType = valuesDataType;
    plan.offsetsDataType = offsetsDataType;
    plan.batchSize = batchSize;
    plan.channels = channels;
    plan.widthCapacity = widthCapacity;
    plan.selectedValueElements = checkedTailZeroMul(checkedTailZeroMul(batchSize, channels), widthCapacity);
    plan.selectedValueBytes = checkedTailZeroMul(plan.selectedValueElements, elementBytes);
    plan.maxTailBytes = checkedTailZeroMul(widthCapacity, elementBytes);
    plan.elementBytes = elementBytes;
    plan.elementShift = paddedElementShift(elementBytes);

    if (widthCapacity != 0) {
        plan.use32BitCoordinateIndexing = batchSize <= std::numeric_limits<uint32_t>::max() &&
                                          channels <= std::numeric_limits<uint32_t>::max() &&
                                          widthCapacity <= std::numeric_limits<uint32_t>::max();
        plan.use32BitIndexing = plan.use32BitCoordinateIndexing &&
                                plan.selectedValueElements <= std::numeric_limits<uint32_t>::max();
        plan.use32BitSpanIndexing = plan.maxTailBytes <= std::numeric_limits<uint32_t>::max();
        plan.lanesPerChannel = paddedTailZeroLanesPerChannel(plan.maxTailBytes);
        plan.laneShift = paddedTailZeroLaneShift(plan.lanesPerChannel);
        plan.channelsPerBlock = static_cast<uint32_t>(std::min<uint64_t>(
            channels, kPaddedTailZeroMaxThreads / plan.lanesPerChannel));
        const uint32_t activeThreads = plan.lanesPerChannel * plan.channelsPerBlock;
        plan.threadsPerBlock = roundTailZeroThreadsToWarp(activeThreads);
        const uint64_t channelGroups =
            1 + (channels - 1) / static_cast<uint64_t>(plan.channelsPerBlock);
        plan.channelBlocks = cappedTailZeroGridDimension(channelGroups);
        plan.rowBlocks = cappedTailZeroGridDimension(batchSize);
    }
    return plan;
}

void launchZeroPaddedRaggedSequenceTail(Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        const PaddedRaggedTailZeroLaunchPlan& plan,
                                        Stream& stream) {
    if (plan.empty()) {
        return;
    }
    validatePaddedTailZeroTensors(paddedValues, rowOffsets, plan, stream);
    switch (plan.offsetsDataType) {
        case DataType::UINT32:
            dispatchPaddedTailZeroIndex<uint32_t>(paddedValues, rowOffsets, plan, stream);
            break;
        case DataType::UINT64:
            dispatchPaddedTailZeroIndex<uint64_t>(paddedValues, rowOffsets, plan, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged tail-zero offsets must use UINT32 or UINT64.");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        const PaddedRaggedPackLaunchPlan& plan,
                                        Stream& stream) {
    if (plan.empty()) {
        return;
    }
    validatePackedToPaddedTensors(packedValues, rowOffsets, paddedValues, plan, stream);
    switch (plan.offsetsDataType) {
        case DataType::UINT32:
            dispatchPackedToPaddedIndex<uint32_t>(packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        case DataType::UINT64:
            dispatchPackedToPaddedIndex<uint64_t>(packedValues, paddedValues, rowOffsets, plan, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged pack offsets must use UINT32 or UINT64.");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream) {
    if (widthCapacity == 0) {
        return;
    }
    if (packedValues.getNumDimensions() != 2 || packedValues.getDimensions()[1] != channels) {
        throw std::invalid_argument("Padded ragged pack packed tensor shape does not match its channel count.");
    }
    const PaddedRaggedPackLaunchPlan plan = preparePaddedRaggedPackLaunchPlan(
        batchSize,
        packedValues.getDimensions()[0],
        channels,
        widthCapacity,
        packedValues.getDataType(),
        rowOffsets.getDataType());
    launchPackedToPaddedRaggedSequence(packedValues, rowOffsets, paddedValues, plan, stream);
}

void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        const PaddedRaggedUnpackLaunchPlan& plan,
                                        Stream& stream) {
    if (plan.empty()) {
        return;
    }
    validatePackedToPaddedTensors(packedValues, rowOffsets, paddedValues, plan, stream);
    switch (plan.offsetsDataType) {
        case DataType::UINT32:
            dispatchPaddedToPackedIndex<uint32_t>(paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        case DataType::UINT64:
            dispatchPaddedToPackedIndex<uint64_t>(paddedValues, packedValues, rowOffsets, plan, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged unpack offsets must use UINT32 or UINT64.");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream) {
    if (widthCapacity == 0) {
        return;
    }
    if (packedValues.getNumDimensions() != 2 || packedValues.getDimensions()[1] != channels) {
        throw std::invalid_argument("Padded ragged unpack packed tensor shape does not match its channel count.");
    }
    const PaddedRaggedUnpackLaunchPlan plan = preparePaddedRaggedUnpackLaunchPlan(
        batchSize,
        packedValues.getDimensions()[0],
        channels,
        widthCapacity,
        packedValues.getDataType(),
        rowOffsets.getDataType());
    launchPaddedToPackedRaggedSequence(paddedValues, rowOffsets, packedValues, plan, stream);
}

}  // namespace ThorImplementation
