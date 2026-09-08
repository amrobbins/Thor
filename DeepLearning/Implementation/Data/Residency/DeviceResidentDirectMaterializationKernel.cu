#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kMaxRowsPerBlock = kThreadsPerBlock;
constexpr uint64_t kTargetBytesPerLane = 32;
constexpr uint32_t kMaxPortableBlocks = 65535;

static_assert(sizeof(ulonglong4_32a) == 32);
static_assert(alignof(ulonglong4_32a) == 32);

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const uint8_t *source,
                                                            const uint8_t *destination) {
    // Field length does not constrain bulk width. A final source packet may
    // extend beyond the logical compact-record field (and, for the final field
    // of the final record, into Thor's 128-byte tensor allocation padding).
    // The destination remainder is always stored with its exact logical size.
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
__device__ __forceinline__ void storeExactTailFromPacket(CopyT packet,
                                                         CopyT *destination) {
    static_assert(TailBytes > 0);
    static_assert(TailBytes < sizeof(CopyT));
    static_assert(sizeof(RawPacketBytes<sizeof(CopyT)>) == sizeof(CopyT));
    static_assert(sizeof(ExactTailBytes<TailBytes>) == TailBytes);

    // One complete aligned source packet is intentionally loaded even when the
    // logical field has fewer bytes left. Reading into the next compact-record
    // field is harmless; at the end of recordStorage, Thor's 128-byte tensor
    // padding makes the same load physically safe. Expose exactly TailBytes to
    // the compiler for the destination aggregate store so the next destination
    // row is never overwritten.
    const RawPacketBytes<sizeof(CopyT)> packetBytes =
        cuda::std::bit_cast<RawPacketBytes<sizeof(CopyT)>>(packet);
    ExactTailBytes<TailBytes> tail;
#pragma unroll
    for (uint32_t byte = 0; byte < TailBytes; ++byte) {
        tail.bytes[byte] = packetBytes.bytes[byte];
    }
    *reinterpret_cast<ExactTailBytes<TailBytes> *>(destination) = tail;
}

template <typename CopyT>
__device__ __attribute__((noinline)) void copyExactTailPacket(const CopyT *source,
                                                              CopyT *destination,
                                                              uint32_t tailBytes) {
    static_assert(sizeof(CopyT) == 2 || sizeof(CopyT) == 4 || sizeof(CopyT) == 8 ||
                  sizeof(CopyT) == 16 || sizeof(CopyT) == 32);
    if (tailBytes == 0 || tailBytes >= sizeof(CopyT)) return;

    // Keep the exact-size cases out of the heavily specialized row-group
    // kernels. A single designated lane performs the full-width source load
    // and one exact logical aggregate copy to the destination.
    const CopyT packet = *source;
#define THOR_DIRECT_TAIL_CASE(N)                  \
    case N:                                       \
        if constexpr (N < sizeof(CopyT)) {        \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                         \
        return
    switch (tailBytes) {
        THOR_DIRECT_TAIL_CASE(1);
        THOR_DIRECT_TAIL_CASE(2);
        THOR_DIRECT_TAIL_CASE(3);
        THOR_DIRECT_TAIL_CASE(4);
        THOR_DIRECT_TAIL_CASE(5);
        THOR_DIRECT_TAIL_CASE(6);
        THOR_DIRECT_TAIL_CASE(7);
        THOR_DIRECT_TAIL_CASE(8);
        THOR_DIRECT_TAIL_CASE(9);
        THOR_DIRECT_TAIL_CASE(10);
        THOR_DIRECT_TAIL_CASE(11);
        THOR_DIRECT_TAIL_CASE(12);
        THOR_DIRECT_TAIL_CASE(13);
        THOR_DIRECT_TAIL_CASE(14);
        THOR_DIRECT_TAIL_CASE(15);
        THOR_DIRECT_TAIL_CASE(16);
        THOR_DIRECT_TAIL_CASE(17);
        THOR_DIRECT_TAIL_CASE(18);
        THOR_DIRECT_TAIL_CASE(19);
        THOR_DIRECT_TAIL_CASE(20);
        THOR_DIRECT_TAIL_CASE(21);
        THOR_DIRECT_TAIL_CASE(22);
        THOR_DIRECT_TAIL_CASE(23);
        THOR_DIRECT_TAIL_CASE(24);
        THOR_DIRECT_TAIL_CASE(25);
        THOR_DIRECT_TAIL_CASE(26);
        THOR_DIRECT_TAIL_CASE(27);
        THOR_DIRECT_TAIL_CASE(28);
        THOR_DIRECT_TAIL_CASE(29);
        THOR_DIRECT_TAIL_CASE(30);
        THOR_DIRECT_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_DIRECT_TAIL_CASE
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyAlignedDirectField(const uint8_t *sourceBytes,
                                                       uint8_t *destinationBytes,
                                                       ItemIndexT fieldBytes,
                                                       uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);
    const ItemIndexT fieldItems = fieldBytes / static_cast<ItemIndexT>(sizeof(CopyT));

    ItemIndexT item = static_cast<ItemIndexT>(lane);
    while (item < fieldItems) {
        destination[item] = source[item];

        // Break before the increment when this was the final item owned by the
        // lane. Besides avoiding one unnecessary add, this keeps the 32-bit
        // fast path correct even when fieldItems is close to UINT32_MAX.
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerRow);
        if (fieldItems - item <= laneStride) break;
        item += laneStride;
    }

    if constexpr (sizeof(CopyT) > 1) {
        if (lane == 0) {
            const uint32_t tailBytes = static_cast<uint32_t>(
                fieldBytes % static_cast<ItemIndexT>(sizeof(CopyT)));
            if (tailBytes != 0) {
                copyExactTailPacket(source + fieldItems, destination + fieldItems, tailBytes);
            }
        }
    }
}

template <typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyDirectField(const uint8_t *sourceBytes,
                                                uint8_t *destinationBytes,
                                                ItemIndexT fieldBytes,
                                                uint32_t lane) {
    // Compact-record and packed destination row strides may be odd, so copy
    // alignment is a property of this selected row, not of the launch as a
    // whole. Choose the widest safe packet from the actual addresses. Field
    // length is intentionally absent from this decision; only the final exact
    // Tail<N> is narrow.
    switch (widestAlignedCopyWidth(sourceBytes, destinationBytes)) {
        case 32:
            copyAlignedDirectField<ulonglong4_32a, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
        case 16:
            copyAlignedDirectField<uint4, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
        case 8:
            copyAlignedDirectField<uint64_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
        case 4:
            copyAlignedDirectField<uint32_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
        case 2:
            copyAlignedDirectField<uint16_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
        default:
            copyAlignedDirectField<uint8_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, fieldBytes, lane);
            break;
    }
}

/**
 * Copy independent compact-record fields with a compile-time CUDA lane group
 * per destination row. The row grouping follows payload bytes rather than the
 * selected transaction width, so a 16-byte-aligned field and a 32-byte-aligned
 * field of the same size receive the same amount of lane-level parallelism.
 *
 * The transaction width is selected independently for each selected row from
 * its actual source/destination alignment. This matters for compact byte-packed
 * records and odd destination row strides: aligned rows can still use wide
 * bulk packets even when neighboring rows cannot.
 */
template <typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
__global__ void materializeDirectFieldKernel(
    const uint8_t *__restrict__ records,
    const uint64_t *__restrict__ rowIndices,
    uint8_t *__restrict__ destination,
    RowIndexT batchSize,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    ItemIndexT fieldBytes) {
    static_assert(RowsPerBlock >= 1 && RowsPerBlock <= kMaxRowsPerBlock,
                  "Device-resident direct rows per CTA are out of range.");
    static_assert(kThreadsPerBlock % RowsPerBlock == 0,
                  "Device-resident direct row groups must tile a CTA.");
    constexpr uint32_t kLanesPerRow = kThreadsPerBlock / RowsPerBlock;

    const uint32_t rowSlot = threadIdx.x / kLanesPerRow;
    const uint32_t lane = threadIdx.x - rowSlot * kLanesPerRow;
    const uint32_t rowStride = gridDim.x * RowsPerBlock;

    RowIndexT rowBase = static_cast<RowIndexT>(blockIdx.x) * RowsPerBlock;
    while (rowBase < batchSize) {
        if (static_cast<RowIndexT>(rowSlot) < batchSize - rowBase) {
            const RowIndexT batchRow = rowBase + static_cast<RowIndexT>(rowSlot);

            // Lanes assigned to the same row read the same index. CUDA can
            // broadcast/cache this uniform address efficiently, so publishing
            // sourceRow through shared memory would add synchronization without
            // removing meaningful global-memory traffic.
            const uint64_t sourceRow = rowIndices[batchRow];
            if (sourceRow < numExamples) {
                const uint8_t *source =
                    records + sourceRow * recordSizeBytes + fieldOffsetBytes;
                uint8_t *rowDestination =
                    destination + static_cast<uint64_t>(batchRow) * fieldBytes;
                copyDirectField<ItemIndexT, kLanesPerRow>(
                    source, rowDestination, fieldBytes, lane);
            }
            // Preserve the historical contract: an invalid resident row index
            // leaves the corresponding destination row untouched.
        }

        // Use subtraction and break-before-increment so the normal UINT32 row
        // specialization cannot wrap on its terminal grid-stride iteration.
        if (static_cast<RowIndexT>(rowStride) >= batchSize - rowBase) break;
        rowBase += static_cast<RowIndexT>(rowStride);
    }
}

uint32_t rowsPerBlockFor(uint64_t fieldBytes, uint64_t batchSize) {
    // Target about 32 bytes of useful row payload per lane. Transaction width
    // is deliberately not part of this policy: narrower alignment may require
    // multiple 16/8/4/2/1-byte transactions per lane, but consecutive lanes
    // still operate on adjacent contiguous items.
    uint32_t rowsByPayload = 1;
    if (fieldBytes <= kTargetBytesPerLane) {
        rowsByPayload = 256;
    } else if (fieldBytes <= 2 * kTargetBytesPerLane) {
        rowsByPayload = 128;
    } else if (fieldBytes <= 4 * kTargetBytesPerLane) {
        rowsByPayload = 64;
    } else if (fieldBytes <= 8 * kTargetBytesPerLane) {
        rowsByPayload = 32;
    } else if (fieldBytes <= 16 * kTargetBytesPerLane) {
        rowsByPayload = 16;
    } else if (fieldBytes <= 32 * kTargetBytesPerLane) {
        rowsByPayload = 8;
    } else if (fieldBytes <= 64 * kTargetBytesPerLane) {
        rowsByPayload = 4;
    } else if (fieldBytes <= 128 * kTargetBytesPerLane) {
        rowsByPayload = 2;
    }

    // Keep roughly 64 CTAs available for small fields instead of packing a
    // small batch into only one or two blocks. Once the batch is large enough,
    // payload geometry alone determines the row grouping.
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

template <uint32_t RowsPerBlock>
uint32_t blocksForRows(uint64_t batchSize) {
    const uint64_t blocks =
        batchSize / RowsPerBlock + (batchSize % RowsPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
void launchGrouped(const uint8_t *records,
                   const uint64_t *rowIndices,
                   uint8_t *destination,
                   RowIndexT batchSize,
                   uint64_t numExamples,
                   uint64_t recordSizeBytes,
                   uint64_t fieldOffsetBytes,
                   ItemIndexT fieldBytes,
                   cudaStream_t stream) {
    const uint32_t blocks =
        blocksForRows<RowsPerBlock>(static_cast<uint64_t>(batchSize));
    materializeDirectFieldKernel<RowIndexT, ItemIndexT, RowsPerBlock>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes);
    CUDA_CHECK(cudaGetLastError());
}

template <typename RowIndexT, typename ItemIndexT>
void launchForGrouping(const uint8_t *records,
                       const uint64_t *rowIndices,
                       uint8_t *destination,
                       RowIndexT batchSize,
                       uint64_t numExamples,
                       uint64_t recordSizeBytes,
                       uint64_t fieldOffsetBytes,
                       ItemIndexT fieldBytes,
                       cudaStream_t stream) {
    switch (rowsPerBlockFor(static_cast<uint64_t>(fieldBytes),
                            static_cast<uint64_t>(batchSize))) {
        case 1:
            launchGrouped<RowIndexT, ItemIndexT, 1>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 2:
            launchGrouped<RowIndexT, ItemIndexT, 2>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 4:
            launchGrouped<RowIndexT, ItemIndexT, 4>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 8:
            launchGrouped<RowIndexT, ItemIndexT, 8>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 16:
            launchGrouped<RowIndexT, ItemIndexT, 16>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 32:
            launchGrouped<RowIndexT, ItemIndexT, 32>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 64:
            launchGrouped<RowIndexT, ItemIndexT, 64>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 128:
            launchGrouped<RowIndexT, ItemIndexT, 128>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        case 256:
            launchGrouped<RowIndexT, ItemIndexT, 256>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, stream);
            return;
        default:
            break;
    }
    throw std::logic_error("Invalid device-resident direct rows-per-CTA selection.");
}

template <typename RowIndexT>
void launchForItemIndexType(const uint8_t *records,
                            const uint64_t *rowIndices,
                            uint8_t *destination,
                            RowIndexT batchSize,
                            uint64_t numExamples,
                            uint64_t recordSizeBytes,
                            uint64_t fieldOffsetBytes,
                            uint64_t fieldBytes,
                            cudaStream_t stream) {
    // Keep ordinary per-row copy arithmetic 32-bit. Leave headroom for the
    // largest 256-lane stride so the terminal increment cannot wrap.
    if (fieldBytes <=
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - kThreadsPerBlock) {
        launchForGrouping<RowIndexT, uint32_t>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            static_cast<uint32_t>(fieldBytes),
            stream);
    } else {
        launchForGrouping<RowIndexT, uint64_t>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            stream);
    }
}

void launchForRowIndexType(const uint8_t *records,
                           const uint64_t *rowIndices,
                           uint8_t *destination,
                           uint64_t batchSize,
                           uint64_t numExamples,
                           uint64_t recordSizeBytes,
                           uint64_t fieldOffsetBytes,
                           uint64_t fieldBytes,
                           cudaStream_t stream) {
    // Batch-row arithmetic is 32-bit for the overwhelmingly common case.
    // Source byte addresses remain 64-bit because a valid resident dataset may
    // exceed 4 GiB even when its row count is comfortably within UINT32.
    if (batchSize <= std::numeric_limits<uint32_t>::max()) {
        launchForItemIndexType<uint32_t>(
            records,
            rowIndices,
            destination,
            static_cast<uint32_t>(batchSize),
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            stream);
    } else {
        launchForItemIndexType<uint64_t>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            stream);
    }
}

}  // namespace

void launchDeviceResidentDirectMaterializationKernel(
    const Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    Tensor &destination,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destination.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes > 0);
    THOR_THROW_IF_FALSE(fieldBytes > 0);
    THOR_THROW_IF_FALSE(fieldOffsetBytes <= recordSizeBytes);
    THOR_THROW_IF_FALSE(fieldBytes <= recordSizeBytes - fieldOffsetBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() % recordSizeBytes == 0);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() / recordSizeBytes == numExamples);

    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(destination.getDimensions().front() == batchSize);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() % fieldBytes == 0);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() / fieldBytes == batchSize);

    const uint8_t *records = recordStorage.getMemPtr<uint8_t>();
    const uint64_t *rowIndices = rowIndicesDevice.getMemPtr<uint64_t>();
    uint8_t *destinationBytes = static_cast<uint8_t *>(destination.getMemPtr());
    const cudaStream_t cudaStream = stream.getStream();

    // Compact records and packed destination rows may have byte strides that
    // change alignment from one selected row to the next. Select vector width
    // inside the kernel from each row's actual addresses; fieldBytes affects
    // launch geometry and exact tail size, but no longer forces a whole-launch
    // downgrade to a narrow CopyT.
    launchForRowIndexType(
        records,
        rowIndices,
        destinationBytes,
        batchSize,
        numExamples,
        recordSizeBytes,
        fieldOffsetBytes,
        fieldBytes,
        cudaStream);
}
