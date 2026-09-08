#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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
    // Row length does not constrain bulk width. A final source packet may
    // extend beyond the logical row (and, for the final source row, into Thor's
    // 128-byte tensor allocation padding). The destination remainder is always
    // stored with its exact logical size.
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
    // logical row has fewer bytes left. Reading into the next source row is
    // harmless; at the end of source storage, Thor's 128-byte tensor padding
    // makes the same load physically safe. Expose exactly TailBytes to the
    // compiler for the destination aggregate store so the next destination row
    // is never overwritten.
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
#define THOR_NAMED_GATHER_TAIL_CASE(N)             \
    case N:                                         \
        if constexpr (N < sizeof(CopyT)) {          \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                           \
        return
    switch (tailBytes) {
        THOR_NAMED_GATHER_TAIL_CASE(1);
        THOR_NAMED_GATHER_TAIL_CASE(2);
        THOR_NAMED_GATHER_TAIL_CASE(3);
        THOR_NAMED_GATHER_TAIL_CASE(4);
        THOR_NAMED_GATHER_TAIL_CASE(5);
        THOR_NAMED_GATHER_TAIL_CASE(6);
        THOR_NAMED_GATHER_TAIL_CASE(7);
        THOR_NAMED_GATHER_TAIL_CASE(8);
        THOR_NAMED_GATHER_TAIL_CASE(9);
        THOR_NAMED_GATHER_TAIL_CASE(10);
        THOR_NAMED_GATHER_TAIL_CASE(11);
        THOR_NAMED_GATHER_TAIL_CASE(12);
        THOR_NAMED_GATHER_TAIL_CASE(13);
        THOR_NAMED_GATHER_TAIL_CASE(14);
        THOR_NAMED_GATHER_TAIL_CASE(15);
        THOR_NAMED_GATHER_TAIL_CASE(16);
        THOR_NAMED_GATHER_TAIL_CASE(17);
        THOR_NAMED_GATHER_TAIL_CASE(18);
        THOR_NAMED_GATHER_TAIL_CASE(19);
        THOR_NAMED_GATHER_TAIL_CASE(20);
        THOR_NAMED_GATHER_TAIL_CASE(21);
        THOR_NAMED_GATHER_TAIL_CASE(22);
        THOR_NAMED_GATHER_TAIL_CASE(23);
        THOR_NAMED_GATHER_TAIL_CASE(24);
        THOR_NAMED_GATHER_TAIL_CASE(25);
        THOR_NAMED_GATHER_TAIL_CASE(26);
        THOR_NAMED_GATHER_TAIL_CASE(27);
        THOR_NAMED_GATHER_TAIL_CASE(28);
        THOR_NAMED_GATHER_TAIL_CASE(29);
        THOR_NAMED_GATHER_TAIL_CASE(30);
        THOR_NAMED_GATHER_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_NAMED_GATHER_TAIL_CASE
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyAlignedNamedRow(const uint8_t *sourceBytes,
                                                    uint8_t *destinationBytes,
                                                    ItemIndexT rowBytes,
                                                    uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);
    const ItemIndexT rowItems = rowBytes / static_cast<ItemIndexT>(sizeof(CopyT));

    ItemIndexT item = static_cast<ItemIndexT>(lane);
    while (item < rowItems) {
        destination[item] = source[item];

        // Break before the increment when this was the final item owned by the
        // lane. Besides avoiding one unnecessary add, this keeps the 32-bit
        // fast path correct even when rowItems is close to UINT32_MAX.
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerRow);
        if (rowItems - item <= laneStride) break;
        item += laneStride;
    }

    if constexpr (sizeof(CopyT) > 1) {
        if (lane == 0) {
            const uint32_t tailBytes = static_cast<uint32_t>(
                rowBytes % static_cast<ItemIndexT>(sizeof(CopyT)));
            if (tailBytes != 0) {
                copyExactTailPacket(source + rowItems, destination + rowItems, tailBytes);
            }
        }
    }
}

template <typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyNamedRow(const uint8_t *sourceBytes,
                                             uint8_t *destinationBytes,
                                             ItemIndexT rowBytes,
                                             uint32_t lane) {
    // An odd row stride means different gathered source/destination row pairs
    // can have different alignment. Choose the widest bulk packet from the
    // actual pair of addresses. Row length is intentionally absent from this
    // decision; only the final exact Tail<N> is narrow.
    switch (widestAlignedCopyWidth(sourceBytes, destinationBytes)) {
        case 32:
            copyAlignedNamedRow<ulonglong4_32a, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
        case 16:
            copyAlignedNamedRow<uint4, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
        case 8:
            copyAlignedNamedRow<uint64_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
        case 4:
            copyAlignedNamedRow<uint32_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
        case 2:
            copyAlignedNamedRow<uint16_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
        default:
            copyAlignedNamedRow<uint8_t, ItemIndexT, LanesPerRow>(
                sourceBytes, destinationBytes, rowBytes, lane);
            break;
    }
}

/**
 * Gather independent tensor rows with a compile-time CUDA lane group per
 * destination row. The row grouping follows payload bytes rather than the
 * selected transaction width, so narrower alignment does not also reduce the
 * amount of lane-level parallelism assigned to a row.
 *
 * Transaction width is selected independently for each gathered row pair from
 * its actual source/destination alignment. Complete packets are copied in
 * parallel; one designated lane stores any remainder with an exact Tail<N>.
 */
template <typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
__global__ void gatherRowsKernel(const uint8_t *__restrict__ sourceBytes,
                                 uint8_t *__restrict__ destinationBytes,
                                 const uint64_t *__restrict__ rowIndices,
                                 RowIndexT batchSize,
                                 ItemIndexT rowBytes,
                                 uint64_t sourceRows) {
    static_assert(RowsPerBlock >= 1 && RowsPerBlock <= kMaxRowsPerBlock,
                  "Device-resident named gather rows per CTA are out of range.");
    static_assert(kThreadsPerBlock % RowsPerBlock == 0,
                  "Device-resident named gather row groups must tile a CTA.");
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
            if (sourceRow < sourceRows) {
                const uint64_t rowBytes64 = static_cast<uint64_t>(rowBytes);
                const uint8_t *source = sourceBytes + sourceRow * rowBytes64;
                uint8_t *destination =
                    destinationBytes + static_cast<uint64_t>(batchRow) * rowBytes64;
                copyNamedRow<ItemIndexT, kLanesPerRow>(
                    source, destination, rowBytes, lane);
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

uint32_t rowsPerBlockFor(uint64_t rowBytes, uint64_t batchSize) {
    // Target about 32 bytes of useful row payload per lane. Transaction width
    // is deliberately not part of this policy: a 16/8/4/2/1-byte fallback may
    // require multiple transactions per lane, but consecutive lanes continue
    // to operate on adjacent contiguous items.
    uint32_t rowsByPayload = 1;
    if (rowBytes <= kTargetBytesPerLane) {
        rowsByPayload = 256;
    } else if (rowBytes <= 2 * kTargetBytesPerLane) {
        rowsByPayload = 128;
    } else if (rowBytes <= 4 * kTargetBytesPerLane) {
        rowsByPayload = 64;
    } else if (rowBytes <= 8 * kTargetBytesPerLane) {
        rowsByPayload = 32;
    } else if (rowBytes <= 16 * kTargetBytesPerLane) {
        rowsByPayload = 16;
    } else if (rowBytes <= 32 * kTargetBytesPerLane) {
        rowsByPayload = 8;
    } else if (rowBytes <= 64 * kTargetBytesPerLane) {
        rowsByPayload = 4;
    } else if (rowBytes <= 128 * kTargetBytesPerLane) {
        rowsByPayload = 2;
    }

    // Keep roughly 64 CTAs available for small rows instead of packing a small
    // batch into only one or two blocks. Once the batch is large enough,
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
void launchGrouped(const uint8_t *source,
                   uint8_t *destination,
                   const uint64_t *rowIndices,
                   RowIndexT batchSize,
                   ItemIndexT rowBytes,
                   uint64_t sourceRows,
                   cudaStream_t stream) {
    const uint32_t blocks =
        blocksForRows<RowsPerBlock>(static_cast<uint64_t>(batchSize));
    gatherRowsKernel<RowIndexT, ItemIndexT, RowsPerBlock>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows);
    CUDA_CHECK(cudaGetLastError());
}

template <typename RowIndexT, typename ItemIndexT>
void launchForGrouping(const uint8_t *source,
                       uint8_t *destination,
                       const uint64_t *rowIndices,
                       RowIndexT batchSize,
                       ItemIndexT rowBytes,
                       uint64_t sourceRows,
                       cudaStream_t stream) {
    switch (rowsPerBlockFor(static_cast<uint64_t>(rowBytes),
                            static_cast<uint64_t>(batchSize))) {
        case 1:
            launchGrouped<RowIndexT, ItemIndexT, 1>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 2:
            launchGrouped<RowIndexT, ItemIndexT, 2>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 4:
            launchGrouped<RowIndexT, ItemIndexT, 4>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 8:
            launchGrouped<RowIndexT, ItemIndexT, 8>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 16:
            launchGrouped<RowIndexT, ItemIndexT, 16>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 32:
            launchGrouped<RowIndexT, ItemIndexT, 32>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 64:
            launchGrouped<RowIndexT, ItemIndexT, 64>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 128:
            launchGrouped<RowIndexT, ItemIndexT, 128>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        case 256:
            launchGrouped<RowIndexT, ItemIndexT, 256>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows, stream);
            return;
        default:
            break;
    }
    throw std::logic_error("Invalid device-resident named gather rows-per-CTA selection.");
}

template <typename RowIndexT>
void launchForItemIndexType(const uint8_t *source,
                            uint8_t *destination,
                            const uint64_t *rowIndices,
                            RowIndexT batchSize,
                            uint64_t rowBytes,
                            uint64_t sourceRows,
                            cudaStream_t stream) {
    // Keep ordinary per-row copy arithmetic 32-bit. Leave headroom for the
    // largest 256-lane stride so the terminal increment cannot wrap.
    if (rowBytes <=
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - kThreadsPerBlock) {
        launchForGrouping<RowIndexT, uint32_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            static_cast<uint32_t>(rowBytes),
            sourceRows,
            stream);
    } else {
        launchForGrouping<RowIndexT, uint64_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            stream);
    }
}

void launchForRowIndexType(const uint8_t *source,
                           uint8_t *destination,
                           const uint64_t *rowIndices,
                           uint64_t batchSize,
                           uint64_t rowBytes,
                           uint64_t sourceRows,
                           cudaStream_t stream) {
    // Batch-row arithmetic is 32-bit for the overwhelmingly common case.
    // Source/destination byte addresses remain 64-bit because valid tensors may
    // exceed 4 GiB even when their row counts fit comfortably in UINT32.
    if (batchSize <= std::numeric_limits<uint32_t>::max()) {
        launchForItemIndexType<uint32_t>(
            source,
            destination,
            rowIndices,
            static_cast<uint32_t>(batchSize),
            rowBytes,
            sourceRows,
            stream);
    } else {
        launchForItemIndexType<uint64_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            stream);
    }
}

void validateGatherTensorShapes(const Tensor &source, const Tensor &destination, const Tensor &rowIndicesDevice) {
    THOR_THROW_IF_FALSE(source.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(source.getPlacement() == destination.getPlacement());
    THOR_THROW_IF_FALSE(source.getPlacement() == rowIndicesDevice.getPlacement());
    THOR_THROW_IF_FALSE(source.getDataType() == destination.getDataType());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == ThorImplementation::DataType::UINT64);

    const std::vector<uint64_t> sourceDims = source.getDimensions();
    const std::vector<uint64_t> destDims = destination.getDimensions();
    const std::vector<uint64_t> indexDims = rowIndicesDevice.getDimensions();
    THOR_THROW_IF_FALSE(!sourceDims.empty());
    THOR_THROW_IF_FALSE(!destDims.empty());
    THOR_THROW_IF_FALSE(sourceDims.size() == destDims.size());
    THOR_THROW_IF_FALSE(indexDims.size() == 1);
    THOR_THROW_IF_FALSE(indexDims.at(0) == destDims.at(0));
    THOR_THROW_IF_FALSE(sourceDims.at(0) > 0);
    THOR_THROW_IF_FALSE(destDims.at(0) > 0);
    for (uint64_t i = 1; i < sourceDims.size(); ++i) {
        THOR_THROW_IF_FALSE(sourceDims.at(i) == destDims.at(i));
    }
}

}  // namespace

void launchDeviceResidentNamedGatherKernel(const Tensor &source, Tensor &destination, const Tensor &rowIndicesDevice, Stream &stream) {
    validateGatherTensorShapes(source, destination, rowIndicesDevice);

    const uint64_t batchSize = destination.getDimensions().at(0);
    const uint64_t sourceRows = source.getDimensions().at(0);
    const uint64_t rowBytes = destination.getArraySizeInBytes() / batchSize;
    THOR_THROW_IF_FALSE(rowBytes > 0);
    const uint64_t totalBytes = destination.getArraySizeInBytes();
    THOR_THROW_IF_FALSE(totalBytes == batchSize * rowBytes);
    THOR_THROW_IF_FALSE(source.getArraySizeInBytes() == sourceRows * rowBytes);

    const uint8_t *sourceBytes = static_cast<const uint8_t *>(source.getMemPtr());
    uint8_t *destinationBytes = static_cast<uint8_t *>(destination.getMemPtr());
    const uint64_t *rowIndices = rowIndicesDevice.getMemPtr<uint64_t>();
    const cudaStream_t cudaStream = stream.getStream();

    // Copy width is selected independently for each gathered source/destination
    // row pair from its actual alignment. This lets odd row strides retain wide
    // bulk transactions on aligned pairs while the final remainder is stored as
    // one exact Tail<N> by a designated lane.
    launchForRowIndexType(
        sourceBytes,
        destinationBytes,
        rowIndices,
        batchSize,
        rowBytes,
        sourceRows,
        cudaStream);
}
