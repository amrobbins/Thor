#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

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

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyDirectField(const uint8_t *sourceBytes,
                                                uint8_t *destinationBytes,
                                                ItemIndexT fieldItems,
                                                uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);

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
}

/**
 * Copy independent compact-record fields with a compile-time CUDA lane group
 * per destination row. The row grouping follows payload bytes rather than the
 * selected transaction width, so a 16-byte-aligned field and a 32-byte-aligned
 * field of the same size receive the same amount of lane-level parallelism.
 *
 * CopyT is selected on the host only when every source/destination row start is
 * aligned for that type. UINT8 remains the exact fallback for byte-packed
 * record layouts.
 */
template <typename CopyT,
          typename RowIndexT,
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
    uint64_t fieldBytes,
    ItemIndexT fieldItems) {
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
                copyDirectField<CopyT, ItemIndexT, kLanesPerRow>(
                    source, rowDestination, fieldItems, lane);
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

template <typename CopyT,
          typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
void launchGrouped(const uint8_t *records,
                   const uint64_t *rowIndices,
                   uint8_t *destination,
                   RowIndexT batchSize,
                   uint64_t numExamples,
                   uint64_t recordSizeBytes,
                   uint64_t fieldOffsetBytes,
                   uint64_t fieldBytes,
                   ItemIndexT fieldItems,
                   cudaStream_t stream) {
    const uint32_t blocks =
        blocksForRows<RowsPerBlock>(static_cast<uint64_t>(batchSize));
    materializeDirectFieldKernel<CopyT, RowIndexT, ItemIndexT, RowsPerBlock>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            fieldItems);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CopyT, typename RowIndexT, typename ItemIndexT>
void launchForGrouping(const uint8_t *records,
                       const uint64_t *rowIndices,
                       uint8_t *destination,
                       RowIndexT batchSize,
                       uint64_t numExamples,
                       uint64_t recordSizeBytes,
                       uint64_t fieldOffsetBytes,
                       uint64_t fieldBytes,
                       ItemIndexT fieldItems,
                       cudaStream_t stream) {
    switch (rowsPerBlockFor(fieldBytes, static_cast<uint64_t>(batchSize))) {
        case 1:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 1>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 2:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 2>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 4:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 4>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 8:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 8>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 16:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 16>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 32:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 32>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 64:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 64>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 128:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 128>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        case 256:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 256>(
                records, rowIndices, destination, batchSize, numExamples,
                recordSizeBytes, fieldOffsetBytes, fieldBytes, fieldItems, stream);
            return;
        default:
            break;
    }
    throw std::logic_error("Invalid device-resident direct rows-per-CTA selection.");
}

template <typename CopyT, typename RowIndexT>
void launchForItemIndexType(const uint8_t *records,
                            const uint64_t *rowIndices,
                            uint8_t *destination,
                            RowIndexT batchSize,
                            uint64_t numExamples,
                            uint64_t recordSizeBytes,
                            uint64_t fieldOffsetBytes,
                            uint64_t fieldBytes,
                            cudaStream_t stream) {
    const uint64_t fieldItems = fieldBytes / sizeof(CopyT);
    if (fieldItems <= std::numeric_limits<uint32_t>::max()) {
        launchForGrouping<CopyT, RowIndexT, uint32_t>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            static_cast<uint32_t>(fieldItems),
            stream);
    } else {
        launchForGrouping<CopyT, RowIndexT, uint64_t>(
            records,
            rowIndices,
            destination,
            batchSize,
            numExamples,
            recordSizeBytes,
            fieldOffsetBytes,
            fieldBytes,
            fieldItems,
            stream);
    }
}

template <typename CopyT>
void launchForCopyType(const uint8_t *records,
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
        launchForItemIndexType<CopyT, uint32_t>(
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
        launchForItemIndexType<CopyT, uint64_t>(
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

template <typename CopyT>
bool canUseCopyType(const uint8_t *records,
                    const uint8_t *destination,
                    uint64_t recordSizeBytes,
                    uint64_t fieldOffsetBytes,
                    uint64_t fieldBytes) {
    constexpr uint64_t alignment = alignof(CopyT);
    return fieldBytes % sizeof(CopyT) == 0 &&
           recordSizeBytes % alignment == 0 &&
           fieldOffsetBytes % alignment == 0 &&
           fieldBytes % alignment == 0 &&
           (reinterpret_cast<uintptr_t>(records) % alignment) == 0 &&
           (reinterpret_cast<uintptr_t>(destination) % alignment) == 0;
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

    // Compact records are byte-packed and are not guaranteed to align tensor
    // fields. Pick the widest type that is provably aligned for every resident
    // source row and destination row. The launch geometry is chosen separately
    // from fieldBytes, so falling back to a narrower transaction does not also
    // reduce the amount of lane-level parallelism assigned to the row.
    if (canUseCopyType<ulonglong4_32a>(
            records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<ulonglong4_32a>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint4>(
                   records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint4>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint64_t>(
                   records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint64_t>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint32_t>(
                   records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint32_t>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint16_t>(
                   records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint16_t>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else {
        launchForCopyType<uint8_t>(
            records, rowIndices, destinationBytes, batchSize, numExamples,
            recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    }
}
