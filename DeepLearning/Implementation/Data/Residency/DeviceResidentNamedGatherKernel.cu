#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

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

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyNamedRow(const uint8_t *sourceBytes,
                                             uint8_t *destinationBytes,
                                             ItemIndexT rowItems,
                                             uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);

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
}

/**
 * Gather independent tensor rows with a compile-time CUDA lane group per
 * destination row. The row grouping follows payload bytes rather than the
 * selected transaction width, so narrower alignment does not also reduce the
 * amount of lane-level parallelism assigned to a row.
 *
 * CopyT is selected on the host only when every source and destination row
 * start is aligned for that type. UINT8 is the exact fallback for odd row
 * widths.
 */
template <typename CopyT,
          typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
__global__ void gatherRowsKernel(const uint8_t *__restrict__ sourceBytes,
                                 uint8_t *__restrict__ destinationBytes,
                                 const uint64_t *__restrict__ rowIndices,
                                 RowIndexT batchSize,
                                 uint64_t rowBytes,
                                 uint64_t sourceRows,
                                 ItemIndexT rowItems) {
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
                const uint8_t *source = sourceBytes + sourceRow * rowBytes;
                uint8_t *destination =
                    destinationBytes + static_cast<uint64_t>(batchRow) * rowBytes;
                copyNamedRow<CopyT, ItemIndexT, kLanesPerRow>(
                    source, destination, rowItems, lane);
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

template <typename CopyT,
          typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock>
void launchGrouped(const uint8_t *source,
                   uint8_t *destination,
                   const uint64_t *rowIndices,
                   RowIndexT batchSize,
                   uint64_t rowBytes,
                   uint64_t sourceRows,
                   ItemIndexT rowItems,
                   cudaStream_t stream) {
    const uint32_t blocks =
        blocksForRows<RowsPerBlock>(static_cast<uint64_t>(batchSize));
    gatherRowsKernel<CopyT, RowIndexT, ItemIndexT, RowsPerBlock>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            rowItems);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CopyT, typename RowIndexT, typename ItemIndexT>
void launchForGrouping(const uint8_t *source,
                       uint8_t *destination,
                       const uint64_t *rowIndices,
                       RowIndexT batchSize,
                       uint64_t rowBytes,
                       uint64_t sourceRows,
                       ItemIndexT rowItems,
                       cudaStream_t stream) {
    switch (rowsPerBlockFor(rowBytes, static_cast<uint64_t>(batchSize))) {
        case 1:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 1>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 2:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 2>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 4:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 4>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 8:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 8>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 16:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 16>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 32:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 32>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 64:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 64>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 128:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 128>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        case 256:
            launchGrouped<CopyT, RowIndexT, ItemIndexT, 256>(
                source, destination, rowIndices, batchSize, rowBytes, sourceRows,
                rowItems, stream);
            return;
        default:
            break;
    }
    throw std::logic_error("Invalid device-resident named gather rows-per-CTA selection.");
}

template <typename CopyT, typename RowIndexT>
void launchForItemIndexType(const uint8_t *source,
                            uint8_t *destination,
                            const uint64_t *rowIndices,
                            RowIndexT batchSize,
                            uint64_t rowBytes,
                            uint64_t sourceRows,
                            cudaStream_t stream) {
    const uint64_t rowItems = rowBytes / sizeof(CopyT);
    if (rowItems <= std::numeric_limits<uint32_t>::max()) {
        launchForGrouping<CopyT, RowIndexT, uint32_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            static_cast<uint32_t>(rowItems),
            stream);
    } else {
        launchForGrouping<CopyT, RowIndexT, uint64_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            rowItems,
            stream);
    }
}

template <typename CopyT>
void launchForCopyType(const uint8_t *source,
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
        launchForItemIndexType<CopyT, uint32_t>(
            source,
            destination,
            rowIndices,
            static_cast<uint32_t>(batchSize),
            rowBytes,
            sourceRows,
            stream);
    } else {
        launchForItemIndexType<CopyT, uint64_t>(
            source,
            destination,
            rowIndices,
            batchSize,
            rowBytes,
            sourceRows,
            stream);
    }
}

template <typename CopyT>
bool canUseCopyType(const uint8_t *source,
                    const uint8_t *destination,
                    uint64_t rowBytes) {
    constexpr uint64_t alignment = alignof(CopyT);
    return rowBytes % sizeof(CopyT) == 0 &&
           rowBytes % alignment == 0 &&
           (reinterpret_cast<uintptr_t>(source) % alignment) == 0 &&
           (reinterpret_cast<uintptr_t>(destination) % alignment) == 0;
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

    // Named tensors have identical contiguous row geometry on source and
    // destination. Pick the widest transaction that is aligned for every row.
    // Launch geometry is selected independently from rowBytes, so falling back
    // to a narrower transaction does not also reduce row-level parallelism.
    if (canUseCopyType<ulonglong4_32a>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<ulonglong4_32a>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    } else if (canUseCopyType<uint4>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint4>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    } else if (canUseCopyType<uint64_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint64_t>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    } else if (canUseCopyType<uint32_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint32_t>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    } else if (canUseCopyType<uint16_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint16_t>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    } else {
        launchForCopyType<uint8_t>(
            sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes,
            sourceRows, cudaStream);
    }
}
