#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

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
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFULL;

static_assert(sizeof(ulonglong4_32a) == 32);
static_assert(alignof(ulonglong4_32a) == 32);

template <typename LoadT>
__device__ __forceinline__ uint64_t readReferenceUint64(const uint8_t *bytes) {
    static_assert(sizeof(uint64_t) % sizeof(LoadT) == 0,
                  "Device-resident ragged reference loads must tile UINT64 metadata.");
    const LoadT *typed = reinterpret_cast<const LoadT *>(bytes);
    uint64_t value = 0;
#pragma unroll
    for (uint32_t i = 0; i < sizeof(uint64_t) / sizeof(LoadT); ++i) {
        value |= static_cast<uint64_t>(typed[i]) << (8U * sizeof(LoadT) * i);
    }
    return value;
}

template <typename LoadT>
bool canUseReferenceLoadType(const uint8_t *records,
                             uint64_t recordSizeBytes,
                             uint64_t referenceOffsetBytes) {
    constexpr uint64_t alignment = alignof(LoadT);
    return (reinterpret_cast<uintptr_t>(records) % alignment) == 0 &&
           recordSizeBytes % alignment == 0 &&
           referenceOffsetBytes % alignment == 0;
}

template <typename OffsetT, typename ReferenceLoadT, typename RowIndexT>
__global__ void buildRaggedOffsetsKernel(
    const uint8_t *__restrict__ records,
    const uint64_t *__restrict__ rowIndices,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    RowIndexT logicalRows,
    RowIndexT batchSize,
    OffsetT *__restrict__ offsets) {
    using BlockScan = cub::BlockScan<OffsetT, kThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ OffsetT tileBase;

    if (threadIdx.x == 0) {
        offsets[0] = static_cast<OffsetT>(0);
        tileBase = static_cast<OffsetT>(0);
    }
    __syncthreads();

    // The caller has already validated the selected cumulative value count
    // against the destination offsets dtype. Keep UINT32 partitions entirely
    // in 32-bit scan arithmetic; UINT64 partitions retain their full range.
    RowIndexT tile = 0;
    while (tile < logicalRows) {
        const RowIndexT remaining = logicalRows - tile;
        const uint32_t lane = threadIdx.x;
        const bool active = static_cast<RowIndexT>(lane) < remaining;

        RowIndexT row = 0;
        OffsetT count = 0;
        if (active) {
            row = tile + static_cast<RowIndexT>(lane);
            const uint64_t sourceRow = rowIndices[row];
            const uint8_t *reference =
                records + sourceRow * recordSizeBytes + referenceOffsetBytes;
            count = static_cast<OffsetT>(
                readReferenceUint64<ReferenceLoadT>(reference + sizeof(uint64_t)));
        }

        OffsetT exclusive = 0;
        OffsetT tileTotal = 0;
        BlockScan(scanStorage).ExclusiveSum(count, exclusive, tileTotal);

        const OffsetT base = tileBase;
        if (active) {
            offsets[row + 1] = base + exclusive + count;
        }

        // BlockScan reuses shared storage on the next tile. Synchronize both to
        // protect that storage and to publish the next tile's prefix base.
        __syncthreads();
        if (threadIdx.x == 0) {
            tileBase = base + tileTotal;
        }
        __syncthreads();

        if (remaining <= static_cast<RowIndexT>(kThreadsPerBlock)) break;
        tile += static_cast<RowIndexT>(kThreadsPerBlock);
    }

    const OffsetT finalOffset = tileBase;
    if (logicalRows < batchSize) {
        const RowIndexT trailingOffsets = batchSize - logicalRows;
        const uint32_t lane = threadIdx.x;
        if (static_cast<RowIndexT>(lane) < trailingOffsets) {
            RowIndexT row = logicalRows + static_cast<RowIndexT>(1 + lane);
            while (row <= batchSize) {
                offsets[row] = finalOffset;
                const RowIndexT remaining = batchSize - row;
                if (remaining < static_cast<RowIndexT>(kThreadsPerBlock)) break;
                row += static_cast<RowIndexT>(kThreadsPerBlock);
            }
        }
    }
}

struct RowCopyMetadata {
    const uint8_t *source;
    uint8_t *destination;
    uint64_t byteCount;
    uint32_t copyWidthBytes;
};

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const uint8_t *source,
                                                            const uint8_t *destination,
                                                            uint64_t byteCount) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination) |
                               static_cast<uintptr_t>(byteCount);
    if ((combined & 31U) == 0) return 32;
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyAligned(const uint8_t *source,
                                            uint8_t *destination,
                                            ItemIndexT byteCount,
                                            uint32_t lane) {
    const CopyT *typedSource = reinterpret_cast<const CopyT *>(source);
    CopyT *typedDestination = reinterpret_cast<CopyT *>(destination);
    const ItemIndexT items = byteCount / sizeof(CopyT);
    for (ItemIndexT item = static_cast<ItemIndexT>(lane);
         item < items;
         item += static_cast<ItemIndexT>(LanesPerRow)) {
        typedDestination[item] = typedSource[item];
    }
}

template <typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyRowIndexed(const RowCopyMetadata &metadata,
                                               uint32_t lane) {
    const ItemIndexT byteCount = static_cast<ItemIndexT>(metadata.byteCount);
    switch (metadata.copyWidthBytes) {
        case 32:
            copyAligned<ulonglong4_32a, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        case 16:
            copyAligned<uint4, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        case 8:
            copyAligned<uint64_t, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        case 4:
            copyAligned<uint32_t, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        case 2:
            copyAligned<uint16_t, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
        default:
            copyAligned<uint8_t, ItemIndexT, LanesPerRow>(
                metadata.source, metadata.destination, byteCount, lane);
            break;
    }
}

template <uint32_t LanesPerRow>
__device__ __forceinline__ void copyRow(const RowCopyMetadata &metadata,
                                       uint32_t lane) {
    // Keep the ordinary copy loop 32-bit. Leave enough headroom for the
    // increment so a near-4-GiB copy cannot wrap; only an extreme row uses the
    // 64-bit fallback.
    if (metadata.byteCount <= kMaxUint32 - LanesPerRow) {
        copyRowIndexed<uint32_t, LanesPerRow>(metadata, lane);
    } else {
        copyRowIndexed<uint64_t, LanesPerRow>(metadata, lane);
    }
}

template <typename OffsetT,
          typename ReferenceLoadT,
          typename RowIndexT,
          uint32_t RowsPerBlock>
__global__ void gatherRaggedValuesKernel(
    const uint8_t *__restrict__ records,
    const uint8_t *__restrict__ packedValues,
    const uint64_t *__restrict__ rowIndices,
    const OffsetT *__restrict__ offsets,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    RowIndexT logicalRows,
    uint8_t *__restrict__ destination) {
    static_assert(RowsPerBlock >= 1 && RowsPerBlock <= kMaxRowsPerBlock,
                  "Device-resident ragged rows per CTA are out of range.");
    static_assert(kThreadsPerBlock % RowsPerBlock == 0,
                  "Device-resident ragged row groups must tile a CTA.");
    constexpr uint32_t kLanesPerRow = kThreadsPerBlock / RowsPerBlock;

    // The one-lane specialization resolves and copies each row entirely in
    // registers. Wider row groups publish one producer lane's metadata through
    // shared memory before cooperating on the contiguous copy.
    __shared__ RowCopyMetadata rowMetadata[kLanesPerRow == 1 ? 1 : RowsPerBlock];

    const uint32_t rowSlot = threadIdx.x / kLanesPerRow;
    const uint32_t lane = threadIdx.x - rowSlot * kLanesPerRow;
    const uint32_t rowStride = gridDim.x * RowsPerBlock;

    RowIndexT rowBase = static_cast<RowIndexT>(blockIdx.x) * RowsPerBlock;
    while (rowBase < logicalRows) {
        if constexpr (kLanesPerRow == 1) {
            if (static_cast<RowIndexT>(rowSlot) < logicalRows - rowBase) {
                const RowIndexT row = rowBase + static_cast<RowIndexT>(rowSlot);
                RowCopyMetadata metadata{nullptr, nullptr, 0, 1};
                const uint64_t sourceRow = rowIndices[row];
                const uint8_t *reference =
                    records + sourceRow * recordSizeBytes + referenceOffsetBytes;
                const uint64_t start = readReferenceUint64<ReferenceLoadT>(reference);
                const OffsetT destinationValue = offsets[row];
                const OffsetT nextDestinationValue = offsets[row + 1];
                const OffsetT count = nextDestinationValue - destinationValue;

                if (count != 0) {
                    metadata.source = packedValues + start * valueBytes;
                    metadata.destination =
                        destination + static_cast<uint64_t>(destinationValue) * valueBytes;
                    metadata.byteCount = static_cast<uint64_t>(count) * valueBytes;
                    metadata.copyWidthBytes = widestAlignedCopyWidth(
                        metadata.source, metadata.destination, metadata.byteCount);
                    copyRow<1>(metadata, 0);
                }
            }
        } else {
            if (lane == 0) {
                RowCopyMetadata metadata{nullptr, nullptr, 0, 1};
                if (static_cast<RowIndexT>(rowSlot) < logicalRows - rowBase) {
                    const RowIndexT row = rowBase + static_cast<RowIndexT>(rowSlot);
                    const uint64_t sourceRow = rowIndices[row];
                    const uint8_t *reference =
                        records + sourceRow * recordSizeBytes + referenceOffsetBytes;
                    const uint64_t start = readReferenceUint64<ReferenceLoadT>(reference);
                    const OffsetT destinationValue = offsets[row];
                    const OffsetT nextDestinationValue = offsets[row + 1];
                    const OffsetT count = nextDestinationValue - destinationValue;

                    if (count != 0) {
                        metadata.source = packedValues + start * valueBytes;
                        metadata.destination =
                            destination + static_cast<uint64_t>(destinationValue) * valueBytes;
                        metadata.byteCount = static_cast<uint64_t>(count) * valueBytes;
                        metadata.copyWidthBytes = widestAlignedCopyWidth(
                            metadata.source, metadata.destination, metadata.byteCount);
                    }
                }
                rowMetadata[rowSlot] = metadata;
            }

            // Sub-warp and one-warp row groups are contained within one warp,
            // so warp synchronization is sufficient. Groups spanning multiple
            // warps require the CTA-wide barrier.
            if constexpr (kLanesPerRow <= 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
            const RowCopyMetadata metadata = rowMetadata[rowSlot];

            // Once every lane has copied metadata into registers, the next
            // grid-stride iteration may reuse the shared slot while this row
            // copy is still in flight. The condition is uniform for the CTA.
            const bool hasNextRowBase =
                static_cast<RowIndexT>(rowStride) < logicalRows - rowBase;
            if (hasNextRowBase) {
                if constexpr (kLanesPerRow <= 32) {
                    __syncwarp();
                } else {
                    __syncthreads();
                }
            }

            if (metadata.byteCount != 0) copyRow<kLanesPerRow>(metadata, lane);
        }

        const bool hasNextRowBase =
            static_cast<RowIndexT>(rowStride) < logicalRows - rowBase;
        if (!hasNextRowBase) break;
        rowBase += static_cast<RowIndexT>(rowStride);
    }
}

uint32_t rowsPerBlockFor(uint64_t expectedRowBytes, uint64_t logicalRows) {
    // Give each lane about 32 bytes of expected row payload. A 32-byte-aligned
    // row can issue one ulonglong4_32a transaction per iteration; a row with
    // narrower alignment still reaches the same per-lane payload with multiple
    // 16/8/4/2/1-byte transactions. This yields the full power-of-two ladder
    // from one row using the whole CTA through 256 independent one-thread rows.
    uint32_t rowsByPayload = 1;
    if (expectedRowBytes <= kTargetBytesPerLane) {
        rowsByPayload = 256;
    } else if (expectedRowBytes <= 2 * kTargetBytesPerLane) {
        rowsByPayload = 128;
    } else if (expectedRowBytes <= 4 * kTargetBytesPerLane) {
        rowsByPayload = 64;
    } else if (expectedRowBytes <= 8 * kTargetBytesPerLane) {
        rowsByPayload = 32;
    } else if (expectedRowBytes <= 16 * kTargetBytesPerLane) {
        rowsByPayload = 16;
    } else if (expectedRowBytes <= 32 * kTargetBytesPerLane) {
        rowsByPayload = 8;
    } else if (expectedRowBytes <= 64 * kTargetBytesPerLane) {
        rowsByPayload = 4;
    } else if (expectedRowBytes <= 128 * kTargetBytesPerLane) {
        rowsByPayload = 2;
    }

    // Do not sacrifice block-level parallelism just because rows are tiny.
    // This is the same ~64-CTA floor encoded by the previous row-count-only
    // selector, extended through the new 16/32/64/128/256-row specializations.
    uint32_t rowsByParallelism = 1;
    if (logicalRows >= 16384) {
        rowsByParallelism = 256;
    } else if (logicalRows >= 8192) {
        rowsByParallelism = 128;
    } else if (logicalRows >= 4096) {
        rowsByParallelism = 64;
    } else if (logicalRows >= 2048) {
        rowsByParallelism = 32;
    } else if (logicalRows >= 1024) {
        rowsByParallelism = 16;
    } else if (logicalRows >= 512) {
        rowsByParallelism = 8;
    } else if (logicalRows >= 256) {
        rowsByParallelism = 4;
    } else if (logicalRows >= 128) {
        rowsByParallelism = 2;
    }

    return std::min(rowsByPayload, rowsByParallelism);
}

template <uint32_t RowsPerBlock>
uint32_t blocksForRows(uint64_t logicalRows) {
    const uint64_t blocks =
        logicalRows / RowsPerBlock + (logicalRows % RowsPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <typename OffsetT,
          typename ReferenceLoadT,
          typename RowIndexT,
          uint32_t RowsPerBlock>
void launchGatherGrouped(const uint8_t *records,
                         const uint8_t *packedValues,
                         const uint64_t *rowIndices,
                         const OffsetT *offsets,
                         uint64_t recordSizeBytes,
                         uint64_t referenceOffsetBytes,
                         uint64_t valueBytes,
                         RowIndexT logicalRows,
                         uint8_t *destination,
                         cudaStream_t stream) {
    const uint32_t blocks = blocksForRows<RowsPerBlock>(
        static_cast<uint64_t>(logicalRows));
    gatherRaggedValuesKernel<OffsetT, ReferenceLoadT, RowIndexT, RowsPerBlock>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            records,
            packedValues,
            rowIndices,
            offsets,
            recordSizeBytes,
            referenceOffsetBytes,
            valueBytes,
            logicalRows,
            destination);
    CUDA_CHECK(cudaGetLastError());
}

template <bool BuildOffsets, typename OffsetT, typename ReferenceLoadT, typename RowIndexT>
void launchForConfiguration(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    uint64_t expectedRowBytes,
    RowIndexT logicalRows,
    RowIndexT batchSize,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    const uint8_t *records = recordStorage.getMemPtr<uint8_t>();
    const uint64_t *rowIndices = rowIndicesDevice.getMemPtr<uint64_t>();
    OffsetT *offsets = destinationOffsets.getMemPtr<OffsetT>();
    const cudaStream_t cudaStream = stream.getStream();

    if constexpr (BuildOffsets) {
        buildRaggedOffsetsKernel<OffsetT, ReferenceLoadT, RowIndexT>
            <<<1, kThreadsPerBlock, 0, cudaStream>>>(
                records,
                rowIndices,
                recordSizeBytes,
                referenceOffsetBytes,
                logicalRows,
                batchSize,
                offsets);
        CUDA_CHECK(cudaGetLastError());
    } else {
        (void)batchSize;
    }

    const uint8_t *packedValues =
        static_cast<const uint8_t *>(packedValuesStorage.getMemPtr<void>());
    uint8_t *destination = static_cast<uint8_t *>(destinationValues.getMemPtr());
    switch (rowsPerBlockFor(expectedRowBytes, static_cast<uint64_t>(logicalRows))) {
        case 1:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 1>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 2:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 2>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 4:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 4>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 8:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 8>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 16:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 16>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 32:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 32>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 64:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 64>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 128:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 128>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        case 256:
            launchGatherGrouped<OffsetT, ReferenceLoadT, RowIndexT, 256>(
                records, packedValues, rowIndices, offsets, recordSizeBytes,
                referenceOffsetBytes, valueBytes, logicalRows, destination, cudaStream);
            return;
        default:
            break;
    }
    throw std::logic_error("Invalid device-resident ragged rows-per-CTA selection.");
}

template <bool BuildOffsets, typename OffsetT, typename RowIndexT>
void launchForRowIndexType(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    uint64_t expectedRowBytes,
    RowIndexT logicalRows,
    RowIndexT batchSize,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    const uint8_t *records = recordStorage.getMemPtr<uint8_t>();

    // Compact records are byte-packed. Select the widest metadata load that is
    // aligned for every record; UINT8 is the exact unaligned fallback.
    if (canUseReferenceLoadType<uint64_t>(records, recordSizeBytes, referenceOffsetBytes)) {
        launchForConfiguration<BuildOffsets, OffsetT, uint64_t, RowIndexT>(
            recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
            valueBytes, expectedRowBytes, logicalRows, batchSize, destinationValues,
            destinationOffsets, rowIndicesDevice, stream);
    } else if (canUseReferenceLoadType<uint32_t>(records, recordSizeBytes, referenceOffsetBytes)) {
        launchForConfiguration<BuildOffsets, OffsetT, uint32_t, RowIndexT>(
            recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
            valueBytes, expectedRowBytes, logicalRows, batchSize, destinationValues,
            destinationOffsets, rowIndicesDevice, stream);
    } else if (canUseReferenceLoadType<uint16_t>(records, recordSizeBytes, referenceOffsetBytes)) {
        launchForConfiguration<BuildOffsets, OffsetT, uint16_t, RowIndexT>(
            recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
            valueBytes, expectedRowBytes, logicalRows, batchSize, destinationValues,
            destinationOffsets, rowIndicesDevice, stream);
    } else {
        launchForConfiguration<BuildOffsets, OffsetT, uint8_t, RowIndexT>(
            recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
            valueBytes, expectedRowBytes, logicalRows, batchSize, destinationValues,
            destinationOffsets, rowIndicesDevice, stream);
    }
}

template <bool BuildOffsets, typename OffsetT>
void launchTyped(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t valueBytes,
    uint64_t expectedRowBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();

    // Keep row/grid arithmetic 32-bit for normal batches. The UINT64 fallback
    // preserves the existing API range without imposing 64-bit arithmetic on
    // every CUDA thread.
    if (batchSize <= std::numeric_limits<uint32_t>::max()) {
        launchForRowIndexType<BuildOffsets, OffsetT, uint32_t>(
            recordStorage,
            packedValuesStorage,
            recordSizeBytes,
            referenceOffsetBytes,
            valueBytes,
            expectedRowBytes,
            static_cast<uint32_t>(logicalRows),
            static_cast<uint32_t>(batchSize),
            destinationValues,
            destinationOffsets,
            rowIndicesDevice,
            stream);
    } else {
        launchForRowIndexType<BuildOffsets, OffsetT, uint64_t>(
            recordStorage,
            packedValuesStorage,
            recordSizeBytes,
            referenceOffsetBytes,
            valueBytes,
            expectedRowBytes,
            logicalRows,
            batchSize,
            destinationValues,
            destinationOffsets,
            rowIndicesDevice,
            stream);
    }
}

uint64_t expectedResidentRowBytes(uint64_t storedValueCount,
                                  uint64_t valueBytes,
                                  uint64_t numExamples) {
    const uint64_t totalValueBytes = storedValueCount * valueBytes;
    return totalValueBytes / numExamples +
           (totalValueBytes % numExamples != 0 ? 1 : 0);
}

template <bool BuildOffsets>
void launchValidated(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(packedValuesStorage.isInitialized() || storedValueCount == 0);
    THOR_THROW_IF_FALSE(destinationValues.isInitialized());
    THOR_THROW_IF_FALSE(destinationOffsets.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!packedValuesStorage.isInitialized() ||
                        packedValuesStorage.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationValues.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationOffsets.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destinationValues.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes >= 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(referenceOffsetBytes <= recordSizeBytes - 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(valueBytes > 0);
    THOR_THROW_IF_FALSE(
        storedValueCount <= std::numeric_limits<uint64_t>::max() / valueBytes);
    THOR_THROW_IF_FALSE(
        (packedValuesStorage.isInitialized() ? packedValuesStorage.getArraySizeInBytes() : 0) ==
        storedValueCount * valueBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() == numExamples * recordSizeBytes);

    const uint64_t batchSize = rowIndicesDevice.getDimensions().front();
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= batchSize);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().front() == batchSize + 1);
    THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() > 0);

    const uint64_t expectedRowBytes =
        expectedResidentRowBytes(storedValueCount, valueBytes, numExamples);

    switch (destinationOffsets.getDataType()) {
        case DataType::UINT32:
            THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() <=
                                std::numeric_limits<uint32_t>::max());
            if (storedValueCount == 0) {
                if constexpr (BuildOffsets) {
                    CUDA_CHECK(cudaMemsetAsync(
                        destinationOffsets.getMemPtr<void>(),
                        0,
                        destinationOffsets.getArraySizeInBytes(),
                        stream.getStream()));
                }
                return;
            }
            launchTyped<BuildOffsets, uint32_t>(
                recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
                valueBytes, expectedRowBytes, logicalRows, destinationValues,
                destinationOffsets, rowIndicesDevice, stream);
            return;
        case DataType::UINT64:
            if (storedValueCount == 0) {
                if constexpr (BuildOffsets) {
                    CUDA_CHECK(cudaMemsetAsync(
                        destinationOffsets.getMemPtr<void>(),
                        0,
                        destinationOffsets.getArraySizeInBytes(),
                        stream.getStream()));
                }
                return;
            }
            launchTyped<BuildOffsets, uint64_t>(
                recordStorage, packedValuesStorage, recordSizeBytes, referenceOffsetBytes,
                valueBytes, expectedRowBytes, logicalRows, destinationValues,
                destinationOffsets, rowIndicesDevice, stream);
            return;
        default:
            break;
    }
    throw std::runtime_error(
        "Device resident ragged offsets must use canonical UINT32 or UINT64 dtype.");
}

}  // namespace

void launchDeviceResidentRaggedMaterializationKernel(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    launchValidated<true>(
        recordStorage,
        packedValuesStorage,
        numExamples,
        recordSizeBytes,
        referenceOffsetBytes,
        storedValueCount,
        valueBytes,
        logicalRows,
        destinationValues,
        destinationOffsets,
        rowIndicesDevice,
        stream);
}

void launchDeviceResidentRaggedValuesMaterializationKernel(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    Tensor &destinationOffsets,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    launchValidated<false>(
        recordStorage,
        packedValuesStorage,
        numExamples,
        recordSizeBytes,
        referenceOffsetBytes,
        storedValueCount,
        valueBytes,
        logicalRows,
        destinationValues,
        destinationOffsets,
        rowIndicesDevice,
        stream);
}
