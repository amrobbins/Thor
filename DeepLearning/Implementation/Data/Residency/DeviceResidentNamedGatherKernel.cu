#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

constexpr int kThreadsPerBlock = 256;

/**
 * Gather independent tensor rows with a power-of-two number of CUDA threads per
 * destination row. Small rows therefore keep many independent gathers resident
 * in one CTA, while wider rows expose their contiguous trailing bytes to more
 * lanes.
 *
 * CopyT is selected on the host only when every source and destination row start
 * is aligned for that type. UINT8 is the alignment-safe fallback for rows whose
 * byte width prevents a wider transaction.
 */
template <typename CopyT>
__global__ void gatherRowsKernel(const uint8_t *__restrict__ sourceBytes,
                                 uint8_t *__restrict__ destinationBytes,
                                 const uint64_t *__restrict__ rowIndices,
                                 uint64_t batchSize,
                                 uint64_t rowBytes,
                                 uint64_t sourceRows,
                                 uint32_t threadsPerRow,
                                 uint32_t threadsPerRowShift) {
    const uint64_t rowsPerBlock =
        static_cast<uint64_t>(kThreadsPerBlock >> threadsPerRowShift);
    const uint64_t rowSlot =
        static_cast<uint64_t>(threadIdx.x >> threadsPerRowShift);
    const uint64_t lane =
        static_cast<uint64_t>(threadIdx.x & (threadsPerRow - 1));
    const uint64_t rowStride = static_cast<uint64_t>(gridDim.x) * rowsPerBlock;
    const uint64_t itemsPerRow = rowBytes / sizeof(CopyT);

    for (uint64_t rowBase = static_cast<uint64_t>(blockIdx.x) * rowsPerBlock;
         rowBase < batchSize;
         rowBase += rowStride) {
        const uint64_t batchRow = rowBase + rowSlot;
        if (batchRow >= batchSize) {
            continue;
        }

        // Lanes assigned to the same row read the same index. CUDA coalesces the
        // duplicate addresses, while neighboring row groups read neighboring
        // index entries. This avoids a CTA barrier solely to publish sourceRow.
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow >= sourceRows) {
            // Preserve the historical contract: invalid resident row indices do
            // not overwrite the corresponding destination row.
            continue;
        }

        const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(
            sourceBytes + sourceRow * rowBytes);
        CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(
            destinationBytes + batchRow * rowBytes);

        for (uint64_t item = lane; item < itemsPerRow; item += threadsPerRow) {
            destination[item] = source[item];
        }
    }
}

void chooseThreadsPerRow(uint64_t itemsPerRow,
                         uint32_t &threadsPerRow,
                         uint32_t &threadsPerRowShift) {
    if (itemsPerRow <= 1) {
        threadsPerRow = 1;
        threadsPerRowShift = 0;
    } else if (itemsPerRow <= 2) {
        threadsPerRow = 2;
        threadsPerRowShift = 1;
    } else if (itemsPerRow <= 4) {
        threadsPerRow = 4;
        threadsPerRowShift = 2;
    } else if (itemsPerRow <= 8) {
        threadsPerRow = 8;
        threadsPerRowShift = 3;
    } else if (itemsPerRow <= 16) {
        threadsPerRow = 16;
        threadsPerRowShift = 4;
    } else {
        threadsPerRow = 32;
        threadsPerRowShift = 5;
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
    const uint64_t itemsPerRow = rowBytes / sizeof(CopyT);
    uint32_t threadsPerRow = 0;
    uint32_t threadsPerRowShift = 0;
    chooseThreadsPerRow(itemsPerRow, threadsPerRow, threadsPerRowShift);

    const uint64_t rowsPerBlock =
        static_cast<uint64_t>(kThreadsPerBlock >> threadsPerRowShift);
    uint64_t blocks64 = (batchSize + rowsPerBlock - 1) / rowsPerBlock;
    blocks64 = std::max<uint64_t>(1, std::min<uint64_t>(blocks64, 65535));
    const int blocks = static_cast<int>(blocks64);

    gatherRowsKernel<CopyT><<<blocks, kThreadsPerBlock, 0, stream>>>(
        source,
        destination,
        rowIndices,
        batchSize,
        rowBytes,
        sourceRows,
        threadsPerRow,
        threadsPerRowShift);
    CUDA_CHECK(cudaGetLastError());
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
    // destination. Pick the widest transaction that is aligned for every row;
    // UINT8 remains the exact fallback for byte-width rows.
    if (canUseCopyType<uint4>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint4>(sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes, sourceRows, cudaStream);
    } else if (canUseCopyType<uint64_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint64_t>(sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes, sourceRows, cudaStream);
    } else if (canUseCopyType<uint32_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint32_t>(sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes, sourceRows, cudaStream);
    } else if (canUseCopyType<uint16_t>(sourceBytes, destinationBytes, rowBytes)) {
        launchForCopyType<uint16_t>(sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes, sourceRows, cudaStream);
    } else {
        launchForCopyType<uint8_t>(sourceBytes, destinationBytes, rowIndices, batchSize, rowBytes, sourceRows, cudaStream);
    }
}
