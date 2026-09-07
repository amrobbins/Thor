#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

constexpr int kThreadsPerBlock = 256;

/**
 * Copy independent record fields with a power-of-two number of CUDA threads per
 * destination row. Small fields therefore keep many rows resident in one CTA,
 * while wider fields expose their contiguous trailing bytes to more lanes.
 *
 * CopyT is selected on the host only when every source/destination row start is
 * aligned for that type. The UINT8 specialization is the alignment-safe fallback
 * for byte-packed record layouts.
 */
template <typename CopyT>
__global__ void materializeDirectFieldKernel(
    const uint8_t *__restrict__ records,
    const uint64_t *__restrict__ rowIndices,
    uint8_t *__restrict__ destination,
    uint64_t batchSize,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint32_t threadsPerRow,
    uint32_t threadsPerRowShift) {
    const uint64_t rowsPerBlock =
        static_cast<uint64_t>(kThreadsPerBlock >> threadsPerRowShift);
    const uint64_t rowSlot =
        static_cast<uint64_t>(threadIdx.x >> threadsPerRowShift);
    const uint64_t lane =
        static_cast<uint64_t>(threadIdx.x & (threadsPerRow - 1));
    const uint64_t rowStride = static_cast<uint64_t>(gridDim.x) * rowsPerBlock;
    const uint64_t fieldItems = fieldBytes / sizeof(CopyT);

    for (uint64_t rowBase = static_cast<uint64_t>(blockIdx.x) * rowsPerBlock;
         rowBase < batchSize;
         rowBase += rowStride) {
        const uint64_t batchRow = rowBase + rowSlot;
        if (batchRow >= batchSize) {
            continue;
        }

        // Lanes assigned to the same row read the same index. CUDA coalesces the
        // duplicate addresses, while neighboring row groups read neighboring
        // index entries. This avoids a block barrier solely to publish sourceRow.
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow >= numExamples) {
            // Preserve the historical contract: invalid resident row indices do
            // not overwrite the corresponding destination row.
            continue;
        }

        const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(
            records + sourceRow * recordSizeBytes + fieldOffsetBytes);
        CopyT *__restrict__ rowDestination = reinterpret_cast<CopyT *>(
            destination + batchRow * fieldBytes);

        for (uint64_t item = lane; item < fieldItems; item += threadsPerRow) {
            rowDestination[item] = source[item];
        }
    }
}

void chooseThreadsPerRow(
    uint64_t fieldItems,
    uint32_t &threadsPerRow,
    uint32_t &threadsPerRowShift) {
    if (fieldItems <= 1) {
        threadsPerRow = 1;
        threadsPerRowShift = 0;
    } else if (fieldItems <= 2) {
        threadsPerRow = 2;
        threadsPerRowShift = 1;
    } else if (fieldItems <= 4) {
        threadsPerRow = 4;
        threadsPerRowShift = 2;
    } else if (fieldItems <= 8) {
        threadsPerRow = 8;
        threadsPerRowShift = 3;
    } else if (fieldItems <= 16) {
        threadsPerRow = 16;
        threadsPerRowShift = 4;
    } else {
        threadsPerRow = 32;
        threadsPerRowShift = 5;
    }
}

template <typename CopyT>
void launchForCopyType(
    const uint8_t *records,
    const uint64_t *rowIndices,
    uint8_t *destination,
    uint64_t batchSize,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    cudaStream_t stream) {
    const uint64_t fieldItems = fieldBytes / sizeof(CopyT);
    uint32_t threadsPerRow = 0;
    uint32_t threadsPerRowShift = 0;
    chooseThreadsPerRow(fieldItems, threadsPerRow, threadsPerRowShift);

    const uint64_t rowsPerBlock =
        static_cast<uint64_t>(kThreadsPerBlock >> threadsPerRowShift);
    uint64_t blocks64 = (batchSize + rowsPerBlock - 1) / rowsPerBlock;
    blocks64 = std::max<uint64_t>(1, std::min<uint64_t>(blocks64, 65535));
    const int blocks = static_cast<int>(blocks64);

    materializeDirectFieldKernel<CopyT><<<blocks, kThreadsPerBlock, 0, stream>>>(
        records,
        rowIndices,
        destination,
        batchSize,
        numExamples,
        recordSizeBytes,
        fieldOffsetBytes,
        fieldBytes,
        threadsPerRow,
        threadsPerRowShift);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CopyT>
bool canUseCopyType(
    const uint8_t *records,
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
    // source row and destination row; UINT8 remains the exact fallback.
    if (canUseCopyType<uint4>(records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint4>(records, rowIndices, destinationBytes, batchSize, numExamples,
                                 recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint64_t>(records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint64_t>(records, rowIndices, destinationBytes, batchSize, numExamples,
                                    recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint32_t>(records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint32_t>(records, rowIndices, destinationBytes, batchSize, numExamples,
                                    recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else if (canUseCopyType<uint16_t>(records, destinationBytes, recordSizeBytes, fieldOffsetBytes, fieldBytes)) {
        launchForCopyType<uint16_t>(records, rowIndices, destinationBytes, batchSize, numExamples,
                                    recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    } else {
        launchForCopyType<uint8_t>(records, rowIndices, destinationBytes, batchSize, numExamples,
                                   recordSizeBytes, fieldOffsetBytes, fieldBytes, cudaStream);
    }
}
