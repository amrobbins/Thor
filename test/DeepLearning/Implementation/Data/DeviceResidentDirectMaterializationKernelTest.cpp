#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for device-resident direct materialization tests.";                \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t> &dimensions, const std::vector<T> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("DeviceResidentDirectMaterializationKernel test tensor value count mismatch.");
    }
    T *hostValues = host.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];
    Tensor device(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T *values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

void runMaterializationCase(
    uint64_t numExamples,
    uint64_t batchSize,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t logicalRows = 0) {
    if (logicalRows == 0) logicalRows = batchSize;
    ASSERT_LE(logicalRows, batchSize);
    constexpr uint8_t recordSentinel = 0x5aU;
    constexpr uint8_t destinationSentinel = 0xcdU;

    std::vector<uint8_t> records(numExamples * recordSizeBytes, recordSentinel);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
            records[sourceRow * recordSizeBytes + fieldOffsetBytes + byte] =
                static_cast<uint8_t>((sourceRow * 29 + byte * 17 + 11) & 0xffU);
        }
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
        rowIndices[batchRow] = (batchRow * 137 + 19) % numExamples;
    }
    // A row whose index is outside the records is left untouched.
    if (batchSize > 7) rowIndices[7] = numExamples + 3;
    if (batchSize > 263) rowIndices[263] = numExamples;

    std::vector<uint8_t> expected(batchSize * fieldBytes, destinationSentinel);
    for (uint64_t batchRow = 0; batchRow < logicalRows; ++batchRow) {
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow >= numExamples) continue;
        for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
            expected[batchRow * fieldBytes + byte] =
                records[sourceRow * recordSizeBytes + fieldOffsetBytes + byte];
        }
    }

    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchSize, fieldBytes},
        std::vector<uint8_t>(batchSize * fieldBytes, destinationSentinel),
        stream);

    launchDeviceResidentDirectMaterializationKernel(
        recordStorage,
        numExamples,
        recordSizeBytes,
        fieldOffsetBytes,
        fieldBytes,
        logicalRows,
        destination,
        rowIndicesDevice,
        stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        EXPECT_EQ(actual[offset], expected[offset]) << "byte offset " << offset;
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, LogicalRowsLeaveInactiveBatchCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    runMaterializationCase(/*numExamples=*/97,
                           /*batchSize=*/257,
                           /*recordSizeBytes=*/64,
                           /*fieldOffsetBytes=*/32,
                           /*fieldBytes=*/32,
                           /*logicalRows=*/17);
}

TEST(DeviceResidentDirectMaterializationKernelTest, EveryPacketWidthGathersFieldsExactlyAtEveryDepth) {
    REQUIRE_CUDA_DEVICE();

    struct Layout {
        uint64_t recordSizeBytes;
        uint64_t fieldOffsetBytes;
        uint64_t fieldBytes;
        uint64_t packetsPerField;
    };
    // The gather takes the largest power of two up to 16 that divides the field bytes, the record size and both
    // base addresses (the bases are allocation aligned, so the field offset counts): 16, 16, 8, 4, 2, 1 bytes here.
    constexpr std::array<Layout, 6> layouts{{
        {64, 32, 32, 2},
        {48, 16, 32, 2},
        {24, 8, 8, 1},
        {12, 4, 4, 1},
        {6, 2, 2, 1},
        {5, 1, 3, 3},
    }};
    // It takes 1, 2, 4 or 8 packets per thread by the batch's packet count (steps of 196608; 16-byte packets stop at
    // 4), so these packet counts put every width on each of its launches.
    for (const Layout layout : layouts) {
        for (const uint64_t packets : {1551ULL, 250000ULL, 500000ULL, 1000000ULL}) {
            runMaterializationCase(/*numExamples=*/607,
                                   /*batchSize=*/packets / layout.packetsPerField,
                                   layout.recordSizeBytes,
                                   layout.fieldOffsetBytes,
                                   layout.fieldBytes);
        }
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, EveryFieldLengthUpToSixtyFiveBytesGathersExactly) {
    REQUIRE_CUDA_DEVICE();

    for (uint64_t fieldBytes = 1; fieldBytes <= 65; ++fieldBytes) {
        // Packed: 3 bytes before the field and 5 after, so fields start at every address.
        runMaterializationCase(/*numExamples=*/97, /*batchSize=*/300, fieldBytes + 8, /*fieldOffsetBytes=*/3, fieldBytes);
        // The whole record: the last field ends on the last byte of the storage.
        runMaterializationCase(/*numExamples=*/97, /*batchSize=*/300, fieldBytes, /*fieldOffsetBytes=*/0, fieldBytes);
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, LongFieldsGatherExactlyFromPackedAndPaddedRecords) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t fieldBytes;
        uint64_t batchSize;
    };
    constexpr std::array<LaunchCase, 9> cases{{
        {4097, 128},
        {4096, 128},
        {2049, 256},
        {1025, 512},
        {1024, 512},
        {513, 1024},
        {257, 2048},
        {129, 4096},
        {100, 4096},
    }};
    for (const LaunchCase launchCase : cases) {
        runMaterializationCase(/*numExamples=*/17, launchCase.batchSize, launchCase.fieldBytes + 8, /*fieldOffsetBytes=*/3,
                               launchCase.fieldBytes);
        runMaterializationCase(/*numExamples=*/17,
                               launchCase.batchSize,
                               /*recordSizeBytes=*/(launchCase.fieldBytes + 31) / 32 * 32 + 32,
                               /*fieldOffsetBytes=*/32,
                               launchCase.fieldBytes);
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, OneByteFieldsGatherExactlyAtEveryBatchSize) {
    REQUIRE_CUDA_DEVICE();

    for (const uint64_t batchSize : {1ULL, 64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL}) {
        runMaterializationCase(/*numExamples=*/31, batchSize, /*recordSizeBytes=*/1, /*fieldOffsetBytes=*/0, /*fieldBytes=*/1);
    }
}

}  // namespace
