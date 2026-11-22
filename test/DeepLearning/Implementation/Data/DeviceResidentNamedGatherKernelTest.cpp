#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

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
            GTEST_SKIP() << "CUDA device is required for device-resident named gather tests.";                          \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t> &dimensions, const std::vector<T> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("DeviceResidentNamedGatherKernel test tensor value count mismatch.");
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

void runByteGatherCase(
    uint64_t sourceRows, uint64_t batchSize, uint64_t rowBytes, uint64_t logicalRows = 0) {
    if (logicalRows == 0) logicalRows = batchSize;
    ASSERT_LE(logicalRows, batchSize);
    constexpr uint8_t destinationSentinel = 0xcdU;

    std::vector<uint8_t> source(sourceRows * rowBytes);
    for (uint64_t sourceRow = 0; sourceRow < sourceRows; ++sourceRow) {
        for (uint64_t byte = 0; byte < rowBytes; ++byte) {
            source[sourceRow * rowBytes + byte] =
                static_cast<uint8_t>((sourceRow * 29 + byte * 17 + 11) & 0xffU);
        }
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
        rowIndices[batchRow] = (batchRow * 137 + 19) % sourceRows;
    }
    // An invalid resident row index leaves the destination row untouched.
    if (batchSize > 7) rowIndices[7] = sourceRows + 3;
    if (batchSize > 263) rowIndices[263] = sourceRows;

    std::vector<uint8_t> expected(batchSize * rowBytes, destinationSentinel);
    for (uint64_t batchRow = 0; batchRow < logicalRows; ++batchRow) {
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow >= sourceRows) continue;
        for (uint64_t byte = 0; byte < rowBytes; ++byte) {
            expected[batchRow * rowBytes + byte] = source[sourceRow * rowBytes + byte];
        }
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({sourceRows, rowBytes}, source, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchSize, rowBytes},
        std::vector<uint8_t>(batchSize * rowBytes, destinationSentinel),
        stream);

    launchDeviceResidentNamedGatherKernel(
        sourceDevice, destination, rowIndicesDevice, logicalRows, stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        EXPECT_EQ(actual[offset], expected[offset]) << "byte offset " << offset;
    }
}

}  // namespace

TEST(DeviceResidentNamedGatherKernelTest, LogicalRowsLeaveInactiveBatchCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    runByteGatherCase(
        /*sourceRows=*/97, /*batchSize=*/257, /*rowBytes=*/32, /*logicalRows=*/17);
}

TEST(DeviceResidentNamedGatherKernelTest, EveryPacketWidthGathersExactly) {
    REQUIRE_CUDA_DEVICE();
    // The packet width is the largest power of two up to 16 dividing the row bytes (the bases are allocation
    // aligned): 32 and 48 use 16-byte packets, 24 eight, 12 four, 6 two, 5 one. The launcher takes 1, 2, 4 or 8
    // packets per thread by the batch's packet count (up to 196608 packets per step; 16-byte packets stop at 4), and
    // these batch sizes put every width on each of its launches.
    for (const uint64_t batchSize : {8192ULL, 65536ULL, 131072ULL, 262144ULL, 524288ULL}) {
        for (const uint64_t rowBytes : {32ULL, 48ULL, 24ULL, 12ULL, 6ULL, 5ULL}) {
            runByteGatherCase(/*sourceRows=*/607, batchSize, rowBytes);
        }
    }
}

TEST(DeviceResidentNamedGatherKernelTest, EveryRowLengthUpToSixtyFiveBytesGathersExactly) {
    REQUIRE_CUDA_DEVICE();
    // Every residue of every packet width, with invalid rows (7, 263) that must stay untouched next to valid ones.
    for (uint64_t rowBytes = 1; rowBytes <= 65; ++rowBytes) {
        runByteGatherCase(/*sourceRows=*/97, /*batchSize=*/517, rowBytes);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, WideAndOddRowLengthsGatherExactly) {
    REQUIRE_CUDA_DEVICE();
    struct LaunchCase {
        uint64_t rowBytes;
        uint64_t batchSize;
    };
    // Powers of two and their odd neighbours from 64 bytes to past 4 KiB; each destination is about 512 KiB or less.
    constexpr std::array<LaunchCase, 17> cases{{
        {4097, 128},
        {4096, 128},
        {2049, 256},
        {2048, 256},
        {1025, 512},
        {1024, 512},
        {1000, 512},
        {513, 1024},
        {512, 1024},
        {257, 2048},
        {256, 2048},
        {129, 4096},
        {128, 4096},
        {100, 4096},
        {65, 8192},
        {64, 8192},
        {33, 16384},
    }};
    for (const LaunchCase launchCase : cases) {
        runByteGatherCase(/*sourceRows=*/17, launchCase.batchSize, launchCase.rowBytes);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, OneByteRowsGatherExactlyAtEveryBatchSize) {
    REQUIRE_CUDA_DEVICE();
    for (const uint64_t batchSize :
         {64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL}) {
        runByteGatherCase(/*sourceRows=*/31, batchSize, /*rowBytes=*/1);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, Uint32RowsGatherExactly) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t sourceRows = 1031;
    constexpr uint64_t batchSize = 1025;
    constexpr uint32_t destinationSentinel = 0xdeadbeefU;

    std::vector<uint32_t> source(sourceRows);
    for (uint64_t row = 0; row < sourceRows; ++row) {
        source[row] = static_cast<uint32_t>(row * 1009 + 17);
    }
    std::vector<uint64_t> rowIndices(batchSize);
    for (uint64_t row = 0; row < batchSize; ++row) {
        rowIndices[row] = (row * 193 + 23) % sourceRows;
    }
    rowIndices[13] = sourceRows + 1;

    std::vector<uint32_t> expected(batchSize, destinationSentinel);
    for (uint64_t row = 0; row < batchSize; ++row) {
        if (rowIndices[row] < sourceRows) expected[row] = source[rowIndices[row]];
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint32_t>({sourceRows}, source, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destination = makeGpuTensor<uint32_t>(
        {batchSize}, std::vector<uint32_t>(batchSize, destinationSentinel), stream);

    launchDeviceResidentNamedGatherKernel(
        sourceDevice, destination, rowIndicesDevice, batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint32_t>(destination, stream), expected);
}

TEST(DeviceResidentNamedGatherKernelTest, RowIndicesSpanAtAnOffsetIntoASharedMetadataBuffer) {
    REQUIRE_CUDA_DEVICE();
    // The file batch session packs the row indices and the window plans into one selection-metadata buffer and passes
    // each as a span (offset, count) of it. The kernel must read exactly the span: the words before and after it hold
    // indices that would gather the wrong rows.
    constexpr uint64_t sourceRows = 9;
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t spanOffset = 3;
    constexpr uint32_t destinationSentinel = 0xdeadbeefU;
    std::vector<uint32_t> source(sourceRows);
    for (uint64_t row = 0; row < sourceRows; ++row) source[row] = static_cast<uint32_t>(1000 + row);
    const std::vector<uint64_t> metadata = {0, 0, 0, 5, 2, 8, 1, 0, 0};

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint32_t>({sourceRows}, source, stream);
    Tensor metadataDevice = makeGpuTensor<uint64_t>({metadata.size()}, metadata, stream);
    Tensor destination = makeGpuTensor<uint32_t>(
        {batchSize}, std::vector<uint32_t>(batchSize, destinationSentinel), stream);

    launchDeviceResidentNamedGatherKernel(
        sourceDevice, destination, TensorSpan(metadataDevice, spanOffset, batchSize), batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint32_t>(destination, stream), (std::vector<uint32_t>{1005, 1002, 1008, 1001}));

    // A span that runs past the buffer, or whose count is not the batch size, is rejected.
    EXPECT_THROW(TensorSpan(metadataDevice, 6, batchSize), std::invalid_argument);
    EXPECT_THROW(launchDeviceResidentNamedGatherKernel(sourceDevice, destination, TensorSpan(metadataDevice, spanOffset, 3),
                                                       batchSize, stream),
                 std::exception);
}
