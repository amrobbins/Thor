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

void runByteGatherCase(uint64_t sourceRows, uint64_t batchSize, uint64_t rowBytes) {
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
    // Invalid resident row indices historically leave the destination row
    // untouched; retain that behavior under the row/chunk mapping.
    if (batchSize > 7) rowIndices[7] = sourceRows + 3;
    if (batchSize > 263) rowIndices[263] = sourceRows;

    std::vector<uint8_t> expected(batchSize * rowBytes, destinationSentinel);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
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

    launchDeviceResidentNamedGatherKernel(sourceDevice, destination, rowIndicesDevice, stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        EXPECT_EQ(actual[offset], expected[offset]) << "byte offset " << offset;
    }
}

TEST(DeviceResidentNamedGatherKernelTest, AlignedRowsUseThirtyTwoByteTransactions) {
    REQUIRE_CUDA_DEVICE();
    // CUDA tensor allocations are naturally wide-aligned and a 32-byte row
    // stride preserves that alignment for every source/destination row. This
    // exercises the ulonglong4_32a path.
    runByteGatherCase(/*sourceRows=*/97, /*batchSize=*/16384, /*rowBytes=*/32);
}

TEST(DeviceResidentNamedGatherKernelTest, SixteenByteFallbackRemainsContiguousAcrossLanes) {
    REQUIRE_CUDA_DEVICE();
    // A 48-byte row stride prevents a globally safe 32-byte transaction but
    // preserves 16-byte alignment. Launch geometry still follows the 48-byte
    // row payload, so fallback width does not reduce lane-level parallelism.
    runByteGatherCase(/*sourceRows=*/101, /*batchSize=*/8192, /*rowBytes=*/48);
}

TEST(DeviceResidentNamedGatherKernelTest, EveryNarrowerTransactionWidthAndByteFallbackPreserveRows) {
    REQUIRE_CUDA_DEVICE();

    // Each row width deliberately defeats the next-wider transaction while
    // preserving the requested width: 8, 4, 2, then exact byte fallback.
    runByteGatherCase(/*sourceRows=*/607, /*batchSize=*/517, /*rowBytes=*/24);
    runByteGatherCase(/*sourceRows=*/607, /*batchSize=*/517, /*rowBytes=*/12);
    runByteGatherCase(/*sourceRows=*/607, /*batchSize=*/517, /*rowBytes=*/6);
    runByteGatherCase(/*sourceRows=*/607, /*batchSize=*/517, /*rowBytes=*/5);
}

TEST(DeviceResidentNamedGatherKernelTest, PayloadAwareGroupingCoversFullLaneLadder) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t rowBytes;
        uint64_t batchSize;
    };

    // These exact payload thresholds select 1/2/4/8/16/32/64/128/256 rows per
    // CTA respectively once the matching batch-size parallelism floor permits
    // them. Each destination remains about 512 KiB or smaller.
    constexpr std::array<LaunchCase, 9> cases{{
        {4097, 64},
        {4096, 128},
        {2048, 256},
        {1024, 512},
        {512, 1024},
        {256, 2048},
        {128, 4096},
        {64, 8192},
        {32, 16384},
    }};

    for (const LaunchCase launchCase : cases) {
        runByteGatherCase(/*sourceRows=*/17,
                          launchCase.batchSize,
                          launchCase.rowBytes);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, PayloadThresholdTransitionsUseNextWiderRowGrouping) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t rowBytes;
        uint64_t batchSize;
    };

    // Exercise the first byte immediately above every 32-bytes/lane boundary.
    // Each transition halves rows/CTA and doubles lanes/row independently of
    // the transaction width selected for the tensor row.
    constexpr std::array<LaunchCase, 8> cases{{
        {33, 16384},
        {65, 8192},
        {129, 4096},
        {257, 2048},
        {513, 1024},
        {1025, 512},
        {2049, 256},
        {4097, 128},
    }};

    for (const LaunchCase launchCase : cases) {
        runByteGatherCase(/*sourceRows=*/17,
                          launchCase.batchSize,
                          launchCase.rowBytes);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, SmallRowsRetainBlockParallelismFloor) {
    REQUIRE_CUDA_DEVICE();

    // A one-byte row always prefers 256 rows/CTA by payload. The batch-size
    // guard intentionally walks the full 1/2/4/8/16/32/64/128/256 rows/CTA
    // ladder so small batches still expose roughly 64 CTAs of parallelism.
    for (const uint64_t batchSize :
         {64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL}) {
        runByteGatherCase(/*sourceRows=*/31, batchSize, /*rowBytes=*/1);
    }
}

TEST(DeviceResidentNamedGatherKernelTest, ScalarTypedRowsKeepManyIndependentRowsPerBlock) {
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

    launchDeviceResidentNamedGatherKernel(sourceDevice, destination, rowIndicesDevice, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint32_t>(destination, stream), expected);
}

}  // namespace
