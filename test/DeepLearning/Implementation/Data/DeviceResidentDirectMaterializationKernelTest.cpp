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
    uint64_t fieldBytes) {
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
    // Invalid row indices historically leave their destination row untouched.
    // Keep that behavior under the row/chunk mapping as well.
    if (batchSize > 7) rowIndices[7] = numExamples + 3;
    if (batchSize > 263) rowIndices[263] = numExamples;

    std::vector<uint8_t> expected(batchSize * fieldBytes, destinationSentinel);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
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

void runExactTailCase(uint32_t copyWidth, uint32_t tailBytes) {
    ASSERT_TRUE(copyWidth == 32 || copyWidth == 16 || copyWidth == 8 ||
                copyWidth == 4 || copyWidth == 2);
    ASSERT_GT(tailBytes, 0U);
    ASSERT_LT(tailBytes, copyWidth);

    constexpr uint64_t numExamples = 1;
    constexpr uint64_t batchSize = 2;
    constexpr uint8_t prefixSentinel = 0x5aU;
    constexpr uint8_t destinationSentinel = 0xcdU;

    // CUDA tensor allocations are at least 32-byte aligned. Offset the one
    // valid source field by exactly the requested width (except for 32 itself)
    // so source+destination jointly support exactly copyWidth and no wider
    // transaction. The field reaches the logical end of recordStorage, making
    // the final full-width source packet rely on Thor's 128-byte allocation
    // padding for its legal over-read.
    const uint64_t fieldOffsetBytes = copyWidth == 32 ? 0 : copyWidth;
    const uint64_t fieldBytes = copyWidth + tailBytes;
    const uint64_t recordSizeBytes = fieldOffsetBytes + fieldBytes;

    std::vector<uint8_t> records(recordSizeBytes, prefixSentinel);
    for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
        records[fieldOffsetBytes + byte] =
            static_cast<uint8_t>((byte * 37 + copyWidth * 11 + tailBytes) & 0xffU);
    }

    // Row 1 is intentionally invalid. It must remain sentinel-filled, which
    // makes any over-wide destination tail store from row 0 immediately visible
    // instead of allowing a neighboring valid row to overwrite the corruption.
    const std::vector<uint64_t> rowIndices{0, numExamples};
    std::vector<uint8_t> expected(batchSize * fieldBytes, destinationSentinel);
    for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
        expected[byte] = records[fieldOffsetBytes + byte];
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
        destination,
        rowIndicesDevice,
        stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        EXPECT_EQ(actual[offset], expected[offset])
            << "copyWidth=" << copyWidth << " tailBytes=" << tailBytes
            << " byte offset=" << offset;
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, AlignedWideFieldsUseThirtyTwoByteTransactions) {
    REQUIRE_CUDA_DEVICE();
    // CUDA allocations are naturally wide-aligned and every compact-record row,
    // field start, and destination row is 32-byte aligned. This exercises the
    // ulonglong4_32a path added by the production optimization pass.
    runMaterializationCase(/*numExamples=*/97,
                           /*batchSize=*/16384,
                           /*recordSizeBytes=*/64,
                           /*fieldOffsetBytes=*/32,
                           /*fieldBytes=*/32);
}

TEST(DeviceResidentDirectMaterializationKernelTest, SixteenByteFallbackRemainsContiguousAcrossLanes) {
    REQUIRE_CUDA_DEVICE();
    // A 48-byte record stride makes alternating source rows 16- versus 32-byte
    // aligned. Row-local width selection therefore exercises the 16-byte path
    // without changing the 32-byte payload-based lane grouping.
    runMaterializationCase(/*numExamples=*/101,
                           /*batchSize=*/8192,
                           /*recordSizeBytes=*/48,
                           /*fieldOffsetBytes=*/16,
                           /*fieldBytes=*/32);
}

TEST(DeviceResidentDirectMaterializationKernelTest, EveryNarrowerTransactionWidthAndByteFallbackPreserveFields) {
    REQUIRE_CUDA_DEVICE();

    // These compact byte strides force selected rows through the 8/4/2/1-byte
    // row-local alignment fallbacks while preserving exact field contents.
    runMaterializationCase(/*numExamples=*/607,
                           /*batchSize=*/517,
                           /*recordSizeBytes=*/24,
                           /*fieldOffsetBytes=*/8,
                           /*fieldBytes=*/8);
    runMaterializationCase(/*numExamples=*/607,
                           /*batchSize=*/517,
                           /*recordSizeBytes=*/12,
                           /*fieldOffsetBytes=*/4,
                           /*fieldBytes=*/4);
    runMaterializationCase(/*numExamples=*/607,
                           /*batchSize=*/517,
                           /*recordSizeBytes=*/6,
                           /*fieldOffsetBytes=*/2,
                           /*fieldBytes=*/2);
    runMaterializationCase(/*numExamples=*/607,
                           /*batchSize=*/517,
                           /*recordSizeBytes=*/5,
                           /*fieldOffsetBytes=*/1,
                           /*fieldBytes=*/3);
}

TEST(DeviceResidentDirectMaterializationKernelTest,
     AlignedBulkCopiesExerciseEveryExactTailSizeAndPreserveFollowingRow) {
    REQUIRE_CUDA_DEVICE();

    // For every vector width, exercise every legal nonzero Tail<N>. Each case
    // also makes the full-width source tail load cross recordStorage's logical
    // end into Thor's tensor padding, while the following destination row is an
    // invalid gather target that must remain untouched.
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        for (uint32_t tailBytes = 1; tailBytes < copyWidth; ++tailBytes) {
            runExactTailCase(copyWidth, tailBytes);
        }
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, PayloadAwareGroupingCoversFullLaneLadder) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t fieldBytes;
        uint64_t batchSize;
    };

    // These exact payload thresholds select 1/2/4/8/16/32/64/128/256 rows per
    // CTA respectively once the matching batch-size parallelism floor permits
    // them. The largest destination is only about 512 KiB.
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
        runMaterializationCase(/*numExamples=*/17,
                               launchCase.batchSize,
                               /*recordSizeBytes=*/launchCase.fieldBytes,
                               /*fieldOffsetBytes=*/0,
                               launchCase.fieldBytes);
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, PayloadThresholdTransitionsUseNextWiderRowGrouping) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t fieldBytes;
        uint64_t batchSize;
    };

    // Exercise the first byte immediately above every 32-bytes/lane boundary.
    // Each transition halves rows/CTA and doubles lanes/row without depending on
    // the transaction width selected for the compact record.
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
        runMaterializationCase(/*numExamples=*/17,
                               launchCase.batchSize,
                               /*recordSizeBytes=*/launchCase.fieldBytes,
                               /*fieldOffsetBytes=*/0,
                               launchCase.fieldBytes);
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, SmallFieldsRetainBlockParallelismFloor) {
    REQUIRE_CUDA_DEVICE();

    // A one-byte field always prefers 256 rows/CTA by payload. The batch-size
    // guard intentionally walks the entire 1/2/4/8/16/32/64/128/256 rows/CTA
    // ladder so small batches still expose roughly 64 CTAs of parallelism.
    for (const uint64_t batchSize :
         {64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL}) {
        runMaterializationCase(/*numExamples=*/31,
                               batchSize,
                               /*recordSizeBytes=*/1,
                               /*fieldOffsetBytes=*/0,
                               /*fieldBytes=*/1);
    }
}

}  // namespace
