#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for device-resident ragged materialization tests.";                \
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
        throw std::runtime_error("DeviceResidentRaggedMaterializationKernel test tensor value count mismatch.");
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

void writeUnalignedUint64(std::vector<uint8_t> &bytes, uint64_t offset, uint64_t value) {
    for (uint64_t i = 0; i < sizeof(uint64_t); ++i) {
        bytes[offset + i] = static_cast<uint8_t>((value >> (8 * i)) & 0xffU);
    }
}

template <typename OffsetT>
void runMultiTileMaterializationCase() {
    constexpr uint64_t numExamples = 600;
    constexpr uint64_t logicalRows = 513;
    constexpr uint64_t batchSize = 520;
    constexpr uint64_t recordSizeBytes = 23;
    constexpr uint64_t referenceOffsetBytes = 3;
    constexpr uint64_t valueBytes = sizeof(uint32_t);
    constexpr uint32_t destinationSentinel = 0xdeadbeefU;

    std::vector<uint64_t> starts(numExamples, 0);
    std::vector<uint64_t> counts(numExamples, 0);
    uint64_t storedValueCount = 0;
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        starts[sourceRow] = storedValueCount;
        counts[sourceRow] = (sourceRow * 7 + 3) % 5;
        storedValueCount += counts[sourceRow];
    }

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0xa5U);
    std::vector<uint32_t> packedValues(storedValueCount);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        const uint64_t reference = sourceRow * recordSizeBytes + referenceOffsetBytes;
        writeUnalignedUint64(records, reference, starts[sourceRow]);
        writeUnalignedUint64(records, reference + sizeof(uint64_t), counts[sourceRow]);
        for (uint64_t item = 0; item < counts[sourceRow]; ++item) {
            packedValues[starts[sourceRow] + item] =
                static_cast<uint32_t>(sourceRow * 1000 + item + 1);
        }
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        // 137 is coprime to 600, so these rows form a permutation prefix rather
        // than accidentally exercising only a small subset of resident records.
        rowIndices[row] = (row * 137 + 17) % numExamples;
    }

    std::vector<uint64_t> expectedOffsets(batchSize + 1, 0);
    std::vector<uint32_t> expectedValues;
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        for (uint64_t item = 0; item < counts[sourceRow]; ++item) {
            expectedValues.push_back(packedValues[starts[sourceRow] + item]);
        }
        expectedOffsets[row + 1] = expectedValues.size();
    }
    for (uint64_t row = logicalRows; row < batchSize; ++row) {
        expectedOffsets[row + 1] = expectedValues.size();
    }

    const uint64_t destinationCapacity = expectedValues.size() + 17;
    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage =
        makeGpuTensor<uint32_t>({packedValues.size()}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint32_t>(
        {destinationCapacity}, std::vector<uint32_t>(destinationCapacity, destinationSentinel), stream);
    std::vector<OffsetT> publishedOffsets(expectedOffsets.size());
    for (size_t i = 0; i < expectedOffsets.size(); ++i) {
        publishedOffsets[i] = static_cast<OffsetT>(expectedOffsets[i]);
    }
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, publishedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
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
    stream.synchronize();

    const std::vector<OffsetT> actualOffsets = copyGpuTensor<OffsetT>(destinationOffsets, stream);
    ASSERT_EQ(actualOffsets.size(), expectedOffsets.size());
    for (uint64_t i = 0; i < expectedOffsets.size(); ++i) {
        EXPECT_EQ(static_cast<uint64_t>(actualOffsets[i]), expectedOffsets[i]) << "offset " << i;
    }

    const std::vector<uint32_t> actualValues = copyGpuTensor<uint32_t>(destinationValues, stream);
    ASSERT_GE(actualValues.size(), expectedValues.size());
    for (uint64_t i = 0; i < expectedValues.size(); ++i) {
        EXPECT_EQ(actualValues[i], expectedValues[i]) << "active value " << i;
    }
    for (uint64_t i = expectedValues.size(); i < actualValues.size(); ++i) {
        EXPECT_EQ(actualValues[i], destinationSentinel) << "inactive value " << i;
    }
}

template <typename OffsetT>
void runPayloadAwareRowGroupingCase(uint64_t expectedResidentRowBytes,
                                    uint64_t logicalRows) {
    constexpr uint64_t numExamples = 2;
    constexpr uint64_t recordSizeBytes = 32;
    constexpr uint64_t referenceOffsetBytes = 8;
    constexpr uint64_t valueBytes = 1;
    constexpr uint64_t selectedRowBytes = 1;
    constexpr uint8_t destinationSentinel = 0xd7U;
    const uint64_t batchSize = logicalRows + 3;

    // Keep the selected batch tiny while independently controlling the
    // resident-field average used by the launch selector. Both examples
    // together contain exactly 2 * expectedResidentRowBytes bytes, but every
    // selected row points at the one-byte example 0.
    ASSERT_GE(expectedResidentRowBytes, 1U);
    const uint64_t secondRowBytes = 2 * expectedResidentRowBytes - selectedRowBytes;
    const uint64_t storedValueCount = selectedRowBytes + secondRowBytes;

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0xa5U);
    std::vector<uint8_t> packedValues(storedValueCount, 0);
    writeUnalignedUint64(records, referenceOffsetBytes, 0);
    writeUnalignedUint64(records, referenceOffsetBytes + sizeof(uint64_t), selectedRowBytes);
    writeUnalignedUint64(records, recordSizeBytes + referenceOffsetBytes, selectedRowBytes);
    writeUnalignedUint64(records,
                         recordSizeBytes + referenceOffsetBytes + sizeof(uint64_t),
                         secondRowBytes);
    packedValues[0] = 0x4dU;
    for (uint64_t byte = selectedRowBytes; byte < storedValueCount; ++byte) {
        packedValues[byte] = static_cast<uint8_t>((byte * 7 + 11) & 0xffU);
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    const uint64_t activeValueCount = logicalRows * selectedRowBytes;
    const uint64_t destinationCapacityValues = activeValueCount + 17;
    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage =
        makeGpuTensor<uint8_t>({storedValueCount}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>(
        {destinationCapacityValues},
        std::vector<uint8_t>(destinationCapacityValues, destinationSentinel),
        stream);
    std::vector<OffsetT> expectedOffsets(batchSize + 1, static_cast<OffsetT>(logicalRows));
    for (uint64_t row = 0; row <= logicalRows; ++row) {
        expectedOffsets[row] = static_cast<OffsetT>(row);
    }
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, expectedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
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
    stream.synchronize();

    const std::vector<OffsetT> actualOffsets = copyGpuTensor<OffsetT>(destinationOffsets, stream);
    EXPECT_EQ(actualOffsets, expectedOffsets)
        << "expectedResidentRowBytes=" << expectedResidentRowBytes
        << " logicalRows=" << logicalRows;

    const std::vector<uint8_t> actualValues = copyGpuTensor<uint8_t>(destinationValues, stream);
    std::vector<uint8_t> expectedValues(destinationCapacityValues, destinationSentinel);
    std::fill(expectedValues.begin(), expectedValues.begin() + activeValueCount, packedValues[0]);
    EXPECT_EQ(actualValues, expectedValues)
        << "expectedResidentRowBytes=" << expectedResidentRowBytes
        << " logicalRows=" << logicalRows;
}

template <typename OffsetT>
void runReferenceLoadAlignmentCase(uint64_t recordSizeBytes, uint64_t referenceOffsetBytes) {
    constexpr uint64_t numExamples = 37;
    constexpr uint64_t logicalRows = 19;
    constexpr uint64_t batchSize = 23;
    constexpr uint64_t valueBytes = 3;
    constexpr uint8_t destinationSentinel = 0xe3U;
    const uint64_t storedValueCount = numExamples;

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0x5aU);
    std::vector<uint8_t> packedValues(storedValueCount * valueBytes, 0);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        const uint64_t reference = sourceRow * recordSizeBytes + referenceOffsetBytes;
        writeUnalignedUint64(records, reference, sourceRow);
        writeUnalignedUint64(records, reference + sizeof(uint64_t), 1);
        for (uint64_t byte = 0; byte < valueBytes; ++byte) {
            packedValues[sourceRow * valueBytes + byte] =
                static_cast<uint8_t>((sourceRow * 13 + byte + 3) & 0xffU);
        }
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        rowIndices[row] = (row * 11 + 5) % numExamples;
    }

    const uint64_t destinationCapacityValues = logicalRows + 7;
    const uint64_t destinationBytes = destinationCapacityValues * valueBytes;
    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage =
        makeGpuTensor<uint8_t>({storedValueCount, valueBytes}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>(
        {destinationCapacityValues, valueBytes},
        std::vector<uint8_t>(destinationBytes, destinationSentinel),
        stream);
    std::vector<OffsetT> expectedOffsets(batchSize + 1, static_cast<OffsetT>(logicalRows));
    for (uint64_t row = 0; row <= logicalRows; ++row) {
        expectedOffsets[row] = static_cast<OffsetT>(row);
    }
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, expectedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
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
    stream.synchronize();

    const std::vector<OffsetT> actualOffsets = copyGpuTensor<OffsetT>(destinationOffsets, stream);
    EXPECT_EQ(actualOffsets, expectedOffsets)
        << "recordSize=" << recordSizeBytes << " referenceOffset=" << referenceOffsetBytes;

    const std::vector<uint8_t> actualValues = copyGpuTensor<uint8_t>(destinationValues, stream);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        for (uint64_t byte = 0; byte < valueBytes; ++byte) {
            EXPECT_EQ(actualValues[row * valueBytes + byte],
                      packedValues[sourceRow * valueBytes + byte])
                << "recordSize=" << recordSizeBytes << " referenceOffset=" << referenceOffsetBytes
                << " row=" << row << " byte=" << byte;
        }
    }
    for (uint64_t byte = logicalRows * valueBytes; byte < actualValues.size(); ++byte) {
        EXPECT_EQ(actualValues[byte], destinationSentinel);
    }
}

template <typename OffsetT>
void runAlignedBulkWithEveryExactTailCase(uint32_t expectedCopyWidth) {
    constexpr uint64_t recordSizeBytes = 32;
    constexpr uint64_t referenceOffsetBytes = 8;
    constexpr uint64_t valueBytes = 1;
    constexpr uint8_t destinationSentinel = 0xd9U;

    ASSERT_TRUE(expectedCopyWidth == 2 || expectedCopyWidth == 4 ||
                expectedCopyWidth == 8 || expectedCopyWidth == 16 ||
                expectedCopyWidth == 32);

    // Keep each target row at exactly expectedCopyWidth alignment. For widths
    // below 32 an initial row shifts both packed source and destination starts
    // to W modulo 2W. Each target/filler pair then consumes exactly 2W bytes,
    // preserving that alignment for the next target. Target row W + tail thus
    // exercises every Tail<1..W-1> specialization while the W - tail filler
    // restores the next target's alignment.
    std::vector<uint64_t> rowCounts;
    if (expectedCopyWidth < 32) rowCounts.push_back(expectedCopyWidth);
    for (uint32_t tail = 1; tail < expectedCopyWidth; ++tail) {
        rowCounts.push_back(expectedCopyWidth + tail);
        rowCounts.push_back(expectedCopyWidth - tail);
    }

    // End with another aligned odd-sized target rather than its filler. Its
    // full-width source tail load therefore crosses the logical end of the
    // packed tensor into Thor's 128-byte backing padding.
    const uint32_t finalTail = std::min<uint32_t>(13, expectedCopyWidth - 1);
    rowCounts.push_back(expectedCopyWidth + finalTail);

    const uint64_t numExamples = rowCounts.size();
    const uint64_t logicalRows = numExamples;
    const uint64_t batchSize = logicalRows + 2;
    std::vector<uint64_t> starts(numExamples, 0);
    uint64_t storedValueCount = 0;
    for (uint64_t row = 0; row < numExamples; ++row) {
        starts[row] = storedValueCount;
        storedValueCount += rowCounts[row];
    }

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0xa5U);
    for (uint64_t row = 0; row < numExamples; ++row) {
        const uint64_t reference = row * recordSizeBytes + referenceOffsetBytes;
        writeUnalignedUint64(records, reference, starts[row]);
        // value_count is not part of the materializer contract. Poison it so
        // this test also cannot accidentally regress toward resident-count use.
        writeUnalignedUint64(records,
                             reference + sizeof(uint64_t),
                             std::numeric_limits<uint64_t>::max() - row);
    }

    std::vector<uint8_t> packedValues(storedValueCount);
    for (uint64_t byte = 0; byte < packedValues.size(); ++byte) {
        packedValues[byte] =
            static_cast<uint8_t>((byte * 29 + expectedCopyWidth * 7 + 3) & 0xffU);
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t row = 0; row < logicalRows; ++row) rowIndices[row] = row;

    std::vector<OffsetT> expectedOffsets(batchSize + 1,
                                         static_cast<OffsetT>(storedValueCount));
    for (uint64_t row = 0; row < logicalRows; ++row) {
        expectedOffsets[row] = static_cast<OffsetT>(starts[row]);
    }
    expectedOffsets[logicalRows] = static_cast<OffsetT>(storedValueCount);

    const uint64_t destinationCapacity = storedValueCount + 64;
    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage =
        makeGpuTensor<uint8_t>({storedValueCount}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>(
        {destinationCapacity},
        std::vector<uint8_t>(destinationCapacity, destinationSentinel),
        stream);
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, expectedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
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
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(destinationOffsets, stream), expectedOffsets);
    const std::vector<uint8_t> actualValues = copyGpuTensor<uint8_t>(destinationValues, stream);
    ASSERT_GE(actualValues.size(), packedValues.size());
    for (uint64_t byte = 0; byte < packedValues.size(); ++byte) {
        EXPECT_EQ(actualValues[byte], packedValues[byte])
            << "copyWidth=" << expectedCopyWidth << " active byte=" << byte;
    }
    for (uint64_t byte = packedValues.size(); byte < actualValues.size(); ++byte) {
        EXPECT_EQ(actualValues[byte], destinationSentinel)
            << "copyWidth=" << expectedCopyWidth << " inactive byte=" << byte;
    }
}

template <typename OffsetT>
void runAllEmptyFastPathCase() {
    constexpr uint64_t numExamples = 7;
    constexpr uint64_t logicalRows = 5;
    constexpr uint64_t batchSize = 9;
    constexpr uint64_t recordSizeBytes = 23;
    constexpr uint64_t referenceOffsetBytes = 3;
    constexpr uint64_t valueBytes = 3;
    constexpr uint8_t destinationSentinel = 0xb9U;

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0xa5U);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        const uint64_t reference = sourceRow * recordSizeBytes + referenceOffsetBytes;
        writeUnalignedUint64(records, reference, 0);
        writeUnalignedUint64(records, reference + sizeof(uint64_t), 0);
    }
    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        rowIndices[row] = (row * 3 + 1) % numExamples;
    }

    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage;
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>(
        {31, valueBytes}, std::vector<uint8_t>(31 * valueBytes, destinationSentinel), stream);
    const std::vector<OffsetT> publishedOffsets(batchSize + 1, static_cast<OffsetT>(0));
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, publishedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
        recordStorage,
        packedValuesStorage,
        numExamples,
        recordSizeBytes,
        referenceOffsetBytes,
        0,
        valueBytes,
        logicalRows,
        destinationValues,
        destinationOffsets,
        rowIndicesDevice,
        stream);
    stream.synchronize();

    const std::vector<OffsetT> actualOffsets = copyGpuTensor<OffsetT>(destinationOffsets, stream);
    for (uint64_t i = 0; i < actualOffsets.size(); ++i) {
        EXPECT_EQ(actualOffsets[i], static_cast<OffsetT>(0)) << "offset " << i;
    }
    const std::vector<uint8_t> actualValues = copyGpuTensor<uint8_t>(destinationValues, stream);
    for (uint64_t i = 0; i < actualValues.size(); ++i) {
        EXPECT_EQ(actualValues[i], destinationSentinel) << "destination byte " << i;
    }
}

template <typename OffsetT>
void runPublishedOffsetsCase() {
    constexpr uint64_t numExamples = 3;
    constexpr uint64_t logicalRows = 3;
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t recordSizeBytes = 32;
    constexpr uint64_t referenceOffsetBytes = 0;
    constexpr uint64_t valueBytes = 32;
    constexpr uint64_t storedValueCount = 6;
    constexpr uint8_t destinationSentinel = 0xc7U;

    // True resident row lengths are {2,1,3}, but value_count is deliberately
    // poisoned. Materialization must derive selected row lengths solely
    // from the already-published destination offsets and read only start_value
    // from resident metadata.
    const std::vector<uint64_t> starts{0, 2, 3};
    const std::vector<uint64_t> rowIndices{2, 0, 1, 1};
    const std::vector<uint64_t> expectedOffsets64{0, 3, 5, 6, 6};
    std::vector<OffsetT> expectedOffsets(expectedOffsets64.size());
    for (size_t i = 0; i < expectedOffsets64.size(); ++i) {
        expectedOffsets[i] = static_cast<OffsetT>(expectedOffsets64[i]);
    }

    std::vector<uint8_t> records(numExamples * recordSizeBytes, 0xa5U);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        const uint64_t reference = sourceRow * recordSizeBytes + referenceOffsetBytes;
        writeUnalignedUint64(records, reference, starts[sourceRow]);
        writeUnalignedUint64(records,
                             reference + sizeof(uint64_t),
                             std::numeric_limits<uint64_t>::max() - sourceRow);
    }

    std::vector<uint8_t> packedValues(storedValueCount * valueBytes);
    for (uint64_t value = 0; value < storedValueCount; ++value) {
        for (uint64_t byte = 0; byte < valueBytes; ++byte) {
            packedValues[value * valueBytes + byte] =
                static_cast<uint8_t>((value * 37 + byte * 5 + 11) & 0xffU);
        }
    }

    const uint64_t destinationCapacityValues = storedValueCount + 2;
    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage =
        makeGpuTensor<uint8_t>({storedValueCount, valueBytes}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>(
        {destinationCapacityValues, valueBytes},
        std::vector<uint8_t>(destinationCapacityValues * valueBytes, destinationSentinel),
        stream);
    Tensor destinationOffsets =
        makeGpuTensor<OffsetT>({batchSize + 1}, expectedOffsets, stream);

    launchDeviceResidentRaggedMaterializationKernel(
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
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(destinationOffsets, stream), expectedOffsets);

    const std::vector<uint8_t> actualValues = copyGpuTensor<uint8_t>(destinationValues, stream);
    std::vector<uint8_t> expectedValues(
        destinationCapacityValues * valueBytes, destinationSentinel);
    uint64_t destinationValue = 0;
    const std::vector<uint64_t> trueCounts{2, 1, 3};
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        for (uint64_t item = 0; item < trueCounts[sourceRow]; ++item) {
            std::copy_n(
                packedValues.begin() + (starts[sourceRow] + item) * valueBytes,
                valueBytes,
                expectedValues.begin() + destinationValue * valueBytes);
            ++destinationValue;
        }
    }
    EXPECT_EQ(actualValues, expectedValues);
}

TEST(DeviceResidentRaggedMaterializationKernelTest, LargeBatchGatherPreservesPublishedUint32Offsets) {
    REQUIRE_CUDA_DEVICE();
    runMultiTileMaterializationCase<uint32_t>();
}

TEST(DeviceResidentRaggedMaterializationKernelTest, LargeBatchGatherPreservesPublishedUint64Offsets) {
    REQUIRE_CUDA_DEVICE();
    runMultiTileMaterializationCase<uint64_t>();
}

TEST(DeviceResidentRaggedMaterializationKernelTest, PayloadAwareRowGroupingCoversFullLaneLadderForBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();

    // With >=16384 logical rows the parallelism floor permits every grouping,
    // so these resident averages select 256/128/64/32/16/8/4/2/1 rows per CTA
    // respectively with the 32-byte-per-lane payload target. The selected
    // example itself is only one byte, keeping this launch-policy test
    // lightweight even for the >4-KiB resident average.
    for (const uint64_t expectedResidentRowBytes :
         {32ULL, 33ULL, 65ULL, 129ULL, 257ULL, 513ULL, 1025ULL, 2049ULL, 4097ULL}) {
        runPayloadAwareRowGroupingCase<uint32_t>(expectedResidentRowBytes, 16384);
        runPayloadAwareRowGroupingCase<uint64_t>(expectedResidentRowBytes, 16384);
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, PayloadAwareRowGroupingRetainsBlockParallelismFloor) {
    REQUIRE_CUDA_DEVICE();

    // A one-byte resident average always prefers 256 rows/CTA by payload. The
    // row-count guard should nevertheless retain roughly 64 CTAs until enough
    // rows exist, exercising 1/2/4/8/16/32/64/128/256 rows per CTA.
    for (const uint64_t logicalRows :
         {64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL, 8192ULL, 16384ULL}) {
        runPayloadAwareRowGroupingCase<uint32_t>(1, logicalRows);
        runPayloadAwareRowGroupingCase<uint64_t>(1, logicalRows);
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, ReferenceMetadataUsesEveryAlignedLoadWidthAndByteFallback) {
    REQUIRE_CUDA_DEVICE();
    // 8-byte aligned, 4-byte aligned only, 2-byte aligned only, and deliberately unaligned.
    runReferenceLoadAlignmentCase<uint32_t>(32, 8);
    runReferenceLoadAlignmentCase<uint32_t>(20, 4);
    runReferenceLoadAlignmentCase<uint32_t>(18, 2);
    runReferenceLoadAlignmentCase<uint32_t>(19, 3);
}

TEST(DeviceResidentRaggedMaterializationKernelTest,
     AlignedBulkCopiesExerciseEveryExactTailSizeWithoutOverwritingFollowingCapacity) {
    REQUIRE_CUDA_DEVICE();

    // Logical row length no longer participates in copy-width selection. These
    // batches force every legal nonzero tail for each vector width while keeping
    // the corresponding target row at exactly that alignment. The final target
    // also ends at packedValuesStorage's logical end, requiring the full-width
    // source tail load to rely on Thor's 128-byte allocation padding.
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        runAlignedBulkWithEveryExactTailCase<uint32_t>(copyWidth);
        runAlignedBulkWithEveryExactTailCase<uint64_t>(copyWidth);
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest,
     PublishedOffsetsIgnorePoisonedCountsAndUseWideCopies) {
    REQUIRE_CUDA_DEVICE();
    runPublishedOffsetsCase<uint32_t>();
    runPublishedOffsetsCase<uint64_t>();
}

TEST(DeviceResidentRaggedMaterializationKernelTest,
     AllEmptyStoredFieldPreservesPublishedOffsetsWithoutLaunchingValueGather) {
    REQUIRE_CUDA_DEVICE();
    runAllEmptyFastPathCase<uint32_t>();
    runAllEmptyFastPathCase<uint64_t>();
}

}  // namespace
