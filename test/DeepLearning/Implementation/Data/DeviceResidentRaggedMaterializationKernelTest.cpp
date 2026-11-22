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
#include <string>
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

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

struct RaggedCase {
    uint64_t valueBytes = 1;
    // The stored rows' lengths in values: every row meanValues (Uniform); pseudo-random lengths from 0 to twice it
    // (Varied); or mostly short and empty rows and one row in eight of eight times it (FewLong).
    enum class Lengths { Uniform, Varied, FewLong };
    uint64_t meanValues = 8;
    Lengths lengths = Lengths::Varied;
    uint64_t numExamples = 64;
    uint64_t logicalRows = 64;
    uint64_t recordSizeBytes = 32;
    uint64_t referenceOffsetBytes = 8;
};

// Materialize a batch of pseudo-randomly chosen records and compare every byte with the definition: batch row r holds
// the values of its record, where the published offsets put them; the offsets are unchanged; the batch rows past the
// logical rows and the capacity past the last value are untouched. Every record's value count is poisoned.
template <typename OffsetT>
void runRaggedCase(const RaggedCase &testCase) {
    constexpr uint8_t sentinel = 0xcdU;
    const uint64_t numExamples = testCase.numExamples;
    std::vector<uint64_t> firstValue(numExamples);
    std::vector<uint64_t> valueCount(numExamples);
    uint64_t storedValueCount = 0;
    for (uint64_t example = 0; example < numExamples; ++example) {
        uint64_t count = testCase.meanValues;
        if (testCase.lengths == RaggedCase::Lengths::Varied) {
            count = splitMix64(example * 3 + 1) % (2 * testCase.meanValues + 1);
        } else if (testCase.lengths == RaggedCase::Lengths::FewLong) {
            count = splitMix64(example * 5 + 2) % 8 == 0   ? testCase.meanValues * 8
                    : splitMix64(example * 7 + 3) % 3 == 0 ? 0
                                                           : splitMix64(example * 11 + 4) % (testCase.meanValues / 2 + 2);
        }
        if (example + 1 == numExamples && storedValueCount + count == 0) count = 1;  // a store holds a value
        firstValue[example] = storedValueCount;
        valueCount[example] = count;
        storedValueCount += count;
    }

    std::vector<uint8_t> records(numExamples * testCase.recordSizeBytes, 0xeeU);
    for (uint64_t example = 0; example < numExamples; ++example) {
        const uint64_t reference = example * testCase.recordSizeBytes + testCase.referenceOffsetBytes;
        writeUnalignedUint64(records, reference, firstValue[example]);
        writeUnalignedUint64(records, reference + sizeof(uint64_t), ~valueCount[example]);
    }
    std::vector<uint8_t> packedValues(storedValueCount * testCase.valueBytes);
    for (uint64_t byte = 0; byte < packedValues.size(); ++byte) packedValues[byte] = static_cast<uint8_t>(splitMix64(byte) >> 24);

    const uint64_t logicalRows = testCase.logicalRows;
    const uint64_t batchSize = logicalRows + 2;
    std::vector<uint64_t> rowIndices(batchSize);
    std::vector<OffsetT> offsets(batchSize + 1, 0);
    for (uint64_t row = 0; row < batchSize; ++row) {
        rowIndices[row] = splitMix64(row * 13 + 5) % numExamples;
        const uint64_t count = row < logicalRows ? valueCount[rowIndices[row]] : 0;
        offsets[row + 1] = static_cast<OffsetT>(static_cast<uint64_t>(offsets[row]) + count);
    }
    const uint64_t batchValues = static_cast<uint64_t>(offsets[logicalRows]);
    const uint64_t capacityBytes = (batchValues + 3) * testCase.valueBytes;
    std::vector<uint8_t> expected(capacityBytes, sentinel);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        std::copy_n(packedValues.begin() + firstValue[rowIndices[row]] * testCase.valueBytes,
                    valueCount[rowIndices[row]] * testCase.valueBytes,
                    expected.begin() + static_cast<uint64_t>(offsets[row]) * testCase.valueBytes);
    }

    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor packedValuesStorage = makeGpuTensor<uint8_t>({packedValues.size()}, packedValues, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destinationOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, offsets, stream);
    Tensor destinationValues = makeGpuTensor<uint8_t>({capacityBytes}, std::vector<uint8_t>(capacityBytes, sentinel), stream);
    launchDeviceResidentRaggedMaterializationKernel(
        recordStorage, packedValuesStorage, numExamples, testCase.recordSizeBytes, testCase.referenceOffsetBytes, storedValueCount,
        testCase.valueBytes, logicalRows, destinationValues, destinationOffsets, rowIndicesDevice, stream);
    stream.synchronize();

    const std::string what = "valueBytes=" + std::to_string(testCase.valueBytes) + " meanValues=" + std::to_string(testCase.meanValues) +
                             " lengths=" + std::to_string(static_cast<int>(testCase.lengths)) + " examples=" +
                             std::to_string(numExamples) + " rows=" + std::to_string(logicalRows) + " record=" +
                             std::to_string(testCase.recordSizeBytes) + "+" + std::to_string(testCase.referenceOffsetBytes);
    EXPECT_EQ(copyGpuTensor<OffsetT>(destinationOffsets, stream), offsets) << what;
    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destinationValues, stream);
    ASSERT_EQ(actual.size(), expected.size()) << what;
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        ADD_FAILURE() << what << ": first wrong byte at " << (mismatch.first - actual.begin()) << " of " << actual.size() << ", got "
                      << static_cast<uint32_t>(*mismatch.first) << ", expected " << static_cast<uint32_t>(*mismatch.second);
    }
}

void runRaggedCaseWithBothOffsetWidths(const RaggedCase &testCase) {
    runRaggedCase<uint32_t>(testCase);
    runRaggedCase<uint64_t>(testCase);
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

TEST(DeviceResidentRaggedMaterializationKernelTest, EveryValueWidthMaterializesRaggedRowsExactlyAtEveryDepth) {
    REQUIRE_CUDA_DEVICE();

    // A value is whole packets of 2 to 16 bytes (12-byte values are 4-byte packets) and a lane takes 1, 2 or 4 of them
    // by the batch's row count. These row counts put every width on each of its depths between them; the largest
    // batches of 3-byte values are past them and move as 16-byte packets (as do all of
    // LargeBatchesOfByteAndTwoByteValuesMaterializeExactlyWhereverRowsStartAndEnd).
    for (const uint64_t valueBytes : {1ULL, 2ULL, 3ULL, 4ULL, 8ULL, 12ULL, 16ULL, 32ULL}) {
        for (const uint64_t logicalRows : {300ULL, 5000ULL, 10000ULL, 20000ULL, 40000ULL}) {
            RaggedCase testCase;
            testCase.valueBytes = valueBytes;
            testCase.meanValues = 16;
            testCase.numExamples = 3000;
            testCase.logicalRows = logicalRows;
            runRaggedCaseWithBothOffsetWidths(testCase);
        }
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, StoresOfEveryMeanRowLengthMaterializeExactly) {
    REQUIRE_CUDA_DEVICE();

    // The store's mean row sizes a row's share of a block, from 256 rows a block to one; rows shorter and longer
    // than the mean, empty rows and rows of several passes all come out the same.
    for (const uint64_t meanValues : {1ULL, 2ULL, 3ULL, 5ULL, 9ULL, 17ULL, 33ULL, 65ULL, 129ULL, 257ULL, 513ULL, 1025ULL, 4097ULL}) {
        for (const RaggedCase::Lengths lengths :
             {RaggedCase::Lengths::Uniform, RaggedCase::Lengths::Varied, RaggedCase::Lengths::FewLong}) {
            RaggedCase testCase;
            testCase.meanValues = meanValues;
            testCase.lengths = lengths;
            testCase.numExamples = 200;
            testCase.logicalRows = 300;
            runRaggedCaseWithBothOffsetWidths(testCase);
        }
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, LargeBatchesOfByteAndTwoByteValuesMaterializeExactlyWhereverRowsStartAndEnd) {
    REQUIRE_CUDA_DEVICE();

    // A batch of values on a 1- or 2-byte boundary that four packets a lane do not keep within 768 blocks moves in
    // 16-byte packets, and a row's first and last byte are anywhere in them: its ends are every mix of aligned 1-,
    // 2-, 4- and 8-byte pieces. Rows inside one packet, rows across a boundary with no whole packet, rows of many
    // packets and empty rows, in every mix the varied lengths give; each mean row length with a row count past
    // that bound, the last two with two packets to a lane.
    struct MeanAndRows {
        uint64_t meanValues;
        uint64_t logicalRows;
    };
    for (const uint64_t valueBytes : {1ULL, 2ULL, 3ULL}) {
        for (const MeanAndRows shape : {MeanAndRows{1, 200000}, MeanAndRows{4, 200000}, MeanAndRows{16, 200000}, MeanAndRows{20, 100000},
                                        MeanAndRows{100, 40000}, MeanAndRows{1000, 4000}}) {
            for (const RaggedCase::Lengths lengths : {RaggedCase::Lengths::Varied, RaggedCase::Lengths::FewLong}) {
                RaggedCase testCase;
                testCase.valueBytes = valueBytes;
                testCase.meanValues = shape.meanValues;
                testCase.lengths = lengths;
                testCase.numExamples = 500;
                testCase.logicalRows = shape.logicalRows;
                runRaggedCaseWithBothOffsetWidths(testCase);
            }
        }
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, BatchesOfEverySizeMaterializeExactly) {
    REQUIRE_CUDA_DEVICE();

    for (const uint64_t logicalRows : {1ULL, 2ULL, 3ULL, 255ULL, 256ULL, 257ULL, 1023ULL, 4097ULL, 70000ULL}) {
        for (const uint64_t valueBytes : {1ULL, 4ULL}) {
            RaggedCase testCase;
            testCase.valueBytes = valueBytes;
            testCase.meanValues = 5;
            testCase.numExamples = 1000;
            testCase.logicalRows = logicalRows;
            runRaggedCaseWithBothOffsetWidths(testCase);
        }
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, LongRowsMaterializeExactly) {
    REQUIRE_CUDA_DEVICE();

    for (const uint64_t valueBytes : {1ULL, 4ULL, 16ULL}) {
        RaggedCase testCase;
        testCase.valueBytes = valueBytes;
        testCase.meanValues = 5000;
        testCase.lengths = RaggedCase::Lengths::FewLong;
        testCase.numExamples = 40;
        testCase.logicalRows = 60;
        runRaggedCaseWithBothOffsetWidths(testCase);
    }
}

TEST(DeviceResidentRaggedMaterializationKernelTest, ReferenceIsReadAtEveryByteOfItsRecord) {
    REQUIRE_CUDA_DEVICE();

    // Records are packed, so a reference starts at any byte of the record store: every record size from the
    // smallest to past three words, with the reference at every offset that leaves room for the count after it.
    for (uint64_t recordSizeBytes = 16; recordSizeBytes <= 27; ++recordSizeBytes) {
        for (uint64_t referenceOffsetBytes = 0; referenceOffsetBytes + 16 <= recordSizeBytes; ++referenceOffsetBytes) {
            RaggedCase testCase;
            testCase.recordSizeBytes = recordSizeBytes;
            testCase.referenceOffsetBytes = referenceOffsetBytes;
            testCase.numExamples = 37;
            runRaggedCase<uint32_t>(testCase);
        }
    }
    // 8-byte aligned, 4-byte aligned only, 2-byte aligned only, and deliberately unaligned.
    runReferenceLoadAlignmentCase<uint32_t>(32, 8);
    runReferenceLoadAlignmentCase<uint32_t>(20, 4);
    runReferenceLoadAlignmentCase<uint32_t>(18, 2);
    runReferenceLoadAlignmentCase<uint32_t>(19, 3);
}

TEST(DeviceResidentRaggedMaterializationKernelTest, PublishedOffsetsDecideRowLengthsWhateverTheRecordCountsSay) {
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
