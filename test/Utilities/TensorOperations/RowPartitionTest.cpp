#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <limits>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for row partition tests.";                                         \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <>
DataType dtypeFor<int32_t>() {
    return DataType::INT32;
}

template <typename T>
Tensor makeGpuVector(const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(values.size())}));
    auto* cpu_ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(values.size())}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<T> copyGpuVector(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<T> values(cpu.getTotalNumElements());
    const auto* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

template <typename T>
void expectOffsetsToLengthsDType() {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<T>({T{0}, T{3}, T{3}, T{7}, T{8}}, stream);
    Tensor lengths(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {4}));

    rowPartitionOffsetsToLengths(offsets, lengths, 4, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<T>(lengths, stream), (std::vector<T>{T{3}, T{0}, T{4}, T{1}}));
}

template <typename OffsetT>
void expectOffsetsToInt32LengthsCheckedDType() {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<OffsetT>({OffsetT{0}, OffsetT{3}, OffsetT{3}, OffsetT{7}, OffsetT{8}}, stream);
    Tensor lengths = makeGpuVector<int32_t>({-7, -7, -7, -7}, stream);
    Tensor errorBits = makeGpuVector<uint32_t>({99U}, stream);

    rowPartitionOffsetsToInt32LengthsChecked(offsets, lengths, errorBits, 4, 8, 4, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<int32_t>(lengths, stream), (std::vector<int32_t>{3, 0, 4, 1}));
    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream), (std::vector<uint32_t>{0U}));
}

template <typename T>
void expectLengthsToOffsetsDType() {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lengths = makeGpuVector<T>({T{3}, T{0}, T{4}, T{1}}, stream);
    Tensor offsets(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {5}));

    const RowPartitionLengthsToOffsetsPlan plan = prepareRowPartitionLengthsToOffsets(lengths, offsets, 4);
    Tensor temp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(plan.temp_storage_bytes, 1)}));
    rowPartitionLengthsToOffsets(plan, temp, lengths, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<T>(offsets, stream), (std::vector<T>{T{0}, T{3}, T{3}, T{7}, T{8}}));
}

}  // namespace

TEST(RowPartition, ReportsSupportedOffsetDTypes) {
    EXPECT_TRUE(isRowPartitionOffsetDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isRowPartitionOffsetDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isRowPartitionOffsetDTypeSupported(DataType::INT32));
    EXPECT_FALSE(isRowPartitionOffsetDTypeSupported(DataType::FP32));
}


TEST(RowPartition, ActiveValueCountAliasesLastOffsetWithoutCopy) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U, 5U}, stream);
    Tensor activeCount = rowPartitionActiveValueCount(offsets, 3);

    EXPECT_EQ(activeCount.getDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_EQ(activeCount.getDataType(), DataType::UINT32);
    EXPECT_EQ(activeCount.getPlacement(), offsets.getPlacement());
    EXPECT_EQ(activeCount.getMemPtr<uint32_t>(), offsets.getMemPtr<uint32_t>() + 3);

    uint32_t hostValue = 0U;
    ASSERT_EQ(cudaMemcpyAsync(&hostValue, activeCount.getMemPtr<uint32_t>(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.getStream()),
              cudaSuccess);
    stream.synchronize();
    EXPECT_EQ(hostValue, 5U);
}


TEST(RowPartition, CanonicalAndBackendOffsetDTypePoliciesAreExplicitlyDistinct) {
    EXPECT_EQ(kDefaultRowPartitionOffsetDataType, DataType::UINT32);
    EXPECT_TRUE(isCanonicalRowPartitionOffsetDataType(DataType::UINT32));
    EXPECT_TRUE(isCanonicalRowPartitionOffsetDataType(DataType::UINT64));
    EXPECT_FALSE(isCanonicalRowPartitionOffsetDataType(DataType::INT32));
    EXPECT_TRUE(canonicalRowPartitionOffsetCanRepresent(DataType::UINT32, std::numeric_limits<uint32_t>::max()));
    EXPECT_FALSE(canonicalRowPartitionOffsetCanRepresent(
        DataType::UINT32, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL));
    EXPECT_TRUE(isCudnnRaggedOffsetDataType(DataType::INT32));
    EXPECT_FALSE(isCudnnRaggedOffsetDataType(DataType::UINT32));
    EXPECT_TRUE(isCudnnCtcLengthDataType(DataType::INT32));
}

TEST(RowPartition, RuntimeExtentBuildsFromOffsetsWithStaticCapacityAndTrailingElements) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 1ULL, 4ULL}, stream);
    RaggedRuntimeExtent extent = raggedRuntimeExtentFromOffsets(offsets, 2, 9, 7);

    EXPECT_TRUE(extent.isInitialized());
    EXPECT_EQ(extent.activeValueCount.getMemPtr<uint64_t>(), offsets.getMemPtr<uint64_t>() + 2);
    EXPECT_EQ(extent.maxActiveValues, 9ULL);
    EXPECT_EQ(extent.elementsPerValue, 7ULL);
    EXPECT_EQ(extent.maxLaunchElements(), 63ULL);
    EXPECT_EQ(extent.maxGridDimX(16), 4U);
}

TEST(RowPartition, RaggedTensorRuntimeExtentDerivesTrailingElementsPerValue) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {9, 3, 4}));
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U}, stream);
    RaggedTensor ragged(values, offsets);

    const RaggedRuntimeExtent extent = ragged.getRuntimeExtent();
    EXPECT_EQ(extent.maxActiveValues, 9ULL);
    EXPECT_EQ(extent.elementsPerValue, 12ULL);
    EXPECT_EQ(extent.maxLaunchElements(), 108ULL);
    EXPECT_EQ(extent.activeValueCount.getMemPtr<uint32_t>(), offsets.getMemPtr<uint32_t>() + 2);
}

TEST(RowPartition, RuntimeExtentRejectsInvalidStaticCapacityAndElementsPerValue) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({0U, 1U}, stream);
    EXPECT_THROW(static_cast<void>(raggedRuntimeExtentFromOffsets(offsets, 1, 0, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedRuntimeExtentFromOffsets(offsets, 1, 1, 0)), std::invalid_argument);
}

TEST(RowPartition, OffsetsToLengthsUint32) { expectOffsetsToLengthsDType<uint32_t>(); }

TEST(RowPartition, OffsetsToLengthsUint64) { expectOffsetsToLengthsDType<uint64_t>(); }

TEST(RowPartition, OffsetsToInt32LengthsCheckedUint32) { expectOffsetsToInt32LengthsCheckedDType<uint32_t>(); }

TEST(RowPartition, OffsetsToInt32LengthsCheckedUint64) { expectOffsetsToInt32LengthsCheckedDType<uint64_t>(); }

TEST(RowPartition, OffsetsToInt32LengthsCheckedReportsMalformedPartitionAndZerosAllLengths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({1U, 3U, 2U, 9U}, stream);
    Tensor lengths = makeGpuVector<int32_t>({91, 92, 93}, stream);
    Tensor errorBits = makeGpuVector<uint32_t>({0xffffffffU}, stream);

    rowPartitionOffsetsToInt32LengthsChecked(offsets, lengths, errorBits, 3, 8, 8, stream);
    stream.synchronize();

    const uint32_t expected = static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO) |
                              static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC) |
                              static_cast<uint32_t>(ROW_PARTITION_OFFSETS_EXCEED_CAPACITY);
    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream), (std::vector<uint32_t>{expected}));
    EXPECT_EQ(copyGpuVector<int32_t>(lengths, stream), (std::vector<int32_t>{0, 0, 0}));
}

TEST(RowPartition, OffsetsToInt32LengthsCheckedReportsBackendMaximumAndZerosAllLengths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 3ULL, 5ULL}, stream);
    Tensor lengths = makeGpuVector<int32_t>({81, 82}, stream);
    Tensor errorBits = makeGpuVector<uint32_t>({0U}, stream);

    rowPartitionOffsetsToInt32LengthsChecked(offsets, lengths, errorBits, 2, 5, 2, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream),
              (std::vector<uint32_t>{static_cast<uint32_t>(ROW_PARTITION_ROW_LENGTH_EXCEEDS_MAX)}));
    EXPECT_EQ(copyGpuVector<int32_t>(lengths, stream), (std::vector<int32_t>{0, 0}));
}

TEST(RowPartition, OffsetsToInt32LengthsCheckedReportsInt32OverflowWithoutLargeAllocation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const uint64_t tooLargeForInt32 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1ULL;
    Tensor offsets = makeGpuVector<uint64_t>({0ULL, tooLargeForInt32}, stream);
    Tensor lengths = makeGpuVector<int32_t>({73}, stream);
    Tensor errorBits = makeGpuVector<uint32_t>({0U}, stream);

    rowPartitionOffsetsToInt32LengthsChecked(
        offsets, lengths, errorBits, 1, tooLargeForInt32, std::numeric_limits<uint64_t>::max(), stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream),
              (std::vector<uint32_t>{static_cast<uint32_t>(ROW_PARTITION_ROW_LENGTH_EXCEEDS_INT32)}));
    EXPECT_EQ(copyGpuVector<int32_t>(lengths, stream), (std::vector<int32_t>{0}));
}

TEST(RowPartition, LengthsToOffsetsUint32) { expectLengthsToOffsetsDType<uint32_t>(); }

TEST(RowPartition, LengthsToOffsetsUint64) { expectLengthsToOffsetsDType<uint64_t>(); }

TEST(RowPartition, OffsetsToRowIdsLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 2U, 5U}, stream);
    Tensor rowIds = makeGpuVector<uint32_t>({999U, 999U, 999U, 999U, 999U, 999U, 999U}, stream);

    rowPartitionOffsetsToRowIds(offsets, rowIds, 3, 7, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(rowIds, stream), (std::vector<uint32_t>{0U, 0U, 2U, 2U, 2U, 999U, 999U}));
}

TEST(RowPartition, ValidateOffsetsDebugReportsBitMaskWithoutHostSyncInsideCall) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({1U, 3U, 2U, 9U}, stream);
    Tensor errorBits(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    rowPartitionValidateOffsetsDebug(offsets, errorBits, 3, 8, stream);
    stream.synchronize();

    const uint32_t expected = static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO) |
                              static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC) |
                              static_cast<uint32_t>(ROW_PARTITION_OFFSETS_EXCEED_CAPACITY);
    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream), (std::vector<uint32_t>{expected}));
}

TEST(RowPartition, ValidateOffsetsDebugReturnsZeroForValidOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 0ULL, 3ULL, 3ULL, 8ULL}, stream);
    Tensor errorBits(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    rowPartitionValidateOffsetsDebug(offsets, errorBits, 4, 8, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(errorBits, stream), (std::vector<uint32_t>{0U}));
}

TEST(RowPartition, RejectsSignedLengthAndOffsetDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lengths = makeGpuVector<int32_t>({1, 2, 3}, stream);
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::INT32, {4}));

    EXPECT_THROW(static_cast<void>(prepareRowPartitionLengthsToOffsets(lengths, offsets, 3)), std::invalid_argument);
}

TEST(RowPartition, OffsetsToRopePositionIdsUsesPerRowOriginsAndClearsUnusedCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 2U, 5U}, stream);
    Tensor origins = makeGpuVector<int32_t>({10, 20, 30}, stream);
    Tensor positions(gpuPlacement, TensorDescriptor(DataType::FP32, {7}));
    positions.fill(999.0, stream);

    rowPartitionOffsetsToRopePositionIds(offsets, &origins, 123, positions, 3, 7, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<float>(positions, stream), (std::vector<float>{10.0f, 11.0f, 30.0f, 31.0f, 32.0f, 0.0f, 0.0f}));
}

TEST(RowPartition, OffsetsToRopePositionIdsResetsScalarOriginAtEveryRaggedRow) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 2ULL, 4ULL}, stream);
    Tensor positions(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));

    rowPartitionOffsetsToRopePositionIds(offsets, nullptr, 7, positions, 2, 4, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<float>(positions, stream), (std::vector<float>{7.0f, 8.0f, 7.0f, 8.0f}));
}
