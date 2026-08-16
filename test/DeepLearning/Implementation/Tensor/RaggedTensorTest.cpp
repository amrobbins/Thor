#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

#include "gtest/gtest.h"

#include <optional>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(RaggedTensorImplementation, OwnsRowPartitionRuntimeAndDelegatesPartitionMetadata) {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 9;
    Tensor values(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 4}));
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    uint32_t *rawOffsets = offsets.getMemPtr<uint32_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 2;
    rawOffsets[2] = 5;
    rawOffsets[3] = 7;

    RaggedTensor ragged(values, offsets);
    RowPartitionRuntime partition = ragged.getRowPartitionRuntime();

    ASSERT_TRUE(partition.isInitialized());
    EXPECT_EQ(ragged.getOffsets(), offsets);
    EXPECT_EQ(ragged.getBatchSize(), batchSize);
    EXPECT_EQ(ragged.getMaxTotalValues(), maxTotalValues);
    EXPECT_EQ(ragged.getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(ragged.getOffsetsDescriptor(), offsets.getDescriptor());
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));

    rawOffsets[3] = 6;
    partition.setHostActiveValueCount(6);
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));
}

TEST(RaggedTensorImplementation, WithValuesPreservesExactRowPartitionRuntime) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 8;
    Tensor values(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 3}));
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    uint64_t *rawOffsets = offsets.getMemPtr<uint64_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 4;
    rawOffsets[2] = 6;

    RaggedTensor original(values, offsets);
    original.getRowPartitionRuntime().setHostActiveValueCount(6);

    Tensor newValues(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP16, {maxTotalValues, 11}));
    RaggedTensor replaced = original.withValues(newValues);

    EXPECT_EQ(replaced.getValues(), newValues);
    EXPECT_EQ(replaced.getOffsets(), offsets);
    EXPECT_EQ(replaced.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(replaced.getDescriptor().getTrailingDimensions(), (std::vector<uint64_t>{11}));
    EXPECT_TRUE(original.getRowPartitionRuntime().describesSamePartition(replaced.getRowPartitionRuntime()));
    EXPECT_TRUE(original.getRowPartitionRuntime().sharesRuntimeStateWith(replaced.getRowPartitionRuntime()));
    EXPECT_EQ(replaced.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));

    rawOffsets[2] = 5;
    replaced.getRowPartitionRuntime().setHostActiveValueCount(5);
    EXPECT_EQ(original.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));

    Tensor wrongCapacityValues(
        TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues + 1, 11}));
    EXPECT_THROW((void)original.withValues(wrongCapacityValues), std::logic_error);
}

TEST(RaggedTensorImplementation, ExistingRuntimeConstructorSharesStateAcrossIndependentWrappers) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 7;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    uint32_t *rawOffsets = offsets.getMemPtr<uint32_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 1;
    rawOffsets[2] = 4;

    RowPartitionDescriptor descriptor(batchSize, maxTotalValues, DataType::UINT32);
    RowPartitionRuntime firstPartition(offsets, descriptor);
    RowPartitionRuntime independentPartition(offsets, descriptor);

    Tensor firstValues(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 2}));
    Tensor secondValues(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 5}));
    RaggedTensor first(firstValues, firstPartition);
    RaggedTensor second(secondValues, independentPartition);

    rawOffsets[2] = 3;
    first.getRowPartitionRuntime().setHostActiveValueCount(3);
    EXPECT_EQ(second.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(3));
    EXPECT_TRUE(first.getRowPartitionRuntime().describesSamePartition(second.getRowPartitionRuntime()));
    EXPECT_TRUE(first.getRowPartitionRuntime().sharesRuntimeStateWith(second.getRowPartitionRuntime()));
}

TEST(RaggedTensorImplementation, ExistingRuntimeConstructorRejectsValuesCapacityMismatch) {
    constexpr uint64_t batchSize = 2;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, 7, DataType::UINT32));
    Tensor wrongCapacityValues(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {8, 3}));

    EXPECT_THROW((void)RaggedTensor(wrongCapacityValues, partition), std::logic_error);
}

TEST(RaggedTensorImplementation, HostActiveValueCountDelegatesEntirelyToRowPartitionRuntime) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 7;
    Tensor values(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 2}));
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    uint32_t *rawOffsets = offsets.getMemPtr<uint32_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 2;
    rawOffsets[2] = 6;

    RaggedTensor ragged(values, offsets);
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));

    rawOffsets[2] = 5;
    ragged.getRowPartitionRuntime().setHostActiveValueCount(5);
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));

    ragged.getRowPartitionRuntime().clearHostActiveValueCount();
    // With no explicit cache, CPU offsets remain the semantic source of truth.
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));
}
