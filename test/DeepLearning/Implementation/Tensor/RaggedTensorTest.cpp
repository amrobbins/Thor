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
    partition.setHostOffsets({0, 2, 5, 7});

    ASSERT_TRUE(partition.isInitialized());
    EXPECT_EQ(ragged.getOffsets(), offsets);
    EXPECT_EQ(ragged.getBatchSize(), batchSize);
    EXPECT_EQ(ragged.getMaxTotalValues(), maxTotalValues);
    EXPECT_EQ(ragged.getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(ragged.getOffsetsDescriptor(), offsets.getDescriptor());
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));

    rawOffsets[3] = 6;
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));
    partition.setHostOffsets({0, 2, 5, 6});
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
    original.getRowPartitionRuntime().setHostOffsets({0, 4, 6});

    Tensor newValues(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP16, {maxTotalValues, 11}));
    RaggedTensor replaced = original.withValues(newValues);

    EXPECT_EQ(replaced.getValues(), newValues);
    EXPECT_EQ(replaced.getOffsets(), offsets);
    EXPECT_EQ(replaced.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(replaced.getDescriptor().getTrailingDimensions(), (std::vector<uint64_t>{11}));
    EXPECT_EQ(original.getRowPartitionId(), replaced.getRowPartitionId());
    EXPECT_TRUE(original.sharesPartitionWith(replaced));
    EXPECT_TRUE(original.getRowPartitionRuntime().describesSamePartition(replaced.getRowPartitionRuntime()));
    EXPECT_TRUE(original.getRowPartitionRuntime().sharesPartitionWith(replaced.getRowPartitionRuntime()));
    EXPECT_TRUE(original.getRowPartitionRuntime().sharesRuntimeStateWith(replaced.getRowPartitionRuntime()));
    EXPECT_EQ(replaced.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));

    rawOffsets[2] = 5;
    replaced.getRowPartitionRuntime().setHostOffsets({0, 4, 5});
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
    first.getRowPartitionRuntime().setHostOffsets({0, 1, 3});
    EXPECT_EQ(second.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(3));
    EXPECT_EQ(first.getRowPartitionId(), second.getRowPartitionId());
    EXPECT_TRUE(first.sharesPartitionWith(second));
    EXPECT_TRUE(first.getRowPartitionRuntime().describesSamePartition(second.getRowPartitionRuntime()));
    EXPECT_TRUE(first.getRowPartitionRuntime().sharesPartitionWith(second.getRowPartitionRuntime()));
    EXPECT_TRUE(first.getRowPartitionRuntime().sharesRuntimeStateWith(second.getRowPartitionRuntime()));
}


TEST(RaggedTensorImplementation, DistinctOffsetsExecutionRepresentationsHaveDistinctLogicalPartitionIdentity) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 7;
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    Tensor firstOffsets(cpu, TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Tensor secondOffsets(cpu, TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Tensor firstValues(cpu, TensorDescriptor(DataType::FP32, {maxTotalValues, 2}));
    Tensor secondValues(cpu, TensorDescriptor(DataType::FP32, {maxTotalValues, 2}));

    RaggedTensor first(firstValues, firstOffsets);
    RaggedTensor second(secondValues, secondOffsets);

    EXPECT_NE(first.getRowPartitionId(), second.getRowPartitionId());
    EXPECT_FALSE(first.sharesPartitionWith(second));
    EXPECT_FALSE(first.getRowPartitionRuntime().sharesPartitionWith(second.getRowPartitionRuntime()));
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
    ragged.getRowPartitionRuntime().setHostOffsets({0, 2, 6});
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));

    rawOffsets[2] = 5;
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));
    ragged.getRowPartitionRuntime().setHostOffsets({0, 2, 5});
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));

}

TEST(RaggedTensorImplementation, ValuesOffsetsCapacityConstructorPublishesBoundAndCpuRuntimeExtent) {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 10;
    constexpr uint64_t maxValuesPerRow = 4;
    Tensor values(
        TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {maxTotalValues, 2}));
    Tensor offsets(
        TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    uint32_t* rawOffsets = offsets.getMemPtr<uint32_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 3;
    rawOffsets[2] = 3;
    rawOffsets[3] = 7;

    RaggedTensor ragged(values, offsets, maxValuesPerRow);
    ragged.getRowPartitionRuntime().setHostOffsets({0, 3, 3, 7});

    ASSERT_TRUE(ragged.hasMaxValuesPerRow());
    EXPECT_EQ(ragged.getMaxValuesPerRow(), maxValuesPerRow);
    EXPECT_EQ(ragged.getDescriptor().getRowPartition().getMaxValuesPerRow(), maxValuesPerRow);
    EXPECT_EQ(ragged.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));
    EXPECT_EQ(ragged.getHostMaxActiveRowLengthIfAvailable(), std::optional<uint64_t>(4));
    EXPECT_EQ(ragged.getRowPartitionRuntime().requireHostMaxActiveRowLength(), 4u);

    EXPECT_THROW((void)RaggedTensor(values, offsets, 0), std::logic_error);
    EXPECT_THROW((void)RaggedTensor(values, offsets, maxTotalValues + 1), std::logic_error);
}
