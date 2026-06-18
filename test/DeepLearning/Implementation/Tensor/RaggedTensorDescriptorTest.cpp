#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace std;

TEST(RaggedTensorDescriptor, ConstructsFromShapeMetadata) {
    RaggedTensorDescriptor descriptor(DataType::FP32, {16, 4}, 3, 11, DataType::UINT32);

    EXPECT_EQ(descriptor.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(descriptor.getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(descriptor.getBatchSize(), 3u);
    EXPECT_EQ(descriptor.getMaxTotalValues(), 11u);
    EXPECT_EQ(descriptor.getRaggedRank(), 1u);
    EXPECT_EQ(descriptor.getValuesDimensions(), (vector<uint64_t>{11, 16, 4}));
    EXPECT_EQ(descriptor.getTrailingDimensions(), (vector<uint64_t>{16, 4}));
    EXPECT_EQ(descriptor.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{4}));
}

TEST(RaggedTensorDescriptor, SupportsScalarValues) {
    RaggedTensorDescriptor descriptor(DataType::INT32, {}, 2, 5, DataType::UINT64);

    EXPECT_EQ(descriptor.getValuesDimensions(), (vector<uint64_t>{5}));
    EXPECT_TRUE(descriptor.getTrailingDimensions().empty());
    EXPECT_EQ(descriptor.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{3}));
    EXPECT_EQ(descriptor.getOffsetsDataType(), DataType::UINT64);
}

TEST(RaggedTensorDescriptor, RejectsInvalidOffsetTypes) {
    EXPECT_THROW((RowPartitionDescriptor(2, 5, DataType::INT32)), logic_error);
    EXPECT_THROW((RowPartitionDescriptor(2, 5, DataType::FP32)), logic_error);
    EXPECT_THROW((RaggedTensorDescriptor(DataType::FP32, {}, 2, 5, DataType::BOOLEAN)), logic_error);
}

TEST(RaggedTensorDescriptor, RejectsInvalidCapacityAndTrailingDimensions) {
    EXPECT_THROW((RowPartitionDescriptor(2, 0, DataType::UINT32)), logic_error);
    EXPECT_THROW((RaggedTensorDescriptor(DataType::FP32, {8, 0}, 2, 5, DataType::UINT32)), logic_error);
}

TEST(RaggedTensorDescriptor, RejectsValuesCapacityMismatch) {
    TensorDescriptor valuesDescriptor(DataType::FP16, {8, 32});
    RowPartitionDescriptor rowPartition(3, 9, DataType::UINT32);

    EXPECT_THROW((RaggedTensorDescriptor(valuesDescriptor, rowPartition)), logic_error);
}

TEST(RaggedTensorImplementation, ConstructsFromPhysicalValuesAndOffsets) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor values(cpuPlacement, TensorDescriptor(DataType::FP32, {12, 8}));
    Tensor offsets(cpuPlacement, TensorDescriptor(DataType::UINT32, {5}));

    RaggedTensor ragged(values, offsets);

    ASSERT_TRUE(ragged.isInitialized());
    EXPECT_EQ(ragged.getValues(), values);
    EXPECT_EQ(ragged.getOffsets(), offsets);
    EXPECT_EQ(ragged.getBatchSize(), 4u);
    EXPECT_EQ(ragged.getMaxTotalValues(), 12u);
    EXPECT_EQ(ragged.getValuesDescriptor().getDimensions(), (vector<uint64_t>{12, 8}));
    EXPECT_EQ(ragged.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{5}));
    EXPECT_EQ(ragged.getPlacement(), cpuPlacement);
}

TEST(RaggedTensorImplementation, RejectsBadPhysicalOffsets) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor values(cpuPlacement, TensorDescriptor(DataType::FP32, {12, 8}));
    Tensor rankTwoOffsets(cpuPlacement, TensorDescriptor(DataType::UINT32, {5, 1}));
    Tensor signedOffsets(cpuPlacement, TensorDescriptor(DataType::INT32, {5}));

    EXPECT_THROW((RaggedTensor(values, rankTwoOffsets)), logic_error);
    EXPECT_THROW((RaggedTensor(values, signedOffsets)), logic_error);
}
