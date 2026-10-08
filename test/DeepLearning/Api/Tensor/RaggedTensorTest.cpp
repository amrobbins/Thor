#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(RaggedTensorApi, ConstructsFromMetadata) {
    RaggedTensor labels(DataType::INT32, {}, 3, 7, DataType::UINT32);

    ASSERT_TRUE(labels.isInitialized());
    EXPECT_EQ(labels.getBatchSize(), 3u);
    EXPECT_EQ(labels.getMaxTotalValues(), 7u);
    EXPECT_EQ(labels.getRaggedRank(), 1u);
    EXPECT_EQ(labels.getValuesDataType(), DataType::INT32);
    EXPECT_EQ(labels.getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(labels.getValuesDimensions(), (vector<uint64_t>{7}));
    EXPECT_EQ(labels.getOffsetsDimensions(), (vector<uint64_t>{4}));
    EXPECT_TRUE(labels.getTrailingDimensions().empty());
}

TEST(RaggedTensorApi, ConstructsFromValuesAndOffsets) {
    Tensor values(DataType::FP16, {17, 64});
    Tensor offsets(DataType::UINT64, {6});

    RaggedTensor ragged(values, offsets);

    EXPECT_EQ(ragged.getValues(), values);
    EXPECT_EQ(ragged.getOffsets(), offsets);
    EXPECT_EQ(ragged.getBatchSize(), 5u);
    EXPECT_EQ(ragged.getMaxTotalValues(), 17u);
    EXPECT_EQ(ragged.getTrailingDimensions(), (vector<uint64_t>{64}));
    EXPECT_EQ(ragged.getDescriptor().getValuesDimensions(), (vector<uint64_t>{17, 64}));
}

TEST(RaggedTensorApi, PlacementNeutralGeometryCarriesOnlySemanticRowBoundUntilPlacement) {
    Tensor values(DataType::FP32, {5, 3});
    Tensor partitionToken(DataType::UINT32, {1});

    RaggedTensor ragged = RaggedTensor::placementNeutral(values, partitionToken, 5);

    ASSERT_TRUE(ragged.isPlacementNeutral());
    EXPECT_EQ(ragged.getValuesDimensions(), (vector<uint64_t>{5, 3}));
    EXPECT_EQ(ragged.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(ragged.getTrailingDimensions(), (vector<uint64_t>{3}));
    EXPECT_EQ(ragged.getMaxValuesPerRow(), 5u);
    EXPECT_THROW(static_cast<void>(ragged.getBatchSize()), logic_error);
    EXPECT_THROW(static_cast<void>(ragged.getMaxTotalValues()), logic_error);
    EXPECT_THROW(static_cast<void>(ragged.getDescriptor()), logic_error);
    EXPECT_THROW(static_cast<void>(ragged.architectureJson()), logic_error);

    const ThorImplementation::RaggedTensorDescriptor batchTwo = ragged.getDescriptorForPlacement(2);
    EXPECT_EQ(batchTwo.getBatchSize(), 2u);
    EXPECT_EQ(batchTwo.getMaxTotalValues(), 10u);
    EXPECT_EQ(batchTwo.getValuesDimensions(), (vector<uint64_t>{10, 3}));
    EXPECT_EQ(batchTwo.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{3}));
    EXPECT_EQ(batchTwo.getMaxValuesPerRow(), 5u);

    const ThorImplementation::RaggedTensorDescriptor batchSeven = ragged.getDescriptorForPlacement(7);
    EXPECT_EQ(batchSeven.getBatchSize(), 7u);
    EXPECT_EQ(batchSeven.getMaxTotalValues(), 35u);
    EXPECT_EQ(batchSeven.getValuesDimensions(), (vector<uint64_t>{35, 3}));
    EXPECT_EQ(batchSeven.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{8}));
}


TEST(RaggedTensorApi, RowPartitionIdentityIsDistinctFromRaggedTensorIdentityAndPreservedByWithValues) {
    Tensor offsets(DataType::UINT32, {4});
    RaggedTensor first(Tensor(DataType::FP32, {7, 3}), offsets);
    RaggedTensor sibling(Tensor(DataType::FP16, {7, 5}), offsets);
    RaggedTensor replaced = first.withValues(Tensor(DataType::BF16, {7, 9}));
    RaggedTensor different(Tensor(DataType::FP32, {7, 3}), Tensor(DataType::UINT32, {4}));

    EXPECT_NE(first.getId(), sibling.getId());
    EXPECT_EQ(first.getRowPartitionId(), sibling.getRowPartitionId());
    EXPECT_TRUE(first.sharesPartitionWith(sibling));
    EXPECT_TRUE(first.sharesPartitionWith(replaced));
    EXPECT_EQ(first.getRowPartitionId(), replaced.getRowPartitionId());

    EXPECT_FALSE(first.sharesPartitionWith(different));
    EXPECT_NE(first.getRowPartitionId(), different.getRowPartitionId());
}

TEST(RaggedTensorApi, RejectsInvalidOffsets) {
    Tensor values(DataType::INT32, {7});
    Tensor rankTwoOffsets(DataType::UINT32, {4, 1});
    Tensor signedOffsets(DataType::INT32, {4});
    Tensor floatOffsets(DataType::FP32, {4});

    EXPECT_THROW((RaggedTensor(values, rankTwoOffsets)), logic_error);
    EXPECT_THROW((RaggedTensor(values, signedOffsets)), logic_error);
    EXPECT_THROW((RaggedTensor(values, floatOffsets)), logic_error);
    EXPECT_THROW((RaggedTensor(DataType::INT32, {}, 3, 7, DataType::INT32)), logic_error);
}

TEST(RaggedTensorApi, ArchitectureJsonRoundTrips) {
    RaggedTensor original(DataType::FP32, {8}, 4, 13, 7, DataType::UINT64);

    json architecture = original.architectureJson();
    RaggedTensor copy = RaggedTensor::deserialize(architecture);

    ASSERT_TRUE(copy.isInitialized());
    EXPECT_NE(copy.getId(), original.getId());
    EXPECT_EQ(copy.getOriginalId(), original.getId());
    EXPECT_NE(copy.getRowPartitionId(), original.getRowPartitionId());
    EXPECT_FALSE(copy.sharesPartitionWith(original));
    EXPECT_EQ(copy.getBatchSize(), original.getBatchSize());
    EXPECT_EQ(copy.getMaxTotalValues(), original.getMaxTotalValues());
    ASSERT_TRUE(copy.hasMaxValuesPerRow());
    EXPECT_EQ(copy.getMaxValuesPerRow(), 7u);
    EXPECT_EQ(copy.getValuesDataType(), original.getValuesDataType());
    EXPECT_EQ(copy.getOffsetsDataType(), original.getOffsetsDataType());
    EXPECT_EQ(copy.getValuesDimensions(), original.getValuesDimensions());
    EXPECT_EQ(copy.getOffsetsDimensions(), original.getOffsetsDimensions());
}

TEST(RaggedTensorApi, PlacementNeutralDescriptorRejectsOverflowAndOffsetDTypeOverflow) {
    RaggedTensor uint64Offsets = RaggedTensor::placementNeutral(
        Tensor(DataType::FP32, {4, 2}), Tensor(DataType::UINT64, {1}), 4);
    EXPECT_THROW(
        static_cast<void>(uint64Offsets.getDescriptorForPlacement(std::numeric_limits<uint64_t>::max())),
        std::logic_error);

    RaggedTensor uint32Offsets = RaggedTensor::placementNeutral(
        Tensor(DataType::FP32, {3}), Tensor(DataType::UINT32, {1}), 3);
    const uint64_t batchPastUint32Capacity = std::numeric_limits<uint32_t>::max() / 3ULL + 1ULL;
    EXPECT_THROW(
        static_cast<void>(uint32Offsets.getDescriptorForPlacement(batchPastUint32Capacity)),
        std::logic_error);
}

TEST(RaggedTensorApi, ArchitectureJsonContainsStructureButNoRuntimeCacheState) {
    RaggedTensor ragged(DataType::FP32, {8}, 4, 13, DataType::UINT64);

    json architecture = ragged.architectureJson();

    EXPECT_EQ(architecture.at("batch_size"), 4u);
    EXPECT_EQ(architecture.at("max_total_values"), 13u);
    EXPECT_FALSE(architecture.contains("max_values_per_row"));
    EXPECT_FALSE(architecture.contains("host_active_value_count"));
    EXPECT_FALSE(architecture.contains("row_partition_runtime"));
    EXPECT_FALSE(architecture.contains("runtime_extent"));
}

TEST(RaggedTensorApi, MaxValuesPerRowIsOptionalStructuralMetadataAndRoundTrips) {
    RaggedTensor ragged(DataType::FP32, {8}, 4, 23, 7, DataType::UINT64);

    ASSERT_TRUE(ragged.hasMaxValuesPerRow());
    EXPECT_EQ(ragged.getMaxValuesPerRow(), 7u);
    EXPECT_EQ(ragged.getMaxTotalValues(), 23u);
    json architecture = ragged.architectureJson();
    EXPECT_EQ(architecture.at("max_values_per_row"), 7u);

    RaggedTensor loaded = RaggedTensor::deserialize(architecture);
    ASSERT_TRUE(loaded.hasMaxValuesPerRow());
    EXPECT_EQ(loaded.getMaxValuesPerRow(), 7u);
}

TEST(RaggedTensorApi, LegacyVersionWithoutMaxValuesPerRowStillLoads) {
    RaggedTensor original(DataType::FP32, {8}, 4, 23, DataType::UINT64);
    json legacy = original.architectureJson();
    legacy["version"] = "1.0.0";
    legacy.erase("max_values_per_row");

    RaggedTensor loaded = RaggedTensor::deserialize(legacy);
    EXPECT_FALSE(loaded.hasMaxValuesPerRow());
    EXPECT_EQ(loaded.getBatchSize(), 4u);
    EXPECT_EQ(loaded.getMaxTotalValues(), 23u);
}

TEST(RaggedTensorApi, DeserializeRejectsMetadataMismatch) {
    RaggedTensor original(DataType::FP32, {8}, 4, 13, DataType::UINT32);
    json architecture = original.architectureJson();
    architecture["max_total_values"] = 12;

    EXPECT_THROW((RaggedTensor::deserialize(architecture)), logic_error);
}

TEST(RaggedTensorApi, DeserializeRejectsFutureRaggedRank) {
    RaggedTensor original(DataType::FP32, {8}, 4, 13, DataType::UINT32);
    json architecture = original.architectureJson();
    architecture["ragged_rank"] = 2;

    EXPECT_THROW((RaggedTensor::deserialize(architecture)), runtime_error);
}
