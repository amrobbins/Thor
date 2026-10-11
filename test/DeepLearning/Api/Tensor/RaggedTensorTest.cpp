#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

// Deleted model-geometry accessors must remain absent; placement-only
// methods with the same names on physical descriptors are still valid.
template <typename T, typename = void>
struct HasModelBatchSizeAccessor : std::false_type {};
template <typename T>
struct HasModelBatchSizeAccessor<T, std::void_t<decltype(std::declval<const T&>().getBatchSize())>>
    : std::true_type {};
template <typename T, typename = void>
struct HasModelTotalCapacityAccessor : std::false_type {};
template <typename T>
struct HasModelTotalCapacityAccessor<T, std::void_t<decltype(std::declval<const T&>().getMaxTotalValues())>>
    : std::true_type {};

TEST(RaggedTensorApi, HistoricalPhysicalMetadataIsNotAConstructionSignature) {
    static_assert(!HasModelBatchSizeAccessor<RaggedTensor>::value);
    static_assert(!HasModelTotalCapacityAccessor<RaggedTensor>::value);
    // A model cannot be constructed from physical B and total capacity, even
    // when the caller supplies M alongside those two obsolete arguments.
    static_assert(!std::is_constructible_v<RaggedTensor, DataType, const vector<uint64_t>&,
                                           uint64_t, uint64_t, DataType>);
    static_assert(!std::is_constructible_v<RaggedTensor, DataType, const vector<uint64_t>&,
                                           uint64_t, uint64_t, uint64_t, DataType>);
}

TEST(RaggedTensorApi, RequiresExplicitMAndCanonicalPartitionToken) {
    Tensor values(DataType::FP16, {17, 64});
    Tensor physicalOffsets(DataType::UINT64, {6});
    Tensor logicalToken(DataType::UINT64, {1});

    static_assert(!std::is_constructible_v<RaggedTensor, Tensor, Tensor>);
    EXPECT_THROW((RaggedTensor(values, physicalOffsets, 17)), logic_error);
    EXPECT_THROW((RaggedTensor(values, logicalToken, 16)), logic_error);

    RaggedTensor logical = RaggedTensor::logical(values, logicalToken, 17);
    EXPECT_EQ(logical.getValues(), values);
    EXPECT_EQ(logical.getRowPartitionToken(), logicalToken);
    EXPECT_EQ(logical.getMaxValuesPerRow(), 17u);
    EXPECT_EQ(logical.getValuesDimensions(), (vector<uint64_t>{17, 64}));
    EXPECT_EQ(logical.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(logical.getDescriptorForPlacement(3).getValuesDimensions(), (vector<uint64_t>{51, 64}));
}

TEST(RaggedTensorApi, PlacementNeutralGeometryCarriesOnlySemanticRowBoundUntilPlacement) {
    Tensor values(DataType::FP32, {5, 3});
    Tensor partitionToken(DataType::UINT32, {1});

    RaggedTensor ragged = RaggedTensor::logical(values, partitionToken, 5);

    ASSERT_EQ(ragged.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(ragged.getValuesDimensions(), (vector<uint64_t>{5, 3}));
    EXPECT_EQ(ragged.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(ragged.getTrailingDimensions(), (vector<uint64_t>{3}));
    EXPECT_EQ(ragged.getMaxValuesPerRow(), 5u);
    // The logical tensor exposes only semantic M; B and packed capacity
    // are properties of the placement descriptors checked below.
    json logical = ragged.architectureJson();
    EXPECT_EQ(logical.at("max_values_per_row"), 5u);
    EXPECT_FALSE(logical.contains("batch_size"));
    EXPECT_FALSE(logical.contains("max_total_values"));

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
    Tensor partitionToken(DataType::UINT32, {1});
    RaggedTensor first = RaggedTensor::logical(Tensor(DataType::FP32, {7, 3}), partitionToken, 7);
    RaggedTensor sibling = RaggedTensor::logical(Tensor(DataType::FP16, {7, 5}), partitionToken, 7);
    RaggedTensor replaced = first.withValues(Tensor(DataType::BF16, {7, 9}));
    RaggedTensor different = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 3}), Tensor(DataType::UINT32, {1}), 7);

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

    EXPECT_THROW((RaggedTensor(values, rankTwoOffsets, 7)), logic_error);
    EXPECT_THROW((RaggedTensor(values, signedOffsets, 7)), logic_error);
    EXPECT_THROW((RaggedTensor(values, floatOffsets, 7)), logic_error);
}

TEST(RaggedTensorApi, ArchitectureJsonRoundTrips) {
    RaggedTensor original = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT64, {1}), 7);

    json architecture = original.architectureJson();
    RaggedTensor copy = RaggedTensor::deserialize(architecture);

    ASSERT_EQ(copy.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_NE(copy.getId(), original.getId());
    EXPECT_EQ(copy.getOriginalId(), original.getId());
    EXPECT_EQ(copy.getValues().getOriginalId(), original.getValues().getId());
    EXPECT_EQ(copy.getOffsets().getOriginalId(), original.getOffsets().getId());
    EXPECT_NE(copy.getRowPartitionId(), original.getRowPartitionId());
    EXPECT_FALSE(copy.sharesPartitionWith(original));
    EXPECT_EQ(copy.getMaxValuesPerRow(), 7u);
    EXPECT_EQ(copy.getValuesDataType(), original.getValuesDataType());
    EXPECT_EQ(copy.getOffsetsDataType(), original.getOffsetsDataType());
    EXPECT_EQ(copy.getValuesDimensions(), (vector<uint64_t>{7, 8}));
    EXPECT_EQ(copy.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(copy.getDescriptorForPlacement(2).getMaxTotalValues(), 14u);
    EXPECT_EQ(copy.getDescriptorForPlacement(7).getMaxTotalValues(), 49u);
}

TEST(RaggedTensorApi, PlacementNeutralDescriptorRejectsOverflowAndOffsetDTypeOverflow) {
    RaggedTensor uint64Offsets = RaggedTensor::logical(
        Tensor(DataType::FP32, {4, 2}), Tensor(DataType::UINT64, {1}), 4);
    EXPECT_THROW(
        static_cast<void>(uint64Offsets.getDescriptorForPlacement(std::numeric_limits<uint64_t>::max())),
        std::logic_error);

    RaggedTensor uint32Offsets = RaggedTensor::logical(
        Tensor(DataType::FP32, {3}), Tensor(DataType::UINT32, {1}), 3);
    const uint64_t batchPastUint32Capacity = std::numeric_limits<uint32_t>::max() / 3ULL + 1ULL;
    EXPECT_THROW(
        static_cast<void>(uint32Offsets.getDescriptorForPlacement(batchPastUint32Capacity)),
        std::logic_error);
}

TEST(RaggedTensorApi, ArchitectureJsonContainsStructureButNoRuntimeCacheState) {
    RaggedTensor ragged = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT64, {1}), 7);

    json architecture = ragged.architectureJson();

    EXPECT_FALSE(architecture.contains("batch_size"));
    EXPECT_FALSE(architecture.contains("max_total_values"));
    EXPECT_EQ(architecture.at("max_values_per_row"), 7u);
    EXPECT_EQ(architecture.at("values").at("dimensions"), (json::array({7, 8})));
    EXPECT_EQ(architecture.at("offsets").at("dimensions"), (json::array({1})));
    EXPECT_FALSE(architecture.contains("host_active_value_count"));
    EXPECT_FALSE(architecture.contains("row_partition_runtime"));
    EXPECT_FALSE(architecture.contains("runtime_extent"));
}

TEST(RaggedTensorApi, MaxValuesPerRowIsRequiredStructuralMetadataAndRoundTrips) {
    RaggedTensor ragged = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT64, {1}), 7);

    json architecture = ragged.architectureJson();
    EXPECT_EQ(architecture.at("max_values_per_row"), 7u);
    RaggedTensor loaded = RaggedTensor::deserialize(architecture);
    ASSERT_EQ(loaded.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(loaded.getMaxValuesPerRow(), 7u);
    EXPECT_EQ(loaded.getDescriptorForPlacement(2).getMaxTotalValues(), 14u);
}

TEST(RaggedTensorApi, LegacyVersionWithoutMaxValuesPerRowFailsExplicitly) {
    RaggedTensor original = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT64, {1}), 7);
    json legacy = original.architectureJson();
    legacy["version"] = "1.0.0";
    legacy["batch_size"] = 4;
    legacy["max_total_values"] = 23;
    legacy["values"]["dimensions"] = json::array({23, 8});
    legacy["offsets"]["dimensions"] = json::array({5});
    legacy.erase("max_values_per_row");

    EXPECT_THROW((RaggedTensor::deserialize(legacy)), runtime_error);
}

TEST(RaggedTensorApi, RecoverableLegacyIgnoresBatchAndCapacityButPreservesSemanticIdentity) {
    RaggedTensor original = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT64, {1}), 7);
    json legacy = original.architectureJson();
    legacy["version"] = "1.0.0";
    legacy["batch_size"] = 900;
    legacy["max_total_values"] = 2;
    legacy["values"]["dimensions"] = json::array({23, 8});
    legacy["offsets"]["dimensions"] = json::array({5});

    RaggedTensor loaded = RaggedTensor::deserialize(legacy);
    EXPECT_EQ(loaded.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(loaded.getMaxValuesPerRow(), 7u);
    EXPECT_EQ(loaded.getValuesDimensions(), (vector<uint64_t>{7, 8}));
    EXPECT_EQ(loaded.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(loaded.getValues().getOriginalId(), original.getValues().getId());
    EXPECT_EQ(loaded.getOffsets().getOriginalId(), original.getOffsets().getId());
    EXPECT_EQ(loaded.getDescriptorForPlacement(2).getMaxTotalValues(), 14u);
    EXPECT_EQ(loaded.getDescriptorForPlacement(7).getMaxTotalValues(), 49u);
    RaggedTensor replaced = loaded.withValues(Tensor(DataType::BF16, {7, 9}));
    EXPECT_EQ(replaced.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_TRUE(replaced.sharesPartitionWith(loaded));
    EXPECT_EQ(replaced.getDescriptorForPlacement(2).getValuesDimensions(), (vector<uint64_t>{14, 9}));
}

TEST(RaggedTensorApi, DeserializeRejectsSemanticMetadataMismatch) {
    RaggedTensor original = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT32, {1}), 7);
    json architecture = original.architectureJson();
    architecture["max_values_per_row"] = 8;
    EXPECT_THROW((RaggedTensor::deserialize(architecture)), logic_error);

    architecture = original.architectureJson();
    architecture["offsets"]["data_type"] = json(DataType::FP32);
    EXPECT_THROW((RaggedTensor::deserialize(architecture)), logic_error);
}

TEST(RaggedTensorApi, DeserializeRejectsFutureRaggedRank) {
    RaggedTensor original = RaggedTensor::logical(
        Tensor(DataType::FP32, {7, 8}), Tensor(DataType::UINT32, {1}), 7);
    json architecture = original.architectureJson();
    architecture["ragged_rank"] = 2;

    EXPECT_THROW((RaggedTensor::deserialize(architecture)), runtime_error);
}
