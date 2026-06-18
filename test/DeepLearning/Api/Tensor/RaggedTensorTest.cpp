#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

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
    RaggedTensor original(DataType::FP32, {8}, 4, 13, DataType::UINT64);

    json architecture = original.architectureJson();
    RaggedTensor copy = RaggedTensor::deserialize(architecture);

    ASSERT_TRUE(copy.isInitialized());
    EXPECT_NE(copy.getId(), original.getId());
    EXPECT_EQ(copy.getOriginalId(), original.getId());
    EXPECT_EQ(copy.getBatchSize(), original.getBatchSize());
    EXPECT_EQ(copy.getMaxTotalValues(), original.getMaxTotalValues());
    EXPECT_EQ(copy.getValuesDataType(), original.getValuesDataType());
    EXPECT_EQ(copy.getOffsetsDataType(), original.getOffsetsDataType());
    EXPECT_EQ(copy.getValuesDimensions(), original.getValuesDimensions());
    EXPECT_EQ(copy.getOffsetsDimensions(), original.getOffsetsDimensions());
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
