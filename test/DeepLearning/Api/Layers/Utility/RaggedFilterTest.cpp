#include "DeepLearning/Api/Layers/Utility/RaggedFilter.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

RaggedTensor makeSharedMask(Network& network,
                            const string& name,
                            const RaggedTensor& feature,
                            DataType maskDataType = DataType::BOOLEAN,
                            const vector<uint64_t>& trailingDimensions = {}) {
    return RaggedNetworkInput::Builder()
        .network(network)
        .name(name)
        .valuesDataType(maskDataType)
        .trailingDimensions(trailingDimensions)
        .partition(feature)
        .build();
}

}  // namespace

TEST(UtilityApiLayers, RaggedFilterBuildsNewCanonicalPartitionAndDeduplicatesSharedOffsetsPort) {
    Network network("ragged_filter_build");
    RaggedTensor feature = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("feature")
                               .valuesDataType(DataType::FP32)
                               .offsetsDataType(DataType::UINT64)
                               .trailingDimensions({2, 3})
                               .maxTotalValues(12)
                               .maxValuesPerRow(5)
                               .batchSize(4)
                               .build();
    RaggedTensor mask = makeSharedMask(network, "mask", feature);

    RaggedFilter filter = RaggedFilter::Builder().network(network).featureInput(feature).maskInput(mask).build();

    ASSERT_TRUE(filter.isInitialized());
    EXPECT_EQ(filter.getRaggedFeatureInput(), feature);
    EXPECT_EQ(filter.getRaggedMaskInput(), mask);
    const RaggedTensor output = filter.getRaggedFeatureOutput();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{12, 2, 3}));
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{5}));
    EXPECT_EQ(output.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(output.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(output.getBatchSize(), 4u);
    EXPECT_EQ(output.getMaxTotalValues(), 12u);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 5u);
    EXPECT_NE(output.getOffsets(), feature.getOffsets());

    const vector<Tensor> graphInputs = filter.getFeatureInputs();
    ASSERT_EQ(graphInputs.size(), 3u);
    EXPECT_EQ(graphInputs[0], feature.getValues());
    EXPECT_EQ(graphInputs[1], mask.getValues());
    EXPECT_EQ(graphInputs[2], feature.getOffsets());
    const vector<Tensor> graphOutputs = filter.getFeatureOutputs();
    ASSERT_EQ(graphOutputs.size(), 2u);
    EXPECT_EQ(graphOutputs[0], output.getValues());
    EXPECT_EQ(graphOutputs[1], output.getOffsets());

    const json architecture = filter.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "ragged_filter");
    EXPECT_EQ(architecture.at("ragged_mask").at("values").at("data_type").get<string>(), "boolean");
    EXPECT_EQ(architecture.at("ragged_mask").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_input").at("offsets").at("id").get<uint64_t>());
    EXPECT_NE(architecture.at("ragged_output").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_input").at("offsets").at("id").get<uint64_t>());
}

TEST(UtilityApiLayers, RaggedFilterRejectsNonBooleanNonScalarOrDifferentPartitionMasks) {
    Network network("ragged_filter_validation");
    RaggedTensor feature = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("feature")
                               .valuesDataType(DataType::FP32)
                               .trailingDimensions({2})
                               .maxTotalValues(8)
                               .maxValuesPerRow(4)
                               .batchSize(3)
                               .build();

    RaggedTensor fpMask = makeSharedMask(network, "fp_mask", feature, DataType::FP32);
    EXPECT_THROW((void)RaggedFilter::Builder().network(network).featureInput(feature).maskInput(fpMask).build(),
                 std::invalid_argument);

    RaggedTensor vectorMask = makeSharedMask(network, "vector_mask", feature, DataType::BOOLEAN, {1});
    EXPECT_THROW((void)RaggedFilter::Builder().network(network).featureInput(feature).maskInput(vectorMask).build(),
                 std::invalid_argument);

    RaggedTensor otherMask = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("other_mask")
                                 .valuesDataType(DataType::BOOLEAN)
                                 .trailingDimensions({})
                                 .maxTotalValues(8)
                                 .maxValuesPerRow(4)
                                 .batchSize(3)
                                 .build();
    EXPECT_THROW((void)RaggedFilter::Builder().network(network).featureInput(feature).maskInput(otherMask).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RaggedFilterSubgraphCloneRemapsFeatureMaskAndAllocatesFreshOutputPartition) {
    Network source("ragged_filter_clone_source");
    RaggedTensor sourceFeature = RaggedNetworkInput::Builder()
                                     .network(source)
                                     .name("feature")
                                     .valuesDataType(DataType::FP32)
                                     .offsetsDataType(DataType::UINT32)
                                     .trailingDimensions({2})
                                     .maxTotalValues(9)
                                     .maxValuesPerRow(4)
                                     .batchSize(3)
                                     .build();
    RaggedTensor sourceMask = makeSharedMask(source, "mask", sourceFeature);
    RaggedFilter sourceFilter =
        RaggedFilter::Builder().network(source).featureInput(sourceFeature).maskInput(sourceMask).build();
    (void)RaggedNetworkOutput::Builder()
        .network(source)
        .name("filtered")
        .inputTensor(sourceFilter.getRaggedFeatureOutput())
        .build();

    Network destination("ragged_filter_clone_destination");
    RaggedTensor destinationFeature = RaggedNetworkInput::Builder()
                                          .network(destination)
                                          .name("feature")
                                          .valuesDataType(DataType::FP32)
                                          .offsetsDataType(DataType::UINT32)
                                          .trailingDimensions({2})
                                          .maxTotalValues(9)
                                          .maxValuesPerRow(4)
                                          .batchSize(3)
                                          .build();
    RaggedTensor destinationMask = RaggedNetworkInput::Builder()
                                       .network(destination)
                                       .name("mask")
                                       .valuesDataType(DataType::BOOLEAN)
                                       .trailingDimensions({})
                                       .partition(destinationFeature)
                                       .build();

    ApiTensorRemap remap;
    remap.map(sourceFeature.getValues(), destinationFeature.getValues());
    remap.map(sourceFeature.getOffsets(), destinationFeature.getOffsets());
    remap.map(sourceMask.getValues(), destinationMask.getValues());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        source,
        {"__thor_ragged_output.filtered.values", "__thor_ragged_output.filtered.offsets"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 2u);
    Tensor clonedValues = clone.outputTensorsByName.at("__thor_ragged_output.filtered.values");
    Tensor clonedOffsets = clone.outputTensorsByName.at("__thor_ragged_output.filtered.offsets");
    EXPECT_EQ(clonedValues.getDimensions(), (vector<uint64_t>{9, 2}));
    EXPECT_EQ(clonedOffsets.getDimensions(), (vector<uint64_t>{4}));
    EXPECT_NE(clonedOffsets, destinationFeature.getOffsets());
}
