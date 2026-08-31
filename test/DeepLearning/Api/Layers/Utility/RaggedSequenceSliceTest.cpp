#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedSequenceSliceBuildsNewCanonicalPartitionDescriptor) {
    Network network("ragged_sequence_slice_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({2, 3})
                             .maxTotalValues(12)
                             .maxValuesPerRow(5)
                             .batchSize(4)
                             .build();

    RaggedSequenceSlice slice = RaggedSequenceSlice::Builder()
                                    .network(network)
                                    .featureInput(input)
                                    .start(1)
                                    .length(2)
                                    .build();

    ASSERT_TRUE(slice.isInitialized());
    EXPECT_EQ(slice.getStart(), 1u);
    EXPECT_EQ(slice.getLength(), 2u);
    const RaggedTensor output = slice.getRaggedFeatureOutput();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{8, 2, 3}));
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{5}));
    EXPECT_EQ(output.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(output.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(output.getBatchSize(), 4u);
    EXPECT_EQ(output.getMaxTotalValues(), 8u);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 2u);
    EXPECT_NE(output.getOffsets(), input.getOffsets());

    const vector<Tensor> graphInputs = slice.getFeatureInputs();
    ASSERT_EQ(graphInputs.size(), 2u);
    EXPECT_EQ(graphInputs[0], input.getValues());
    EXPECT_EQ(graphInputs[1], input.getOffsets());
    const vector<Tensor> graphOutputs = slice.getFeatureOutputs();
    ASSERT_EQ(graphOutputs.size(), 2u);
    EXPECT_EQ(graphOutputs[0], output.getValues());
    EXPECT_EQ(graphOutputs[1], output.getOffsets());

    const json architecture = slice.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "ragged_sequence_slice");
    EXPECT_EQ(architecture.at("start").get<uint64_t>(), 1u);
    EXPECT_EQ(architecture.at("length").get<uint64_t>(), 2u);
    EXPECT_NE(architecture.at("ragged_output").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_input").at("offsets").at("id").get<uint64_t>());

    shared_ptr<Layer> cloneBase = slice.clone();
    auto* clone = dynamic_cast<RaggedSequenceSlice*>(cloneBase.get());
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getId(), slice.getId());
    EXPECT_EQ(clone->getRaggedFeatureOutput().getOffsets(), output.getOffsets());
}

TEST(UtilityApiLayers, RaggedSequenceSliceUsesConservativePositiveBoundsForStaticallyEmptyWindow) {
    Network network("ragged_sequence_slice_empty_bound");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({4})
                             .maxTotalValues(9)
                             .maxValuesPerRow(3)
                             .batchSize(3)
                             .build();

    RaggedSequenceSlice slice = RaggedSequenceSlice::Builder()
                                    .network(network)
                                    .featureInput(input)
                                    .start(3)
                                    .length(7)
                                    .build();
    const RaggedTensor output = slice.getRaggedFeatureOutput();
    EXPECT_EQ(output.getMaxTotalValues(), 1u);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 1u);
}

TEST(UtilityApiLayers, RaggedSequenceSliceSubgraphCloneRemapsInputAndAllocatesNewOutputPartition) {
    Network source("ragged_sequence_slice_clone_source");
    RaggedTensor sourceInput = RaggedNetworkInput::Builder()
                                   .network(source)
                                   .name("input")
                                   .valuesDataType(DataType::FP32)
                                   .offsetsDataType(DataType::UINT32)
                                   .trailingDimensions({2})
                                   .maxTotalValues(9)
                                   .maxValuesPerRow(4)
                                   .batchSize(3)
                                   .build();
    RaggedSequenceSlice sourceSlice = RaggedSequenceSlice::Builder()
                                          .network(source)
                                          .featureInput(sourceInput)
                                          .start(1)
                                          .length(2)
                                          .build();
    (void)RaggedNetworkOutput::Builder()
        .network(source)
        .name("sliced")
        .inputTensor(sourceSlice.getRaggedFeatureOutput())
        .build();

    Network destination("ragged_sequence_slice_clone_destination");
    RaggedTensor destinationInput = RaggedNetworkInput::Builder()
                                        .network(destination)
                                        .name("input")
                                        .valuesDataType(DataType::FP32)
                                        .offsetsDataType(DataType::UINT32)
                                        .trailingDimensions({2})
                                        .maxTotalValues(9)
                                        .maxValuesPerRow(4)
                                        .batchSize(3)
                                        .build();

    ApiTensorRemap remap;
    remap.map(sourceInput.getValues(), destinationInput.getValues());
    remap.map(sourceInput.getOffsets(), destinationInput.getOffsets());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        source,
        {"__thor_ragged_output.sliced.values", "__thor_ragged_output.sliced.offsets"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 2u);
    Tensor clonedValues = clone.outputTensorsByName.at("__thor_ragged_output.sliced.values");
    Tensor clonedOffsets = clone.outputTensorsByName.at("__thor_ragged_output.sliced.offsets");
    EXPECT_EQ(clonedValues.getDimensions(), (vector<uint64_t>{6, 2}));
    EXPECT_EQ(clonedOffsets.getDimensions(), (vector<uint64_t>{4}));
    EXPECT_NE(clonedOffsets, destinationInput.getOffsets());
}

TEST(UtilityApiLayers, RaggedSequenceSliceRejectsMissingOrZeroLengthConfiguration) {
    Network network("ragged_sequence_slice_validation");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({2})
                             .maxTotalValues(5)
                             .batchSize(2)
                             .build();

    EXPECT_THROW((void)RaggedSequenceSlice::Builder()
                     .network(network)
                     .featureInput(input)
                     .start(0)
                     .length(0)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)RaggedSequenceSlice::Builder().network(network).featureInput(input).start(0).build(),
                 std::runtime_error);
}
