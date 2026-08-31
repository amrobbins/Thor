#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedSequenceConcatenateBuildsNewCanonicalPartitionDescriptor) {
    Network network("ragged_sequence_concatenate_build");
    RaggedTensor left = RaggedNetworkInput::Builder()
                            .network(network)
                            .name("left")
                            .valuesDataType(DataType::FP32)
                            .offsetsDataType(DataType::UINT64)
                            .trailingDimensions({2, 3})
                            .maxTotalValues(5)
                            .maxValuesPerRow(3)
                            .batchSize(3)
                            .build();
    RaggedTensor right = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("right")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({2, 3})
                             .maxTotalValues(7)
                             .maxValuesPerRow(4)
                             .batchSize(3)
                             .build();

    RaggedSequenceConcatenate concatenate = RaggedSequenceConcatenate::Builder()
                                                  .network(network)
                                                  .featureInput(left)
                                                  .featureInput(right)
                                                  .build();

    ASSERT_TRUE(concatenate.isInitialized());
    const RaggedTensor output = concatenate.getRaggedFeatureOutput();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{12, 2, 3}));
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{4}));
    EXPECT_EQ(output.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(output.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(output.getBatchSize(), 3u);
    EXPECT_EQ(output.getMaxTotalValues(), 12u);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 7u);
    EXPECT_NE(output.getOffsets(), left.getOffsets());
    EXPECT_NE(output.getOffsets(), right.getOffsets());

    const vector<Tensor> graphInputs = concatenate.getFeatureInputs();
    ASSERT_EQ(graphInputs.size(), 4u);
    EXPECT_EQ(graphInputs[0], left.getValues());
    EXPECT_EQ(graphInputs[1], right.getValues());
    EXPECT_EQ(graphInputs[2], left.getOffsets());
    EXPECT_EQ(graphInputs[3], right.getOffsets());

    const vector<Tensor> graphOutputs = concatenate.getFeatureOutputs();
    ASSERT_EQ(graphOutputs.size(), 2u);
    EXPECT_EQ(graphOutputs[0], output.getValues());
    EXPECT_EQ(graphOutputs[1], output.getOffsets());

    const json architecture = concatenate.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "ragged_sequence_concatenate");
    ASSERT_EQ(architecture.at("ragged_inputs").size(), 2u);
    const uint64_t outputOffsetsId = architecture.at("ragged_output").at("offsets").at("id").get<uint64_t>();
    EXPECT_NE(outputOffsetsId, architecture.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>());
    EXPECT_NE(outputOffsetsId, architecture.at("ragged_inputs").at(1).at("offsets").at("id").get<uint64_t>());

    shared_ptr<Layer> cloneBase = concatenate.clone();
    auto *clone = dynamic_cast<RaggedSequenceConcatenate *>(cloneBase.get());
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getId(), concatenate.getId());
    EXPECT_EQ(clone->getRaggedFeatureOutput().getOffsets(), output.getOffsets());
}

TEST(UtilityApiLayers, RaggedSequenceConcatenateDeduplicatesSharedStructuralPartitionPorts) {
    Network network("ragged_sequence_concatenate_shared_partition");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .offsetsDataType(DataType::UINT32)
                              .trailingDimensions({4})
                              .maxTotalValues(6)
                              .maxValuesPerRow(3)
                              .batchSize(2)
                              .build();
    Slice left = Slice::Builder().network(network).featureInput(source).axis(0).start(0).length(2).build();
    Slice right = Slice::Builder().network(network).featureInput(source).axis(0).start(2).length(2).build();
    ASSERT_TRUE(left.getRaggedFeatureOutput().has_value());
    ASSERT_TRUE(right.getRaggedFeatureOutput().has_value());
    ASSERT_EQ(left.getRaggedFeatureOutput()->getOffsets(), right.getRaggedFeatureOutput()->getOffsets());

    RaggedSequenceConcatenate concatenate = RaggedSequenceConcatenate::Builder()
                                                  .network(network)
                                                  .featureInput(left.getRaggedFeatureOutput().value())
                                                  .featureInput(right.getRaggedFeatureOutput().value())
                                                  .build();

    // Two distinct value ports plus one shared structural offsets port.
    const vector<Tensor> graphInputs = concatenate.getFeatureInputs();
    ASSERT_EQ(graphInputs.size(), 3u);
    EXPECT_EQ(graphInputs[2], source.getOffsets());

    const RaggedTensor output = concatenate.getRaggedFeatureOutput();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{12, 2}));
    EXPECT_EQ(output.getMaxValuesPerRow(), 6u);
    EXPECT_NE(output.getOffsets(), source.getOffsets());
}


TEST(UtilityApiLayers, RaggedSequenceConcatenateSubgraphCloneRemapsInputsAndAllocatesNewOutputPartition) {
    Network source("ragged_sequence_concatenate_clone_source");
    RaggedTensor sourceLeft = RaggedNetworkInput::Builder()
                                  .network(source)
                                  .name("left")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .maxTotalValues(5)
                                  .maxValuesPerRow(3)
                                  .batchSize(3)
                                  .build();
    RaggedTensor sourceRight = RaggedNetworkInput::Builder()
                                   .network(source)
                                   .name("right")
                                   .valuesDataType(DataType::FP32)
                                   .offsetsDataType(DataType::UINT32)
                                   .trailingDimensions({2})
                                   .maxTotalValues(7)
                                   .maxValuesPerRow(4)
                                   .batchSize(3)
                                   .build();
    RaggedSequenceConcatenate sourceConcatenate = RaggedSequenceConcatenate::Builder()
                                                           .network(source)
                                                           .featureInput(sourceLeft)
                                                           .featureInput(sourceRight)
                                                           .build();
    (void)RaggedNetworkOutput::Builder()
        .network(source)
        .name("joined")
        .inputTensor(sourceConcatenate.getRaggedFeatureOutput())
        .build();

    Network destination("ragged_sequence_concatenate_clone_destination");
    RaggedTensor destinationLeft = RaggedNetworkInput::Builder()
                                       .network(destination)
                                       .name("left")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(DataType::UINT32)
                                       .trailingDimensions({2})
                                       .maxTotalValues(5)
                                       .maxValuesPerRow(3)
                                       .batchSize(3)
                                       .build();
    RaggedTensor destinationRight = RaggedNetworkInput::Builder()
                                        .network(destination)
                                        .name("right")
                                        .valuesDataType(DataType::FP32)
                                        .offsetsDataType(DataType::UINT32)
                                        .trailingDimensions({2})
                                        .maxTotalValues(7)
                                        .maxValuesPerRow(4)
                                        .batchSize(3)
                                        .build();

    ApiTensorRemap remap;
    remap.map(sourceLeft.getValues(), destinationLeft.getValues());
    remap.map(sourceLeft.getOffsets(), destinationLeft.getOffsets());
    remap.map(sourceRight.getValues(), destinationRight.getValues());
    remap.map(sourceRight.getOffsets(), destinationRight.getOffsets());

    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        source,
        {"__thor_ragged_output.joined.values", "__thor_ragged_output.joined.offsets"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 2u);
    Tensor clonedValues = clone.outputTensorsByName.at("__thor_ragged_output.joined.values");
    Tensor clonedOffsets = clone.outputTensorsByName.at("__thor_ragged_output.joined.offsets");
    EXPECT_EQ(clonedValues.getDimensions(), (vector<uint64_t>{12, 2}));
    EXPECT_EQ(clonedOffsets.getDimensions(), (vector<uint64_t>{4}));
    EXPECT_NE(clonedOffsets, destinationLeft.getOffsets());
    EXPECT_NE(clonedOffsets, destinationRight.getOffsets());
}

TEST(UtilityApiLayers, RaggedSequenceConcatenateRejectsIncompatibleDescriptorsAndDuplicateValues) {
    Network network("ragged_sequence_concatenate_validation");
    RaggedTensor reference = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("reference")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxTotalValues(5)
                                 .batchSize(3)
                                 .build();
    RaggedTensor differentTrailing = RaggedNetworkInput::Builder()
                                         .network(network)
                                         .name("different_trailing")
                                         .valuesDataType(DataType::FP32)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({3})
                                         .maxTotalValues(5)
                                         .batchSize(3)
                                         .build();
    RaggedTensor differentOffsetsDtype = RaggedNetworkInput::Builder()
                                             .network(network)
                                             .name("different_offsets_dtype")
                                             .valuesDataType(DataType::FP32)
                                             .offsetsDataType(DataType::UINT64)
                                             .trailingDimensions({2})
                                             .maxTotalValues(5)
                                             .batchSize(3)
                                             .build();
    RaggedTensor differentBatch = RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("different_batch")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({2})
                                      .maxTotalValues(5)
                                      .batchSize(2)
                                      .build();

    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder()
                     .network(network)
                     .featureInput(reference)
                     .featureInput(differentTrailing)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder()
                     .network(network)
                     .featureInput(reference)
                     .featureInput(differentOffsetsDtype)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder()
                     .network(network)
                     .featureInput(reference)
                     .featureInput(differentBatch)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)RaggedSequenceConcatenate::Builder()
                     .network(network)
                     .featureInput(reference)
                     .featureInput(reference)
                     .build(),
                 std::invalid_argument);
}
