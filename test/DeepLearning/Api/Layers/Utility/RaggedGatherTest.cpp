#include "DeepLearning/Api/Layers/Utility/RaggedGather.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedGatherUsesIndicesPartitionQAndSourceValueGeometry) {
    Network network("ragged_gather_build");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .offsetsDataType(DataType::UINT32)
                              .trailingDimensions({2, 3})
                              .maxTotalValues(12)
                              .maxValuesPerRow(5)
                              .batchSize(4)
                              .build();
    RaggedTensor indices = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("indices")
                               .valuesDataType(DataType::UINT64)
                               .offsetsDataType(DataType::UINT64)
                               .trailingDimensions({})
                               .maxTotalValues(9)
                               .maxValuesPerRow(4)
                               .batchSize(4)
                               .build();

    RaggedGather gather = RaggedGather::Builder().network(network).sourceInput(source).indicesInput(indices).build();
    const RaggedTensor output = gather.getRaggedFeatureOutput();

    EXPECT_EQ(output.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{9, 2, 3}));
    EXPECT_EQ(output.getOffsets(), indices.getOffsets());
    EXPECT_EQ(output.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(output.getBatchSize(), 4u);
    EXPECT_EQ(output.getMaxTotalValues(), 9u);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 4u);

    const vector<Tensor> graphInputs = gather.getFeatureInputs();
    ASSERT_EQ(graphInputs.size(), 4u);
    EXPECT_EQ(graphInputs[0], source.getValues());
    EXPECT_EQ(graphInputs[1], indices.getValues());
    EXPECT_EQ(graphInputs[2], source.getOffsets());
    EXPECT_EQ(graphInputs[3], indices.getOffsets());
    ASSERT_EQ(gather.getFeatureOutputs().size(), 1u);
    EXPECT_EQ(gather.getFeatureOutputs()[0], output.getValues());

    const json architecture = gather.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "ragged_gather");
    EXPECT_EQ(architecture.at("ragged_output").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_indices").at("offsets").at("id").get<uint64_t>());
    EXPECT_EQ(architecture.at("ragged_output").at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{9, 2, 3}));
}

TEST(UtilityApiLayers, RaggedGatherDeduplicatesOffsetsWhenSourceAndIndicesSharePartition) {
    Network network("ragged_gather_shared_partition");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({2})
                              .maxTotalValues(8)
                              .maxValuesPerRow(4)
                              .batchSize(3)
                              .build();
    RaggedTensor indices = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("indices")
                               .valuesDataType(DataType::UINT32)
                               .trailingDimensions({})
                               .partition(source)
                               .build();

    RaggedGather gather = RaggedGather::Builder().network(network).sourceInput(source).indicesInput(indices).build();
    ASSERT_EQ(gather.getFeatureInputs().size(), 3u);
    EXPECT_EQ(gather.getFeatureInputs()[2], source.getOffsets());
    EXPECT_EQ(gather.getRaggedFeatureOutput().getOffsets(), source.getOffsets());
}

TEST(UtilityApiLayers, RaggedGatherRejectsInvalidIndicesGeometryOrBatch) {
    Network network("ragged_gather_validation");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({2})
                              .maxTotalValues(8)
                              .maxValuesPerRow(4)
                              .batchSize(3)
                              .build();
    RaggedTensor fpIndices = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("fp_indices")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({})
                                 .maxTotalValues(6)
                                 .maxValuesPerRow(3)
                                 .batchSize(3)
                                 .build();
    EXPECT_THROW((void)RaggedGather::Builder().network(network).sourceInput(source).indicesInput(fpIndices).build(),
                 std::invalid_argument);

    RaggedTensor vectorIndices = RaggedNetworkInput::Builder()
                                     .network(network)
                                     .name("vector_indices")
                                     .valuesDataType(DataType::UINT32)
                                     .trailingDimensions({1})
                                     .maxTotalValues(6)
                                     .maxValuesPerRow(3)
                                     .batchSize(3)
                                     .build();
    EXPECT_THROW((void)RaggedGather::Builder().network(network).sourceInput(source).indicesInput(vectorIndices).build(),
                 std::invalid_argument);

    RaggedTensor wrongBatch = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("wrong_batch")
                                  .valuesDataType(DataType::UINT32)
                                  .trailingDimensions({})
                                  .maxTotalValues(6)
                                  .maxValuesPerRow(3)
                                  .batchSize(2)
                                  .build();
    EXPECT_THROW((void)RaggedGather::Builder().network(network).sourceInput(source).indicesInput(wrongBatch).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RaggedGatherSubgraphCloneRemapsBothPartitionsAndPreservesDestinationQ) {
    Network sourceNetwork("ragged_gather_clone_source");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(sourceNetwork)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .offsetsDataType(DataType::UINT32)
                              .trailingDimensions({2})
                              .maxTotalValues(9)
                              .maxValuesPerRow(4)
                              .batchSize(3)
                              .build();
    RaggedTensor indices = RaggedNetworkInput::Builder()
                               .network(sourceNetwork)
                               .name("indices")
                               .valuesDataType(DataType::UINT32)
                               .offsetsDataType(DataType::UINT64)
                               .trailingDimensions({})
                               .maxTotalValues(7)
                               .maxValuesPerRow(3)
                               .batchSize(3)
                               .build();
    RaggedGather gather =
        RaggedGather::Builder().network(sourceNetwork).sourceInput(source).indicesInput(indices).build();
    (void)RaggedNetworkOutput::Builder()
        .network(sourceNetwork)
        .name("gathered")
        .inputTensor(gather.getRaggedFeatureOutput())
        .build();

    Network destination("ragged_gather_clone_destination");
    RaggedTensor destinationSource = RaggedNetworkInput::Builder()
                                         .network(destination)
                                         .name("source")
                                         .valuesDataType(DataType::FP32)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({2})
                                         .maxTotalValues(9)
                                         .maxValuesPerRow(4)
                                         .batchSize(3)
                                         .build();
    RaggedTensor destinationIndices = RaggedNetworkInput::Builder()
                                          .network(destination)
                                          .name("indices")
                                          .valuesDataType(DataType::UINT32)
                                          .offsetsDataType(DataType::UINT64)
                                          .trailingDimensions({})
                                          .maxTotalValues(7)
                                          .maxValuesPerRow(3)
                                          .batchSize(3)
                                          .build();

    ApiTensorRemap remap;
    remap.map(source.getValues(), destinationSource.getValues());
    remap.map(source.getOffsets(), destinationSource.getOffsets());
    remap.map(indices.getValues(), destinationIndices.getValues());
    remap.map(indices.getOffsets(), destinationIndices.getOffsets());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        sourceNetwork,
        {"__thor_ragged_output.gathered.values", "__thor_ragged_output.gathered.offsets"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 2u);
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.gathered.values").getDimensions(),
              (vector<uint64_t>{7, 2}));
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.gathered.offsets"), destinationIndices.getOffsets());
}
