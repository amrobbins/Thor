#include "DeepLearning/Api/Layers/Utility/PaddedDenseToRagged.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedDenseAdaptersExposeNormalPaddedShapeAndPreserveCanonicalPartition) {
    Network network("ragged_dense_adapter_api");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .offsetsDataType(DataType::UINT64)
                              .trailingDimensions({2, 3})
                              .maxTotalValues(11)
                              .maxValuesPerRow(4)
                              .batchSize(3)
                              .build();

    RaggedToPaddedDense toDense = RaggedToPaddedDense::Builder()
                                      .network(network)
                                      .featureInput(source)
                                      .paddingValue(-7.5)
                                      .build();
    Tensor padded = toDense.getPaddedFeatureOutput();
    EXPECT_EQ(padded.getDataType(), DataType::FP32);
    EXPECT_EQ(padded.getDimensions(), (vector<uint64_t>{4, 2, 3}));
    EXPECT_DOUBLE_EQ(toDense.getPaddingValue(), -7.5);
    ASSERT_EQ(toDense.getFeatureInputs().size(), 2u);
    EXPECT_EQ(toDense.getFeatureInputs()[0], source.getValues());
    EXPECT_EQ(toDense.getFeatureInputs()[1], source.getOffsets());

    PaddedDenseToRagged toRagged = PaddedDenseToRagged::Builder()
                                       .network(network)
                                       .featureInput(padded)
                                       .partitionInput(source)
                                       .build();
    RaggedTensor restored = toRagged.getRaggedFeatureOutput();
    EXPECT_EQ(restored.getValuesDataType(), DataType::FP32);
    EXPECT_EQ(restored.getValuesDimensions(), (vector<uint64_t>{11, 2, 3}));
    EXPECT_EQ(restored.getOffsets(), source.getOffsets());
    EXPECT_EQ(restored.getBatchSize(), 3u);
    EXPECT_EQ(restored.getMaxTotalValues(), 11u);
    ASSERT_TRUE(restored.hasMaxValuesPerRow());
    EXPECT_EQ(restored.getMaxValuesPerRow(), 4u);
    ASSERT_EQ(toRagged.getFeatureInputs().size(), 2u);
    EXPECT_EQ(toRagged.getFeatureInputs()[0], padded);
    EXPECT_EQ(toRagged.getFeatureInputs()[1], source.getOffsets());

    const json denseArchitecture = toDense.architectureJson();
    EXPECT_EQ(denseArchitecture.at("layer_type").get<string>(), "ragged_to_padded_dense");
    EXPECT_EQ(denseArchitecture.at("feature_output").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{4, 2, 3}));

    const json raggedArchitecture = toRagged.architectureJson();
    EXPECT_EQ(raggedArchitecture.at("layer_type").get<string>(), "padded_dense_to_ragged");
    EXPECT_EQ(raggedArchitecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(),
              raggedArchitecture.at("partition_input").at("offsets").at("id").get<uint64_t>());
}

TEST(UtilityApiLayers, RaggedDenseAdaptersRequireFiniteWidthAndSufficientPadding) {
    Network network("ragged_dense_adapter_validation");
    RaggedTensor unbounded = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("unbounded")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({2})
                                 .maxTotalValues(8)
                                 .batchSize(3)
                                 .build();
    EXPECT_THROW((void)RaggedToPaddedDense::Builder().network(network).featureInput(unbounded).build(),
                 std::invalid_argument);

    RaggedTensor bounded = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("bounded")
                               .valuesDataType(DataType::FP32)
                               .trailingDimensions({2})
                               .maxTotalValues(8)
                               .maxValuesPerRow(4)
                               .batchSize(3)
                               .build();
    Tensor tooNarrow(DataType::FP32, {3, 5});
    EXPECT_THROW((void)PaddedDenseToRagged::Builder()
                     .network(network)
                     .featureInput(tooNarrow)
                     .partitionInput(bounded)
                     .build(),
                 std::invalid_argument);
}


TEST(UtilityApiLayers, RaggedDenseAdaptersSubgraphCloneRemapsPartitionWithoutCreatingAnotherOffsetsTensor) {
    Network sourceNetwork("ragged_dense_adapter_clone_source");
    RaggedTensor source = RaggedNetworkInput::Builder()
                              .network(sourceNetwork)
                              .name("source")
                              .valuesDataType(DataType::FP32)
                              .offsetsDataType(DataType::UINT64)
                              .trailingDimensions({2})
                              .maxTotalValues(9)
                              .maxValuesPerRow(4)
                              .batchSize(3)
                              .build();
    Tensor padded = RaggedToPaddedDense::Builder().network(sourceNetwork).featureInput(source).build().getPaddedFeatureOutput();
    RaggedTensor restored = PaddedDenseToRagged::Builder()
                                .network(sourceNetwork)
                                .featureInput(padded)
                                .partitionInput(source)
                                .build()
                                .getRaggedFeatureOutput();
    (void)RaggedNetworkOutput::Builder().network(sourceNetwork).name("restored").inputTensor(restored).build();

    Network destination("ragged_dense_adapter_clone_destination");
    RaggedTensor destinationSource = RaggedNetworkInput::Builder()
                                         .network(destination)
                                         .name("source")
                                         .valuesDataType(DataType::FP32)
                                         .offsetsDataType(DataType::UINT64)
                                         .trailingDimensions({2})
                                         .maxTotalValues(9)
                                         .maxValuesPerRow(4)
                                         .batchSize(3)
                                         .build();

    ApiTensorRemap remap;
    remap.map(source.getValues(), destinationSource.getValues());
    remap.map(source.getOffsets(), destinationSource.getOffsets());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        sourceNetwork,
        {"__thor_ragged_output.restored.values", "__thor_ragged_output.restored.offsets"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 2u);
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.restored.values").getDimensions(),
              (vector<uint64_t>{9, 2}));
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.restored.offsets"), destinationSource.getOffsets());
}
