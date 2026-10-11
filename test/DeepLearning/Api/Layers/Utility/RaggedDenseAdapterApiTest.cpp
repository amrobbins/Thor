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
                              
                              .maxValuesPerRow(4)
                              
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
    EXPECT_EQ(restored.getValuesDimensions(), (vector<uint64_t>{4, 2, 3}));
    EXPECT_EQ(restored.getOffsets(), source.getOffsets());
    EXPECT_EQ(restored.getMaxValuesPerRow(), source.getMaxValuesPerRow());
    EXPECT_TRUE(restored.sharesPartitionWith(source));
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
    EXPECT_THROW((void)RaggedNetworkInput::Builder()
                     .network(network)
                     .name("missing_semantic_m")
                     .valuesDataType(DataType::FP32)
                     .trailingDimensions({2})
                     .build(), std::logic_error);

    RaggedTensor bounded = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("bounded")
                               .valuesDataType(DataType::FP32)
                               .trailingDimensions({2})
                               
                               .maxValuesPerRow(4)
                               
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
                              
                              .maxValuesPerRow(4)
                              
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
                                         
                                         .maxValuesPerRow(4)
                                         
                                         .build();

    ApiTensorRemap remap;
    remap.map(source.getValues(), destinationSource.getValues());
    remap.map(source.getRowPartitionToken(), destinationSource.getRowPartitionToken());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        sourceNetwork,
        {"__thor_ragged_output.restored.values"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 1u);
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.restored.values").getDimensions(),
              (vector<uint64_t>{4, 2}));
    const Tensor clonedPartitionToken = clone.clonedTensorBySourceOriginalId.at(
        restored.getRowPartitionToken().getOriginalId());
    EXPECT_EQ(clonedPartitionToken, destinationSource.getRowPartitionToken());
}

// Historical layer grammars must not bind the recovered logical partition to
// the physical batch/capacity recorded in the archive. The current cross-B
// numerical suites exercise the same physical adapters independently.
TEST(UtilityApiLayers, HistoricalDenseRaggedAdaptersNormalizePackedReferencesWithoutChangingVersions) {
    Network network("ragged_dense_adapters_historical_grammar");
    const RaggedTensor source = RaggedNetworkInput::Builder().network(network).name("source")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
    const auto unpack = RaggedToPaddedDense::Builder().network(network).featureInput(source).build();
    const auto pack = PaddedDenseToRagged::Builder().network(network)
        .featureInput(unpack.getPaddedFeatureOutput()).partitionInput(source).build();

    auto legacyReference = [](json& record, uint64_t packed) {
        record.erase("layer_ragged_reference_version");
        record.erase("geometry_mode");
        record.erase("max_values_per_row");
        record["batch_size"] = 13;
        record["max_total_values"] = 999;
        record["values"]["dimensions"][0] = packed;
        record["offsets"]["dimensions"][0] = 51;
    };
    json unpackJson = unpack.architectureJson();
    unpackJson["version"] = "1.0.0";
    legacyReference(unpackJson["ragged_feature_input"], 52);
    EXPECT_NO_THROW(RaggedToPaddedDense::deserialize(unpackJson, &network));
    auto loadedUnpack = std::dynamic_pointer_cast<RaggedToPaddedDense>(
        network.getLayer(network.getNumLayers() - 1));
    ASSERT_NE(loadedUnpack, nullptr);
    EXPECT_EQ(loadedUnpack->getPaddedFeatureOutput().getDimensions(), (vector<uint64_t>{4, 2}));

    json packJson = pack.architectureJson();
    packJson["version"] = "1.0.0";
    legacyReference(packJson["partition_input"], 52);
    legacyReference(packJson["ragged_feature_output"], 78);
    EXPECT_NO_THROW(PaddedDenseToRagged::deserialize(packJson, &network));
    auto loadedPack = std::dynamic_pointer_cast<PaddedDenseToRagged>(
        network.getLayer(network.getNumLayers() - 1));
    ASSERT_NE(loadedPack, nullptr);
    const RaggedTensor restored = loadedPack->getRaggedFeatureOutput();
    EXPECT_EQ(restored.getValuesDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_EQ(restored.getOffsets(), source.getOffsets());
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(restored.getDescriptorForPlacement(b).getValuesDimensions(),
                  (vector<uint64_t>{b * 4U, 2U}));
        EXPECT_EQ(restored.getDescriptorForPlacement(b).getOffsetsDescriptor().getDimensions(),
                  (vector<uint64_t>{b + 1U}));
    }
}
