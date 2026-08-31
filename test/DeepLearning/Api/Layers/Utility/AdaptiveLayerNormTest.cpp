#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, AdaptiveLayerNormConstructsDefaultLastDimAndOutputPreservesShapeDtype) {
    Network network("adaptive_layer_norm_default_shape");
    Tensor input(DataType::FP16, {8, 16});
    Tensor scale(DataType::FP32, {16});
    Tensor bias(DataType::FP32, {16});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getNormalizedShape(), vector<uint64_t>({16}));
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getScaleBiasDataType(), DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}


TEST(UtilityApiLayers, AdaptiveLayerNormInferencePlacementAcceptsConnectionOwnedStreams) {
    constexpr uint32_t batchSize = 2;

    Network network("adaptive_layer_norm_connection_owned_streams");
    NetworkInput data = NetworkInput::Builder()
                            .network(network)
                            .name("x")
                            .dimensions({4, 32})
                            .dataType(DataType::FP16)
                            .build();
    NetworkInput scale = NetworkInput::Builder()
                             .network(network)
                             .name("scale")
                             .dimensions({32})
                             .dataType(DataType::FP32)
                             .build();
    NetworkInput bias = NetworkInput::Builder()
                            .network(network)
                            .name("bias")
                            .dimensions({32})
                            .dataType(DataType::FP32)
                            .build();

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder()
                                  .network(network)
                                  .featureInput(data.getFeatureOutput().value())
                                  .scaleInput(scale.getFeatureOutput().value())
                                  .biasInput(bias.getFeatureOutput().value())
                                  .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(layer.getFeatureOutput().value())
        .dataType(DataType::FP16)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(
                        batchSize,
                        initDoneEvents,
                        /*inferenceOnly=*/true,
                        vector<int32_t>{0},
                        /*forcedNumStampsPerGpu=*/1));
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents)
        event.synchronize();
}

TEST(UtilityApiLayers, AdaptiveLayerNormAcceptsExplicitTrailingNormalizedShape) {
    Network network("adaptive_layer_norm_explicit_shape");
    Tensor input(DataType::BF16, {2, 3, 4});
    Tensor scale(DataType::FP32, {3, 4});
    Tensor bias(DataType::FP32, {3, 4});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .scaleInput(scale)
                                  .biasInput(bias)
                                  .normalizedShape({3, 4})
                                  .epsilon(1.0e-4)
                                  .build();

    EXPECT_EQ(layer.getNormalizedShape(), vector<uint64_t>({3, 4}));
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, AdaptiveLayerNormRejectsBadNormalizedShape) {
    Network network("adaptive_layer_norm_bad_shape");
    Tensor input(DataType::FP16, {2, 3, 4});
    Tensor scale(DataType::FP32, {4});
    Tensor bias(DataType::FP32, {4});

    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({4, 3}).build(),
                 std::invalid_argument);
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({0}).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, AdaptiveLayerNormRejectsFp32NormalizedFeatureCountsUnsupportedByCudnnPrimaryEngines) {
    Network network("adaptive_layer_norm_bad_cudnn_contract");
    Tensor input(DataType::FP32, {3, 16});
    Tensor scale(DataType::FP32, {16});
    Tensor bias(DataType::FP32, {16});

    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, AdaptiveLayerNormRejectsUnsupportedDtypesOrShapes) {
    Network network("adaptive_layer_norm_bad_dtype");
    Tensor intInput(DataType::INT32, {2, 4});
    Tensor scale(DataType::FP32, {4});
    Tensor bias(DataType::FP32, {4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(intInput).scaleInput(scale).biasInput(bias).build(), std::invalid_argument);

    Tensor fpInput(DataType::FP16, {2, 4});
    Tensor halfScale(DataType::FP16, {4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(fpInput).scaleInput(halfScale).biasInput(bias).build(),
                 std::invalid_argument);

    Tensor wrongShapeBias(DataType::FP32, {2, 4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(fpInput).scaleInput(scale).biasInput(wrongShapeBias).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, AdaptiveLayerNormArchitectureJsonContainsSideInputs) {
    Network network("adaptive_layer_norm_architecture");
    Tensor input(DataType::FP32, {8, 32});
    Tensor scale(DataType::FP32, {32});
    Tensor bias(DataType::FP32, {32});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({32}).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "adaptive_layer_norm");
    EXPECT_EQ(arch.at("normalized_shape").get<vector<uint64_t>>(), vector<uint64_t>({32}));
    ASSERT_EQ(arch.at("inputs").size(), 3);
    EXPECT_EQ(arch.at("inputs")[0].at("port").get<string>(), "feature_input");
    EXPECT_EQ(arch.at("inputs")[1].at("port").get<string>(), "scale_input");
    EXPECT_EQ(arch.at("inputs")[2].at("port").get<string>(), "bias_input");
}

TEST(UtilityApiLayers, RaggedAdaptiveLayerNormUsesDensePerRowConditioningAndPreservesPartition) {
    Network network("ragged_adaptive_layer_norm_contract");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP16)
                             .trailingDimensions({32})
                             .batchSize(3)
                             .maxTotalValues(11)
                             .maxValuesPerRow(7)
                             .offsetsDataType(DataType::UINT64)
                             .build();
    Tensor scale(DataType::FP32, {32});
    Tensor bias(DataType::FP32, {32});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .scaleInput(scale)
                                  .biasInput(bias)
                                  .epsilon(3.0e-5)
                                  .build();

    EXPECT_TRUE(layer.getUseRagged());
    ASSERT_TRUE(layer.getRaggedDataInput().has_value());
    ASSERT_TRUE(layer.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = layer.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getValues().getDimensions(), (vector<uint64_t>{11, 32}));
    EXPECT_EQ(output.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getBatchSize(), input.getBatchSize());
    EXPECT_EQ(output.getMaxTotalValues(), input.getMaxTotalValues());
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), input.getMaxValuesPerRow());
    EXPECT_EQ(layer.getNormalizedShape(), (vector<uint64_t>{32}));

    const vector<Tensor> physicalInputs = layer.getFeatureInputs();
    ASSERT_EQ(physicalInputs.size(), 4U);
    EXPECT_EQ(physicalInputs[0], input.getValues());
    EXPECT_EQ(physicalInputs[1], input.getOffsets());
    EXPECT_EQ(physicalInputs[2], scale);
    EXPECT_EQ(physicalInputs[3], bias);

    const json arch = layer.architectureJson();
    EXPECT_TRUE(arch.at("use_ragged").get<bool>());
    EXPECT_EQ(arch.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              arch.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());
}

TEST(UtilityApiLayers, RaggedAdaptiveLayerNormRejectsNonTokenwiseNormalizedGeometry) {
    Network network("ragged_adaptive_layer_norm_geometry");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP16)
                             .trailingDimensions({2, 16})
                             .batchSize(2)
                             .maxTotalValues(7)
                             .build();
    Tensor scale(DataType::FP32, {16});
    Tensor bias(DataType::FP32, {16});

    EXPECT_THROW(AdaptiveLayerNorm::Builder()
                     .network(network)
                     .featureInput(input)
                     .scaleInput(scale)
                     .biasInput(bias)
                     .normalizedShape({16})
                     .build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RaggedAdaptiveLayerNormInferencePlacementUsesSegmentedRowConditioning) {
    constexpr uint32_t batchSize = 3;
    Network network("ragged_adaptive_layer_norm_placement");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP16)
                             .trailingDimensions({32})
                             .batchSize(batchSize)
                             .maxTotalValues(12)
                             .maxValuesPerRow(6)
                             .offsetsDataType(DataType::UINT32)
                             .build();
    NetworkInput scale = NetworkInput::Builder()
                             .network(network)
                             .name("scale")
                             .dimensions({32})
                             .dataType(DataType::FP32)
                             .build();
    NetworkInput bias = NetworkInput::Builder()
                            .network(network)
                            .name("bias")
                            .dimensions({32})
                            .dataType(DataType::FP32)
                            .build();
    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .scaleInput(scale.getFeatureOutput().value())
                                  .biasInput(bias.getFeatureOutput().value())
                                  .build();
    ASSERT_TRUE(layer.getRaggedFeatureOutput().has_value());
    RaggedNetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(layer.getRaggedFeatureOutput().value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(
                        batchSize, initDoneEvents, /*inferenceOnly=*/true, vector<int32_t>{0}, /*forcedNumStampsPerGpu=*/1));
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();
}
