#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using Api::DataType;

namespace {

constexpr uint32_t kBatchSize = 2;
constexpr uint32_t kFeatures = 16;
constexpr uint64_t kHistoryCapacity = 7;
constexpr uint64_t kFutureLength = 2;
constexpr int64_t kHistoryBoundary = 371;

struct AuditGraph {
    Api::RaggedTensor history;
    Api::NetworkInput future;
    Api::NetworkInput historyOrigins;
    Api::Tensor historyMean;
    Api::RaggedTensor historyStates;
    Api::Tensor futureOutput;
    Api::Tensor mirrorMean;
};

Impl::RotaryPositionEmbeddingOptions auditRopeOptions() {
    Impl::RotaryPositionEmbeddingOptions options;
    options.rotary_dim = kFeatures;
    options.base = 10000.0;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP16;
    return options;
}

AuditGraph buildAuditGraph(Api::Network& network, bool addExternalOutputs) {
    AuditGraph graph{
        .history = Api::RaggedNetworkInput::Builder()
                       .network(network)
                       .name("history")
                       .valuesDataType(DataType::FP16)
                       .offsetsDataType(DataType::UINT32)
                       .trailingDimensions({kFeatures})
                       .maxTotalValues(kHistoryCapacity)
                       .batchSize(kBatchSize)
                       .build(),
        .future = Api::NetworkInput::Builder()
                      .network(network)
                      .name("future")
                      .dimensions({kFutureLength, kFeatures})
                      .dataType(DataType::FP16)
                      .build(),
        .historyOrigins = Api::NetworkInput::Builder()
                              .network(network)
                              .name("history_origins")
                              .dimensions({1})
                              .dataType(DataType::INT32)
                              .build(),
    };

    Api::Slice channel0 = Api::Slice::Builder()
                              .network(network)
                              .featureInput(graph.history)
                              .axis(0)
                              .start(0)
                              .length(1)
                              .build();
    Api::SegmentedReduction historyMean = Api::SegmentedReduction::Builder()
                                                .network(network)
                                                .featureInput(channel0.getRaggedFeatureOutput().value())
                                                .reductionType(Api::SegmentedReduction::Type::MEAN)
                                                .build();
    graph.historyMean = historyMean.getFeatureOutput().value();

    Api::FullyConnected wide = Api::FullyConnected::Builder()
                                   .network(network)
                                   .featureInput(graph.history)
                                   .numOutputFeatures(kFeatures * 2)
                                   .hasBias(true)
                                   .noActivation()
                                   .build();
    std::shared_ptr<Api::Activation> swiglu = Api::Swiglu::Builder()
                                                   .network(network)
                                                   .featureInput(wide.getRaggedFeatureOutput().value())
                                                   .build();
    Api::RaggedTensor encoded = swiglu->getRaggedFeatureOutput().value();

    Api::RMSNorm norm = Api::RMSNorm::Builder()
                            .network(network)
                            .featureInput(encoded)
                            .epsilon(1.0e-5)
                            .build();
    Api::DropOut dropout = Api::DropOut::Builder()
                               .network(network)
                               .featureInput(norm.getRaggedFeatureOutput().value())
                               .dropProportion(0.2f)
                               .build();
    std::shared_ptr<Api::Activation> swish = Api::Swish::Builder()
                                                  .network(network)
                                                  .featureInput(dropout.getRaggedFeatureOutput().value())
                                                  .build();
    encoded = swish->getRaggedFeatureOutput().value();

    const Impl::RotaryPositionEmbeddingOptions rope = auditRopeOptions();
    Api::Attention selfAttention = Api::Attention::Builder()
                                       .network(network)
                                       .queryInput(encoded).keyInput(encoded).valueInput(encoded)
                                       .numHeads(1)
                                       .headDim(kFeatures)
                                       .ropeOptions(rope)
                                       .queryRopePositionOffsetsInput(graph.historyOrigins.getFeatureOutput().value())
                                       .keyRopePositionOffsetsInput(graph.historyOrigins.getFeatureOutput().value())
                                       .build();
    graph.historyStates = selfAttention.getRaggedFeatureOutput().value();

    Api::Attention futureAttention = Api::Attention::Builder()
                                         .network(network)
                                         .queryInput(graph.future.getFeatureOutput().value())
                                         .keyInput(graph.historyStates)
                                         .valueInput(graph.historyStates)
                                         .numHeads(1)
                                         .headDim(kFeatures)
                                         .ropeOptions(rope)
                                         .queryRopePositionOffset(kHistoryBoundary)
                                         .keyRopePositionOffsetsInput(graph.historyOrigins.getFeatureOutput().value())
                                         .build();
    graph.futureOutput = futureAttention.getFeatureOutput().value();

    Api::Attention historyToFuture = Api::Attention::Builder()
                                         .network(network)
                                         .queryInput(graph.historyStates)
                                         .keyInput(graph.future.getFeatureOutput().value())
                                         .valueInput(graph.future.getFeatureOutput().value())
                                         .numHeads(1)
                                         .headDim(kFeatures)
                                         .ropeOptions(rope)
                                         .queryRopePositionOffsetsInput(graph.historyOrigins.getFeatureOutput().value())
                                         .keyRopePositionOffset(kHistoryBoundary)
                                         .build();
    Api::SegmentedReduction mirrorMean = Api::SegmentedReduction::Builder()
                                             .network(network)
                                             .featureInput(historyToFuture.getRaggedFeatureOutput().value())
                                             .reductionType(Api::SegmentedReduction::Type::MEAN)
                                             .build();
    graph.mirrorMean = mirrorMean.getFeatureOutput().value();

    if (addExternalOutputs) {
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("history_mean")
            .inputTensor(graph.historyMean)
            .dataType(DataType::FP16)
            .build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("future_output")
            .inputTensor(graph.futureOutput)
            .dataType(DataType::FP16)
            .build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("mirror_mean")
            .inputTensor(graph.mirrorMean)
            .dataType(DataType::FP16)
            .build();
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("encoded_history")
            .inputTensor(graph.historyStates)
            .build();
    }

    return graph;
}

std::set<std::pair<bool, bool>> attentionModes(const Api::Network& network) {
    std::set<std::pair<bool, bool>> modes;
    const nlohmann::json architecture = network.architectureJson();
    for (const nlohmann::json& layer : architecture.at("layers")) {
        if (layer.at("layer_type").get<std::string>() == "attention") {
            modes.emplace(layer.at("query_ragged").get<bool>(), layer.at("key_value_ragged").get<bool>());
        }
    }
    return modes;
}

void expectCombinedLayerFamilies(const Api::Network& network) {
    const nlohmann::json architecture = network.architectureJson();
    std::multiset<std::string> types;
    for (const nlohmann::json& layer : architecture.at("layers")) {
        types.insert(layer.at("layer_type").get<std::string>());
    }
    EXPECT_TRUE(types.contains("slice"));
    EXPECT_TRUE(types.contains("fully_connected"));
    EXPECT_TRUE(types.contains("swiglu"));
    EXPECT_TRUE(types.contains("rms_norm"));
    EXPECT_TRUE(types.contains("drop_out"));
    EXPECT_TRUE(types.contains("swish"));
    EXPECT_EQ(types.count("segmented_reduction"), 2U);
    EXPECT_EQ(types.count("attention"), 3U);
    EXPECT_EQ(attentionModes(network), (std::set<std::pair<bool, bool>>{{true, true}, {false, true}, {true, false}}));
}

}  // namespace

TEST(RaggedTransformerCompleteness, ArchitectureOnlySaveLoadPreservesCombinedRaggedAndMixedContracts) {
    const std::filesystem::path archiveDir =
        std::filesystem::temp_directory_path() / "thor_ragged_transformer_completeness_architecture";
    std::filesystem::remove_all(archiveDir);

    try {
        Api::Network source("ragged_transformer_completeness_architecture");
        (void)buildAuditGraph(source, true);
        expectCombinedLayerFamilies(source);
        source.save(archiveDir.string(), true);

        Api::Network loaded("ragged_transformer_completeness_architecture");
        loaded.load(archiveDir.string());
        expectCombinedLayerFamilies(loaded);

        const std::vector<Api::RaggedNetworkInputReference> inputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 1U);
        EXPECT_EQ(inputs.front().name, "history");
        const std::vector<Api::RaggedNetworkOutputReference> outputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(outputs.size(), 1U);
        EXPECT_EQ(outputs.front().name, "encoded_history");
        EXPECT_EQ(outputs.front().raggedTensor.getOffsets().getDimensions(),
                  inputs.front().raggedTensor.getOffsets().getDimensions());
        EXPECT_EQ(outputs.front().raggedTensor.getOffsets().getDataType(),
                  inputs.front().raggedTensor.getOffsets().getDataType());

        // RaggedNetworkOutput intentionally inserts a NetworkOutput tensor for the
        // offsets, so the external output tensor has a distinct identity. The
        // serialization contract we care about is that this output remains wired
        // to the authoritative history row partition.
        const nlohmann::json loadedArchitecture = loaded.architectureJson();
        const nlohmann::json* offsetsOutputLayer = nullptr;
        for (const nlohmann::json& layer : loadedArchitecture.at("layers")) {
            if (layer.at("layer_type").get<std::string>() == "network_output" &&
                layer.at("name").get<std::string>() == outputs.front().offsetsOutputName) {
                offsetsOutputLayer = &layer;
                break;
            }
        }
        ASSERT_NE(offsetsOutputLayer, nullptr);
        EXPECT_EQ(offsetsOutputLayer->at("feature_input").at("id").get<uint64_t>(),
                  inputs.front().raggedTensor.getOffsets().getId());
        EXPECT_EQ(offsetsOutputLayer->at("feature_output").at("id").get<uint64_t>(),
                  outputs.front().raggedTensor.getOffsets().getId());
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(RaggedTransformerCompleteness, CloneSubgraphRemapsBothRaggedValuesAndOffsetsAcrossCombinedGraph) {
    Api::Network source("ragged_transformer_completeness_clone_source");
    AuditGraph sourceGraph = buildAuditGraph(source, true);
    expectCombinedLayerFamilies(source);

    Api::Network destination("ragged_transformer_completeness_clone_destination");
    Api::RaggedTensor destinationHistory = Api::RaggedNetworkInput::Builder()
                                               .network(destination)
                                               .name("history")
                                               .valuesDataType(DataType::FP16)
                                               .offsetsDataType(DataType::UINT32)
                                               .trailingDimensions({kFeatures})
                                               .maxTotalValues(kHistoryCapacity)
                                               .batchSize(kBatchSize)
                                               .build();
    Api::NetworkInput destinationFuture = Api::NetworkInput::Builder()
                                              .network(destination)
                                              .name("future")
                                              .dimensions({kFutureLength, kFeatures})
                                              .dataType(DataType::FP16)
                                              .build();
    Api::NetworkInput destinationOrigins = Api::NetworkInput::Builder()
                                               .network(destination)
                                               .name("history_origins")
                                               .dimensions({1})
                                               .dataType(DataType::INT32)
                                               .build();

    Api::ApiTensorRemap remap;
    remap.map(sourceGraph.history.getValues(), destinationHistory.getValues());
    remap.map(sourceGraph.history.getOffsets(), destinationHistory.getOffsets());
    remap.map(sourceGraph.future.getFeatureOutput().value(), destinationFuture.getFeatureOutput().value());
    remap.map(sourceGraph.historyOrigins.getFeatureOutput().value(), destinationOrigins.getFeatureOutput().value());

    Api::ApiSubgraphCloneOptions options;
    options.namePrefix = "clone/";
    options.inferenceOnly = true;
    Api::ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(
        source,
        {"history_mean", "future_output", "mirror_mean"},
        remap,
        options);

    ASSERT_EQ(clone.outputTensorsByName.size(), 3U);
    EXPECT_EQ(clone.outputTensorsByName.at("history_mean").getDimensions(), std::vector<uint64_t>({1}));
    EXPECT_EQ(clone.outputTensorsByName.at("future_output").getDimensions(),
              (std::vector<uint64_t>{kFutureLength, kFeatures}));
    EXPECT_EQ(clone.outputTensorsByName.at("mirror_mean").getDimensions(), std::vector<uint64_t>({kFeatures}));

    (void)Api::NetworkOutput::Builder()
        .network(destination)
        .name("history_mean")
        .inputTensor(clone.outputTensorsByName.at("history_mean"))
        .dataType(DataType::FP16)
        .build();
    (void)Api::NetworkOutput::Builder()
        .network(destination)
        .name("future_output")
        .inputTensor(clone.outputTensorsByName.at("future_output"))
        .dataType(DataType::FP16)
        .build();
    (void)Api::NetworkOutput::Builder()
        .network(destination)
        .name("mirror_mean")
        .inputTensor(clone.outputTensorsByName.at("mirror_mean"))
        .dataType(DataType::FP16)
        .build();

    expectCombinedLayerFamilies(destination);

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = destination.place(kBatchSize, initDoneEvents, true));
    for (Event& event : initDoneEvents) event.synchronize();
    ASSERT_NE(placed, nullptr);
}
