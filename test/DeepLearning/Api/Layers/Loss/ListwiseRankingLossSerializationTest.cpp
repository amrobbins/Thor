#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

NetworkInput fp32Input(Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(DataType::FP32).build();
}

void consumeLoss(Network& network, const string& name, const Tensor& lossTensor) {
    NetworkOutput::Builder().network(network).name(name).inputTensor(lossTensor).dataType(lossTensor.getDataType()).build();
}

template <typename LayerT>
uint32_t countLayersOfType(Network& network) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        if (dynamic_pointer_cast<LayerT>(network.getLayer(i)))
            ++count;
    }
    return count;
}

uint32_t countLayerTypesInArchitecture(const json& architecture, const string& layerType) {
    uint32_t count = 0;
    for (const json& layer : architecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == layerType)
            ++count;
    }
    return count;
}

void expectTensorJsonMatches(const json& tensorJson, const Tensor& tensor) {
    EXPECT_EQ(tensorJson.at("id").get<uint64_t>(), tensor.getOriginalId());
    EXPECT_EQ(tensorJson.at("data_type").get<DataType>(), tensor.getDataType());
    EXPECT_EQ(tensorJson.at("dimensions").get<vector<uint64_t>>(), tensor.getDimensions());
}

struct ListwiseRankingLossNetworkParts {
    ListNetLoss listNetLoss;
    ListwiseSoftmaxCrossEntropyLoss listwiseSoftmaxCrossEntropyLoss;
};

ListwiseRankingLossNetworkParts addListwiseRankingLosses(Network& network) {
    NetworkInput predictions = fp32Input(network, "list_net_predictions", {5});
    NetworkInput labels = fp32Input(network, "list_net_labels", {5});
    ListNetLoss listNetLoss = ListNetLoss::Builder()
                                  .network(network)
                                  .predictions(predictions.getFeatureOutput().value())
                                  .labels(labels.getFeatureOutput().value())
                                  .scoreTemperature(0.75f)
                                  .labelTemperature(0.5f)
                                  .reportsElementwiseLoss()
                                  .lossDataType(DataType::FP32)
                                  .build();
    consumeLoss(network, "list_net_loss", listNetLoss.getLoss());

    NetworkInput listwiseSoftmaxPredictions = fp32Input(network, "listwise_softmax_predictions", {5});
    NetworkInput listwiseSoftmaxLabels = fp32Input(network, "listwise_softmax_labels", {5});
    ListwiseSoftmaxCrossEntropyLoss listwiseSoftmaxCrossEntropyLoss = ListwiseSoftmaxCrossEntropyLoss::Builder()
                                                                         .network(network)
                                                                         .predictions(listwiseSoftmaxPredictions.getFeatureOutput().value())
                                                                         .labels(listwiseSoftmaxLabels.getFeatureOutput().value())
                                                                         .temperature(0.625f)
                                                                         .reportsRawLoss()
                                                                         .lossDataType(DataType::FP32)
                                                                         .build();
    consumeLoss(network, "listwise_softmax_cross_entropy_loss", listwiseSoftmaxCrossEntropyLoss.getLoss());

    return ListwiseRankingLossNetworkParts{listNetLoss, listwiseSoftmaxCrossEntropyLoss};
}

}  // namespace

TEST(ListwiseRankingLossSerialization, PublicArchitectureJsonIncludesRoundTripFields) {
    Network network("listwise_ranking_loss_public_json");
    ListwiseRankingLossNetworkParts losses = addListwiseRankingLosses(network);

    const json listNetJson = losses.listNetLoss.architectureJson();
    EXPECT_EQ(listNetJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(listNetJson.at("layer_type").get<string>(), "list_net_loss");
    EXPECT_EQ(listNetJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    EXPECT_EQ(listNetJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(listNetJson.at("score_temperature").get<float>(), 0.75f);
    EXPECT_FLOAT_EQ(listNetJson.at("label_temperature").get<float>(), 0.5f);
    expectTensorJsonMatches(listNetJson.at("predictions_tensor"), losses.listNetLoss.getPredictions());
    expectTensorJsonMatches(listNetJson.at("labels_tensor"), losses.listNetLoss.getLabels());
    expectTensorJsonMatches(listNetJson.at("loss_tensor"), losses.listNetLoss.getLoss());

    const json listwiseSoftmaxJson = losses.listwiseSoftmaxCrossEntropyLoss.architectureJson();
    EXPECT_EQ(listwiseSoftmaxJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(listwiseSoftmaxJson.at("layer_type").get<string>(), "listwise_softmax_cross_entropy_loss");
    EXPECT_EQ(listwiseSoftmaxJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
    EXPECT_EQ(listwiseSoftmaxJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(listwiseSoftmaxJson.at("temperature").get<float>(), 0.625f);
    expectTensorJsonMatches(listwiseSoftmaxJson.at("predictions_tensor"), losses.listwiseSoftmaxCrossEntropyLoss.getPredictions());
    expectTensorJsonMatches(listwiseSoftmaxJson.at("labels_tensor"), losses.listwiseSoftmaxCrossEntropyLoss.getLabels());
    expectTensorJsonMatches(listwiseSoftmaxJson.at("loss_tensor"), losses.listwiseSoftmaxCrossEntropyLoss.getLoss());
}

TEST(ListwiseRankingLossSerialization, PublicLossJsonDeserializerRebuildsSupportLayers) {
    Network originalNetwork("listwise_ranking_loss_public_deserialize_source");
    ListwiseRankingLossNetworkParts losses = addListwiseRankingLosses(originalNetwork);

    const json originalArchitecture = originalNetwork.architectureJson();
    Network restoredNetwork("listwise_ranking_loss_public_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            NetworkInput::deserialize(layer, &restoredNetwork);
    }

    Loss::deserialize(losses.listNetLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.listwiseSoftmaxCrossEntropyLoss.architectureJson(), &restoredNetwork);

    EXPECT_EQ(countLayersOfType<CustomLoss>(restoredNetwork), 2u);
    EXPECT_EQ(countLayersOfType<LossShaper>(restoredNetwork), 1u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "custom_loss"), 2u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "loss_shaper"), 1u);
}
