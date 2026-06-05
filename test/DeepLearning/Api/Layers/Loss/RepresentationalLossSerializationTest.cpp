#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"
#include "DeepLearning/Api/Layers/Loss/CosineEmbeddingLoss.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/InfoNCELoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MarginRankingLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/TripletLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

filesystem::path makeUniqueTestArchiveDir(const string& testName) {
    const auto now = chrono::high_resolution_clock::now().time_since_epoch().count();
    filesystem::path dir = filesystem::temp_directory_path() / (testName + "_" + to_string(now));
    filesystem::remove_all(dir);
    filesystem::create_directories(dir);
    return dir;
}

NetworkInput fp32Input(Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(DataType::FP32).build();
}

NetworkInput int32Input(Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(DataType::INT32).build();
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

struct RepresentationalLossNetworkParts {
    ContrastiveLoss contrastiveLoss;
    InfoNCELoss infoNCELoss;
    TripletLoss tripletLoss;
    CosineEmbeddingLoss cosineEmbeddingLoss;
    MarginRankingLoss marginRankingLoss;
};

RepresentationalLossNetworkParts addRepresentationalLosses(Network& network) {
    NetworkInput contrastivePredictions = fp32Input(network, "contrastive_predictions", {4});
    NetworkInput contrastiveLabels = fp32Input(network, "contrastive_labels", {4});
    ContrastiveLoss contrastiveLoss = ContrastiveLoss::Builder()
                                          .network(network)
                                          .predictions(contrastivePredictions.getFeatureOutput().value())
                                          .labels(contrastiveLabels.getFeatureOutput().value())
                                          .margin(1.75f)
                                          .reportsElementwiseLoss()
                                          .lossDataType(DataType::FP32)
                                          .build();
    consumeLoss(network, "contrastive_loss", contrastiveLoss.getLoss());

    NetworkInput infoNCELogits = fp32Input(network, "info_nce_logits", {3});
    NetworkInput infoNCELabels = fp32Input(network, "info_nce_labels", {3});
    InfoNCELoss infoNCELoss = InfoNCELoss::Builder()
                                  .network(network)
                                  .predictions(infoNCELogits.getFeatureOutput().value())
                                  .labels(infoNCELabels.getFeatureOutput().value())
                                  .temperature(0.375f)
                                  .reportsPerOutputLoss()
                                  .lossDataType(DataType::FP32)
                                  .build();
    consumeLoss(network, "info_nce_loss", infoNCELoss.getLoss());

    NetworkInput anchor = fp32Input(network, "triplet_anchor", {3});
    NetworkInput positive = fp32Input(network, "triplet_positive", {3});
    NetworkInput negative = fp32Input(network, "triplet_negative", {3});
    TripletLoss tripletLoss = TripletLoss::Builder()
                                  .network(network)
                                  .anchor(anchor.getFeatureOutput().value())
                                  .positive(positive.getFeatureOutput().value())
                                  .negative(negative.getFeatureOutput().value())
                                  .margin(0.625f)
                                  .eps(1.0e-5f)
                                  .reportsBatchLoss()
                                  .lossDataType(DataType::FP32)
                                  .build();
    consumeLoss(network, "triplet_loss", tripletLoss.getLoss());

    NetworkInput cosineInput1 = fp32Input(network, "cosine_input1", {3});
    NetworkInput cosineInput2 = fp32Input(network, "cosine_input2", {3});
    NetworkInput cosineTarget = int32Input(network, "cosine_target", {1});
    CosineEmbeddingLoss cosineEmbeddingLoss = CosineEmbeddingLoss::Builder()
                                                  .network(network)
                                                  .input1(cosineInput1.getFeatureOutput().value())
                                                  .input2(cosineInput2.getFeatureOutput().value())
                                                  .target(cosineTarget.getFeatureOutput().value())
                                                  .margin(0.25f)
                                                  .eps(1.0e-6f)
                                                  .reportsRawLoss()
                                                  .lossDataType(DataType::FP32)
                                                  .build();
    consumeLoss(network, "cosine_embedding_loss", cosineEmbeddingLoss.getLoss());

    NetworkInput rankingInput1 = fp32Input(network, "ranking_input1", {4});
    NetworkInput rankingInput2 = fp32Input(network, "ranking_input2", {4});
    NetworkInput rankingTarget = int32Input(network, "ranking_target", {4});
    MarginRankingLoss marginRankingLoss = MarginRankingLoss::Builder()
                                              .network(network)
                                              .input1(rankingInput1.getFeatureOutput().value())
                                              .input2(rankingInput2.getFeatureOutput().value())
                                              .target(rankingTarget.getFeatureOutput().value())
                                              .margin(0.5f)
                                              .reportsElementwiseLoss()
                                              .lossDataType(DataType::FP32)
                                              .build();
    consumeLoss(network, "margin_ranking_loss", marginRankingLoss.getLoss());

    return RepresentationalLossNetworkParts{contrastiveLoss, infoNCELoss, tripletLoss, cosineEmbeddingLoss, marginRankingLoss};
}

void expectTensorJsonMatches(const json& tensorJson, const Tensor& tensor) {
    EXPECT_EQ(tensorJson.at("id").get<uint64_t>(), tensor.getOriginalId());
    EXPECT_EQ(tensorJson.at("data_type").get<DataType>(), tensor.getDataType());
    EXPECT_EQ(tensorJson.at("dimensions").get<vector<uint64_t>>(), tensor.getDimensions());
}

}  // namespace

TEST(RepresentationalLossSerialization, PublicArchitectureJsonIncludesRoundTripFields) {
    Network network("representational_loss_public_json");
    RepresentationalLossNetworkParts losses = addRepresentationalLosses(network);

    const json contrastiveJson = losses.contrastiveLoss.architectureJson();
    EXPECT_EQ(contrastiveJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(contrastiveJson.at("layer_type").get<string>(), "contrastive_loss");
    EXPECT_EQ(contrastiveJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    EXPECT_EQ(contrastiveJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(contrastiveJson.at("margin").get<float>(), 1.75f);
    expectTensorJsonMatches(contrastiveJson.at("predictions_tensor"), losses.contrastiveLoss.getPredictions());
    expectTensorJsonMatches(contrastiveJson.at("labels_tensor"), losses.contrastiveLoss.getLabels());
    expectTensorJsonMatches(contrastiveJson.at("loss_tensor"), losses.contrastiveLoss.getLoss());

    const json infoNCEJson = losses.infoNCELoss.architectureJson();
    EXPECT_EQ(infoNCEJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(infoNCEJson.at("layer_type").get<string>(), "info_nce_loss");
    EXPECT_EQ(infoNCEJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::CLASSWISE);
    EXPECT_EQ(infoNCEJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(infoNCEJson.at("temperature").get<float>(), 0.375f);
    expectTensorJsonMatches(infoNCEJson.at("predictions_tensor"), losses.infoNCELoss.getPredictions());
    expectTensorJsonMatches(infoNCEJson.at("labels_tensor"), losses.infoNCELoss.getLabels());
    expectTensorJsonMatches(infoNCEJson.at("loss_tensor"), losses.infoNCELoss.getLoss());

    const json tripletJson = losses.tripletLoss.architectureJson();
    EXPECT_EQ(tripletJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(tripletJson.at("layer_type").get<string>(), "triplet_loss");
    EXPECT_EQ(tripletJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::BATCH);
    EXPECT_EQ(tripletJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(tripletJson.at("margin").get<float>(), 0.625f);
    EXPECT_FLOAT_EQ(tripletJson.at("eps").get<float>(), 1.0e-5f);
    expectTensorJsonMatches(tripletJson.at("anchor_tensor"), losses.tripletLoss.getAnchor());
    expectTensorJsonMatches(tripletJson.at("positive_tensor"), losses.tripletLoss.getPositive());
    expectTensorJsonMatches(tripletJson.at("negative_tensor"), losses.tripletLoss.getNegative());
    expectTensorJsonMatches(tripletJson.at("loss_tensor"), losses.tripletLoss.getLoss());

    const json cosineJson = losses.cosineEmbeddingLoss.architectureJson();
    EXPECT_EQ(cosineJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(cosineJson.at("layer_type").get<string>(), "cosine_embedding_loss");
    EXPECT_EQ(cosineJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
    EXPECT_EQ(cosineJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(cosineJson.at("margin").get<float>(), 0.25f);
    EXPECT_FLOAT_EQ(cosineJson.at("eps").get<float>(), 1.0e-6f);
    expectTensorJsonMatches(cosineJson.at("input1_tensor"), losses.cosineEmbeddingLoss.getInput1());
    expectTensorJsonMatches(cosineJson.at("input2_tensor"), losses.cosineEmbeddingLoss.getInput2());
    expectTensorJsonMatches(cosineJson.at("target_tensor"), losses.cosineEmbeddingLoss.getTarget());
    expectTensorJsonMatches(cosineJson.at("loss_tensor"), losses.cosineEmbeddingLoss.getLoss());

    const json rankingJson = losses.marginRankingLoss.architectureJson();
    EXPECT_EQ(rankingJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(rankingJson.at("layer_type").get<string>(), "margin_ranking_loss");
    EXPECT_EQ(rankingJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    EXPECT_EQ(rankingJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(rankingJson.at("margin").get<float>(), 0.5f);
    expectTensorJsonMatches(rankingJson.at("input1_tensor"), losses.marginRankingLoss.getInput1());
    expectTensorJsonMatches(rankingJson.at("input2_tensor"), losses.marginRankingLoss.getInput2());
    expectTensorJsonMatches(rankingJson.at("target_tensor"), losses.marginRankingLoss.getTarget());
    expectTensorJsonMatches(rankingJson.at("loss_tensor"), losses.marginRankingLoss.getLoss());
}


TEST(RepresentationalLossSerialization, PublicLossJsonDeserializersRebuildSupportLayers) {
    Network originalNetwork("representational_loss_public_deserialize_source");
    RepresentationalLossNetworkParts losses = addRepresentationalLosses(originalNetwork);

    const json originalArchitecture = originalNetwork.architectureJson();
    Network restoredNetwork("representational_loss_public_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            NetworkInput::deserialize(layer, &restoredNetwork);
    }

    Loss::deserialize(losses.contrastiveLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.infoNCELoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.tripletLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.cosineEmbeddingLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.marginRankingLoss.architectureJson(), &restoredNetwork);

    EXPECT_EQ(countLayersOfType<CustomLoss>(restoredNetwork), 2u);
    EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(restoredNetwork), 3u);
    EXPECT_EQ(countLayersOfType<LossShaper>(restoredNetwork), 4u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "custom_loss"), 2u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "multi_input_custom_loss"), 3u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "loss_shaper"), 4u);
}

TEST(RepresentationalLossSerialization, NetworkSaveLoadRoundTripRestoresCustomLossSupportLayers) {
    const string networkName = "representational_loss_save_load";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addRepresentationalLosses(network);

        const json beforeArchitecture = network.architectureJson();
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "custom_loss"), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "multi_input_custom_loss"), 3u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "loss_shaper"), 4u);

        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        EXPECT_EQ(loadedNetwork.getNumLayers(), network.getNumLayers());
        EXPECT_EQ(countLayersOfType<CustomLoss>(loadedNetwork), 2u);
        EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(loadedNetwork), 3u);
        EXPECT_EQ(countLayersOfType<LossShaper>(loadedNetwork), 4u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "custom_loss"), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "multi_input_custom_loss"), 3u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "loss_shaper"), 4u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}

TEST(RepresentationalLossSerialization, LoadedMultiInputCustomLossPreservesAuxiliaryInputs) {
    const string networkName = "representational_loss_multi_input_aux_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addRepresentationalLosses(network);
        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        uint32_t sawTripletLikeLoss = 0;
        uint32_t sawAuxiliaryTargetLoss = 0;
        for (uint32_t i = 0; i < loadedNetwork.getNumLayers(); ++i) {
            shared_ptr<MultiInputCustomLoss> loss = dynamic_pointer_cast<MultiInputCustomLoss>(loadedNetwork.getLayer(i));
            if (!loss)
                continue;

            uint32_t differentiableInputs = 0;
            uint32_t auxiliaryInputs = 0;
            for (const MultiInputCustomLoss::InputSpec& input : loss->getInputs()) {
                if (input.gradientName.has_value())
                    ++differentiableInputs;
                else
                    ++auxiliaryInputs;
            }

            if (differentiableInputs == 3 && auxiliaryInputs == 0)
                ++sawTripletLikeLoss;
            if (differentiableInputs == 2 && auxiliaryInputs == 1)
                ++sawAuxiliaryTargetLoss;
        }

        EXPECT_EQ(sawTripletLikeLoss, 1u);
        EXPECT_EQ(sawAuxiliaryTargetLoss, 2u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}
