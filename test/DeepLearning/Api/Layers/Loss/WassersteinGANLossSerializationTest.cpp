#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticGradientPenaltyLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
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

struct WassersteinGANLossNetworkParts {
    WassersteinGANCriticLoss criticLoss;
    WassersteinGANGeneratorLoss generatorLoss;
    WassersteinGANCriticGradientPenaltyLoss gradientPenaltyLoss;
};

WassersteinGANLossNetworkParts addWassersteinGANLosses(Network& network) {
    NetworkInput realScores = fp32Input(network, "real_scores", {3});
    NetworkInput fakeScoresForCritic = fp32Input(network, "fake_scores_critic", {3});
    WassersteinGANCriticLoss criticLoss = WassersteinGANCriticLoss::Builder()
                                              .network(network)
                                              .realScores(realScores.getFeatureOutput().value())
                                              .fakeScores(fakeScoresForCritic.getFeatureOutput().value())
                                              .reportsElementwiseLoss()
                                              .lossDataType(DataType::FP32)
                                              .build();
    consumeLoss(network, "wasserstein_gan_critic_loss", criticLoss.getLoss());

    NetworkInput fakeScoresForGenerator = fp32Input(network, "fake_scores_generator", {3});
    WassersteinGANGeneratorLoss generatorLoss = WassersteinGANGeneratorLoss::Builder()
                                                   .network(network)
                                                   .fakeScores(fakeScoresForGenerator.getFeatureOutput().value())
                                                   .reportsPerOutputLoss()
                                                   .lossDataType(DataType::FP32)
                                                   .build();
    consumeLoss(network, "wasserstein_gan_generator_loss", generatorLoss.getLoss());

    NetworkInput realScalarScores = fp32Input(network, "real_scalar_scores", {1});
    NetworkInput fakeScalarScores = fp32Input(network, "fake_scalar_scores", {1});
    NetworkInput sampleGradients = fp32Input(network, "sample_gradients", {2, 2});
    WassersteinGANCriticGradientPenaltyLoss gradientPenaltyLoss = WassersteinGANCriticGradientPenaltyLoss::Builder()
                                                                     .network(network)
                                                                     .realScores(realScalarScores.getFeatureOutput().value())
                                                                     .fakeScores(fakeScalarScores.getFeatureOutput().value())
                                                                     .sampleGradients(sampleGradients.getFeatureOutput().value())
                                                                     .gradientPenaltyWeight(7.5f)
                                                                     .targetGradientNorm(1.25f)
                                                                     .epsilon(1.0e-8f)
                                                                     .reportsRawLoss()
                                                                     .lossDataType(DataType::FP32)
                                                                     .build();
    consumeLoss(network, "wasserstein_gan_gp_loss", gradientPenaltyLoss.getLoss());

    return WassersteinGANLossNetworkParts{criticLoss, generatorLoss, gradientPenaltyLoss};
}

void expectTensorJsonMatches(const json& tensorJson, const Tensor& tensor) {
    EXPECT_EQ(tensorJson.at("id").get<uint64_t>(), tensor.getOriginalId());
    EXPECT_EQ(tensorJson.at("data_type").get<DataType>(), tensor.getDataType());
    EXPECT_EQ(tensorJson.at("dimensions").get<vector<uint64_t>>(), tensor.getDimensions());
}

}  // namespace

TEST(WassersteinGANLossSerialization, PublicArchitectureJsonIncludesRoundTripFields) {
    Network network("wasserstein_gan_loss_public_json");
    WassersteinGANLossNetworkParts losses = addWassersteinGANLosses(network);

    const json criticJson = losses.criticLoss.architectureJson();
    EXPECT_EQ(criticJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(criticJson.at("layer_type").get<string>(), "wasserstein_gan_critic_loss");
    EXPECT_EQ(criticJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    EXPECT_EQ(criticJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    expectTensorJsonMatches(criticJson.at("real_scores_tensor"), losses.criticLoss.getRealScores());
    expectTensorJsonMatches(criticJson.at("fake_scores_tensor"), losses.criticLoss.getFakeScores());
    expectTensorJsonMatches(criticJson.at("loss_tensor"), losses.criticLoss.getLoss());

    const json generatorJson = losses.generatorLoss.architectureJson();
    EXPECT_EQ(generatorJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(generatorJson.at("layer_type").get<string>(), "wasserstein_gan_generator_loss");
    EXPECT_EQ(generatorJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::CLASSWISE);
    EXPECT_EQ(generatorJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    expectTensorJsonMatches(generatorJson.at("fake_scores_tensor"), losses.generatorLoss.getFakeScores());
    expectTensorJsonMatches(generatorJson.at("loss_tensor"), losses.generatorLoss.getLoss());

    const json gpJson = losses.gradientPenaltyLoss.architectureJson();
    EXPECT_EQ(gpJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(gpJson.at("layer_type").get<string>(), "wasserstein_gan_critic_gradient_penalty_loss");
    EXPECT_EQ(gpJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
    EXPECT_EQ(gpJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_FLOAT_EQ(gpJson.at("gradient_penalty_weight").get<float>(), 7.5f);
    EXPECT_FLOAT_EQ(gpJson.at("target_gradient_norm").get<float>(), 1.25f);
    EXPECT_FLOAT_EQ(gpJson.at("eps").get<float>(), 1.0e-8f);
    expectTensorJsonMatches(gpJson.at("real_scores_tensor"), losses.gradientPenaltyLoss.getRealScores());
    expectTensorJsonMatches(gpJson.at("fake_scores_tensor"), losses.gradientPenaltyLoss.getFakeScores());
    expectTensorJsonMatches(gpJson.at("sample_gradients_tensor"), losses.gradientPenaltyLoss.getSampleGradients());
    expectTensorJsonMatches(gpJson.at("loss_tensor"), losses.gradientPenaltyLoss.getLoss());
}

TEST(WassersteinGANLossSerialization, PublicLossJsonDeserializersRebuildSupportLayers) {
    Network originalNetwork("wasserstein_gan_loss_public_deserialize_source");
    WassersteinGANLossNetworkParts losses = addWassersteinGANLosses(originalNetwork);

    const json originalArchitecture = originalNetwork.architectureJson();
    Network restoredNetwork("wasserstein_gan_loss_public_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            NetworkInput::deserialize(layer, &restoredNetwork);
    }

    Loss::deserialize(losses.criticLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.generatorLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.gradientPenaltyLoss.architectureJson(), &restoredNetwork);

    EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(restoredNetwork), 3u);
    EXPECT_EQ(countLayersOfType<LossShaper>(restoredNetwork), 2u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "multi_input_custom_loss"), 3u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "loss_shaper"), 2u);
}

TEST(WassersteinGANLossSerialization, NetworkSaveLoadRoundTripRestoresSupportLayers) {
    const string networkName = "wasserstein_gan_loss_save_load";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addWassersteinGANLosses(network);

        const json beforeArchitecture = network.architectureJson();
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "multi_input_custom_loss"), 3u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "loss_shaper"), 2u);

        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        EXPECT_EQ(loadedNetwork.getNumLayers(), network.getNumLayers());
        EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(loadedNetwork), 3u);
        EXPECT_EQ(countLayersOfType<LossShaper>(loadedNetwork), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "multi_input_custom_loss"), 3u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "loss_shaper"), 2u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}

TEST(WassersteinGANLossSerialization, LoadedMultiInputCustomLossPreservesDifferentiableInputs) {
    const string networkName = "wasserstein_gan_loss_multi_input_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addWassersteinGANLosses(network);
        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        uint32_t sawCriticLoss = 0;
        uint32_t sawGeneratorLoss = 0;
        uint32_t sawGradientPenaltyLoss = 0;
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

            if (differentiableInputs == 2 && auxiliaryInputs == 0)
                ++sawCriticLoss;
            if (differentiableInputs == 1 && auxiliaryInputs == 0)
                ++sawGeneratorLoss;
            if (differentiableInputs == 3 && auxiliaryInputs == 0)
                ++sawGradientPenaltyLoss;
        }

        EXPECT_EQ(sawCriticLoss, 1u);
        EXPECT_EQ(sawGeneratorLoss, 1u);
        EXPECT_EQ(sawGradientPenaltyLoss, 1u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}
