#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
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

struct HingeGANLossNetworkParts {
    HingeGANDiscriminatorLoss discriminatorLoss;
    HingeGANGeneratorLoss generatorLoss;
};

HingeGANLossNetworkParts addHingeGANLosses(Network& network) {
    NetworkInput realScores = fp32Input(network, "real_scores", {3});
    NetworkInput fakeScoresForDiscriminator = fp32Input(network, "fake_scores_d", {3});
    HingeGANDiscriminatorLoss discriminatorLoss = HingeGANDiscriminatorLoss::Builder()
                                                       .network(network)
                                                       .realScores(realScores.getFeatureOutput().value())
                                                       .fakeScores(fakeScoresForDiscriminator.getFeatureOutput().value())
                                                       .reportsElementwiseLoss()
                                                       .lossDataType(DataType::FP32)
                                                       .build();
    consumeLoss(network, "hinge_gan_discriminator_loss", discriminatorLoss.getLoss());

    NetworkInput fakeScoresForGenerator = fp32Input(network, "fake_scores_g", {3});
    HingeGANGeneratorLoss generatorLoss = HingeGANGeneratorLoss::Builder()
                                             .network(network)
                                             .fakeScores(fakeScoresForGenerator.getFeatureOutput().value())
                                             .reportsPerOutputLoss()
                                             .lossDataType(DataType::FP32)
                                             .build();
    consumeLoss(network, "hinge_gan_generator_loss", generatorLoss.getLoss());

    return HingeGANLossNetworkParts{discriminatorLoss, generatorLoss};
}

void expectTensorJsonMatches(const json& tensorJson, const Tensor& tensor) {
    EXPECT_EQ(tensorJson.at("id").get<uint64_t>(), tensor.getOriginalId());
    EXPECT_EQ(tensorJson.at("data_type").get<DataType>(), tensor.getDataType());
    EXPECT_EQ(tensorJson.at("dimensions").get<vector<uint64_t>>(), tensor.getDimensions());
}

}  // namespace

TEST(HingeGANLossSerialization, PublicArchitectureJsonIncludesRoundTripFields) {
    Network network("hinge_gan_loss_public_json");
    HingeGANLossNetworkParts losses = addHingeGANLosses(network);

    const json discriminatorJson = losses.discriminatorLoss.architectureJson();
    EXPECT_EQ(discriminatorJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(discriminatorJson.at("layer_type").get<string>(), "hinge_gan_discriminator_loss");
    EXPECT_EQ(discriminatorJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    EXPECT_EQ(discriminatorJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    expectTensorJsonMatches(discriminatorJson.at("real_scores_tensor"), losses.discriminatorLoss.getRealScores());
    expectTensorJsonMatches(discriminatorJson.at("fake_scores_tensor"), losses.discriminatorLoss.getFakeScores());
    expectTensorJsonMatches(discriminatorJson.at("loss_tensor"), losses.discriminatorLoss.getLoss());

    const json generatorJson = losses.generatorLoss.architectureJson();
    EXPECT_EQ(generatorJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(generatorJson.at("layer_type").get<string>(), "hinge_gan_generator_loss");
    EXPECT_EQ(generatorJson.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::CLASSWISE);
    EXPECT_EQ(generatorJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    expectTensorJsonMatches(generatorJson.at("fake_scores_tensor"), losses.generatorLoss.getFakeScores());
    expectTensorJsonMatches(generatorJson.at("loss_tensor"), losses.generatorLoss.getLoss());
}

TEST(HingeGANLossSerialization, PublicLossJsonDeserializersRebuildSupportLayers) {
    Network originalNetwork("hinge_gan_loss_public_deserialize_source");
    HingeGANLossNetworkParts losses = addHingeGANLosses(originalNetwork);

    const json originalArchitecture = originalNetwork.architectureJson();
    Network restoredNetwork("hinge_gan_loss_public_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            NetworkInput::deserialize(layer, &restoredNetwork);
    }

    Loss::deserialize(losses.discriminatorLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.generatorLoss.architectureJson(), &restoredNetwork);

    EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(restoredNetwork), 2u);
    EXPECT_EQ(countLayersOfType<LossShaper>(restoredNetwork), 2u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "multi_input_custom_loss"), 2u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "loss_shaper"), 2u);
}

TEST(HingeGANLossSerialization, NetworkSaveLoadRoundTripRestoresSupportLayers) {
    const string networkName = "hinge_gan_loss_save_load";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addHingeGANLosses(network);

        const json beforeArchitecture = network.architectureJson();
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "multi_input_custom_loss"), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "loss_shaper"), 2u);

        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        EXPECT_EQ(loadedNetwork.getNumLayers(), network.getNumLayers());
        EXPECT_EQ(countLayersOfType<MultiInputCustomLoss>(loadedNetwork), 2u);
        EXPECT_EQ(countLayersOfType<LossShaper>(loadedNetwork), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "multi_input_custom_loss"), 2u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "loss_shaper"), 2u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}

TEST(HingeGANLossSerialization, LoadedMultiInputCustomLossPreservesDifferentiableInputs) {
    const string networkName = "hinge_gan_loss_multi_input_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addHingeGANLosses(network);
        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        uint32_t sawDiscriminatorLoss = 0;
        uint32_t sawGeneratorLoss = 0;
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
                ++sawDiscriminatorLoss;
            if (differentiableInputs == 1 && auxiliaryInputs == 0)
                ++sawGeneratorLoss;
        }

        EXPECT_EQ(sawDiscriminatorLoss, 1u);
        EXPECT_EQ(sawGeneratorLoss, 1u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}
