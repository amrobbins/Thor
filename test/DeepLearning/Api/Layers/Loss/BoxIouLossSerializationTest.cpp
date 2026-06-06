#include "DeepLearning/Api/Layers/Loss/BoxIouLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
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

struct BoxLossNetworkParts {
    IoULoss iouLoss;
    GIoULoss giouLoss;
    DIoULoss diouLoss;
    CIoULoss ciouLoss;
};

BoxLossNetworkParts addBoxLosses(Network& network) {
    NetworkInput iouPredictions = fp32Input(network, "iou_predictions", {2, 4});
    NetworkInput iouLabels = fp32Input(network, "iou_labels", {2, 4});
    IoULoss iouLoss = IoULoss::Builder()
                           .network(network)
                           .predictions(iouPredictions.getFeatureOutput().value())
                           .labels(iouLabels.getFeatureOutput().value())
                           .eps(1.0e-6f)
                           .reportsBatchLoss()
                           .lossDataType(DataType::FP32)
                           .build();
    consumeLoss(network, "iou_loss", iouLoss.getLoss());

    NetworkInput giouPredictions = fp32Input(network, "giou_predictions", {2, 4});
    NetworkInput giouLabels = fp32Input(network, "giou_labels", {2, 4});
    GIoULoss giouLoss = GIoULoss::Builder()
                            .network(network)
                            .predictions(giouPredictions.getFeatureOutput().value())
                            .labels(giouLabels.getFeatureOutput().value())
                            .eps(2.0e-6f)
                            .reportsElementwiseLoss()
                            .lossDataType(DataType::FP32)
                            .build();
    consumeLoss(network, "giou_loss", giouLoss.getLoss());

    NetworkInput diouPredictions = fp32Input(network, "diou_predictions", {2, 4});
    NetworkInput diouLabels = fp32Input(network, "diou_labels", {2, 4});
    DIoULoss diouLoss = DIoULoss::Builder()
                            .network(network)
                            .predictions(diouPredictions.getFeatureOutput().value())
                            .labels(diouLabels.getFeatureOutput().value())
                            .eps(3.0e-6f)
                            .reportsPerOutputLoss()
                            .lossDataType(DataType::FP32)
                            .build();
    consumeLoss(network, "diou_loss", diouLoss.getLoss());

    NetworkInput ciouPredictions = fp32Input(network, "ciou_predictions", {4});
    NetworkInput ciouLabels = fp32Input(network, "ciou_labels", {4});
    CIoULoss ciouLoss = CIoULoss::Builder()
                            .network(network)
                            .predictions(ciouPredictions.getFeatureOutput().value())
                            .labels(ciouLabels.getFeatureOutput().value())
                            .eps(4.0e-6f)
                            .reportsRawLoss()
                            .lossDataType(DataType::FP32)
                            .build();
    consumeLoss(network, "ciou_loss", ciouLoss.getLoss());

    return BoxLossNetworkParts{iouLoss, giouLoss, diouLoss, ciouLoss};
}

void expectBoxLossJson(const json& lossJson,
                       const string& layerType,
                       Loss::LossShape lossShape,
                       float eps,
                       const Tensor& predictions,
                       const Tensor& labels,
                       const Tensor& loss) {
    EXPECT_EQ(lossJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    EXPECT_EQ(lossJson.at("layer_type").get<string>(), layerType);
    EXPECT_EQ(lossJson.at("loss_shape").get<Loss::LossShape>(), lossShape);
    EXPECT_EQ(lossJson.at("loss_data_type").get<DataType>(), DataType::FP32);
    EXPECT_EQ(lossJson.at("box_format").get<string>(), "xyxy");
    EXPECT_FLOAT_EQ(lossJson.at("eps").get<float>(), eps);
    expectTensorJsonMatches(lossJson.at("predictions_tensor"), predictions);
    expectTensorJsonMatches(lossJson.at("labels_tensor"), labels);
    expectTensorJsonMatches(lossJson.at("loss_tensor"), loss);
}

}  // namespace

TEST(BoxIouLossSerialization, PublicArchitectureJsonIncludesRoundTripFields) {
    Network network("box_iou_loss_public_json");
    BoxLossNetworkParts losses = addBoxLosses(network);

    expectBoxLossJson(losses.iouLoss.architectureJson(),
                      "iou_loss",
                      Loss::LossShape::BATCH,
                      1.0e-6f,
                      losses.iouLoss.getPredictions(),
                      losses.iouLoss.getLabels(),
                      losses.iouLoss.getLoss());
    expectBoxLossJson(losses.giouLoss.architectureJson(),
                      "giou_loss",
                      Loss::LossShape::ELEMENTWISE,
                      2.0e-6f,
                      losses.giouLoss.getPredictions(),
                      losses.giouLoss.getLabels(),
                      losses.giouLoss.getLoss());
    expectBoxLossJson(losses.diouLoss.architectureJson(),
                      "diou_loss",
                      Loss::LossShape::CLASSWISE,
                      3.0e-6f,
                      losses.diouLoss.getPredictions(),
                      losses.diouLoss.getLabels(),
                      losses.diouLoss.getLoss());
    expectBoxLossJson(losses.ciouLoss.architectureJson(),
                      "ciou_loss",
                      Loss::LossShape::RAW,
                      4.0e-6f,
                      losses.ciouLoss.getPredictions(),
                      losses.ciouLoss.getLabels(),
                      losses.ciouLoss.getLoss());
}

TEST(BoxIouLossSerialization, PublicLossJsonDeserializersRebuildSupportLayers) {
    Network originalNetwork("box_iou_loss_public_deserialize_source");
    BoxLossNetworkParts losses = addBoxLosses(originalNetwork);

    const json originalArchitecture = originalNetwork.architectureJson();
    Network restoredNetwork("box_iou_loss_public_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            NetworkInput::deserialize(layer, &restoredNetwork);
    }

    Loss::deserialize(losses.iouLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.giouLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.diouLoss.architectureJson(), &restoredNetwork);
    Loss::deserialize(losses.ciouLoss.architectureJson(), &restoredNetwork);

    EXPECT_EQ(countLayersOfType<BoxIouLoss>(restoredNetwork), 4u);
    EXPECT_EQ(countLayersOfType<LossShaper>(restoredNetwork), 3u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "iou_loss"), 1u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "giou_loss"), 1u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "diou_loss"), 1u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "ciou_loss"), 1u);
    EXPECT_EQ(countLayerTypesInArchitecture(restoredNetwork.architectureJson(), "loss_shaper"), 3u);
}

TEST(BoxIouLossSerialization, NetworkSaveLoadRoundTripRestoresBoxLossSupportLayers) {
    const string networkName = "box_iou_loss_save_load";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network network(networkName);
        addBoxLosses(network);

        const json beforeArchitecture = network.architectureJson();
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "iou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "giou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "diou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "ciou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(beforeArchitecture, "loss_shaper"), 3u);

        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        EXPECT_EQ(loadedNetwork.getNumLayers(), network.getNumLayers());
        EXPECT_EQ(countLayersOfType<BoxIouLoss>(loadedNetwork), 4u);
        EXPECT_EQ(countLayersOfType<LossShaper>(loadedNetwork), 3u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "iou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "giou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "diou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "ciou_loss"), 1u);
        EXPECT_EQ(countLayerTypesInArchitecture(loadedNetwork.architectureJson(), "loss_shaper"), 3u);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}
