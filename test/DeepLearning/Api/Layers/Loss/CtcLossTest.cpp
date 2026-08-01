#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {

Api::NetworkInput input(Api::Network& network, const string& name, const vector<uint64_t>& dimensions, Api::DataType dataType) {
    return Api::NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(dataType).build();
}

Api::RaggedTensor raggedInput(Api::Network& network,
                              const string& name,
                              Api::DataType valuesType = Api::DataType::INT32,
                              Api::DataType offsetsType = Api::DataType::UINT32,
                              const vector<uint64_t>& trailingDimensions = {},
                              uint64_t batchSize = 2,
                              uint64_t maxTotalValues = 4) {
    return Api::RaggedNetworkInput::Builder()
        .network(network)
        .name(name)
        .valuesDataType(valuesType)
        .offsetsDataType(offsetsType)
        .trailingDimensions(trailingDimensions)
        .batchSize(batchSize)
        .maxTotalValues(maxTotalValues)
        .build();
}

struct CtcApiInputs {
    Api::NetworkInput logits;
    Api::RaggedTensor labels;
    Api::NetworkInput inputLengths;
};

CtcApiInputs makeInputs(Api::Network& network,
                        Api::DataType logitsType = Api::DataType::FP32,
                        Api::DataType labelsType = Api::DataType::INT32,
                        Api::DataType offsetsType = Api::DataType::UINT32,
                        Api::DataType lengthsType = Api::DataType::INT32) {
    return CtcApiInputs{input(network, "ctc_logits", {4, 3}, logitsType),
                        raggedInput(network, "ctc_labels", labelsType, offsetsType),
                        input(network, "ctc_input_lengths", {1}, lengthsType)};
}

size_t countLayersOfType(const json& architecture, const string& layerType) {
    size_t count = 0;
    for (const json& layer : architecture.at("layers")) {
        if (layer.value("layer_type", string{}) == layerType)
            ++count;
    }
    return count;
}

const json& findOnlyLayerOfType(const json& architecture, const string& layerType) {
    const json* result = nullptr;
    for (const json& layer : architecture.at("layers")) {
        if (layer.value("layer_type", string{}) != layerType)
            continue;
        if (result != nullptr)
            throw runtime_error("Expected exactly one layer of type " + layerType);
        result = &layer;
    }
    if (result == nullptr)
        throw runtime_error("Expected a layer of type " + layerType);
    return *result;
}

}  // namespace

TEST(CtcLossApiLayer, BuildsRawCudnnCtcLossWithCanonicalRaggedLabels) {
    Api::Network network("ctc_raw_api");
    CtcApiInputs tensors = makeInputs(network);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels)
                            .inputLengths(tensors.inputLengths.getFeatureOutput().value())
                            .reportsRawLoss()
                            .build();

    Api::NetworkOutput::Builder()
        .network(network)
        .name("ctc_loss")
        .inputTensor(loss.getLoss())
        .dataType(Api::DataType::FP32)
        .build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_EQ(loss.getLayerType(), "CtcLoss");
    EXPECT_EQ(loss.getLayerVersion(), "2.0.0");
    EXPECT_EQ(loss.getLoss().getDataType(), Api::DataType::FP32);
    EXPECT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>{1});
    EXPECT_EQ(loss.getRaggedLabels(), tensors.labels);
    EXPECT_EQ(loss.getLossInputTensors().size(), 4u);

    EXPECT_EQ(loss.getConnectionType(tensors.logits.getFeatureOutput().value()),
              static_cast<int>(Impl::Loss::ConnectionType::FORWARD_BACKWARD));
    EXPECT_EQ(loss.getConnectionType(tensors.labels.getValues()), static_cast<int>(Impl::Loss::ConnectionType::LABELS));
    EXPECT_EQ(loss.getConnectionType(tensors.labels.getOffsets()), Impl::CtcLoss::LABEL_OFFSETS_CONNECTION_TYPE);
    EXPECT_EQ(loss.getConnectionType(tensors.inputLengths.getFeatureOutput().value()), Impl::CtcLoss::INPUT_LENGTHS_CONNECTION_TYPE);
    EXPECT_EQ(loss.getInputPortName(tensors.labels.getValues()), optional<string>("labels.values"));
    EXPECT_EQ(loss.getInputPortName(tensors.labels.getOffsets()), optional<string>("labels.offsets"));

    const json architecture = network.architectureJson();
    EXPECT_EQ(countLayersOfType(architecture, "ctc_loss"), 1u);
    EXPECT_EQ(countLayersOfType(architecture, "loss_shaper"), 0u);

    const json& ctcJson = findOnlyLayerOfType(architecture, "ctc_loss");
    EXPECT_EQ(ctcJson.at("version").get<string>(), "2.0.0");
    EXPECT_EQ(ctcJson.at("loss_shape").get<Api::Loss::LossShape>(), Api::Loss::LossShape::RAW);
    EXPECT_EQ(ctcJson.at("loss_data_type").get<Api::DataType>(), Api::DataType::FP32);
    EXPECT_FALSE(ctcJson.contains("max_label_length"));
    EXPECT_EQ(ctcJson.at("oob_gradient_mode").get<string>(), "zero");
    EXPECT_TRUE(ctcJson.contains("labels_ragged_tensor"));
    EXPECT_FALSE(ctcJson.contains("labels_tensor"));
    EXPECT_FALSE(ctcJson.contains("label_lengths_tensor"));
    EXPECT_FALSE(ctcJson.contains("loss_weight"));
}

TEST(CtcLossApiLayer, ReportsBatchLossThroughLossShaper) {
    Api::Network network("ctc_batch_api");
    CtcApiInputs tensors = makeInputs(network, Api::DataType::FP32, Api::DataType::INT32, Api::DataType::UINT64);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels)
                            .inputLengths(tensors.inputLengths.getFeatureOutput().value())
                            .reportsBatchLoss()
                            .lossWeight(2.0f)
                            .skipOutOfBoundsGradients()
                            .build();

    Api::NetworkOutput::Builder()
        .network(network)
        .name("ctc_batch_loss")
        .inputTensor(loss.getLoss())
        .dataType(Api::DataType::FP32)
        .build();

    EXPECT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>{1});

    const json architecture = network.architectureJson();
    EXPECT_EQ(countLayersOfType(architecture, "ctc_loss"), 1u);
    EXPECT_EQ(countLayersOfType(architecture, "loss_shaper"), 1u);

    const json& ctcJson = findOnlyLayerOfType(architecture, "ctc_loss");
    ASSERT_TRUE(ctcJson.contains("loss_weight"));
    EXPECT_FLOAT_EQ(ctcJson.at("loss_weight").get<float>(), 2.0f);
    EXPECT_EQ(ctcJson.at("oob_gradient_mode").get<string>(), "skip");
    EXPECT_EQ(ctcJson.at("labels_ragged_tensor").at("offsets").at("data_type").get<Api::DataType>(), Api::DataType::UINT64);

    const json& shaperJson = findOnlyLayerOfType(architecture, "loss_shaper");
    EXPECT_EQ(shaperJson.at("loss_shape").get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::BATCH);
}

TEST(CtcLossApiLayer, ReportsPerExampleLossThroughLossShaper) {
    Api::Network network("ctc_per_example_api");
    CtcApiInputs tensors = makeInputs(network);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels)
                            .inputLengths(tensors.inputLengths.getFeatureOutput().value())
                            .reportsPerExampleLoss()
                            .build();

    EXPECT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>{1});
    const json architecture = network.architectureJson();
    EXPECT_EQ(countLayersOfType(architecture, "ctc_loss"), 1u);
    EXPECT_EQ(countLayersOfType(architecture, "loss_shaper"), 1u);
    const json& shaperJson = findOnlyLayerOfType(architecture, "loss_shaper");
    EXPECT_EQ(shaperJson.at("loss_shape").get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::PER_EXAMPLE);
}


TEST(CtcLossApiLayer, JsonDeserializerRestoresCanonicalRaggedInputsAndRejectsLegacyVersion) {
    Api::Network originalNetwork("ctc_ragged_deserialize_source");
    CtcApiInputs tensors = makeInputs(originalNetwork, Api::DataType::FP32, Api::DataType::INT32, Api::DataType::UINT64);
    Api::CtcLoss originalLoss = Api::CtcLoss::Builder()
                                    .network(originalNetwork)
                                    .logits(tensors.logits.getFeatureOutput().value())
                                    .labels(tensors.labels)
                                    .inputLengths(tensors.inputLengths.getFeatureOutput().value())
                                    .reportsRawLoss()
                                    .skipOutOfBoundsGradients()
                                    .build();

    const json originalArchitecture = originalNetwork.architectureJson();
    const json originalCtcJson = originalLoss.architectureJson();
    EXPECT_EQ(originalCtcJson.at("version").get<string>(), "2.0.0");
    EXPECT_TRUE(originalCtcJson.contains("labels_ragged_tensor"));
    EXPECT_FALSE(originalCtcJson.contains("labels_tensor"));
    EXPECT_FALSE(originalCtcJson.contains("label_lengths_tensor"));
    EXPECT_FALSE(originalCtcJson.contains("max_label_length"));

    Api::Network restoredNetwork("ctc_ragged_deserialize_restored");
    for (const json& layer : originalArchitecture.at("layers")) {
        if (layer.at("layer_type").get<string>() == "network_input")
            Api::NetworkInput::deserialize(layer, &restoredNetwork);
    }
    Api::Loss::deserialize(originalCtcJson, &restoredNetwork);

    shared_ptr<Api::CtcLoss> restoredLoss;
    for (uint32_t i = 0; i < restoredNetwork.getNumLayers(); ++i) {
        shared_ptr<Api::CtcLoss> candidate = dynamic_pointer_cast<Api::CtcLoss>(restoredNetwork.getLayer(i));
        if (candidate != nullptr) {
            ASSERT_EQ(restoredLoss, nullptr);
            restoredLoss = std::move(candidate);
        }
    }
    ASSERT_NE(restoredLoss, nullptr);
    EXPECT_EQ(restoredLoss->getLayerVersion(), "2.0.0");
    EXPECT_EQ(restoredLoss->getRaggedLabels().getValuesDataType(), Api::DataType::INT32);
    EXPECT_EQ(restoredLoss->getRaggedLabels().getOffsetsDataType(), Api::DataType::UINT64);
    EXPECT_TRUE(restoredLoss->getRaggedLabels().getTrailingDimensions().empty());
    EXPECT_EQ(restoredLoss->getRaggedLabels().getBatchSize(), 2u);
    EXPECT_EQ(restoredLoss->getRaggedLabels().getMaxTotalValues(), 4u);
    EXPECT_EQ(restoredLoss->getOobGradientMode(), Impl::CtcLossOobGradientMode::SKIP);

    json legacyJson = originalCtcJson;
    legacyJson["version"] = "1.0.0";
    Api::Network legacyNetwork("ctc_legacy_deserialize_rejected");
    EXPECT_THROW(Api::Loss::deserialize(legacyJson, &legacyNetwork), runtime_error);
}

TEST(CtcLossApiLayer, RejectsUnsupportedPublicApiContracts) {
    {
        Api::Network network("ctc_reject_logits_dtype");
        CtcApiInputs tensors = makeInputs(network, Api::DataType::FP16);
        EXPECT_THROW(Api::CtcLoss::Builder().network(network).logits(tensors.logits.getFeatureOutput().value()), std::logic_error);
    }
    {
        Api::Network network("ctc_reject_label_dtype");
        CtcApiInputs tensors = makeInputs(network, Api::DataType::FP32, Api::DataType::UINT32);
        EXPECT_THROW(Api::CtcLoss::Builder()
                         .network(network)
                         .logits(tensors.logits.getFeatureOutput().value())
                         .labels(tensors.labels),
                     std::logic_error);
    }
    {
        Api::Network network("ctc_reject_vector_label_values");
        Api::NetworkInput logits = input(network, "logits", {4, 3}, Api::DataType::FP32);
        Api::RaggedTensor labels = raggedInput(network, "labels", Api::DataType::INT32, Api::DataType::UINT32, {2});
        EXPECT_THROW(Api::CtcLoss::Builder().network(network).logits(logits.getFeatureOutput().value()).labels(labels), std::logic_error);
    }
    {
        Api::Network network("ctc_reject_length_shape");
        Api::NetworkInput logits = input(network, "logits", {4, 3}, Api::DataType::FP32);
        Api::RaggedTensor labels = raggedInput(network, "labels");
        Api::NetworkInput badLengths = input(network, "bad_lengths", {2}, Api::DataType::INT32);
        EXPECT_THROW(Api::CtcLoss::Builder()
                         .network(network)
                         .logits(logits.getFeatureOutput().value())
                         .labels(labels)
                         .inputLengths(badLengths.getFeatureOutput().value())
                         .build(),
                     std::logic_error);
    }
}
