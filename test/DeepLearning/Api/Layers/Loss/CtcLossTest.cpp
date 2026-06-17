#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {

Api::NetworkInput input(Api::Network& network, const string& name, const vector<uint64_t>& dimensions, Api::DataType dataType) {
    return Api::NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(dataType).build();
}

struct CtcApiInputs {
    Api::NetworkInput logits;
    Api::NetworkInput labels;
    Api::NetworkInput labelLengths;
    Api::NetworkInput inputLengths;
};

CtcApiInputs makeInputs(Api::Network& network,
                        Api::DataType logitsType = Api::DataType::FP32,
                        Api::DataType labelsType = Api::DataType::INT32,
                        Api::DataType lengthsType = Api::DataType::INT32) {
    return CtcApiInputs{input(network, "ctc_logits", {4, 3}, logitsType),
                        input(network, "ctc_labels", {2}, labelsType),
                        input(network, "ctc_label_lengths", {1}, lengthsType),
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

TEST(CtcLossApiLayer, BuildsRawCudnnCtcLossWithFourInputs) {
    Api::Network network("ctc_raw_api");
    CtcApiInputs tensors = makeInputs(network);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels.getFeatureOutput().value())
                            .labelLengths(tensors.labelLengths.getFeatureOutput().value())
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
    EXPECT_EQ(loss.getLoss().getDataType(), Api::DataType::FP32);
    EXPECT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>{1});
    EXPECT_EQ(loss.getMaxLabelLength(), 2u);
    EXPECT_EQ(loss.getLossInputTensors().size(), 4u);

    EXPECT_EQ(loss.getConnectionType(tensors.logits.getFeatureOutput().value()),
              static_cast<int>(Impl::Loss::ConnectionType::FORWARD_BACKWARD));
    EXPECT_EQ(loss.getConnectionType(tensors.labels.getFeatureOutput().value()), static_cast<int>(Impl::Loss::ConnectionType::LABELS));
    EXPECT_EQ(loss.getConnectionType(tensors.labelLengths.getFeatureOutput().value()), Impl::CtcLoss::LABEL_LENGTHS_CONNECTION_TYPE);
    EXPECT_EQ(loss.getConnectionType(tensors.inputLengths.getFeatureOutput().value()), Impl::CtcLoss::INPUT_LENGTHS_CONNECTION_TYPE);

    const json architecture = network.architectureJson();
    EXPECT_EQ(countLayersOfType(architecture, "ctc_loss"), 1u);
    EXPECT_EQ(countLayersOfType(architecture, "loss_shaper"), 0u);

    const json& ctcJson = findOnlyLayerOfType(architecture, "ctc_loss");
    EXPECT_EQ(ctcJson.at("loss_shape").get<Api::Loss::LossShape>(), Api::Loss::LossShape::RAW);
    EXPECT_EQ(ctcJson.at("loss_data_type").get<Api::DataType>(), Api::DataType::FP32);
    EXPECT_EQ(ctcJson.at("max_label_length").get<uint32_t>(), 2u);
    EXPECT_EQ(ctcJson.at("oob_gradient_mode").get<string>(), "zero");
    EXPECT_FALSE(ctcJson.contains("loss_weight"));
}

TEST(CtcLossApiLayer, ReportsBatchLossThroughLossShaper) {
    Api::Network network("ctc_batch_api");
    CtcApiInputs tensors = makeInputs(network);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels.getFeatureOutput().value())
                            .labelLengths(tensors.labelLengths.getFeatureOutput().value())
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

    const json& shaperJson = findOnlyLayerOfType(architecture, "loss_shaper");
    EXPECT_EQ(shaperJson.at("loss_shape").get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::BATCH);
}

TEST(CtcLossApiLayer, ReportsElementwiseLossThroughLossShaper) {
    Api::Network network("ctc_elementwise_api");
    CtcApiInputs tensors = makeInputs(network);

    Api::CtcLoss loss = Api::CtcLoss::Builder()
                            .network(network)
                            .logits(tensors.logits.getFeatureOutput().value())
                            .labels(tensors.labels.getFeatureOutput().value())
                            .labelLengths(tensors.labelLengths.getFeatureOutput().value())
                            .inputLengths(tensors.inputLengths.getFeatureOutput().value())
                            .reportsElementwiseLoss()
                            .build();

    EXPECT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>{1});
    const json architecture = network.architectureJson();
    EXPECT_EQ(countLayersOfType(architecture, "ctc_loss"), 1u);
    EXPECT_EQ(countLayersOfType(architecture, "loss_shaper"), 1u);
    const json& shaperJson = findOnlyLayerOfType(architecture, "loss_shaper");
    EXPECT_EQ(shaperJson.at("loss_shape").get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::ELEMENTWISE);
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
                         .labels(tensors.labels.getFeatureOutput().value()),
                     std::logic_error);
    }
    {
        Api::Network network("ctc_reject_length_shape");
        Api::NetworkInput logits = input(network, "logits", {4, 3}, Api::DataType::FP32);
        Api::NetworkInput labels = input(network, "labels", {2}, Api::DataType::INT32);
        Api::NetworkInput badLengths = input(network, "bad_lengths", {2}, Api::DataType::INT32);
        Api::NetworkInput inputLengths = input(network, "input_lengths", {1}, Api::DataType::INT32);
        EXPECT_THROW(Api::CtcLoss::Builder()
                         .network(network)
                         .logits(logits.getFeatureOutput().value())
                         .labels(labels.getFeatureOutput().value())
                         .labelLengths(badLengths.getFeatureOutput().value())
                         .inputLengths(inputLengths.getFeatureOutput().value())
                         .build(),
                     std::logic_error);
    }
    {
        Api::Network network("ctc_reject_label_longer_than_time");
        Api::NetworkInput logits = input(network, "logits", {2, 3}, Api::DataType::FP32);
        Api::NetworkInput labels = input(network, "labels", {3}, Api::DataType::INT32);
        Api::NetworkInput labelLengths = input(network, "label_lengths", {1}, Api::DataType::INT32);
        Api::NetworkInput inputLengths = input(network, "input_lengths", {1}, Api::DataType::INT32);
        EXPECT_THROW(Api::CtcLoss::Builder()
                         .network(network)
                         .logits(logits.getFeatureOutput().value())
                         .labels(labels.getFeatureOutput().value())
                         .labelLengths(labelLengths.getFeatureOutput().value())
                         .inputLengths(inputLengths.getFeatureOutput().value())
                         .build(),
                     std::logic_error);
    }
}
