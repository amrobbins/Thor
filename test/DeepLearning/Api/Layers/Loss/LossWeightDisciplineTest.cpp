#include "DeepLearning/Api/Layers/Loss/LSGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/WeightedLossExpression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {

Impl::DynamicExpression makeSerializableSquareLossExpression() {
    Impl::Expression scores = Impl::Expression::input("scores", Api::DataType::FP32, Api::DataType::FP32);
    Impl::Expression loss = (scores * scores).withOutputDType(Api::DataType::FP32);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"loss", loss}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

size_t nodeCountAfterApplyingLossWeight(optional<float> lossWeight) {
    Impl::TensorPlacement placement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, {2, 3});
    Impl::Tensor scores(placement, descriptor);
    Stream stream(0);

    Impl::DynamicExpression weightedExpression = Impl::applyLossWeightToDynamicExpression(
        makeSerializableSquareLossExpression(), {{"loss", Api::DataType::FP32}}, lossWeight, "test loss");
    Impl::DynamicExpressionBuild build = weightedExpression.build({{"scores", scores}}, {}, stream);
    const Impl::PhysicalOutputs& outputs = build.equation->physicalOutputs();
    if (!outputs.expr) {
        throw runtime_error("Expected non-conditional physical outputs.");
    }
    return outputs.expr->nodes.size();
}

Api::NetworkInput fp32Input(Api::Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return Api::NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(Api::DataType::FP32).build();
}

shared_ptr<Api::MultiInputCustomLoss> findOnlyRawMultiInputCustomLoss(Api::Network& network) {
    shared_ptr<Api::MultiInputCustomLoss> result;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::MultiInputCustomLoss> loss = dynamic_pointer_cast<Api::MultiInputCustomLoss>(network.getLayer(i));
        if (!loss)
            continue;
        if (result) {
            throw runtime_error("Expected exactly one MultiInputCustomLoss support layer.");
        }
        result = loss;
    }
    if (!result) {
        throw runtime_error("Expected a MultiInputCustomLoss support layer.");
    }
    return result;
}

void expectNoLayerSerializesLossWeight(Api::Network& network) {
    const json architecture = network.architectureJson();
    for (const json& layer : architecture.at("layers")) {
        EXPECT_FALSE(layer.contains("loss_weight")) << layer.dump();
    }
}

}  // namespace

TEST(LossWeightDiscipline, DefaultNulloptAndExplicitOneDoNotAddExpressionCompute) {
    const size_t unweightedNodeCount = nodeCountAfterApplyingLossWeight(nullopt);
    EXPECT_EQ(nodeCountAfterApplyingLossWeight(1.0f), unweightedNodeCount);
    EXPECT_GT(nodeCountAfterApplyingLossWeight(2.0f), unweightedNodeCount);
}

TEST(LossWeightDiscipline, PublicAndSupportLayerSerializationOmitDefaultLossWeight) {
    Api::Network network("loss_weight_default_omits_json");
    Api::NetworkInput fakeScores = fp32Input(network, "fake_scores", {3});

    Api::LSGANGeneratorLoss loss = Api::LSGANGeneratorLoss::Builder()
                                       .network(network)
                                       .fakeScores(fakeScores.getFeatureOutput().value())
                                       .lossDataType(Api::DataType::FP32)
                                       .reportsRawLoss()
                                       .build();

    EXPECT_FALSE(loss.getLossWeight().has_value());
    EXPECT_FALSE(loss.architectureJson().contains("loss_weight"));

    shared_ptr<Api::MultiInputCustomLoss> rawLoss = findOnlyRawMultiInputCustomLoss(network);
    EXPECT_FALSE(rawLoss->getLossWeight().has_value());
    EXPECT_FALSE(rawLoss->architectureJson().contains("loss_weight"));
    expectNoLayerSerializesLossWeight(network);
}

TEST(LossWeightDiscipline, PublicAndSupportLayerSerializationOmitExplicitOneLossWeight) {
    Api::Network network("loss_weight_explicit_one_omits_json");
    Api::NetworkInput fakeScores = fp32Input(network, "fake_scores", {3});

    Api::LSGANGeneratorLoss loss = Api::LSGANGeneratorLoss::Builder()
                                       .network(network)
                                       .fakeScores(fakeScores.getFeatureOutput().value())
                                       .lossDataType(Api::DataType::FP32)
                                       .reportsRawLoss()
                                       .lossWeight(1.0f)
                                       .build();

    EXPECT_FALSE(loss.getLossWeight().has_value());
    EXPECT_FALSE(loss.architectureJson().contains("loss_weight"));

    shared_ptr<Api::MultiInputCustomLoss> rawLoss = findOnlyRawMultiInputCustomLoss(network);
    EXPECT_FALSE(rawLoss->getLossWeight().has_value());
    EXPECT_FALSE(rawLoss->architectureJson().contains("loss_weight"));
    expectNoLayerSerializesLossWeight(network);
}

TEST(LossWeightDiscipline, PublicAndSupportLayerSerializationPreserveNonNoopLossWeight) {
    Api::Network network("loss_weight_non_noop_json");
    Api::NetworkInput fakeScores = fp32Input(network, "fake_scores", {3});
    const float lossWeight = 2.5f;

    Api::LSGANGeneratorLoss loss = Api::LSGANGeneratorLoss::Builder()
                                       .network(network)
                                       .fakeScores(fakeScores.getFeatureOutput().value())
                                       .lossDataType(Api::DataType::FP32)
                                       .reportsRawLoss()
                                       .lossWeight(lossWeight)
                                       .build();

    ASSERT_TRUE(loss.getLossWeight().has_value());
    EXPECT_FLOAT_EQ(loss.getLossWeight().value(), lossWeight);
    ASSERT_TRUE(loss.architectureJson().contains("loss_weight"));
    EXPECT_FLOAT_EQ(loss.architectureJson().at("loss_weight").get<float>(), lossWeight);

    shared_ptr<Api::MultiInputCustomLoss> rawLoss = findOnlyRawMultiInputCustomLoss(network);
    ASSERT_TRUE(rawLoss->getLossWeight().has_value());
    EXPECT_FLOAT_EQ(rawLoss->getLossWeight().value(), lossWeight);
    ASSERT_TRUE(rawLoss->architectureJson().contains("loss_weight"));
    EXPECT_FLOAT_EQ(rawLoss->architectureJson().at("loss_weight").get<float>(), lossWeight);
}
