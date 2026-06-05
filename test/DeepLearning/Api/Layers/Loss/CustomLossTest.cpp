#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {

Impl::DynamicExpression makeSerializableSquaredErrorLossExpression(const std::string& predictionsName = "predictions",
                                                                    const std::string& labelsName = "labels",
                                                                    const std::string& lossName = "loss",
                                                                    DataType lossDataType = DataType::FP32) {
    Impl::Expression predictions = Impl::Expression::input(predictionsName, DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input(labelsName, DataType::FP32, DataType::FP32);
    Impl::Expression diff = predictions - labels;
    Impl::Expression loss = (diff * diff).withOutputDType(lossDataType);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{lossName, loss}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeSerializableSquaredErrorGradientExpression(const std::string& predictionsName = "predictions",
                                                                        const std::string& labelsName = "labels",
                                                                        const std::string& gradientName = "predictions_grad",
                                                                        DataType predictionsDataType = DataType::FP32) {
    Impl::Expression predictions = Impl::Expression::input(predictionsName, DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input(labelsName, DataType::FP32, DataType::FP32);
    Impl::Expression gradient = ((predictions - labels) * Impl::Expression(2.0f * Impl::Loss::getLossScalingFactor()))
                                    .withOutputDType(predictionsDataType);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{gradientName, gradient}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeNonSerializableSquaredErrorLossExpression() {
    return Impl::DynamicExpression({"predictions", "labels"},
                                   {"loss"},
                                   [](const Impl::DynamicExpression::TensorMap& inputs,
                                      const Impl::DynamicExpression::TensorMap& outputs,
                                      Stream& stream) -> Impl::DynamicExpressionBuild {
                                       auto predictions = Impl::Expression::input("predictions");
                                       auto labels = Impl::Expression::input("labels");
                                       auto diff = predictions - labels;
                                       auto expressionOutputs = Impl::Expression::outputs({{"loss", diff * diff}});
                                       return Impl::DynamicExpressionBuild{
                                           std::make_shared<Impl::FusedEquation>(Impl::FusedEquation::compile(
                                               expressionOutputs.physicalOutputs(), stream.getGpuNum())),
                                           inputs,
                                           {},
                                           outputs,
                                           {}};
                                   });
}

}  // namespace

TEST(CustomLossApi, BuildsAndSerializesExpressionBackedRawLoss) {
    Network network("custom_loss_builds");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomLoss customLoss = CustomLoss::Builder()
                                .network(network)
                                .lossExpression(makeSerializableSquaredErrorLossExpression())
                                .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                .predictions(predictions)
                                .labels(labels)
                                .reportsRawLoss()
                                .build();

    ASSERT_TRUE(customLoss.isInitialized());
    ASSERT_EQ(customLoss.getPredictions(), predictions);
    ASSERT_EQ(customLoss.getLabels(), labels);
    ASSERT_EQ(customLoss.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(customLoss.getLoss().getDimensions(), vector<uint64_t>({3}));
    ASSERT_EQ(customLoss.getPredictionsName(), string("predictions"));
    ASSERT_EQ(customLoss.getLabelsName(), string("labels"));
    ASSERT_EQ(customLoss.getLossName(), string("loss"));
    ASSERT_EQ(customLoss.getGradientName(), string("predictions_grad"));

    shared_ptr<Layer> cloneLayer = customLoss.clone();
    CustomLoss* clone = dynamic_cast<CustomLoss*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());
    ASSERT_EQ(clone->getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(clone->getLoss().getDimensions(), vector<uint64_t>({3}));

    json lossJson = customLoss.architectureJson();
    ASSERT_EQ(lossJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    ASSERT_EQ(lossJson.at("layer_type").get<string>(), string("custom_loss"));
    ASSERT_EQ(lossJson.at("predictions_name").get<string>(), string("predictions"));
    ASSERT_EQ(lossJson.at("labels_name").get<string>(), string("labels"));
    ASSERT_EQ(lossJson.at("loss_name").get<string>(), string("loss"));
    ASSERT_EQ(lossJson.at("gradient_name").get<string>(), string("predictions_grad"));
    ASSERT_TRUE(lossJson.contains("loss_expression"));
    ASSERT_TRUE(lossJson.contains("gradient_expression"));
}

TEST(CustomLossApi, BuilderInfersNonDefaultExpressionNames) {
    Network network("custom_loss_infers_names");
    Tensor predictions(DataType::FP32, {5});
    Tensor labels(DataType::FP32, {5});

    CustomLoss customLoss = CustomLoss::Builder()
                                .network(network)
                                .lossExpression(makeSerializableSquaredErrorLossExpression("y_hat", "target", "mse"))
                                .gradientExpression(makeSerializableSquaredErrorGradientExpression("y_hat", "target", "y_hat_grad"))
                                .predictions(predictions)
                                .labels(labels)
                                .reportsRawLoss()
                                .build();

    ASSERT_EQ(customLoss.getPredictionsName(), string("y_hat"));
    ASSERT_EQ(customLoss.getLabelsName(), string("target"));
    ASSERT_EQ(customLoss.getLossName(), string("mse"));
    ASSERT_EQ(customLoss.getGradientName(), string("y_hat_grad"));
    ASSERT_EQ(customLoss.getLoss().getDimensions(), vector<uint64_t>({5}));
}

TEST(CustomLossApi, RejectsGradientDescriptorMismatch) {
    Network network("custom_loss_rejects_gradient_mismatch");
    Tensor predictions(DataType::FP16, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(CustomLoss::Builder()
                     .network(network)
                     .lossExpression(makeSerializableSquaredErrorLossExpression())
                     .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .reportsRawLoss()
                     .build(),
                 std::runtime_error);
}

TEST(CustomLossApi, RejectsSavingNonSerializableLossExpression) {
    Network network("custom_loss_rejects_nonserializable_expression");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomLoss customLoss = CustomLoss::Builder()
                                .network(network)
                                .lossExpression(makeNonSerializableSquaredErrorLossExpression())
                                .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                .predictions(predictions)
                                .labels(labels)
                                .reportsRawLoss()
                                .build();

    EXPECT_THROW(customLoss.architectureJson(), std::runtime_error);
}

TEST(BinaryCrossEntropyApi, PublicBuilderBacksRawLossWithCustomLoss) {
    Network network("bce_backed_by_custom_loss");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});

    BinaryCrossEntropy bce = BinaryCrossEntropy::Builder()
                                 .network(network)
                                 .predictions(predictions)
                                 .labels(labels)
                                 .reportsElementwiseLoss()
                                 .lossDataType(DataType::FP32)
                                 .build();

    ASSERT_TRUE(bce.isInitialized());
    ASSERT_EQ(bce.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(bce.getLoss().getDimensions(), vector<uint64_t>({1}));

    bool foundCustomLoss = false;
    bool foundRawBinaryCrossEntropy = false;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Layer> layer = network.getLayer(i);
        foundCustomLoss = foundCustomLoss || static_cast<bool>(dynamic_pointer_cast<CustomLoss>(layer));
        foundRawBinaryCrossEntropy = foundRawBinaryCrossEntropy || static_cast<bool>(dynamic_pointer_cast<BinaryCrossEntropy>(layer));
    }
    ASSERT_TRUE(foundCustomLoss);
    ASSERT_FALSE(foundRawBinaryCrossEntropy);
}


TEST(MAEApi, PublicBuilderBacksRawLossWithCustomLoss) {
    Network network("mae_backed_by_custom_loss");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    MAE mae = MAE::Builder()
                                .network(network)
                                .predictions(predictions)
                                .labels(labels)
                                .reportsRawLoss()
                                .lossDataType(DataType::FP32)
                                .build();

    ASSERT_TRUE(mae.isInitialized());
    ASSERT_EQ(mae.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(mae.getLoss().getDimensions(), vector<uint64_t>({3}));

    bool foundCustomLoss = false;
    bool foundRawMAE = false;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Layer> layer = network.getLayer(i);
        foundCustomLoss = foundCustomLoss || static_cast<bool>(dynamic_pointer_cast<CustomLoss>(layer));
        foundRawMAE = foundRawMAE || static_cast<bool>(dynamic_pointer_cast<MAE>(layer));
    }
    ASSERT_TRUE(foundCustomLoss);
    ASSERT_FALSE(foundRawMAE);
}

TEST(MSEApi, PublicBuilderBacksRawLossWithCustomLoss) {
    Network network("mse_backed_by_custom_loss");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    MSE mse = MSE::Builder()
                              .network(network)
                              .predictions(predictions)
                              .labels(labels)
                              .reportsRawLoss()
                              .lossDataType(DataType::FP32)
                              .build();

    ASSERT_TRUE(mse.isInitialized());
    ASSERT_EQ(mse.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(mse.getLoss().getDimensions(), vector<uint64_t>({3}));

    bool foundCustomLoss = false;
    bool foundRawMSE = false;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Layer> layer = network.getLayer(i);
        foundCustomLoss = foundCustomLoss || static_cast<bool>(dynamic_pointer_cast<CustomLoss>(layer));
        foundRawMSE = foundRawMSE || static_cast<bool>(dynamic_pointer_cast<MSE>(layer));
    }
    ASSERT_TRUE(foundCustomLoss);
    ASSERT_FALSE(foundRawMSE);
}
