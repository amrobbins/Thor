#include "DeepLearning/Api/BatchValidity.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {


std::filesystem::path makeUniqueCustomLossArchiveDir(const std::string& testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir =
        std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

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

Impl::DynamicExpression makeSerializableValidityAwareLossExpression() {
    Impl::Expression predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression validity =
        Impl::Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
    Impl::Expression validCount = validity.reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression diff = predictions - labels;
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"loss", diff * diff + validCount}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeSerializableValidityAwareGradientExpression() {
    Impl::Expression predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression validity =
        Impl::Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
    Impl::Expression validCount = validity.reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression gradient = ((predictions - labels) * validCount * Impl::Expression(Impl::Loss::getLossScalingFactor()))
                                    .withOutputDType(DataType::FP32);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"predictions_grad", gradient}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

vector<float> runValidityAwareRawLoss(bool multiInput) {
    constexpr uint32_t batchCapacity = 4;
    constexpr uint32_t validExampleCount = 2;
    Network network(multiInput ? "multi_input_validity_aware_loss" : "validity_aware_loss");
    NetworkInput predictions = NetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .dimensions({1})
                                   .dataType(DataType::FP32)
                                   .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();

    Tensor rawLoss;
    if (multiInput) {
        MultiInputCustomLoss loss = MultiInputCustomLoss::Builder()
                                        .network(network)
                                        .lossExpression(makeSerializableValidityAwareLossExpression())
                                        .gradientExpression(makeSerializableValidityAwareGradientExpression())
                                        .input("predictions", predictions.getFeatureOutput().value(), "predictions_grad")
                                        .auxiliaryInput("labels", labels.getFeatureOutput().value())
                                        .usesBatchValidity()
                                        .reportsRawLoss()
                                        .build();
        EXPECT_TRUE(loss.usesBatchValidity());
        EXPECT_TRUE(loss.architectureJson().at("uses_batch_validity").get<bool>());
        EXPECT_FALSE(loss.architectureJson().contains("uses_batch_validity_mask"));
        rawLoss = loss.getRawLoss();
    } else {
        CustomLoss loss = CustomLoss::Builder()
                              .network(network)
                              .lossExpression(makeSerializableValidityAwareLossExpression())
                              .gradientExpression(makeSerializableValidityAwareGradientExpression())
                              .predictions(predictions.getFeatureOutput().value())
                              .labels(labels.getFeatureOutput().value())
                              .usesBatchValidity()
                              .reportsRawLoss()
                              .build();
        EXPECT_TRUE(loss.usesBatchValidity());
        EXPECT_EQ(loss.getPredictionsName(), "predictions");
        EXPECT_EQ(loss.getLabelsName(), "labels");
        EXPECT_TRUE(loss.architectureJson().at("uses_batch_validity").get<bool>());
        EXPECT_FALSE(loss.architectureJson().contains("uses_batch_validity_mask"));
        rawLoss = loss.getRawLoss();
    }
    NetworkOutput::Builder()
        .network(network)
        .name("raw_loss")
        .inputTensor(rawLoss)
        .dataType(DataType::FP32)
        .build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchCapacity, initializationDone, /*inferenceOnly=*/true);
    EXPECT_NE(placed, nullptr);
    for (Event& event : initializationDone)
        event.synchronize();

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor predictionsCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
    Impl::Tensor labelsCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
    const vector<float> predictionValues{1.0f, 3.0f, 100.0f, 200.0f};
    std::copy(predictionValues.begin(), predictionValues.end(), predictionsCpu.getMemPtr<float>());
    std::fill(labelsCpu.getMemPtr<float>(), labelsCpu.getMemPtr<float>() + batchCapacity, 0.0f);

    Batch batch;
    batch.insert("predictions", predictionsCpu);
    batch.insert("labels", labelsCpu);
    batch.setValidExampleCount(validExampleCount);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/true);
    done.synchronize();
    outputReadyEvents.at("raw_loss").synchronize();
    placed->synchronize();

    Impl::Tensor outputCpu = outputs.at("raw_loss").clone(cpuPlacement);
    Stream downloadStream = Stream::getNextDownloadStream(0);
    outputCpu.copyFromAsync(outputs.at("raw_loss"), downloadStream);
    downloadStream.synchronize();
    const float* outputValues = outputCpu.getMemPtr<float>();
    return vector<float>(outputValues, outputValues + batchCapacity);
}

float runValidityAwareGradientUpdate(bool multiInput) {
    constexpr uint32_t batchCapacity = 4;
    constexpr uint32_t validExampleCount = 2;
    constexpr float learningRate = 0.1f;

    Network network(multiInput ? "multi_input_validity_aware_gradient" : "validity_aware_gradient");
    NetworkInput features = NetworkInput::Builder()
                                .network(network)
                                .name("features")
                                .dimensions({1})
                                .dataType(DataType::FP32)
                                .build();
    shared_ptr<Sgd> optimizer = Sgd::Builder()
                                    .initialLearningRate(learningRate)
                                    .decay(0.0f)
                                    .momentum(0.0f)
                                    .build();
    FullyConnected prediction = FullyConnected::Builder()
                                    .network(network)
                                    .featureInput(features.getFeatureOutput().value())
                                    .numOutputFeatures(1)
                                    .hasBias(false)
                                    .weightsDataType(DataType::FP32)
                                    .computeDataType(DataType::FP32)
                                    .outputDataType(DataType::FP32)
                                    .weightsOptimizer(optimizer)
                                    .noActivation()
                                    .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();

    Tensor reportedLoss;
    if (multiInput) {
        MultiInputCustomLoss loss = MultiInputCustomLoss::Builder()
                                        .network(network)
                                        .lossExpression(makeSerializableValidityAwareLossExpression())
                                        .gradientExpression(makeSerializableValidityAwareGradientExpression())
                                        .input("predictions", prediction.getFeatureOutput().value(), "predictions_grad")
                                        .auxiliaryInput("labels", labels.getFeatureOutput().value())
                                        .usesBatchValidity()
                                        .reportsBatchLoss()
                                        .build();
        reportedLoss = loss.getLoss();
    } else {
        CustomLoss loss = CustomLoss::Builder()
                              .network(network)
                              .lossExpression(makeSerializableValidityAwareLossExpression())
                              .gradientExpression(makeSerializableValidityAwareGradientExpression())
                              .predictions(prediction.getFeatureOutput().value())
                              .labels(labels.getFeatureOutput().value())
                              .usesBatchValidity()
                              .reportsBatchLoss()
                              .build();
        reportedLoss = loss.getLoss();
    }
    NetworkOutput::Builder()
        .network(network)
        .name("loss")
        .inputTensor(reportedLoss)
        .dataType(DataType::FP32)
        .build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchCapacity, initializationDone, /*inferenceOnly=*/false);
    if (placed == nullptr)
        throw runtime_error("Validity-aware gradient test failed to place the network.");
    for (Event& event : initializationDone)
        event.synchronize();

    Impl::StampedNetwork& stampedNetwork = placed->getStampedNetwork(0);
    auto physicalPrediction = dynamic_pointer_cast<Impl::CustomLayer>(
        stampedNetwork.getPhysicalLayerFromApiLayer(prediction.getId()));
    if (physicalPrediction == nullptr)
        throw runtime_error("Validity-aware gradient test could not find the physical prediction layer.");
    shared_ptr<Impl::PhysicalParameter> weightsParameter = physicalPrediction->getParameter("weights");
    if (weightsParameter == nullptr || !weightsParameter->getStorage().has_value())
        throw runtime_error("Validity-aware gradient test could not access prediction weights.");

    Impl::Tensor weights = weightsParameter->getStorage().value();
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor weightsCpu = weights.clone(cpuPlacement);
    weightsCpu.getMemPtr<float>()[0] = 1.0f;
    Stream parameterStream = physicalPrediction->getStreams().front();
    weights.copyFromAsync(weightsCpu, parameterStream);
    parameterStream.synchronize();

    Impl::Tensor featuresCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
    Impl::Tensor labelsCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
    std::fill(featuresCpu.getMemPtr<float>(), featuresCpu.getMemPtr<float>() + batchCapacity, 1.0f);
    std::fill(labelsCpu.getMemPtr<float>(), labelsCpu.getMemPtr<float>() + batchCapacity, 0.0f);

    Batch batch;
    batch.insert("features", featuresCpu);
    batch.insert("labels", labelsCpu);
    batch.setValidExampleCount(validExampleCount);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    Stream downloadStream = Stream::getNextDownloadStream(0);
    weightsCpu.copyFromAsync(weights, downloadStream);
    downloadStream.synchronize();
    return weightsCpu.getMemPtr<float>()[0];
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

TEST(CustomLossApi, NoReportingKeepsRawLossButGetLossThrows) {
    Network network("custom_loss_no_report");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomLoss customLoss = CustomLoss::Builder()
                                .network(network)
                                .lossExpression(makeSerializableSquaredErrorLossExpression())
                                .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                .predictions(predictions)
                                .labels(labels)
                                .reportsNoLoss()
                                .build();

    EXPECT_FALSE(customLoss.reportsLoss());
    EXPECT_EQ(customLoss.getRawLoss().getDimensions(), vector<uint64_t>({3}));
    EXPECT_THROW((void)customLoss.getLoss(), std::runtime_error);
}

TEST(MultiInputCustomLossApi, NoReportingSerializesPhysicalLossAsRaw) {
    Network network("multi_input_custom_loss_no_report");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    MultiInputCustomLoss customLoss = MultiInputCustomLoss::Builder()
                                          .network(network)
                                          .lossExpression(makeSerializableSquaredErrorLossExpression())
                                          .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                          .input("predictions", predictions, "predictions_grad")
                                          .auxiliaryInput("labels", labels)
                                          .reportsNoLoss()
                                          .build();

    EXPECT_FALSE(customLoss.reportsLoss());
    EXPECT_THROW((void)customLoss.getLoss(), std::runtime_error);
    EXPECT_EQ(customLoss.architectureJson().at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
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
    Tensor predictions(DataType::FP32, {100});
    Tensor labels(DataType::FP32, {100});

    BinaryCrossEntropy bce = BinaryCrossEntropy::Builder()
                                 .network(network)
                                 .predictions(predictions)
                                 .labels(labels)
                                 .reportsRawLoss()
                                 .lossDataType(DataType::FP32)
                                 .build();

    ASSERT_TRUE(bce.isInitialized());
    ASSERT_EQ(bce.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(bce.getLoss().getDimensions(), vector<uint64_t>({100}));

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
    Tensor predictions(DataType::FP32, {100});
    Tensor labels(DataType::FP32, {100});

    MAE mae = MAE::Builder()
                                .network(network)
                                .predictions(predictions)
                                .labels(labels)
                                .reportsRawLoss()
                                .lossDataType(DataType::FP32)
                                .build();

    ASSERT_TRUE(mae.isInitialized());
    ASSERT_EQ(mae.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(mae.getLoss().getDimensions(), vector<uint64_t>({100}));

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
    Tensor predictions(DataType::FP32, {100});
    Tensor labels(DataType::FP32, {100});

    MSE mse = MSE::Builder()
                              .network(network)
                              .predictions(predictions)
                              .labels(labels)
                              .reportsRawLoss()
                              .lossDataType(DataType::FP32)
                              .build();

    ASSERT_TRUE(mse.isInitialized());
    ASSERT_EQ(mse.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(mse.getLoss().getDimensions(), vector<uint64_t>({100}));

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

TEST(CustomLossApi, PartialBatchMasksFusedGradientAndNormalizesUpdateByValidExamples) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLoss partial-batch execution test requires a GPU";

    constexpr uint32_t batchCapacity = 4;
    constexpr uint32_t validExampleCount = 2;
    constexpr float learningRate = 0.1f;

    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .initialLearningRate(learningRate)
                              .decay(0.0f)
                              .momentum(0.0f)
                              .build();

    Network network("custom_loss_partial_batch_fused_gradient");
    NetworkInput features = NetworkInput::Builder()
                                .network(network)
                                .name("features")
                                .dimensions({1})
                                .dataType(DataType::FP32)
                                .build();
    FullyConnected fc = FullyConnected::Builder()
                            .network(network)
                            .featureInput(features.getFeatureOutput().value())
                            .numOutputFeatures(2)
                            .hasBias(false)
                            .weightsDataType(DataType::FP32)
                            .computeDataType(DataType::FP32)
                            .outputDataType(DataType::FP32)
                            .weightsOptimizer(sgd)
                            .noActivation()
                            .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();
    MSE mse = MSE::Builder()
                  .network(network)
                  .predictions(fc.getFeatureOutput().value())
                  .labels(labels.getFeatureOutput().value())
                  .lossDataType(DataType::FP32)
                  .reportsBatchLoss()
                  .build();
    NetworkOutput lossOutput = NetworkOutput::Builder()
                                   .network(network)
                                   .name("loss")
                                   .inputTensor(mse.getLoss())
                                   .dataType(DataType::FP32)
                                   .build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchCapacity, initializationDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone)
        event.synchronize();

    Impl::StampedNetwork& stampedNetwork = placed->getStampedNetwork(0);
    auto physicalFc = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(fc.getId()));
    ASSERT_NE(physicalFc, nullptr);
    ASSERT_EQ(physicalFc->getNumFusedCustomLossGradients(), 1u);

    shared_ptr<Impl::PhysicalParameter> weightsParameter = physicalFc->getParameter("weights");
    ASSERT_NE(weightsParameter, nullptr);
    ASSERT_TRUE(weightsParameter->getStorage().has_value());
    Impl::Tensor weights = weightsParameter->getStorage().value();
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor weightsCpu = weights.clone(cpuPlacement);
    weightsCpu.getMemPtr<float>()[0] = 1.0f;
    weightsCpu.getMemPtr<float>()[1] = 2.0f;
    Stream parameterStream = physicalFc->getStreams().front();
    weights.copyFromAsync(weightsCpu, parameterStream);
    parameterStream.synchronize();

    Impl::Tensor featuresCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
    Impl::Tensor labelsCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 2}));
    const float featureValues[] = {1.0f, 2.0f, 100.0f, 200.0f};
    for (uint32_t i = 0; i < batchCapacity; ++i) {
        featuresCpu.getMemPtr<float>()[i] = featureValues[i];
        labelsCpu.getMemPtr<float>()[2 * i] = 0.0f;
        labelsCpu.getMemPtr<float>()[2 * i + 1] = 0.0f;
    }

    Batch batch;
    batch.insert("features", featuresCpu);
    batch.insert("labels", labelsCpu);
    batch.setValidExampleCount(validExampleCount);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    Impl::Tensor lossCpu = outputs.at("loss").clone(cpuPlacement);
    Stream downloadStream = Stream::getNextDownloadStream(0);
    lossCpu.copyFromAsync(outputs.at("loss"), downloadStream);
    weightsCpu.copyFromAsync(weights, downloadStream);
    downloadStream.synchronize();

    EXPECT_FLOAT_EQ(lossCpu.getMemPtr<float>()[0], 12.5f);
    EXPECT_NEAR(weightsCpu.getMemPtr<float>()[0], 0.5f, 1.0e-5f);
    EXPECT_NEAR(weightsCpu.getMemPtr<float>()[1], 1.0f, 1.0e-5f);
}


TEST(CustomLossApi, OptionalValidityMaskIsAvailableInsideLossExpression) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLoss validity-mask execution test requires a GPU";
    EXPECT_EQ(runValidityAwareRawLoss(/*multiInput=*/false), (vector<float>{3.0f, 11.0f, 0.0f, 0.0f}));
}

TEST(MultiInputCustomLossApi, OptionalValidityMaskIsAvailableInsideLossExpression) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "MultiInputCustomLoss validity-mask execution test requires a GPU";
    EXPECT_EQ(runValidityAwareRawLoss(/*multiInput=*/true), (vector<float>{3.0f, 11.0f, 0.0f, 0.0f}));
}


TEST(CustomLossApi, OptionalValidityMaskIsAvailableInsideFusedGradientExpression) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLoss validity-mask gradient test requires a GPU";
    EXPECT_NEAR(runValidityAwareGradientUpdate(/*multiInput=*/false), 0.8f, 1.0e-5f);
}

TEST(MultiInputCustomLossApi, OptionalValidityMaskIsAvailableInsideGradientExpression) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "MultiInputCustomLoss validity-mask gradient test requires a GPU";
    EXPECT_NEAR(runValidityAwareGradientUpdate(/*multiInput=*/true), 0.8f, 1.0e-5f);
}

TEST(CustomLossApi, BatchValidityDeclarationMustMatchBothExpressionContracts) {
    Network network("custom_loss_validity_mask_contract");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});

    EXPECT_THROW((void)CustomLoss::Builder()
                     .network(network)
                     .lossExpression(makeSerializableValidityAwareLossExpression())
                     .gradientExpression(makeSerializableValidityAwareGradientExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .reportsRawLoss()
                     .build(),
                 std::runtime_error);
    EXPECT_THROW((void)CustomLoss::Builder()
                     .network(network)
                     .lossExpression(makeSerializableSquaredErrorLossExpression())
                     .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .usesBatchValidity()
                     .reportsRawLoss()
                     .build(),
                 std::runtime_error);
}

TEST(MultiInputCustomLossApi, BatchValidityDeclarationMustMatchBothExpressionContracts) {
    Network network("multi_input_custom_loss_validity_mask_contract");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});

    EXPECT_THROW((void)MultiInputCustomLoss::Builder()
                     .network(network)
                     .lossExpression(makeSerializableValidityAwareLossExpression())
                     .gradientExpression(makeSerializableValidityAwareGradientExpression())
                     .input("predictions", predictions, "predictions_grad")
                     .auxiliaryInput("labels", labels)
                     .reportsRawLoss()
                     .build(),
                 std::runtime_error);
    EXPECT_THROW((void)MultiInputCustomLoss::Builder()
                     .network(network)
                     .lossExpression(makeSerializableSquaredErrorLossExpression())
                     .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                     .input("predictions", predictions, "predictions_grad")
                     .auxiliaryInput("labels", labels)
                     .usesBatchValidity()
                     .reportsRawLoss()
                     .build(),
                 std::runtime_error);
}

TEST(CustomLossApi, FullBatchRequirementIsSerializedAndPropagatedToPhysicalLoss) {
    Network network("custom_loss_full_batch_requirement");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});
    CustomLoss loss = CustomLoss::Builder()
                          .network(network)
                          .lossExpression(makeSerializableSquaredErrorLossExpression())
                          .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                          .predictions(predictions)
                          .labels(labels)
                          .requiresFullBatch()
                          .reportsRawLoss()
                          .build();
    EXPECT_TRUE(loss.requiresFullBatch());
    EXPECT_TRUE(loss.architectureJson().at("requires_full_batch").get<bool>());

    Impl::CustomLoss physical(makeSerializableSquaredErrorLossExpression(),
                              makeSerializableSquaredErrorGradientExpression(),
                              "predictions",
                              "labels",
                              "loss",
                              "predictions_grad",
                              DataType::FP32,
                              std::nullopt,
                              /*usesBatchValidity=*/false,
                              /*requiresFullBatch=*/true);
    EXPECT_FALSE(physical.supportsPartialBatches());
}

TEST(MultiInputCustomLossApi, FullBatchRequirementIsSerializedAndPropagatedToPhysicalLoss) {
    Network network("multi_input_custom_loss_full_batch_requirement");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});
    MultiInputCustomLoss loss = MultiInputCustomLoss::Builder()
                                    .network(network)
                                    .lossExpression(makeSerializableSquaredErrorLossExpression())
                                    .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                    .input("predictions", predictions, "predictions_grad")
                                    .auxiliaryInput("labels", labels)
                                    .requiresFullBatch()
                                    .reportsRawLoss()
                                    .build();
    EXPECT_TRUE(loss.requiresFullBatch());
    EXPECT_TRUE(loss.architectureJson().at("requires_full_batch").get<bool>());

    Impl::MultiInputCustomLoss physical(makeSerializableSquaredErrorLossExpression(),
                                        makeSerializableSquaredErrorGradientExpression(),
                                        {"predictions", "labels"},
                                        {std::optional<std::string>("predictions_grad"), std::nullopt},
                                        "loss",
                                        DataType::FP32,
                                        std::nullopt,
                                        /*usesBatchValidity=*/false,
                                        /*requiresFullBatch=*/true);
    EXPECT_FALSE(physical.supportsPartialBatches());
}

TEST(CustomLossApi, BatchValidityAndFullBatchRequirementAreMutuallyExclusive) {
    Network network("custom_loss_conflicting_partial_batch_contract");
    Tensor predictions(DataType::FP32, {1});
    Tensor labels(DataType::FP32, {1});

    EXPECT_ANY_THROW((void)CustomLoss::Builder()
                         .network(network)
                         .lossExpression(makeSerializableValidityAwareLossExpression())
                         .gradientExpression(makeSerializableValidityAwareGradientExpression())
                         .predictions(predictions)
                         .labels(labels)
                         .usesBatchValidity()
                         .requiresFullBatch());
    EXPECT_ANY_THROW((void)MultiInputCustomLoss::Builder()
                         .network(network)
                         .lossExpression(makeSerializableSquaredErrorLossExpression())
                         .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                         .input("predictions", predictions, "predictions_grad")
                         .auxiliaryInput("labels", labels)
                         .requiresFullBatch()
                         .usesBatchValidity());
}

TEST(CustomLossApi, FullBatchRequirementsSurviveNetworkSaveLoad) {
    const std::string networkName = "custom_loss_full_batch_round_trip";
    const std::filesystem::path archiveDir = makeUniqueCustomLossArchiveDir(networkName);

    try {
        Network network(networkName);
        NetworkInput predictions = NetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .dimensions({1})
                                       .dataType(DataType::FP32)
                                       .build();
        NetworkInput labels = NetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .dimensions({1})
                                  .dataType(DataType::FP32)
                                  .build();

        CustomLoss customLoss = CustomLoss::Builder()
                                    .network(network)
                                    .lossExpression(makeSerializableSquaredErrorLossExpression())
                                    .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                    .predictions(predictions.getFeatureOutput().value())
                                    .labels(labels.getFeatureOutput().value())
                                    .requiresFullBatch()
                                    .reportsRawLoss()
                                    .build();
        NetworkOutput::Builder()
            .network(network)
            .name("custom_raw_loss")
            .inputTensor(customLoss.getRawLoss())
            .dataType(DataType::FP32)
            .build();

        MultiInputCustomLoss multiInputLoss = MultiInputCustomLoss::Builder()
                                                   .network(network)
                                                   .lossExpression(makeSerializableSquaredErrorLossExpression())
                                                   .gradientExpression(makeSerializableSquaredErrorGradientExpression())
                                                   .input("predictions",
                                                          predictions.getFeatureOutput().value(),
                                                          "predictions_grad")
                                                   .auxiliaryInput("labels", labels.getFeatureOutput().value())
                                                   .requiresFullBatch()
                                                   .reportsRawLoss()
                                                   .build();
        NetworkOutput::Builder()
            .network(network)
            .name("multi_input_raw_loss")
            .inputTensor(multiInputLoss.getRawLoss())
            .dataType(DataType::FP32)
            .build();

        network.save(archiveDir.string(), true);

        Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        std::shared_ptr<CustomLoss> loadedCustomLoss;
        std::shared_ptr<MultiInputCustomLoss> loadedMultiInputCustomLoss;
        for (uint32_t i = 0; i < loadedNetwork.getNumLayers(); ++i) {
            if (auto layer = std::dynamic_pointer_cast<CustomLoss>(loadedNetwork.getLayer(i)); layer != nullptr) {
                ASSERT_EQ(loadedCustomLoss, nullptr);
                loadedCustomLoss = std::move(layer);
            }
            if (auto layer = std::dynamic_pointer_cast<MultiInputCustomLoss>(loadedNetwork.getLayer(i)); layer != nullptr) {
                ASSERT_EQ(loadedMultiInputCustomLoss, nullptr);
                loadedMultiInputCustomLoss = std::move(layer);
            }
        }

        ASSERT_NE(loadedCustomLoss, nullptr);
        ASSERT_NE(loadedMultiInputCustomLoss, nullptr);
        EXPECT_TRUE(loadedCustomLoss->requiresFullBatch());
        EXPECT_TRUE(loadedMultiInputCustomLoss->requiresFullBatch());
        EXPECT_TRUE(loadedCustomLoss->architectureJson().at("requires_full_batch").get<bool>());
        EXPECT_TRUE(loadedMultiInputCustomLoss->architectureJson().at("requires_full_batch").get<bool>());
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
