#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/DeepLearning/Api/Layers/Loss/LossNumericalTestTolerance.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint32_t kBatchSize = 2;
constexpr uint32_t kWidth = 100;
constexpr double kFiniteDifferenceEpsilon = 1.0e-3;

Impl::Tensor makeCpuTensor(const vector<float>& values) {
    THOR_THROW_IF_FALSE(values.size() == static_cast<size_t>(kBatchSize * kWidth));
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32,
                                      {static_cast<unsigned long>(kBatchSize), static_cast<unsigned long>(kWidth)});
    Impl::Tensor tensor(cpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.getMemPtr()));
    return tensor;
}

vector<float> tensorToVector(const Impl::Tensor& tensor) {
    const float* begin = static_cast<const float*>(tensor.getMemPtr());
    return vector<float>(begin, begin + tensor.getTotalNumElements());
}

vector<float> copyGpuTensorToVector(const Impl::Tensor& tensor, Stream stream) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpu(cpuPlacement, tensor.getDescriptor());
    cpu.copyFromAsync(tensor, stream);
    stream.synchronize();
    return tensorToVector(cpu);
}

shared_ptr<Api::CustomLoss> findRawCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::CustomLoss> customLoss = dynamic_pointer_cast<Api::CustomLoss>(layer);
        if (customLoss != nullptr)
            return customLoss;
    }
    return nullptr;
}

struct RawLossRunResult {
    vector<float> loss;
    vector<float> predictionGradient;
};

struct PlacedRawLossNetwork {
    Api::Network network;
    Api::NetworkInput predictionsInput;
    Api::NetworkInput labelsInput;
    Api::GradientRivet predictionsRivet;
    Api::NetworkOutput lossOutput;
    shared_ptr<Api::PlacedNetwork> placedNetwork;

    PlacedRawLossNetwork(const string& name,
                         const function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor)>& buildLoss)
        : network(name),
          predictionsInput(Api::NetworkInput::Builder()
                               .network(network)
                               .name("predictions")
                               .dimensions({kWidth})
                               .dataType(Api::DataType::FP32)
                               .build()),
          labelsInput(Api::NetworkInput::Builder()
                          .network(network)
                          .name("labels")
                          .dimensions({kWidth})
                          .dataType(Api::DataType::FP32)
                          .build()),
          predictionsRivet(Api::GradientRivet::Builder()
                               .network(network)
                               .tensor(predictionsInput.getFeatureOutput().value())
                               .build()),
          lossOutput([this, &buildLoss]() {
              Api::Tensor lossTensor = buildLoss(network, predictionsRivet.getFeatureOutput().value(), labelsInput.getFeatureOutput().value());
              return Api::NetworkOutput::Builder()
                  .network(network)
                  .name("loss")
                  .inputTensor(lossTensor)
                  .dataType(Api::DataType::FP32)
                  .build();
          }()) {
        vector<Event> initDoneEvents;
        placedNetwork = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
        THOR_THROW_IF_FALSE(placedNetwork != nullptr);
        Stream stream(0);
        for (Event& event : initDoneEvents)
            stream.waitEvent(event);
        stream.synchronize();
    }
};

RawLossRunResult runPlacedRawLossNetwork(PlacedRawLossNetwork& rawLossNetwork,
                                         uint64_t rawLossApiLayerId,
                                         const vector<float>& predictions,
                                         const vector<float>& labels) {
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(kBatchSize * kWidth));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Impl::StampedNetwork& stampedNetwork = rawLossNetwork.placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalPredictionsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(rawLossNetwork.predictionsInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalLabelsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(rawLossNetwork.labelsInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(rawLossNetwork.lossOutput.getId()));
    shared_ptr<Impl::Layer> physicalRawLoss = stampedNetwork.getPhysicalLayerFromApiLayer(rawLossApiLayerId);
    THOR_THROW_IF_FALSE(physicalPredictionsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::Tensor predictionsCpu = makeCpuTensor(predictions);
    Impl::Tensor labelsCpu = makeCpuTensor(labels);
    physicalPredictionsInput->forward(predictionsCpu, false, kBatchSize);
    physicalLabelsInput->forward(labelsCpu, false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss = tensorToVector(outputLossCpu);

    optional<Impl::Tensor> errorOutputGpu = physicalRawLoss->getErrorOutput();
    THOR_THROW_IF_FALSE(errorOutputGpu.has_value());
    vector<float> outputGradient = copyGpuTensorToVector(errorOutputGpu.value(), physicalRawLoss->getStream());
    return {outputLoss, outputGradient};
}

RawLossRunResult runCustomBackedRawLossNetwork(const string& networkName,
                                               const vector<float>& predictions,
                                               const vector<float>& labels,
                                               const function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor)>& buildLoss) {
    PlacedRawLossNetwork rawLossNetwork(networkName, buildLoss);
    shared_ptr<Api::CustomLoss> rawCustomLoss = findRawCustomLoss(rawLossNetwork.network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);
    return runPlacedRawLossNetwork(rawLossNetwork, rawCustomLoss->getId(), predictions, labels);
}

RawLossRunResult runMapeRawLossNetwork(const vector<float>& predictions, const vector<float>& labels) {
    uint64_t mapeLayerId = 0;
    PlacedRawLossNetwork rawLossNetwork(
        "mape_width_100_gradient_numerical",
        [&](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::MAPE loss = Api::MAPE::Builder()
                                 .network(network)
                                 .predictions(predictionsTensor)
                                 .labels(labelsTensor)
                                 .lossDataType(Api::DataType::FP32)
                                 .reportsRawLoss()
                                 .build();
            mapeLayerId = loss.getId();
            return loss.getLoss();
        });
    THOR_THROW_IF_FALSE(mapeLayerId != 0);
    return runPlacedRawLossNetwork(rawLossNetwork, mapeLayerId, predictions, labels);
}


struct SharedPredictionDualLossRunResult {
    vector<float> maeLoss;
    vector<float> quantileLoss;
    vector<float> predictionGradient;
};

SharedPredictionDualLossRunResult runSharedPredictionMaeAndQuantileLossNetwork(const vector<float>& predictions,
                                                                               const vector<float>& labels,
                                                                               float maeWeight,
                                                                               float quantileWeight,
                                                                               float quantile) {
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(kBatchSize * kWidth));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network("shared_prediction_dual_loss_gradient_numerical");
    Api::NetworkInput predictionsInput = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("predictions")
                                             .dimensions({kWidth})
                                             .dataType(Api::DataType::FP32)
                                             .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({kWidth})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::GradientRivet predictionsRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(predictionsInput.getFeatureOutput().value())
                                             .build();

    Api::Tensor sharedPredictions = predictionsRivet.getFeatureOutput().value();
    Api::Tensor sharedLabels = labelsInput.getFeatureOutput().value();
    Api::MAE maeLoss = Api::MAE::Builder()
                           .network(network)
                           .predictions(sharedPredictions)
                           .labels(sharedLabels)
                           .lossDataType(Api::DataType::FP32)
                           .lossWeight(maeWeight)
                           .reportsRawLoss()
                           .build();
    Api::QuantileLoss quantileLoss = Api::QuantileLoss::Builder()
                                         .network(network)
                                         .predictions(sharedPredictions)
                                         .labels(sharedLabels)
                                         .quantile(quantile)
                                         .lossDataType(Api::DataType::FP32)
                                         .lossWeight(quantileWeight)
                                         .reportsRawLoss()
                                         .build();

    Api::NetworkOutput maeOutput = Api::NetworkOutput::Builder()
                                       .network(network)
                                       .name("mae_loss")
                                       .inputTensor(maeLoss.getLoss())
                                       .dataType(Api::DataType::FP32)
                                       .build();
    Api::NetworkOutput quantileOutput = Api::NetworkOutput::Builder()
                                            .network(network)
                                            .name("quantile_loss")
                                            .inputTensor(quantileLoss.getLoss())
                                            .dataType(Api::DataType::FP32)
                                            .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    Stream stream(0);
    for (Event& event : initDoneEvents)
        stream.waitEvent(event);
    stream.synchronize();

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalPredictionsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(predictionsInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalLabelsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(labelsInput.getId()));
    shared_ptr<Impl::GradientRivet> physicalPredictionsRivet =
        dynamic_pointer_cast<Impl::GradientRivet>(stampedNetwork.getPhysicalLayerFromApiLayer(predictionsRivet.getId()));
    shared_ptr<Impl::NetworkOutput> physicalMaeOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(maeOutput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalQuantileOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(quantileOutput.getId()));
    THOR_THROW_IF_FALSE(physicalPredictionsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalPredictionsRivet != nullptr);
    THOR_THROW_IF_FALSE(physicalMaeOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalQuantileOutput != nullptr);

    Impl::Tensor predictionsCpu = makeCpuTensor(predictions);
    Impl::Tensor labelsCpu = makeCpuTensor(labels);
    physicalPredictionsInput->forward(predictionsCpu, false, kBatchSize);
    physicalLabelsInput->forward(labelsCpu, false, kBatchSize);

    Stream maeStream = physicalMaeOutput->getStream();
    Stream quantileStream = physicalQuantileOutput->getStream();
    maeStream.waitEvent(physicalMaeOutput->getOutputReadyEvent());
    quantileStream.waitEvent(physicalQuantileOutput->getOutputReadyEvent());
    maeStream.synchronize();
    quantileStream.synchronize();
    physicalPredictionsRivet->getStream().synchronize();

    Impl::Tensor maeLossCpu = physicalMaeOutput->getFeatureOutput().value();
    Impl::Tensor quantileLossCpu = physicalQuantileOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(maeLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(quantileLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);

    optional<Impl::Tensor> summedPredictionGradientGpu = physicalPredictionsRivet->getErrorInput();
    THOR_THROW_IF_FALSE(summedPredictionGradientGpu.has_value());
    vector<float> summedPredictionGradient = copyGpuTensorToVector(summedPredictionGradientGpu.value(), physicalPredictionsRivet->getStream());
    return {tensorToVector(maeLossCpu), tensorToVector(quantileLossCpu), summedPredictionGradient};
}

vector<float> makePredictions() {
    vector<float> values(kBatchSize * kWidth, 0.0f);
    for (size_t i = 0; i < values.size(); ++i) {
        const float base = 0.25f + static_cast<float>(i % 17) * 0.07f;
        const float signOffset = (i % 2 == 0) ? 0.19f : -0.23f;
        values[i] = base + signOffset;
    }
    return values;
}

vector<float> makeLabels() {
    vector<float> values(kBatchSize * kWidth, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = 1.5f + static_cast<float>(i % 19) * 0.11f;
    return values;
}

vector<float> makeBinaryLabels() {
    vector<float> values(kBatchSize * kWidth, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    return values;
}

vector<float> referenceMaeRawLoss(const vector<float>& predictions, const vector<float>& labels) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        loss[i] = std::fabs(predictions[i] - labels[i]);
    return loss;
}

vector<float> referenceMseRawLoss(const vector<float>& predictions, const vector<float>& labels) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float diff = predictions[i] - labels[i];
        loss[i] = diff * diff;
    }
    return loss;
}

vector<float> referenceExpectileRawLoss(const vector<float>& predictions, const vector<float>& labels, float expectile) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float error = labels[i] - predictions[i];
        const float weight = error > 0.0f ? 2.0f * expectile : 2.0f * (1.0f - expectile);
        loss[i] = weight * error * error;
    }
    return loss;
}

vector<float> referenceAsymmetricPowerRawLoss(const vector<float>& predictions,
                                               const vector<float>& labels,
                                               float level,
                                               float exponent) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float error = labels[i] - predictions[i];
        const float weight = error > 0.0f ? 2.0f * level : 2.0f * (1.0f - level);
        if (exponent == 1.0f)
            loss[i] = weight * std::fabs(error);
        else if (exponent == 2.0f)
            loss[i] = weight * error * error;
        else
            loss[i] = weight * std::pow(std::fabs(error), exponent);
    }
    return loss;
}

vector<float> referenceBceRawLoss(const vector<float>& logits, const vector<float>& labels) {
    THOR_THROW_IF_FALSE(logits.size() == labels.size());
    vector<float> loss(logits.size(), 0.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        const double logit = logits[i];
        const double label = labels[i];
        loss[i] = static_cast<float>(std::max(logit, 0.0) - (logit * label) + std::log1p(std::exp(-std::fabs(logit))));
    }
    return loss;
}

vector<float> referenceMapeRawLoss(const vector<float>& predictions, const vector<float>& labels) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        loss[i] = std::fabs((labels[i] - predictions[i]) / labels[i]) * 100.0f;
    return loss;
}

vector<float> makeQuantilePredictions() {
    vector<float> values(kBatchSize * kWidth, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = -1.25f + static_cast<float>(i % 23) * 0.13f;
    return values;
}

vector<float> makeQuantileLabels() {
    vector<float> values(kBatchSize * kWidth, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = -0.783f + static_cast<float>((i * 7) % 29) * 0.09f;
    return values;
}

vector<float> referenceQuantileRawLoss(const vector<float>& predictions, const vector<float>& labels, float quantile) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float error = labels[i] - predictions[i];
        loss[i] = (error > 0.0f) ? quantile * error : (quantile - 1.0f) * error;
    }
    return loss;
}


vector<float> scaleValues(const vector<float>& values, float scale) {
    vector<float> scaled = values;
    for (float& value : scaled)
        value *= scale;
    return scaled;
}


double sumAsDouble(const vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0, [](double total, float value) { return total + static_cast<double>(value); });
}

vector<float> finiteDifferenceGradient(const vector<float>& predictions, const function<double(const vector<float>&)>& totalLoss) {
    vector<float> gradient(predictions.size(), 0.0f);
    vector<float> perturbed = predictions;
    for (size_t i = 0; i < predictions.size(); ++i) {
        perturbed[i] = predictions[i] + static_cast<float>(kFiniteDifferenceEpsilon);
        const double lossPlus = totalLoss(perturbed);
        perturbed[i] = predictions[i] - static_cast<float>(kFiniteDifferenceEpsilon);
        const double lossMinus = totalLoss(perturbed);
        perturbed[i] = predictions[i];
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

void scaleByLossScalingFactor(vector<float>& values) {
    const float scale = Impl::Loss::getLossScalingFactor();
    for (float& value : values)
        value *= scale;
}

vector<float> referenceCombinedMaeQuantilePredictionGradient(const vector<float>& predictions,
                                                             const vector<float>& labels,
                                                             float maeWeight,
                                                             float quantileWeight,
                                                             float quantile) {
    vector<float> gradient = finiteDifferenceGradient(predictions, [&](const vector<float>& perturbed) {
        const double mae = sumAsDouble(referenceMaeRawLoss(perturbed, labels));
        const double quantileLoss = sumAsDouble(referenceQuantileRawLoss(perturbed, labels, quantile));
        return static_cast<double>(maeWeight) * mae + static_cast<double>(quantileWeight) * quantileLoss;
    });
    scaleByLossScalingFactor(gradient);
    return gradient;
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(RegressionVectorLossApi, Width100RawMAEForwardAndBackwardMatchNumericalReference) {
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "mae_width_100_gradient_numerical", predictions, labels, [](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::MAE loss = Api::MAE::Builder()
                                .network(network)
                                .predictions(predictionsTensor)
                                .labels(labelsTensor)
                                .lossDataType(Api::DataType::FP32)
                                .reportsRawLoss()
                                .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceMaeRawLoss(predictions, labels);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceMaeRawLoss(perturbed, labels)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(1.0e-2f));
}

TEST(RegressionVectorLossApi, Width100RawMSEForwardAndBackwardMatchNumericalReference) {
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "mse_width_100_gradient_numerical", predictions, labels, [](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::MSE loss = Api::MSE::Builder()
                                .network(network)
                                .predictions(predictionsTensor)
                                .labels(labelsTensor)
                                .lossDataType(Api::DataType::FP32)
                                .reportsRawLoss()
                                .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceMseRawLoss(predictions, labels);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceMseRawLoss(perturbed, labels)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(1.0e-2f));
}

TEST(RegressionVectorLossApi, Width100RawExpectileForwardAndBackwardMatchNumericalReference) {
    constexpr float expectile = 0.9f;
    const vector<float> predictions = makeQuantilePredictions();
    const vector<float> labels = makeQuantileLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "expectile_width_100_gradient_numerical",
        predictions,
        labels,
        [expectile](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::ExpectileLoss loss = Api::ExpectileLoss::Builder()
                                           .network(network)
                                           .predictions(predictionsTensor)
                                           .labels(labelsTensor)
                                           .expectile(expectile)
                                           .lossDataType(Api::DataType::FP32)
                                           .reportsRawLoss()
                                           .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceExpectileRawLoss(predictions, labels, expectile);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions,
        [&](const vector<float>& perturbed) { return sumAsDouble(referenceExpectileRawLoss(perturbed, labels, expectile)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(2.0e-2f));
}

TEST(RegressionVectorLossApi, Width100CenterExpectileExactlyMatchesMse) {
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    const vector<float> expectileLoss = referenceExpectileRawLoss(predictions, labels, 0.5f);
    const vector<float> mseLoss = referenceMseRawLoss(predictions, labels);
    expectClose(expectileLoss, mseLoss, 0.0f);
}

TEST(RegressionVectorLossApi, Width100RawAsymmetricPowerForwardAndBackwardMatchNumericalReference) {
    constexpr float level = 0.9f;
    constexpr float exponent = 1.5f;
    const vector<float> predictions = makeQuantilePredictions();
    const vector<float> labels = makeQuantileLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "asymmetric_power_width_100_gradient_numerical",
        predictions,
        labels,
        [level, exponent](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::AsymmetricPowerLoss loss = Api::AsymmetricPowerLoss::Builder()
                                                .network(network)
                                                .predictions(predictionsTensor)
                                                .labels(labelsTensor)
                                                .level(level)
                                                .exponent(exponent)
                                                .lossDataType(Api::DataType::FP32)
                                                .reportsRawLoss()
                                                .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceAsymmetricPowerRawLoss(predictions, labels, level, exponent);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions,
        [&](const vector<float>& perturbed) {
            return sumAsDouble(referenceAsymmetricPowerRawLoss(perturbed, labels, level, exponent));
        });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-2f));
}

TEST(RegressionVectorLossApi, Width100CenterAsymmetricPowerExactlyMatchesMeanPowerReference) {
    constexpr float exponent = 1.5f;
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    const vector<float> asymmetricLoss = referenceAsymmetricPowerRawLoss(predictions, labels, 0.5f, exponent);
    vector<float> meanPowerLoss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        meanPowerLoss[i] = std::pow(std::fabs(predictions[i] - labels[i]), exponent);
    expectClose(asymmetricLoss, meanPowerLoss, 0.0f);
}

TEST(RegressionVectorLossApi, Width100ExponentTwoAsymmetricPowerExactlyMatchesExpectileReference) {
    constexpr float level = 0.1f;
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    const vector<float> asymmetricLoss = referenceAsymmetricPowerRawLoss(predictions, labels, level, 2.0f);
    const vector<float> expectileLoss = referenceExpectileRawLoss(predictions, labels, level);
    expectClose(asymmetricLoss, expectileLoss, 0.0f);
}

TEST(RegressionVectorLossApi, Width100RawBCEForwardAndBackwardMatchNumericalReference) {
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeBinaryLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "bce_width_100_gradient_numerical", predictions, labels, [](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::BinaryCrossEntropy loss = Api::BinaryCrossEntropy::Builder()
                                               .network(network)
                                               .predictions(predictionsTensor)
                                               .labels(labelsTensor)
                                               .lossDataType(Api::DataType::FP32)
                                               .reportsElementwiseLoss()
                                               .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceBceRawLoss(predictions, labels);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceBceRawLoss(perturbed, labels)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(1.0e-2f));
}

TEST(RegressionVectorLossApi, Width100RawMAPEForwardAndBackwardMatchNumericalReference) {
    const vector<float> predictions = makePredictions();
    const vector<float> labels = makeLabels();

    RawLossRunResult actual = runMapeRawLossNetwork(predictions, labels);

    vector<float> expectedLoss = referenceMapeRawLoss(predictions, labels);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceMapeRawLoss(perturbed, labels)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-4f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-2f));
}


TEST(RegressionVectorLossApi, Width100RawQuantileForwardAndBackwardMatchNumericalReference) {
    constexpr float quantile = 0.9f;
    const vector<float> predictions = makeQuantilePredictions();
    const vector<float> labels = makeQuantileLabels();

    RawLossRunResult actual = runCustomBackedRawLossNetwork(
        "quantile_width_100_gradient_numerical", predictions, labels, [quantile](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::QuantileLoss loss = Api::QuantileLoss::Builder()
                                         .network(network)
                                         .predictions(predictionsTensor)
                                         .labels(labelsTensor)
                                         .quantile(quantile)
                                         .lossDataType(Api::DataType::FP32)
                                         .reportsRawLoss()
                                         .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceQuantileRawLoss(predictions, labels, quantile);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceQuantileRawLoss(perturbed, labels, quantile)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(1.0e-2f));
}

TEST(RegressionVectorLossApi, SharedPredictionTensorAccumulatesMultipleLossGradientsNumerically) {
    constexpr float maeWeight = 10.0f;
    constexpr float quantileWeight = 2.6667f;
    constexpr float quantile = 0.9f;
    const vector<float> predictions = makeQuantilePredictions();
    const vector<float> labels = makeQuantileLabels();

    SharedPredictionDualLossRunResult actual =
        runSharedPredictionMaeAndQuantileLossNetwork(predictions, labels, maeWeight, quantileWeight, quantile);

    const vector<float> expectedMaeLoss = scaleValues(referenceMaeRawLoss(predictions, labels), maeWeight);
    const vector<float> expectedQuantileLoss = scaleValues(referenceQuantileRawLoss(predictions, labels, quantile), quantileWeight);
    const vector<float> expectedGradient =
        referenceCombinedMaeQuantilePredictionGradient(predictions, labels, maeWeight, quantileWeight, quantile);

    expectClose(actual.maeLoss, expectedMaeLoss, 1.0e-5f);
    expectClose(actual.quantileLoss, expectedQuantileLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-2f));
}

