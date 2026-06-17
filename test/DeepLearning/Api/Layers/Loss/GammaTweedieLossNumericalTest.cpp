#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
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
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr double kFiniteDifferenceEpsilon = 1.0e-3;
constexpr double kSpecialPowerTolerance = 1.0e-6;

shared_ptr<Api::CustomLoss> findRawCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::CustomLoss> customLoss = dynamic_pointer_cast<Api::CustomLoss>(layer);
        if (customLoss != nullptr)
            return customLoss;
    }
    return nullptr;
}

vector<float> copyGpuTensorToVector(Impl::Tensor tensor, Stream stream) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpu(cpuPlacement, tensor.getDescriptor());
    cpu.copyFromAsync(tensor, stream);
    stream.synchronize();
    return vector<float>(static_cast<float*>(cpu.getMemPtr()), static_cast<float*>(cpu.getMemPtr()) + tensor.getTotalNumElements());
}

Impl::Tensor makeCpuTensor(const vector<float>& values, uint32_t batchSize, uint32_t numFeatures) {
    THOR_THROW_IF_FALSE(values.size() == static_cast<size_t>(batchSize * numFeatures));
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32,
                                      {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(numFeatures)});
    Impl::Tensor tensor(cpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.getMemPtr()));
    return tensor;
}

struct SingleInputLossRunResult {
    vector<float> loss;
    vector<float> predictionGradient;
};

using BuildLossFn = function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor)>;

SingleInputLossRunResult runRawSingleInputLossNetwork(const string& networkName,
                                                      const vector<float>& predictions,
                                                      const vector<float>& labels,
                                                      const BuildLossFn& buildLoss) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 4;
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * numFeatures));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network(networkName);
    Api::NetworkInput predictionsInput = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("predictions")
                                             .dimensions({numFeatures})
                                             .dataType(Api::DataType::FP32)
                                             .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({numFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();

    Api::GradientRivet predictionsRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(predictionsInput.getFeatureOutput().value())
                                             .build();

    Api::Tensor lossTensor = buildLoss(network, predictionsRivet.getFeatureOutput().value(), labelsInput.getFeatureOutput().value());
    shared_ptr<Api::CustomLoss> rawCustomLoss = findRawCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(lossTensor)
                                        .dataType(Api::DataType::FP32)
                                        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {0}, 1);
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    Stream stream(0);
    for (Event& event : initDoneEvents)
        stream.waitEvent(event);
    stream.synchronize();
    initDoneEvents.clear();

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalPredictionsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(predictionsInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalLabelsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(labelsInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::CustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::CustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalPredictionsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::Tensor predictionsCpu = makeCpuTensor(predictions, batchSize, numFeatures);
    Impl::Tensor labelsCpu = makeCpuTensor(labels, batchSize, numFeatures);

    physicalPredictionsInput->forward(predictionsCpu, false, batchSize);
    physicalLabelsInput->forward(labelsCpu, false, batchSize);

    Stream labelsStream = physicalLabelsInput->getStream();
    labelsStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    labelsStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + predictions.size());

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> predictionGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput().value(), rawLossStream);

    return SingleInputLossRunResult{outputLoss, predictionGradient};
}

float referenceGammaNLLLossElement(float prediction, float target, float eps) {
    const float mean = std::max(prediction, eps);
    return std::log(mean) + target / mean;
}

vector<float> referenceGammaNLLRawLoss(const vector<float>& predictions, const vector<float>& labels, float eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        loss[i] = referenceGammaNLLLossElement(predictions[i], labels[i], eps);
    return loss;
}

double referenceGammaNLLLossElementDouble(double prediction, double target, double eps) {
    const double mean = std::max(prediction, eps);
    return std::log(mean) + target / mean;
}

double totalGammaNLLLoss(const vector<double>& predictions, const vector<double>& labels, double eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    double total = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i)
        total += referenceGammaNLLLossElementDouble(predictions[i], labels[i], eps);
    return total;
}

vector<float> numericalGammaNLLPredictionGradient(const vector<float>& predictions, const vector<float>& labels, float eps) {
    vector<double> predictionValues(predictions.begin(), predictions.end());
    vector<double> labelValues(labels.begin(), labels.end());
    vector<float> gradient(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictionValues.size(); ++i) {
        predictionValues[i] += kFiniteDifferenceEpsilon;
        const double lossPlus = totalGammaNLLLoss(predictionValues, labelValues, eps);
        predictionValues[i] -= 2.0 * kFiniteDifferenceEpsilon;
        const double lossMinus = totalGammaNLLLoss(predictionValues, labelValues, eps);
        predictionValues[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

bool isSpecialPower(double power, double special) { return std::fabs(power - special) <= kSpecialPowerTolerance; }

float referenceTweedieLossElement(float prediction, float targetValue, float powerValue, float eps) {
    const double mean = std::max(static_cast<double>(prediction), static_cast<double>(eps));
    const double target = std::max(static_cast<double>(targetValue), 0.0);
    const double safeTarget = std::max(target, static_cast<double>(eps));
    const double power = powerValue;

    double loss = 0.0;
    if (isSpecialPower(power, 0.0)) {
        const double diff = target - mean;
        loss = diff * diff;
    } else if (isSpecialPower(power, 1.0)) {
        loss = 2.0 * (target * std::log(safeTarget / mean) - target + mean);
    } else if (isSpecialPower(power, 2.0)) {
        loss = 2.0 * (std::log(mean / safeTarget) + target / mean - 1.0);
    } else {
        loss = 2.0 * (std::pow(safeTarget, 2.0 - power) / ((1.0 - power) * (2.0 - power)) -
                      target * std::pow(mean, 1.0 - power) / (1.0 - power) + std::pow(mean, 2.0 - power) / (2.0 - power));
    }
    return static_cast<float>(loss);
}

vector<float> referenceTweedieRawLoss(const vector<float>& predictions, const vector<float>& labels, float power, float eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        loss[i] = referenceTweedieLossElement(predictions[i], labels[i], power, eps);
    return loss;
}

double referenceTweedieLossElementDouble(double prediction, double targetValue, double power, double eps) {
    const double mean = std::max(prediction, eps);
    const double target = std::max(targetValue, 0.0);
    const double safeTarget = std::max(target, eps);

    if (isSpecialPower(power, 0.0)) {
        const double diff = target - mean;
        return diff * diff;
    }
    if (isSpecialPower(power, 1.0)) {
        return 2.0 * (target * std::log(safeTarget / mean) - target + mean);
    }
    if (isSpecialPower(power, 2.0)) {
        return 2.0 * (std::log(mean / safeTarget) + target / mean - 1.0);
    }
    return 2.0 * (std::pow(safeTarget, 2.0 - power) / ((1.0 - power) * (2.0 - power)) -
                  target * std::pow(mean, 1.0 - power) / (1.0 - power) + std::pow(mean, 2.0 - power) / (2.0 - power));
}

double totalTweedieLoss(const vector<double>& predictions, const vector<double>& labels, double power, double eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    double total = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i)
        total += referenceTweedieLossElementDouble(predictions[i], labels[i], power, eps);
    return total;
}

vector<float> numericalTweediePredictionGradient(const vector<float>& predictions, const vector<float>& labels, float power, float eps) {
    vector<double> predictionValues(predictions.begin(), predictions.end());
    vector<double> labelValues(labels.begin(), labels.end());
    vector<float> gradient(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictionValues.size(); ++i) {
        predictionValues[i] += kFiniteDifferenceEpsilon;
        const double lossPlus = totalTweedieLoss(predictionValues, labelValues, power, eps);
        predictionValues[i] -= 2.0 * kFiniteDifferenceEpsilon;
        const double lossMinus = totalTweedieLoss(predictionValues, labelValues, power, eps);
        predictionValues[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(GammaNLLLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    constexpr float eps = 1.0e-5f;
    const vector<float> predictions = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};
    const vector<float> labels = {0.25f, 1.0f, 2.0f, 4.0f, 3.0f, 0.5f, 5.0f, 1.0f};

    vector<float> referenceLoss = referenceGammaNLLRawLoss(predictions, labels, eps);
    vector<float> referenceGradient = numericalGammaNLLPredictionGradient(predictions, labels, eps);
    for (float& value : referenceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    SingleInputLossRunResult actual = runRawSingleInputLossNetwork(
        "gamma_nll_numerical",
        predictions,
        labels,
        [eps](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::GammaNLLLoss loss = Api::GammaNLLLoss::Builder()
                                         .network(network)
                                         .predictions(predictionsTensor)
                                         .labels(labelsTensor)
                                         .eps(eps)
                                         .lossDataType(Api::DataType::FP32)
                                         .reportsRawLoss()
                                         .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.predictionGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-3f));
}

TEST(TweedieLossApi, NumericalRawLossAndBackwardGradientMatchReferenceForCompoundPoissonGammaPower) {
    constexpr float power = 1.5f;
    constexpr float eps = 1.0e-5f;
    const vector<float> predictions = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};
    const vector<float> labels = {0.25f, 1.0f, 2.0f, 4.0f, 3.0f, 0.5f, 5.0f, 1.0f};

    vector<float> referenceLoss = referenceTweedieRawLoss(predictions, labels, power, eps);
    vector<float> referenceGradient = numericalTweediePredictionGradient(predictions, labels, power, eps);
    for (float& value : referenceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    SingleInputLossRunResult actual = runRawSingleInputLossNetwork(
        "tweedie_numerical_power_1_5",
        predictions,
        labels,
        [power, eps](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::TweedieLoss loss = Api::TweedieLoss::Builder()
                                        .network(network)
                                        .predictions(predictionsTensor)
                                        .labels(labelsTensor)
                                        .power(power)
                                        .eps(eps)
                                        .lossDataType(Api::DataType::FP32)
                                        .reportsRawLoss()
                                        .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, referenceLoss, 3.0e-5f);
    expectClose(actual.predictionGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-3f));
}

TEST(TweedieLossApi, NumericalRawLossAndBackwardGradientMatchReferenceForNormalPoissonAndGammaPowers) {
    constexpr float eps = 1.0e-5f;
    const vector<float> predictions = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};
    const vector<float> labels = {0.25f, 1.0f, 2.0f, 4.0f, 3.0f, 0.5f, 5.0f, 1.0f};

    for (float power : {0.0f, 1.0f, 2.0f}) {
        vector<float> referenceLoss = referenceTweedieRawLoss(predictions, labels, power, eps);
        vector<float> referenceGradient = numericalTweediePredictionGradient(predictions, labels, power, eps);
        for (float& value : referenceGradient)
            value *= Impl::Loss::getLossScalingFactor();

        SingleInputLossRunResult actual = runRawSingleInputLossNetwork(
            string("tweedie_numerical_special_power_") + std::to_string(static_cast<int>(power)),
            predictions,
            labels,
            [power, eps](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
                Api::TweedieLoss loss = Api::TweedieLoss::Builder()
                                            .network(network)
                                            .predictions(predictionsTensor)
                                            .labels(labelsTensor)
                                            .power(power)
                                            .eps(eps)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
                return loss.getLoss();
            });

        expectClose(actual.loss, referenceLoss, 3.0e-5f);
        expectClose(actual.predictionGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(4.0e-3f));
    }
}
