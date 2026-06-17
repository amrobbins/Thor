#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/DeepLearning/Api/Layers/Loss/LossNumericalTestTolerance.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr double kFiniteDifferenceEpsilon = 1.0e-3;
constexpr double kPi = 3.14159265358979323846;

float referencePoissonNLLLossElement(float prediction, float label, bool logInput, bool full, float eps) {
    float loss = logInput ? (std::exp(prediction) - label * prediction) : (prediction - label * std::log(prediction + eps));
    if (full && label > 1.0f) {
        loss += label * std::log(label) - label + 0.5f * std::log(2.0f * kPi * label);
    }
    return loss;
}

vector<float> referencePoissonNLLRawLoss(const vector<float>& predictions,
                                         const vector<float>& labels,
                                         bool logInput,
                                         bool full,
                                         float eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    vector<float> loss(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i)
        loss[i] = referencePoissonNLLLossElement(predictions[i], labels[i], logInput, full, eps);
    return loss;
}

double referencePoissonNLLLossElementDouble(double prediction, double label, bool logInput, bool full, double eps) {
    double loss = logInput ? (std::exp(prediction) - label * prediction) : (prediction - label * std::log(prediction + eps));
    if (full && label > 1.0) {
        loss += label * std::log(label) - label + 0.5 * std::log(2.0 * kPi * label);
    }
    return loss;
}

double totalPoissonNLLLoss(const vector<double>& predictions, const vector<double>& labels, bool logInput, bool full, double eps) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    double total = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i)
        total += referencePoissonNLLLossElementDouble(predictions[i], labels[i], logInput, full, eps);
    return total;
}

vector<float> numericalPoissonNLLPredictionGradient(const vector<float>& predictions,
                                                    const vector<float>& labels,
                                                    bool logInput,
                                                    bool full,
                                                    float eps) {
    vector<double> predictionValues(predictions.begin(), predictions.end());
    vector<double> labelValues(labels.begin(), labels.end());
    vector<float> gradient(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictionValues.size(); ++i) {
        predictionValues[i] += kFiniteDifferenceEpsilon;
        const double lossPlus = totalPoissonNLLLoss(predictionValues, labelValues, logInput, full, eps);
        predictionValues[i] -= 2.0 * kFiniteDifferenceEpsilon;
        const double lossMinus = totalPoissonNLLLoss(predictionValues, labelValues, logInput, full, eps);
        predictionValues[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

float referenceGaussianNLLLossElement(float mean, float target, float variance, bool full, float eps) {
    const float safeVariance = std::max(variance, eps);
    const float diff = mean - target;
    float loss = 0.5f * (std::log(safeVariance) + diff * diff / safeVariance);
    if (full)
        loss += 0.5f * std::log(2.0f * kPi);
    return loss;
}

vector<float> referenceGaussianNLLRawLoss(const vector<float>& mean,
                                          const vector<float>& target,
                                          const vector<float>& variance,
                                          bool full,
                                          float eps) {
    THOR_THROW_IF_FALSE(mean.size() == target.size());
    THOR_THROW_IF_FALSE(mean.size() == variance.size());
    vector<float> loss(mean.size(), 0.0f);
    for (size_t i = 0; i < mean.size(); ++i)
        loss[i] = referenceGaussianNLLLossElement(mean[i], target[i], variance[i], full, eps);
    return loss;
}

double referenceGaussianNLLLossElementDouble(double mean, double target, double variance, bool full, double eps) {
    const double safeVariance = std::max(variance, eps);
    const double diff = mean - target;
    double loss = 0.5 * (std::log(safeVariance) + diff * diff / safeVariance);
    if (full)
        loss += 0.5 * std::log(2.0 * kPi);
    return loss;
}

double totalGaussianNLLLoss(const vector<double>& mean, const vector<double>& target, const vector<double>& variance, bool full, double eps) {
    THOR_THROW_IF_FALSE(mean.size() == target.size());
    THOR_THROW_IF_FALSE(mean.size() == variance.size());
    double total = 0.0;
    for (size_t i = 0; i < mean.size(); ++i)
        total += referenceGaussianNLLLossElementDouble(mean[i], target[i], variance[i], full, eps);
    return total;
}

vector<float> numericalGaussianNLLGradient(const vector<float>& mean,
                                           const vector<float>& target,
                                           const vector<float>& variance,
                                           bool full,
                                           float eps,
                                           char wrt) {
    vector<double> meanValues(mean.begin(), mean.end());
    vector<double> targetValues(target.begin(), target.end());
    vector<double> varianceValues(variance.begin(), variance.end());
    vector<double>* tensor = nullptr;
    if (wrt == 'm')
        tensor = &meanValues;
    else if (wrt == 'v')
        tensor = &varianceValues;
    else
        THOR_UNREACHABLE();

    vector<float> gradient(tensor->size(), 0.0f);
    for (size_t i = 0; i < tensor->size(); ++i) {
        (*tensor)[i] += kFiniteDifferenceEpsilon;
        const double lossPlus = totalGaussianNLLLoss(meanValues, targetValues, varianceValues, full, eps);
        (*tensor)[i] -= 2.0 * kFiniteDifferenceEpsilon;
        const double lossMinus = totalGaussianNLLLoss(meanValues, targetValues, varianceValues, full, eps);
        (*tensor)[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
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

shared_ptr<Api::MultiInputCustomLoss> findRawMultiInputCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::MultiInputCustomLoss> customLoss = dynamic_pointer_cast<Api::MultiInputCustomLoss>(layer);
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

struct PoissonNLLRunResult {
    vector<float> loss;
    vector<float> predictionGradient;
};

PoissonNLLRunResult runRawPoissonNLLLossNetwork(const vector<float>& predictions,
                                                const vector<float>& labels,
                                                bool logInput,
                                                bool full,
                                                float eps) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 4;
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * numFeatures));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network("poisson_nll_numerical");
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

    Api::PoissonNLLLoss loss = Api::PoissonNLLLoss::Builder()
                                   .network(network)
                                   .predictions(predictionsRivet.getFeatureOutput().value())
                                   .labels(labelsInput.getFeatureOutput().value())
                                   .logInput(logInput)
                                   .full(full)
                                   .eps(eps)
                                   .lossDataType(Api::DataType::FP32)
                                   .reportsRawLoss()
                                   .build();
    shared_ptr<Api::CustomLoss> rawCustomLoss = findRawCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(loss.getLoss())
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

    return PoissonNLLRunResult{outputLoss, predictionGradient};
}

struct GaussianNLLRunResult {
    vector<float> loss;
    vector<float> meanGradient;
    vector<float> varianceGradient;
};

GaussianNLLRunResult runRawGaussianNLLLossNetwork(const vector<float>& mean,
                                                  const vector<float>& target,
                                                  const vector<float>& variance,
                                                  bool full,
                                                  float eps) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 4;
    THOR_THROW_IF_FALSE(mean.size() == static_cast<size_t>(batchSize * numFeatures));
    THOR_THROW_IF_FALSE(target.size() == mean.size());
    THOR_THROW_IF_FALSE(variance.size() == mean.size());

    Api::Network network("gaussian_nll_numerical");
    Api::NetworkInput meanInput = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("mean")
                                      .dimensions({numFeatures})
                                      .dataType(Api::DataType::FP32)
                                      .build();
    Api::NetworkInput targetInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("target")
                                        .dimensions({numFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::NetworkInput varianceInput = Api::NetworkInput::Builder()
                                          .network(network)
                                          .name("variance")
                                          .dimensions({numFeatures})
                                          .dataType(Api::DataType::FP32)
                                          .build();

    Api::GradientRivet meanRivet = Api::GradientRivet::Builder().network(network).tensor(meanInput.getFeatureOutput().value()).build();
    Api::GradientRivet varianceRivet =
        Api::GradientRivet::Builder().network(network).tensor(varianceInput.getFeatureOutput().value()).build();

    Api::GaussianNLLLoss loss = Api::GaussianNLLLoss::Builder()
                                    .network(network)
                                    .mean(meanRivet.getFeatureOutput().value())
                                    .target(targetInput.getFeatureOutput().value())
                                    .variance(varianceRivet.getFeatureOutput().value())
                                    .full(full)
                                    .eps(eps)
                                    .lossDataType(Api::DataType::FP32)
                                    .reportsRawLoss()
                                    .build();
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(loss.getLoss())
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
    shared_ptr<Impl::NetworkInput> physicalMeanInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(meanInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalTargetInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(targetInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalVarianceInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(varianceInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalMeanInput != nullptr);
    THOR_THROW_IF_FALSE(physicalTargetInput != nullptr);
    THOR_THROW_IF_FALSE(physicalVarianceInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::Tensor meanCpu = makeCpuTensor(mean, batchSize, numFeatures);
    Impl::Tensor targetCpu = makeCpuTensor(target, batchSize, numFeatures);
    Impl::Tensor varianceCpu = makeCpuTensor(variance, batchSize, numFeatures);

    physicalMeanInput->forward(meanCpu, false, batchSize);
    physicalTargetInput->forward(targetCpu, false, batchSize);
    physicalVarianceInput->forward(varianceCpu, false, batchSize);

    Stream outputStream = physicalVarianceInput->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()), static_cast<float*>(outputLossCpu.getMemPtr()) + mean.size());

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> meanGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    THOR_THROW_IF_FALSE(!physicalRawLoss->getErrorOutput(1).has_value());
    vector<float> varianceGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(2).value(), rawLossStream);

    return GaussianNLLRunResult{outputLoss, meanGradient, varianceGradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(PoissonNLLLossApi, NumericalRawLossAndBackwardGradientMatchReferenceForLogInput) {
    constexpr bool logInput = true;
    constexpr bool full = true;
    constexpr float eps = 1.0e-5f;
    const vector<float> predictions = {0.0f, 0.25f, 1.25f, -0.5f, 0.75f, -0.25f, 0.5f, 1.5f};
    const vector<float> labels = {0.0f, 1.0f, 2.0f, 4.0f, 3.0f, 0.0f, 5.0f, 1.0f};

    vector<float> referenceLoss = referencePoissonNLLRawLoss(predictions, labels, logInput, full, eps);
    vector<float> referenceGradient = numericalPoissonNLLPredictionGradient(predictions, labels, logInput, full, eps);
    for (float& value : referenceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    PoissonNLLRunResult actual = runRawPoissonNLLLossNetwork(predictions, labels, logInput, full, eps);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.predictionGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}

TEST(PoissonNLLLossApi, NumericalRawLossAndBackwardGradientMatchReferenceForRateInput) {
    constexpr bool logInput = false;
    constexpr bool full = true;
    constexpr float eps = 1.0e-5f;
    const vector<float> predictions = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};
    const vector<float> labels = {0.0f, 1.0f, 2.0f, 4.0f, 3.0f, 0.0f, 5.0f, 1.0f};

    vector<float> referenceLoss = referencePoissonNLLRawLoss(predictions, labels, logInput, full, eps);
    vector<float> referenceGradient = numericalPoissonNLLPredictionGradient(predictions, labels, logInput, full, eps);
    for (float& value : referenceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    PoissonNLLRunResult actual = runRawPoissonNLLLossNetwork(predictions, labels, logInput, full, eps);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.predictionGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
}

TEST(GaussianNLLLossApi, NumericalRawLossAndBackwardGradientsMatchReference) {
    constexpr bool full = true;
    constexpr float eps = 1.0e-5f;
    const vector<float> mean = {0.0f, 0.25f, 1.5f, -2.0f, -1.0f, 0.75f, 2.25f, -0.5f};
    const vector<float> target = {0.0f, -0.25f, 0.0f, -0.5f, 0.5f, 0.25f, 1.0f, -1.5f};
    const vector<float> variance = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};

    vector<float> referenceLoss = referenceGaussianNLLRawLoss(mean, target, variance, full, eps);
    vector<float> referenceMeanGradient = numericalGaussianNLLGradient(mean, target, variance, full, eps, 'm');
    vector<float> referenceVarianceGradient = numericalGaussianNLLGradient(mean, target, variance, full, eps, 'v');
    for (float& value : referenceMeanGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : referenceVarianceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    GaussianNLLRunResult actual = runRawGaussianNLLLossNetwork(mean, target, variance, full, eps);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.meanGradient, referenceMeanGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
    expectClose(actual.varianceGradient, referenceVarianceGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}
