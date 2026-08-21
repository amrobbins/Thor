#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/LaplaceNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/NegativeBinomialNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
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
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr double kFiniteDifferenceEpsilon = 1.0e-3;
constexpr uint32_t kBatchSize = 2;
constexpr uint32_t kNumFeatures = 4;

shared_ptr<Api::MultiInputCustomLoss> findRawMultiInputCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::MultiInputCustomLoss> customLoss = dynamic_pointer_cast<Api::MultiInputCustomLoss>(layer);
        if (customLoss != nullptr)
            return customLoss;
    }
    return nullptr;
}

Impl::Tensor makeCpuTensor(const vector<float>& values) {
    THOR_THROW_IF_FALSE(values.size() == static_cast<size_t>(kBatchSize * kNumFeatures));
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32,
                                      {static_cast<unsigned long>(kBatchSize), static_cast<unsigned long>(kNumFeatures)});
    Impl::Tensor tensor(cpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.getMemPtr()));
    return tensor;
}

vector<float> copyGpuTensorToVector(Impl::Tensor tensor, Stream stream) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpu(cpuPlacement, tensor.getDescriptor());
    cpu.copyFromAsync(tensor, stream);
    stream.synchronize();
    return vector<float>(static_cast<float*>(cpu.getMemPtr()), static_cast<float*>(cpu.getMemPtr()) + tensor.getTotalNumElements());
}

struct TwoParameterLossRunResult {
    vector<float> loss;
    vector<float> firstGradient;
    vector<float> secondGradient;
};

using BuildLoss = function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor, Api::Tensor)>;

TwoParameterLossRunResult runTwoParameterLoss(const string& name,
                                              const vector<float>& first,
                                              const vector<float>& second,
                                              const vector<float>& labels,
                                              uint32_t secondGradientIndex,
                                              BuildLoss buildLoss) {
    THOR_THROW_IF_FALSE(first.size() == static_cast<size_t>(kBatchSize * kNumFeatures));
    THOR_THROW_IF_FALSE(second.size() == first.size());
    THOR_THROW_IF_FALSE(labels.size() == first.size());

    Api::Network network(name);
    Api::NetworkInput firstInput = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("first")
                                       .dimensions({kNumFeatures})
                                       .dataType(Api::DataType::FP32)
                                       .build();
    Api::NetworkInput secondInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("second")
                                        .dimensions({kNumFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({kNumFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();

    Api::GradientRivet firstRivet =
        Api::GradientRivet::Builder().network(network).tensor(firstInput.getFeatureOutput().value()).build();
    Api::GradientRivet secondRivet =
        Api::GradientRivet::Builder().network(network).tensor(secondInput.getFeatureOutput().value()).build();

    Api::Tensor lossTensor = buildLoss(network,
                                       firstRivet.getFeatureOutput().value(),
                                       secondRivet.getFeatureOutput().value(),
                                       labelsInput.getFeatureOutput().value());
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(lossTensor)
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
    auto physicalFirstInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(firstInput.getId()));
    auto physicalSecondInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(secondInput.getId()));
    auto physicalLabelsInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(labelsInput.getId()));
    auto physicalLossOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    auto physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalFirstInput != nullptr);
    THOR_THROW_IF_FALSE(physicalSecondInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    physicalFirstInput->forward(makeCpuTensor(first), false, kBatchSize);
    physicalSecondInput->forward(makeCpuTensor(second), false, kBatchSize);
    physicalLabelsInput->forward(makeCpuTensor(labels), false, kBatchSize);

    Stream outputStream = physicalLabelsInput->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + first.size());

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> firstGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    vector<float> secondGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(secondGradientIndex).value(), rawLossStream);

    return {outputLoss, firstGradient, secondGradient};
}


struct ThreeParameterLossRunResult {
    vector<float> loss;
    vector<float> firstGradient;
    vector<float> secondGradient;
    vector<float> thirdGradient;
};

using BuildThreeParameterLoss = function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor, Api::Tensor, Api::Tensor)>;

ThreeParameterLossRunResult runThreeParameterLoss(const string& name,
                                                  const vector<float>& first,
                                                  const vector<float>& second,
                                                  const vector<float>& third,
                                                  const vector<float>& labels,
                                                  BuildThreeParameterLoss buildLoss) {
    THOR_THROW_IF_FALSE(first.size() == static_cast<size_t>(kBatchSize * kNumFeatures));
    THOR_THROW_IF_FALSE(second.size() == first.size());
    THOR_THROW_IF_FALSE(third.size() == first.size());
    THOR_THROW_IF_FALSE(labels.size() == first.size());

    Api::Network network(name);
    Api::NetworkInput firstInput = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("first")
                                       .dimensions({kNumFeatures})
                                       .dataType(Api::DataType::FP32)
                                       .build();
    Api::NetworkInput secondInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("second")
                                        .dimensions({kNumFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::NetworkInput thirdInput = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("third")
                                       .dimensions({kNumFeatures})
                                       .dataType(Api::DataType::FP32)
                                       .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({kNumFeatures})
                                        .dataType(Api::DataType::FP32)
                                        .build();

    Api::GradientRivet firstRivet =
        Api::GradientRivet::Builder().network(network).tensor(firstInput.getFeatureOutput().value()).build();
    Api::GradientRivet secondRivet =
        Api::GradientRivet::Builder().network(network).tensor(secondInput.getFeatureOutput().value()).build();
    Api::GradientRivet thirdRivet =
        Api::GradientRivet::Builder().network(network).tensor(thirdInput.getFeatureOutput().value()).build();

    Api::Tensor lossTensor = buildLoss(network,
                                       firstRivet.getFeatureOutput().value(),
                                       secondRivet.getFeatureOutput().value(),
                                       thirdRivet.getFeatureOutput().value(),
                                       labelsInput.getFeatureOutput().value());
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(lossTensor)
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
    auto physicalFirstInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(firstInput.getId()));
    auto physicalSecondInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(secondInput.getId()));
    auto physicalThirdInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(thirdInput.getId()));
    auto physicalLabelsInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(labelsInput.getId()));
    auto physicalLossOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    auto physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalFirstInput != nullptr);
    THOR_THROW_IF_FALSE(physicalSecondInput != nullptr);
    THOR_THROW_IF_FALSE(physicalThirdInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    physicalFirstInput->forward(makeCpuTensor(first), false, kBatchSize);
    physicalSecondInput->forward(makeCpuTensor(second), false, kBatchSize);
    physicalThirdInput->forward(makeCpuTensor(third), false, kBatchSize);
    physicalLabelsInput->forward(makeCpuTensor(labels), false, kBatchSize);

    Stream outputStream = physicalLabelsInput->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + first.size());

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> firstGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    vector<float> secondGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1).value(), rawLossStream);
    vector<float> thirdGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(2).value(), rawLossStream);

    return {outputLoss, firstGradient, secondGradient, thirdGradient};
}

using ReferenceThreeParameterElementLoss = function<double(double, double, double, double)>;

vector<float> referenceThreeParameterRawLoss(const vector<float>& first,
                                             const vector<float>& second,
                                             const vector<float>& third,
                                             const vector<float>& labels,
                                             const ReferenceThreeParameterElementLoss& lossFn) {
    vector<float> result(first.size());
    for (size_t i = 0; i < first.size(); ++i)
        result[i] = static_cast<float>(lossFn(first[i], second[i], third[i], labels[i]));
    return result;
}

vector<float> numericalThreeParameterGradient(const vector<float>& first,
                                              const vector<float>& second,
                                              const vector<float>& third,
                                              const vector<float>& labels,
                                              uint32_t parameterIndex,
                                              const ReferenceThreeParameterElementLoss& lossFn) {
    vector<double> firstValues(first.begin(), first.end());
    vector<double> secondValues(second.begin(), second.end());
    vector<double> thirdValues(third.begin(), third.end());
    vector<float> gradient(first.size(), 0.0f);

    for (size_t i = 0; i < first.size(); ++i) {
        vector<double>* values = nullptr;
        if (parameterIndex == 0)
            values = &firstValues;
        else if (parameterIndex == 1)
            values = &secondValues;
        else {
            THOR_THROW_IF_FALSE(parameterIndex == 2);
            values = &thirdValues;
        }
        (*values)[i] += kFiniteDifferenceEpsilon;
        double plus = 0.0;
        for (size_t j = 0; j < first.size(); ++j)
            plus += lossFn(firstValues[j], secondValues[j], thirdValues[j], labels[j]);
        (*values)[i] -= 2.0 * kFiniteDifferenceEpsilon;
        double minus = 0.0;
        for (size_t j = 0; j < first.size(); ++j)
            minus += lossFn(firstValues[j], secondValues[j], thirdValues[j], labels[j]);
        (*values)[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((plus - minus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

using ReferenceElementLoss = function<double(double, double, double)>;

vector<float> referenceRawLoss(const vector<float>& first,
                               const vector<float>& second,
                               const vector<float>& labels,
                               const ReferenceElementLoss& lossFn) {
    vector<float> result(first.size());
    for (size_t i = 0; i < first.size(); ++i)
        result[i] = static_cast<float>(lossFn(first[i], second[i], labels[i]));
    return result;
}

vector<float> numericalGradient(const vector<float>& first,
                                const vector<float>& second,
                                const vector<float>& labels,
                                bool wrtFirst,
                                const ReferenceElementLoss& lossFn) {
    vector<double> firstValues(first.begin(), first.end());
    vector<double> secondValues(second.begin(), second.end());
    vector<float> gradient(first.size(), 0.0f);

    for (size_t i = 0; i < first.size(); ++i) {
        vector<double>& values = wrtFirst ? firstValues : secondValues;
        values[i] += kFiniteDifferenceEpsilon;
        double plus = 0.0;
        for (size_t j = 0; j < first.size(); ++j)
            plus += lossFn(firstValues[j], secondValues[j], labels[j]);
        values[i] -= 2.0 * kFiniteDifferenceEpsilon;
        double minus = 0.0;
        for (size_t j = 0; j < first.size(); ++j)
            minus += lossFn(firstValues[j], secondValues[j], labels[j]);
        values[i] += kFiniteDifferenceEpsilon;
        gradient[i] = static_cast<float>((plus - minus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_LE(std::fabs(actual[i] - expected[i]), tolerance)
            << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
}

}  // namespace

TEST(NegativeBinomialNLLLossApi, LogMeanAndLogDispersionForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> mean = {0.3f, 1.2f, 5.0f, 12.0f, 2.5f, 0.75f, 7.0f, 20.0f};
    const vector<float> dispersion = {0.15f, 0.4f, 0.8f, 0.2f, 0.6f, 1.1f, 0.35f, 0.1f};
    const vector<float> labels = {0.0f, 1.0f, 7.0f, 9.0f, 4.0f, 0.0f, 12.0f, 25.0f};
    vector<float> logMean(mean.size());
    vector<float> logDispersion(dispersion.size());
    for (size_t i = 0; i < mean.size(); ++i) {
        logMean[i] = std::log(mean[i]);
        logDispersion[i] = std::log(dispersion[i]);
    }

    ReferenceElementLoss reference = [](double logMeanValue, double logDispersionValue, double y) {
        const double mu = std::exp(logMeanValue);
        const double alpha = std::exp(logDispersionValue);
        const double r = 1.0 / alpha;
        return std::lgamma(r) + std::lgamma(y + 1.0) - std::lgamma(y + r) + (r + y) * std::log1p(alpha * mu) -
               y * logDispersionValue - y * logMeanValue;
    };

    vector<float> expectedLoss = referenceRawLoss(logMean, logDispersion, labels, reference);
    vector<float> expectedMeanGradient = numericalGradient(logMean, logDispersion, labels, true, reference);
    vector<float> expectedDispersionGradient = numericalGradient(logMean, logDispersion, labels, false, reference);
    for (float& value : expectedMeanGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedDispersionGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "negative_binomial_log_parameter_numerical",
        logMean,
        logDispersion,
        labels,
        1,
        [](Api::Network& network, Api::Tensor meanTensor, Api::Tensor dispersionTensor, Api::Tensor labelsTensor) {
            Api::NegativeBinomialNLLLoss loss = Api::NegativeBinomialNLLLoss::Builder()
                                                    .network(network)
                                                    .mean(meanTensor)
                                                    .dispersion(dispersionTensor)
                                                    .labels(labelsTensor)
                                                    .logMean(true)
                                                    .logDispersion(true)
                                                    .lossDataType(Api::DataType::FP32)
                                                    .reportsRawLoss()
                                                    .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 5.0e-5f);
    expectClose(actual.firstGradient, expectedMeanGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
    expectClose(actual.secondGradient, expectedDispersionGradient, ThorTest::lossScaleAwareGradientTolerance(7.0e-3f));
}

TEST(GammaNLLLossApi, LearnedLogMeanAndLogDispersionForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> mean = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.35f, 1.5f, 4.0f};
    const vector<float> dispersion = {0.2f, 0.5f, 1.1f, 0.4f, 0.75f, 0.3f, 0.6f, 0.9f};
    const vector<float> labels = {0.25f, 1.0f, 2.0f, 4.0f, 3.0f, 0.5f, 5.0f, 1.0f};
    vector<float> logMean(mean.size());
    vector<float> logDispersion(dispersion.size());
    for (size_t i = 0; i < mean.size(); ++i) {
        logMean[i] = std::log(mean[i]);
        logDispersion[i] = std::log(dispersion[i]);
    }

    ReferenceElementLoss reference = [](double logMeanValue, double logDispersionValue, double y) {
        const double mu = std::exp(logMeanValue);
        const double phi = std::exp(logDispersionValue);
        const double k = 1.0 / phi;
        return std::lgamma(k) + k * (logMeanValue + logDispersionValue) - (k - 1.0) * std::log(y) + y / (mu * phi);
    };

    vector<float> expectedLoss = referenceRawLoss(logMean, logDispersion, labels, reference);
    vector<float> expectedMeanGradient = numericalGradient(logMean, logDispersion, labels, true, reference);
    vector<float> expectedDispersionGradient = numericalGradient(logMean, logDispersion, labels, false, reference);
    for (float& value : expectedMeanGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedDispersionGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "gamma_log_parameter_numerical",
        logMean,
        logDispersion,
        labels,
        2,
        [](Api::Network& network, Api::Tensor meanTensor, Api::Tensor dispersionTensor, Api::Tensor labelsTensor) {
            Api::GammaNLLLoss loss = Api::GammaNLLLoss::Builder()
                                         .network(network)
                                         .mean(meanTensor)
                                         .target(labelsTensor)
                                         .dispersion(dispersionTensor)
                                         .logMean(true)
                                         .logDispersion(true)
                                         .lossDataType(Api::DataType::FP32)
                                         .reportsRawLoss()
                                         .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 5.0e-5f);
    expectClose(actual.firstGradient, expectedMeanGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
    expectClose(actual.secondGradient, expectedDispersionGradient, ThorTest::lossScaleAwareGradientTolerance(7.0e-3f));
}


TEST(LaplaceNLLLossApi, LogScaleForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> location = {-1.5f, 0.25f, 2.0f, 7.5f, 1.25f, -3.0f, 4.5f, 0.75f};
    const vector<float> scale = {0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 2.0f, 0.35f, 1.25f};
    const vector<float> labels = {-0.5f, -1.0f, 2.75f, 4.0f, 0.0f, -1.5f, 6.0f, 1.5f};
    vector<float> logScale(scale.size());
    for (size_t i = 0; i < scale.size(); ++i)
        logScale[i] = std::log(scale[i]);

    ReferenceElementLoss reference = [](double locationValue, double logScaleValue, double target) {
        const double b = std::exp(logScaleValue);
        return std::log(2.0) + logScaleValue + std::abs(target - locationValue) / b;
    };

    vector<float> expectedLoss = referenceRawLoss(location, logScale, labels, reference);
    vector<float> expectedLocationGradient = numericalGradient(location, logScale, labels, true, reference);
    vector<float> expectedLogScaleGradient = numericalGradient(location, logScale, labels, false, reference);
    for (float& value : expectedLocationGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogScaleGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "laplace_log_scale_numerical",
        location,
        logScale,
        labels,
        2,
        [](Api::Network& network, Api::Tensor locationTensor, Api::Tensor logScaleTensor, Api::Tensor labelsTensor) {
            Api::LaplaceNLLLoss loss = Api::LaplaceNLLLoss::Builder()
                                           .network(network)
                                           .location(locationTensor)
                                           .scale(logScaleTensor)
                                           .target(labelsTensor)
                                           .logScale(true)
                                           .lossDataType(Api::DataType::FP32)
                                           .reportsRawLoss()
                                           .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 3.0e-5f);
    expectClose(actual.firstGradient, expectedLocationGradient, ThorTest::lossScaleAwareGradientTolerance(4.0e-3f));
    expectClose(actual.secondGradient, expectedLogScaleGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
}

TEST(LaplaceNLLLossApi, DirectScaleForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> location = {-1.5f, 0.25f, 2.0f, 7.5f, 1.25f, -3.0f, 4.5f, 0.75f};
    const vector<float> scale = {0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 2.0f, 0.35f, 1.25f};
    const vector<float> labels = {-0.5f, -1.0f, 2.75f, 4.0f, 0.0f, -1.5f, 6.0f, 1.5f};

    ReferenceElementLoss reference = [](double locationValue, double scaleValue, double target) {
        return std::log(2.0 * scaleValue) + std::abs(target - locationValue) / scaleValue;
    };

    vector<float> expectedLoss = referenceRawLoss(location, scale, labels, reference);
    vector<float> expectedLocationGradient = numericalGradient(location, scale, labels, true, reference);
    vector<float> expectedScaleGradient = numericalGradient(location, scale, labels, false, reference);
    for (float& value : expectedLocationGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedScaleGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "laplace_direct_scale_numerical",
        location,
        scale,
        labels,
        2,
        [](Api::Network& network, Api::Tensor locationTensor, Api::Tensor scaleTensor, Api::Tensor labelsTensor) {
            Api::LaplaceNLLLoss loss = Api::LaplaceNLLLoss::Builder()
                                           .network(network)
                                           .location(locationTensor)
                                           .scale(scaleTensor)
                                           .target(labelsTensor)
                                           .logScale(false)
                                           .lossDataType(Api::DataType::FP32)
                                           .reportsRawLoss()
                                           .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 3.0e-5f);
    expectClose(actual.firstGradient, expectedLocationGradient, ThorTest::lossScaleAwareGradientTolerance(4.0e-3f));
    expectClose(actual.secondGradient, expectedScaleGradient, ThorTest::lossScaleAwareGradientTolerance(8.0e-3f));
}


TEST(StudentTNLLLossApi, FixedDegreesOfFreedomForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> location = {-1.5f, 0.25f, 2.0f, 7.5f, 1.25f, -3.0f, 4.5f, 0.75f};
    const vector<float> scale = {0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 2.0f, 0.35f, 1.25f};
    const vector<float> labels = {-0.5f, -1.0f, 2.75f, 4.0f, 0.0f, -1.5f, 6.0f, 1.5f};
    vector<float> logScale(scale.size());
    for (size_t i = 0; i < scale.size(); ++i)
        logScale[i] = std::log(scale[i]);
    ReferenceElementLoss reference = [](double locationValue, double logScaleValue, double target) {
        constexpr double fixedNu = 4.5;
        const double z = (locationValue - target) * std::exp(-logScaleValue);
        return logScaleValue + std::lgamma(0.5 * fixedNu) - std::lgamma(0.5 * (fixedNu + 1.0)) +
               0.5 * std::log(fixedNu * 3.1415926535897932384626433832795) + 0.5 * (fixedNu + 1.0) * std::log1p(z * z / fixedNu);
    };

    vector<float> expectedLoss = referenceRawLoss(location, logScale, labels, reference);
    vector<float> expectedLocationGradient = numericalGradient(location, logScale, labels, true, reference);
    vector<float> expectedLogScaleGradient = numericalGradient(location, logScale, labels, false, reference);
    for (float& value : expectedLocationGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogScaleGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "student_t_fixed_df_numerical",
        location,
        logScale,
        labels,
        1,
        [](Api::Network& network, Api::Tensor locationTensor, Api::Tensor logScaleTensor, Api::Tensor labelsTensor) {
            Api::StudentTNLLLoss loss = Api::StudentTNLLLoss::Builder()
                                            .network(network)
                                            .location(locationTensor)
                                            .logScale(logScaleTensor)
                                            .target(labelsTensor)
                                            .degreesOfFreedom(4.5f)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 6.0e-5f);
    expectClose(actual.firstGradient, expectedLocationGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
    expectClose(actual.secondGradient, expectedLogScaleGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
}

TEST(StudentTNLLLossApi, LearnedLogDegreesOfFreedomForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> location = {-1.5f, 0.25f, 2.0f, 7.5f, 1.25f, -3.0f, 4.5f, 0.75f};
    const vector<float> scale = {0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 2.0f, 0.35f, 1.25f};
    const vector<float> degreesOfFreedom = {1.5f, 2.0f, 3.0f, 5.0f, 8.0f, 12.0f, 4.0f, 2.5f};
    const vector<float> labels = {-0.5f, -1.0f, 2.75f, 4.0f, 0.0f, -1.5f, 6.0f, 1.5f};
    vector<float> logScale(scale.size());
    vector<float> logDegreesOfFreedom(degreesOfFreedom.size());
    for (size_t i = 0; i < scale.size(); ++i) {
        logScale[i] = std::log(scale[i]);
        logDegreesOfFreedom[i] = std::log(degreesOfFreedom[i]);
    }

    ReferenceThreeParameterElementLoss reference = [](double locationValue, double logScaleValue, double logNu, double target) {
        const double nu = std::exp(logNu);
        const double z = (locationValue - target) * std::exp(-logScaleValue);
        return logScaleValue + std::lgamma(0.5 * nu) - std::lgamma(0.5 * (nu + 1.0)) +
               0.5 * std::log(nu * 3.1415926535897932384626433832795) + 0.5 * (nu + 1.0) * std::log1p(z * z / nu);
    };

    vector<float> expectedLoss = referenceThreeParameterRawLoss(location, logScale, logDegreesOfFreedom, labels, reference);
    vector<float> expectedLocationGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 0, reference);
    vector<float> expectedLogScaleGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 1, reference);
    vector<float> expectedLogDegreesOfFreedomGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 2, reference);
    for (float& value : expectedLocationGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogScaleGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogDegreesOfFreedomGradient)
        value *= Impl::Loss::getLossScalingFactor();

    ThreeParameterLossRunResult actual = runThreeParameterLoss(
        "student_t_learned_df_numerical",
        location,
        logScale,
        logDegreesOfFreedom,
        labels,
        [](Api::Network& network,
           Api::Tensor locationTensor,
           Api::Tensor logScaleTensor,
           Api::Tensor logDegreesOfFreedomTensor,
           Api::Tensor labelsTensor) {
            Api::StudentTNLLLoss loss = Api::StudentTNLLLoss::Builder()
                                            .network(network)
                                            .location(locationTensor)
                                            .logScale(logScaleTensor)
                                            .target(labelsTensor)
                                            .logDegreesOfFreedom(logDegreesOfFreedomTensor)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 8.0e-5f);
    expectClose(actual.firstGradient, expectedLocationGradient, ThorTest::lossScaleAwareGradientTolerance(6.0e-3f));
    expectClose(actual.secondGradient, expectedLogScaleGradient, ThorTest::lossScaleAwareGradientTolerance(6.0e-3f));
    expectClose(actual.thirdGradient,
                expectedLogDegreesOfFreedomGradient,
                ThorTest::lossScaleAwareGradientTolerance(9.0e-3f));
}

TEST(StudentTNLLLossApi, LearnedLogDegreesOfFreedomWithMinimumForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> location = {-1.5f, 0.25f, 2.0f, 7.5f, 1.25f, -3.0f, 4.5f, 0.75f};
    const vector<float> scale = {0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 2.0f, 0.35f, 1.25f};
    constexpr float minimumDegreesOfFreedom = 2.0f;
    const vector<float> degreesOfFreedom = {2.5f, 3.0f, 4.0f, 5.0f, 8.0f, 12.0f, 4.5f, 2.75f};
    const vector<float> labels = {-0.5f, -1.0f, 2.75f, 4.0f, 0.0f, -1.5f, 6.0f, 1.5f};
    vector<float> logScale(scale.size());
    vector<float> logDegreesOfFreedom(degreesOfFreedom.size());
    for (size_t i = 0; i < scale.size(); ++i) {
        logScale[i] = std::log(scale[i]);
        logDegreesOfFreedom[i] = std::log(degreesOfFreedom[i] - minimumDegreesOfFreedom);
    }

    ReferenceThreeParameterElementLoss reference = [](double locationValue, double logScaleValue, double logNu, double target) {
        constexpr double minimumNu = 2.0;
        const double nu = minimumNu + std::exp(logNu);
        const double z = (locationValue - target) * std::exp(-logScaleValue);
        return logScaleValue + std::lgamma(0.5 * nu) - std::lgamma(0.5 * (nu + 1.0)) +
               0.5 * std::log(nu * 3.1415926535897932384626433832795) + 0.5 * (nu + 1.0) * std::log1p(z * z / nu);
    };

    vector<float> expectedLoss = referenceThreeParameterRawLoss(location, logScale, logDegreesOfFreedom, labels, reference);
    vector<float> expectedLocationGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 0, reference);
    vector<float> expectedLogScaleGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 1, reference);
    vector<float> expectedLogDegreesOfFreedomGradient =
        numericalThreeParameterGradient(location, logScale, logDegreesOfFreedom, labels, 2, reference);
    for (float& value : expectedLocationGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogScaleGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogDegreesOfFreedomGradient)
        value *= Impl::Loss::getLossScalingFactor();

    ThreeParameterLossRunResult actual = runThreeParameterLoss(
        "student_t_learned_df_minimum_numerical",
        location,
        logScale,
        logDegreesOfFreedom,
        labels,
        [](Api::Network& network,
           Api::Tensor locationTensor,
           Api::Tensor logScaleTensor,
           Api::Tensor logDegreesOfFreedomTensor,
           Api::Tensor labelsTensor) {
            Api::StudentTNLLLoss loss = Api::StudentTNLLLoss::Builder()
                                            .network(network)
                                            .location(locationTensor)
                                            .logScale(logScaleTensor)
                                            .target(labelsTensor)
                                            .logDegreesOfFreedom(logDegreesOfFreedomTensor)
                                            .minimumDegreesOfFreedom(2.0f)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 8.0e-5f);
    expectClose(actual.firstGradient, expectedLocationGradient, ThorTest::lossScaleAwareGradientTolerance(6.0e-3f));
    expectClose(actual.secondGradient, expectedLogScaleGradient, ThorTest::lossScaleAwareGradientTolerance(6.0e-3f));
    expectClose(actual.thirdGradient,
                expectedLogDegreesOfFreedomGradient,
                ThorTest::lossScaleAwareGradientTolerance(9.0e-3f));
}

TEST(GaussianNLLLossApi, LogVarianceForwardAndBackwardMatchFiniteDifferences) {
    const vector<float> mean = {0.0f, 0.25f, 1.5f, -2.0f, -1.0f, 0.75f, 2.25f, -0.5f};
    const vector<float> variance = {0.5f, 1.25f, 3.5f, 0.75f, 2.0f, 0.25f, 1.5f, 4.0f};
    const vector<float> labels = {0.0f, -0.25f, 0.0f, -0.5f, 0.5f, 0.25f, 1.0f, -1.5f};
    vector<float> logVariance(variance.size());
    for (size_t i = 0; i < variance.size(); ++i)
        logVariance[i] = std::log(variance[i]);

    ReferenceElementLoss reference = [](double meanValue, double logVarianceValue, double target) {
        const double diff = meanValue - target;
        return 0.5 * (logVarianceValue + diff * diff * std::exp(-logVarianceValue));
    };

    vector<float> expectedLoss = referenceRawLoss(mean, logVariance, labels, reference);
    vector<float> expectedMeanGradient = numericalGradient(mean, logVariance, labels, true, reference);
    vector<float> expectedLogVarianceGradient = numericalGradient(mean, logVariance, labels, false, reference);
    for (float& value : expectedMeanGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : expectedLogVarianceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TwoParameterLossRunResult actual = runTwoParameterLoss(
        "gaussian_log_variance_numerical",
        mean,
        logVariance,
        labels,
        2,
        [](Api::Network& network, Api::Tensor meanTensor, Api::Tensor logVarianceTensor, Api::Tensor labelsTensor) {
            Api::GaussianNLLLoss loss = Api::GaussianNLLLoss::Builder()
                                            .network(network)
                                            .mean(meanTensor)
                                            .target(labelsTensor)
                                            .variance(logVarianceTensor)
                                            .logVariance(true)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    expectClose(actual.loss, expectedLoss, 3.0e-5f);
    expectClose(actual.firstGradient, expectedMeanGradient, ThorTest::lossScaleAwareGradientTolerance(4.0e-3f));
    expectClose(actual.secondGradient, expectedLogVarianceGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
}

TEST(DistributionLossApi, LearnedDistributionParametersAreDifferentiableNamedInputsAndWeightsRemainAuxiliary) {
    {
        Api::Network network("gaussian_distribution_input_contract");
        Api::Tensor mean(Api::DataType::FP32, {3});
        Api::Tensor labels(Api::DataType::FP32, {3});
        Api::Tensor logVariance(Api::DataType::FP32, {3});
        Api::Tensor weights(Api::DataType::FP32, {1});
        Api::GaussianNLLLoss loss = Api::GaussianNLLLoss::Builder()
                                        .network(network)
                                        .mean(mean)
                                        .target(labels)
                                        .variance(logVariance)
                                        .logVariance(true)
                                        .exampleWeights(weights)
                                        .reportsRawLoss()
                                        .build();
        EXPECT_EQ(loss.getLossInputTensors(), (vector<Api::Tensor>{mean, labels, logVariance, weights}));
        EXPECT_EQ(loss.getConnectionType(logVariance), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        ASSERT_TRUE(loss.getInputPortName(logVariance).has_value());
        EXPECT_EQ(loss.getInputPortName(logVariance).value(), "variance");
        EXPECT_EQ(loss.getConnectionType(weights), (int)Impl::Loss::ConnectionType::LABELS);
    }

    {
        Api::Network network("gamma_distribution_input_contract");
        Api::Tensor mean(Api::DataType::FP32, {3});
        Api::Tensor labels(Api::DataType::FP32, {3});
        Api::Tensor logDispersion(Api::DataType::FP32, {3});
        Api::Tensor weights(Api::DataType::FP32, {1});
        Api::GammaNLLLoss loss = Api::GammaNLLLoss::Builder()
                                     .network(network)
                                     .mean(mean)
                                     .target(labels)
                                     .dispersion(logDispersion)
                                     .logMean(true)
                                     .logDispersion(true)
                                     .exampleWeights(weights)
                                     .reportsRawLoss()
                                     .build();
        EXPECT_EQ(loss.getLossInputTensors(), (vector<Api::Tensor>{mean, labels, logDispersion, weights}));
        EXPECT_EQ(loss.getConnectionType(logDispersion), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        ASSERT_TRUE(loss.getInputPortName(logDispersion).has_value());
        EXPECT_EQ(loss.getInputPortName(logDispersion).value(), "dispersion");
        EXPECT_EQ(loss.getConnectionType(weights), (int)Impl::Loss::ConnectionType::LABELS);
    }

    {
        Api::Network network("laplace_distribution_input_contract");
        Api::Tensor location(Api::DataType::FP32, {3});
        Api::Tensor labels(Api::DataType::FP32, {3});
        Api::Tensor logScale(Api::DataType::FP32, {3});
        Api::Tensor weights(Api::DataType::FP32, {1});
        Api::LaplaceNLLLoss loss = Api::LaplaceNLLLoss::Builder()
                                       .network(network)
                                       .location(location)
                                       .target(labels)
                                       .scale(logScale)
                                       .logScale(true)
                                       .exampleWeights(weights)
                                       .reportsRawLoss()
                                       .build();
        EXPECT_EQ(loss.getLossInputTensors(), (vector<Api::Tensor>{location, labels, logScale, weights}));
        EXPECT_EQ(loss.getConnectionType(logScale), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        ASSERT_TRUE(loss.getInputPortName(logScale).has_value());
        EXPECT_EQ(loss.getInputPortName(logScale).value(), "scale");
        EXPECT_EQ(loss.getConnectionType(weights), (int)Impl::Loss::ConnectionType::LABELS);
    }

    {
        Api::Network network("student_t_distribution_input_contract");
        Api::Tensor location(Api::DataType::FP32, {3});
        Api::Tensor labels(Api::DataType::FP32, {3});
        Api::Tensor logScale(Api::DataType::FP32, {3});
        Api::Tensor logDegreesOfFreedom(Api::DataType::FP32, {3});
        Api::Tensor weights(Api::DataType::FP32, {1});
        Api::StudentTNLLLoss loss = Api::StudentTNLLLoss::Builder()
                                       .network(network)
                                       .location(location)
                                       .target(labels)
                                       .logScale(logScale)
                                       .logDegreesOfFreedom(logDegreesOfFreedom)
                                       .exampleWeights(weights)
                                       .reportsRawLoss()
                                       .build();
        EXPECT_EQ(loss.getLossInputTensors(), (vector<Api::Tensor>{location, labels, logScale, logDegreesOfFreedom, weights}));
        EXPECT_EQ(loss.getConnectionType(logScale), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        EXPECT_EQ(loss.getConnectionType(logDegreesOfFreedom), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        ASSERT_TRUE(loss.getInputPortName(logScale).has_value());
        EXPECT_EQ(loss.getInputPortName(logScale).value(), "log_scale");
        ASSERT_TRUE(loss.getInputPortName(logDegreesOfFreedom).has_value());
        EXPECT_EQ(loss.getInputPortName(logDegreesOfFreedom).value(), "log_degrees_of_freedom");
        EXPECT_EQ(loss.getConnectionType(weights), (int)Impl::Loss::ConnectionType::LABELS);
    }

    {
        Api::Network network("negative_binomial_distribution_input_contract");
        Api::Tensor mean(Api::DataType::FP32, {3});
        Api::Tensor labels(Api::DataType::FP32, {3});
        Api::Tensor logDispersion(Api::DataType::FP32, {3});
        Api::Tensor weights(Api::DataType::FP32, {1});
        Api::NegativeBinomialNLLLoss loss = Api::NegativeBinomialNLLLoss::Builder()
                                                .network(network)
                                                .mean(mean)
                                                .dispersion(logDispersion)
                                                .labels(labels)
                                                .exampleWeights(weights)
                                                .reportsRawLoss()
                                                .build();
        EXPECT_EQ(loss.getLossInputTensors(), (vector<Api::Tensor>{mean, labels, logDispersion, weights}));
        EXPECT_EQ(loss.getConnectionType(logDispersion), (int)Impl::Loss::ConnectionType::FORWARD_BACKWARD);
        ASSERT_TRUE(loss.getInputPortName(logDispersion).has_value());
        EXPECT_EQ(loss.getInputPortName(logDispersion).value(), "dispersion");
        EXPECT_EQ(loss.getConnectionType(weights), (int)Impl::Loss::ConnectionType::LABELS);
    }
}
