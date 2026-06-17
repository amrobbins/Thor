#include "DeepLearning/Api/Layers/Loss/LSGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
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
#include <optional>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint32_t kBatchSize = 3;
constexpr uint32_t kScoreCount = 3;
constexpr float kFiniteDifferenceEpsilon = 1.0e-3f;

vector<float> referenceDiscriminatorRawLoss(const vector<float>& realScores,
                                            const vector<float>& fakeScores,
                                            float realTarget,
                                            float fakeTarget) {
    THOR_THROW_IF_FALSE(realScores.size() == fakeScores.size());
    vector<float> loss(realScores.size(), 0.0f);
    for (size_t i = 0; i < realScores.size(); ++i) {
        const float realDiff = realScores[i] - realTarget;
        const float fakeDiff = fakeScores[i] - fakeTarget;
        loss[i] = 0.5f * (realDiff * realDiff + fakeDiff * fakeDiff);
    }
    return loss;
}

vector<float> referenceGeneratorRawLoss(const vector<float>& fakeScores, float target) {
    vector<float> loss(fakeScores.size(), 0.0f);
    for (size_t i = 0; i < fakeScores.size(); ++i) {
        const float diff = fakeScores[i] - target;
        loss[i] = 0.5f * diff * diff;
    }
    return loss;
}

double sumValuesAsDouble(const vector<float>& values) {
    double total = 0.0;
    for (float value : values)
        total += static_cast<double>(value);
    return total;
}

double totalDiscriminatorLoss(const vector<float>& realScores,
                              const vector<float>& fakeScores,
                              float realTarget,
                              float fakeTarget) {
    return sumValuesAsDouble(referenceDiscriminatorRawLoss(realScores, fakeScores, realTarget, fakeTarget));
}

double totalGeneratorLoss(const vector<float>& fakeScores, float target) {
    return sumValuesAsDouble(referenceGeneratorRawLoss(fakeScores, target));
}

vector<float> numericalDiscriminatorRealGradient(const vector<float>& realScores,
                                                 const vector<float>& fakeScores,
                                                 float realTarget,
                                                 float fakeTarget) {
    vector<float> gradient(realScores.size(), 0.0f);
    vector<float> perturbed = realScores;
    for (size_t i = 0; i < realScores.size(); ++i) {
        perturbed[i] = realScores[i] + kFiniteDifferenceEpsilon;
        const double lossPlus = totalDiscriminatorLoss(perturbed, fakeScores, realTarget, fakeTarget);
        perturbed[i] = realScores[i] - kFiniteDifferenceEpsilon;
        const double lossMinus = totalDiscriminatorLoss(perturbed, fakeScores, realTarget, fakeTarget);
        perturbed[i] = realScores[i];
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * static_cast<double>(kFiniteDifferenceEpsilon)));
    }
    return gradient;
}

vector<float> numericalDiscriminatorFakeGradient(const vector<float>& realScores,
                                                 const vector<float>& fakeScores,
                                                 float realTarget,
                                                 float fakeTarget) {
    vector<float> gradient(fakeScores.size(), 0.0f);
    vector<float> perturbed = fakeScores;
    for (size_t i = 0; i < fakeScores.size(); ++i) {
        perturbed[i] = fakeScores[i] + kFiniteDifferenceEpsilon;
        const double lossPlus = totalDiscriminatorLoss(realScores, perturbed, realTarget, fakeTarget);
        perturbed[i] = fakeScores[i] - kFiniteDifferenceEpsilon;
        const double lossMinus = totalDiscriminatorLoss(realScores, perturbed, realTarget, fakeTarget);
        perturbed[i] = fakeScores[i];
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * static_cast<double>(kFiniteDifferenceEpsilon)));
    }
    return gradient;
}

vector<float> numericalGeneratorGradient(const vector<float>& fakeScores, float target) {
    vector<float> gradient(fakeScores.size(), 0.0f);
    vector<float> perturbed = fakeScores;
    for (size_t i = 0; i < fakeScores.size(); ++i) {
        perturbed[i] = fakeScores[i] + kFiniteDifferenceEpsilon;
        const double lossPlus = totalGeneratorLoss(perturbed, target);
        perturbed[i] = fakeScores[i] - kFiniteDifferenceEpsilon;
        const double lossMinus = totalGeneratorLoss(perturbed, target);
        perturbed[i] = fakeScores[i];
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * static_cast<double>(kFiniteDifferenceEpsilon)));
    }
    return gradient;
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

vector<float> tensorToVector(const Impl::Tensor& tensor) {
    const float* begin = static_cast<const float*>(tensor.getMemPtr());
    return vector<float>(begin, begin + tensor.getTotalNumElements());
}

Impl::Tensor makeCpuTensor(const vector<float>& values) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, {kBatchSize, kScoreCount});
    Impl::Tensor tensor(cpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.getMemPtr()));
    return tensor;
}

vector<float> copyErrorOutputToCpu(const shared_ptr<Impl::MultiInputCustomLoss>& physicalRawLoss, uint32_t inputIndex) {
    optional<Impl::Tensor> errorOutputGpu = physicalRawLoss->getErrorOutput(inputIndex);
    THOR_THROW_IF_FALSE(errorOutputGpu.has_value());

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor errorOutputCpu(cpuPlacement, errorOutputGpu.value().getDescriptor());
    Stream rawLossStream = physicalRawLoss->getStream();
    errorOutputCpu.copyFromAsync(errorOutputGpu.value(), rawLossStream);
    rawLossStream.synchronize();
    return tensorToVector(errorOutputCpu);
}

struct DiscriminatorRunResult {
    vector<float> loss;
    vector<float> realGradient;
    vector<float> fakeGradient;
};

DiscriminatorRunResult runRawDiscriminatorLossNetwork(const vector<float>& realScores,
                                                  const vector<float>& fakeScores,
                                                  float realTarget,
                                                  float fakeTarget,
                                                  optional<float> lossWeight = nullopt) {
    THOR_THROW_IF_FALSE(realScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));
    THOR_THROW_IF_FALSE(fakeScores.size() == realScores.size());

    Api::Network network("lsgan_discriminator_numerical");
    Api::NetworkInput realScoresInput = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("real_scores")
                                            .dimensions({kScoreCount})
                                            .dataType(Api::DataType::FP32)
                                            .build();
    Api::NetworkInput fakeScoresInput = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("fake_scores")
                                            .dimensions({kScoreCount})
                                            .dataType(Api::DataType::FP32)
                                            .build();

    Api::GradientRivet realScoresRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(realScoresInput.getFeatureOutput().value())
                                             .build();
    Api::GradientRivet fakeScoresRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(fakeScoresInput.getFeatureOutput().value())
                                             .build();

    Api::LSGANDiscriminatorLoss::Builder lossBuilder = Api::LSGANDiscriminatorLoss::Builder()
                                                         .network(network)
                                                         .realScores(realScoresRivet.getFeatureOutput().value())
                                                         .fakeScores(fakeScoresRivet.getFeatureOutput().value())
                                                         .lossDataType(Api::DataType::FP32)
                                                         .realTarget(realTarget)
                                                         .fakeTarget(fakeTarget)
                                                         .reportsRawLoss();
    if (lossWeight.has_value())
        lossBuilder.lossWeight(lossWeight.value());
    Api::LSGANDiscriminatorLoss loss = lossBuilder.build();
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(loss.getLoss())
                                        .dataType(Api::DataType::FP32)
                                        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    Stream stream(0);
    for (Event& event : initDoneEvents)
        stream.waitEvent(event);
    stream.synchronize();
    initDoneEvents.clear();

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalRealScoresInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(realScoresInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalFakeScoresInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(fakeScoresInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalRealScoresInput != nullptr);
    THOR_THROW_IF_FALSE(physicalFakeScoresInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::Tensor realScoresCpu = makeCpuTensor(realScores);
    Impl::Tensor fakeScoresCpu = makeCpuTensor(fakeScores);
    physicalRealScoresInput->forward(realScoresCpu, false, kBatchSize);
    physicalFakeScoresInput->forward(fakeScoresCpu, false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss = tensorToVector(outputLossCpu);

    return DiscriminatorRunResult{outputLoss, copyErrorOutputToCpu(physicalRawLoss, 0), copyErrorOutputToCpu(physicalRawLoss, 1)};
}

struct GeneratorRunResult {
    vector<float> loss;
    vector<float> fakeGradient;
};

GeneratorRunResult runRawGeneratorLossNetwork(const vector<float>& fakeScores, float target, optional<float> lossWeight = nullopt) {
    THOR_THROW_IF_FALSE(fakeScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));

    Api::Network network("lsgan_generator_numerical");
    Api::NetworkInput fakeScoresInput = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("fake_scores")
                                            .dimensions({kScoreCount})
                                            .dataType(Api::DataType::FP32)
                                            .build();
    Api::GradientRivet fakeScoresRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(fakeScoresInput.getFeatureOutput().value())
                                             .build();

    Api::LSGANGeneratorLoss::Builder lossBuilder = Api::LSGANGeneratorLoss::Builder()
                                                    .network(network)
                                                    .fakeScores(fakeScoresRivet.getFeatureOutput().value())
                                                    .lossDataType(Api::DataType::FP32)
                                                    .target(target)
                                                    .reportsRawLoss();
    if (lossWeight.has_value())
        lossBuilder.lossWeight(lossWeight.value());
    Api::LSGANGeneratorLoss loss = lossBuilder.build();
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(loss.getLoss())
                                        .dataType(Api::DataType::FP32)
                                        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    Stream stream(0);
    for (Event& event : initDoneEvents)
        stream.waitEvent(event);
    stream.synchronize();
    initDoneEvents.clear();

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalFakeScoresInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(fakeScoresInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalFakeScoresInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::Tensor fakeScoresCpu = makeCpuTensor(fakeScores);
    physicalFakeScoresInput->forward(fakeScoresCpu, false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss = tensorToVector(outputLossCpu);

    return GeneratorRunResult{outputLoss, copyErrorOutputToCpu(physicalRawLoss, 0)};
}

void scaleValues(vector<float>& values, float scale) {
    for (float& value : values)
        value *= scale;
}

void scaleByLossScalingFactor(vector<float>& values) { scaleValues(values, Impl::Loss::getLossScalingFactor()); }

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(LSGANLossApi, DiscriminatorNumericalRawLossAndBackwardGradientsMatchReference) {
    const vector<float> realScores = {1.5f, 0.2f, -0.5f,
                                      0.9f, 1.2f, 2.0f,
                                      -1.2f, 0.4f, 1.1f};
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    const float realTarget = 0.9f;
    const float fakeTarget = -0.2f;

    vector<float> referenceLoss = referenceDiscriminatorRawLoss(realScores, fakeScores, realTarget, fakeTarget);
    vector<float> referenceRealGradient = numericalDiscriminatorRealGradient(realScores, fakeScores, realTarget, fakeTarget);
    vector<float> referenceFakeGradient = numericalDiscriminatorFakeGradient(realScores, fakeScores, realTarget, fakeTarget);
    scaleByLossScalingFactor(referenceRealGradient);
    scaleByLossScalingFactor(referenceFakeGradient);

    DiscriminatorRunResult actual = runRawDiscriminatorLossNetwork(realScores, fakeScores, realTarget, fakeTarget);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.realGradient, referenceRealGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
    expectClose(actual.fakeGradient, referenceFakeGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}


TEST(LSGANLossApi, DiscriminatorLossWeightScalesReportedLossAndAllPredictionGradients) {
    const vector<float> realScores = {1.5f, 0.2f, -0.5f,
                                      0.9f, 1.2f, 2.0f,
                                      -1.2f, 0.4f, 1.1f};
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    const float realTarget = 0.9f;
    const float fakeTarget = -0.2f;
    const float lossWeight = 2.75f;

    vector<float> referenceLoss = referenceDiscriminatorRawLoss(realScores, fakeScores, realTarget, fakeTarget);
    vector<float> referenceRealGradient = numericalDiscriminatorRealGradient(realScores, fakeScores, realTarget, fakeTarget);
    vector<float> referenceFakeGradient = numericalDiscriminatorFakeGradient(realScores, fakeScores, realTarget, fakeTarget);
    scaleValues(referenceLoss, lossWeight);
    scaleValues(referenceRealGradient, Impl::Loss::getLossScalingFactor() * lossWeight);
    scaleValues(referenceFakeGradient, Impl::Loss::getLossScalingFactor() * lossWeight);

    DiscriminatorRunResult actual = runRawDiscriminatorLossNetwork(realScores, fakeScores, realTarget, fakeTarget, lossWeight);

    expectClose(actual.loss, referenceLoss, 3.0e-5f);
    expectClose(actual.realGradient, referenceRealGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
    expectClose(actual.fakeGradient, referenceFakeGradient, ThorTest::lossScaleAwareGradientTolerance(5.0e-3f));
}

TEST(LSGANLossApi, GeneratorNumericalRawLossAndBackwardGradientMatchReference) {
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    const float target = 0.8f;

    vector<float> referenceLoss = referenceGeneratorRawLoss(fakeScores, target);
    vector<float> referenceGradient = numericalGeneratorGradient(fakeScores, target);
    scaleByLossScalingFactor(referenceGradient);

    GeneratorRunResult actual = runRawGeneratorLossNetwork(fakeScores, target);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.fakeGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}

TEST(LSGANLossApi, GeneratorLossWeightScalesReportedLossAndPredictionGradient) {
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    const float target = 0.8f;
    const float lossWeight = 0.375f;

    vector<float> referenceLoss = referenceGeneratorRawLoss(fakeScores, target);
    vector<float> referenceGradient = numericalGeneratorGradient(fakeScores, target);
    scaleValues(referenceLoss, lossWeight);
    scaleValues(referenceGradient, Impl::Loss::getLossScalingFactor() * lossWeight);

    GeneratorRunResult actual = runRawGeneratorLossNetwork(fakeScores, target, lossWeight);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.fakeGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}
