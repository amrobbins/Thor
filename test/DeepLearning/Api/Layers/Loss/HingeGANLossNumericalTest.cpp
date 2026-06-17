#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANGeneratorLoss.h"
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

vector<float> referenceDiscriminatorRawLoss(const vector<float>& realScores, const vector<float>& fakeScores) {
    THOR_THROW_IF_FALSE(realScores.size() == fakeScores.size());
    vector<float> loss(realScores.size(), 0.0f);
    for (size_t i = 0; i < realScores.size(); ++i)
        loss[i] = std::max(1.0f - realScores[i], 0.0f) + std::max(1.0f + fakeScores[i], 0.0f);
    return loss;
}

vector<float> referenceGeneratorRawLoss(const vector<float>& fakeScores) {
    vector<float> loss(fakeScores.size(), 0.0f);
    for (size_t i = 0; i < fakeScores.size(); ++i)
        loss[i] = -fakeScores[i];
    return loss;
}

float sumValues(const vector<float>& values) {
    float total = 0.0f;
    for (float value : values)
        total += value;
    return total;
}

float totalDiscriminatorLoss(const vector<float>& realScores, const vector<float>& fakeScores) {
    return sumValues(referenceDiscriminatorRawLoss(realScores, fakeScores));
}

float totalGeneratorLoss(const vector<float>& fakeScores) { return sumValues(referenceGeneratorRawLoss(fakeScores)); }

vector<float> numericalDiscriminatorRealGradient(const vector<float>& realScores, const vector<float>& fakeScores) {
    vector<float> gradient(realScores.size(), 0.0f);
    vector<float> perturbed = realScores;
    for (size_t i = 0; i < realScores.size(); ++i) {
        perturbed[i] = realScores[i] + kFiniteDifferenceEpsilon;
        const float lossPlus = totalDiscriminatorLoss(perturbed, fakeScores);
        perturbed[i] = realScores[i] - kFiniteDifferenceEpsilon;
        const float lossMinus = totalDiscriminatorLoss(perturbed, fakeScores);
        perturbed[i] = realScores[i];
        gradient[i] = (lossPlus - lossMinus) / (2.0f * kFiniteDifferenceEpsilon);
    }
    return gradient;
}

vector<float> numericalDiscriminatorFakeGradient(const vector<float>& realScores, const vector<float>& fakeScores) {
    vector<float> gradient(fakeScores.size(), 0.0f);
    vector<float> perturbed = fakeScores;
    for (size_t i = 0; i < fakeScores.size(); ++i) {
        perturbed[i] = fakeScores[i] + kFiniteDifferenceEpsilon;
        const float lossPlus = totalDiscriminatorLoss(realScores, perturbed);
        perturbed[i] = fakeScores[i] - kFiniteDifferenceEpsilon;
        const float lossMinus = totalDiscriminatorLoss(realScores, perturbed);
        perturbed[i] = fakeScores[i];
        gradient[i] = (lossPlus - lossMinus) / (2.0f * kFiniteDifferenceEpsilon);
    }
    return gradient;
}

vector<float> numericalGeneratorGradient(const vector<float>& fakeScores) {
    vector<float> gradient(fakeScores.size(), 0.0f);
    vector<float> perturbed = fakeScores;
    for (size_t i = 0; i < fakeScores.size(); ++i) {
        perturbed[i] = fakeScores[i] + kFiniteDifferenceEpsilon;
        const float lossPlus = totalGeneratorLoss(perturbed);
        perturbed[i] = fakeScores[i] - kFiniteDifferenceEpsilon;
        const float lossMinus = totalGeneratorLoss(perturbed);
        perturbed[i] = fakeScores[i];
        gradient[i] = (lossPlus - lossMinus) / (2.0f * kFiniteDifferenceEpsilon);
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

DiscriminatorRunResult runRawDiscriminatorLossNetwork(const vector<float>& realScores, const vector<float>& fakeScores) {
    THOR_THROW_IF_FALSE(realScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));
    THOR_THROW_IF_FALSE(fakeScores.size() == realScores.size());

    Api::Network network("hinge_gan_discriminator_numerical");
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

    Api::HingeGANDiscriminatorLoss loss = Api::HingeGANDiscriminatorLoss::Builder()
                                              .network(network)
                                              .realScores(realScoresRivet.getFeatureOutput().value())
                                              .fakeScores(fakeScoresRivet.getFeatureOutput().value())
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

GeneratorRunResult runRawGeneratorLossNetwork(const vector<float>& fakeScores) {
    THOR_THROW_IF_FALSE(fakeScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));

    Api::Network network("hinge_gan_generator_numerical");
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

    Api::HingeGANGeneratorLoss loss = Api::HingeGANGeneratorLoss::Builder()
                                          .network(network)
                                          .fakeScores(fakeScoresRivet.getFeatureOutput().value())
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

void scaleByLossScalingFactor(vector<float>& values) {
    for (float& value : values)
        value *= Impl::Loss::getLossScalingFactor();
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(HingeGANLossApi, DiscriminatorNumericalRawLossAndBackwardGradientsMatchReference) {
    const vector<float> realScores = {1.5f, 0.2f, -0.5f,
                                      0.9f, 1.2f, 2.0f,
                                      -1.2f, 0.4f, 1.1f};
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    vector<float> referenceLoss = referenceDiscriminatorRawLoss(realScores, fakeScores);
    vector<float> referenceRealGradient = numericalDiscriminatorRealGradient(realScores, fakeScores);
    vector<float> referenceFakeGradient = numericalDiscriminatorFakeGradient(realScores, fakeScores);
    scaleByLossScalingFactor(referenceRealGradient);
    scaleByLossScalingFactor(referenceFakeGradient);

    DiscriminatorRunResult actual = runRawDiscriminatorLossNetwork(realScores, fakeScores);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.realGradient, referenceRealGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
    expectClose(actual.fakeGradient, referenceFakeGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}

TEST(HingeGANLossApi, GeneratorNumericalRawLossAndBackwardGradientMatchReference) {
    const vector<float> fakeScores = {-1.5f, -0.2f, 0.5f,
                                      -0.9f, -1.2f, -2.0f,
                                      1.2f, -0.4f, -1.1f};

    vector<float> referenceLoss = referenceGeneratorRawLoss(fakeScores);
    vector<float> referenceGradient = numericalGeneratorGradient(fakeScores);
    scaleByLossScalingFactor(referenceGradient);

    GeneratorRunResult actual = runRawGeneratorLossNetwork(fakeScores);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.fakeGradient, referenceGradient, ThorTest::lossScaleAwareGradientTolerance(2.5e-3f));
}
