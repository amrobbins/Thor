#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticGradientPenaltyLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"
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

constexpr uint32_t kBatchSize = 2;
constexpr uint32_t kScoreCount = 3;

vector<unsigned long> tensorDimsWithBatch(uint32_t batchSize, const vector<uint64_t>& featureDims) {
    vector<unsigned long> dims;
    dims.reserve(featureDims.size() + 1);
    dims.push_back(batchSize);
    for (uint64_t dim : featureDims)
        dims.push_back(static_cast<unsigned long>(dim));
    return dims;
}

uint64_t featureElementCount(const vector<uint64_t>& featureDims) {
    uint64_t elements = 1;
    for (uint64_t dim : featureDims)
        elements *= dim;
    return elements;
}

Impl::Tensor makeCpuTensor(const vector<float>& values, uint32_t batchSize, const vector<uint64_t>& featureDims) {
    THOR_THROW_IF_FALSE(values.size() == static_cast<size_t>(batchSize * featureElementCount(featureDims)));
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, tensorDimsWithBatch(batchSize, featureDims));
    Impl::Tensor tensor(cpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), static_cast<float*>(tensor.getMemPtr()));
    return tensor;
}

vector<float> tensorToVector(const Impl::Tensor& tensor) {
    const float* begin = static_cast<const float*>(tensor.getMemPtr());
    return vector<float>(begin, begin + tensor.getTotalNumElements());
}

vector<float> copyGpuTensorToVector(const optional<Impl::Tensor>& tensor, Stream stream) {
    THOR_THROW_IF_FALSE(tensor.has_value());
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpu(cpuPlacement, tensor.value().getDescriptor());
    cpu.copyFromAsync(tensor.value(), stream);
    stream.synchronize();
    return tensorToVector(cpu);
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

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

struct CriticRunResult {
    vector<float> loss;
    vector<float> realGradient;
    vector<float> fakeGradient;
};

CriticRunResult runRawCriticLossNetwork(const vector<float>& realScores, const vector<float>& fakeScores) {
    THOR_THROW_IF_FALSE(realScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));
    THOR_THROW_IF_FALSE(fakeScores.size() == realScores.size());

    Api::Network network("wasserstein_gan_critic_gradient_numerical");
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

    Api::WassersteinGANCriticLoss loss = Api::WassersteinGANCriticLoss::Builder()
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

    physicalRealScoresInput->forward(makeCpuTensor(realScores, kBatchSize, {kScoreCount}), false, kBatchSize);
    physicalFakeScoresInput->forward(makeCpuTensor(fakeScores, kBatchSize, {kScoreCount}), false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);

    return {tensorToVector(outputLossCpu),
            copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0), physicalRawLoss->getStream()),
            copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1), physicalRawLoss->getStream())};
}

struct GeneratorRunResult {
    vector<float> loss;
    vector<float> fakeGradient;
};

GeneratorRunResult runRawGeneratorLossNetwork(const vector<float>& fakeScores) {
    THOR_THROW_IF_FALSE(fakeScores.size() == static_cast<size_t>(kBatchSize * kScoreCount));

    Api::Network network("wasserstein_gan_generator_gradient_numerical");
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

    Api::WassersteinGANGeneratorLoss loss = Api::WassersteinGANGeneratorLoss::Builder()
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

    physicalFakeScoresInput->forward(makeCpuTensor(fakeScores, kBatchSize, {kScoreCount}), false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);

    return {tensorToVector(outputLossCpu), copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0), physicalRawLoss->getStream())};
}

struct GradientPenaltyRunResult {
    vector<float> loss;
    vector<float> realGradient;
    vector<float> fakeGradient;
    vector<float> sampleGradient;
};

GradientPenaltyRunResult runRawGradientPenaltyLossNetwork(const vector<float>& realScores,
                                                           const vector<float>& fakeScores,
                                                           const vector<float>& sampleGradients,
                                                           float gradientPenaltyWeight,
                                                           float targetGradientNorm,
                                                           float eps) {
    constexpr uint32_t gradientRows = 2;
    constexpr uint32_t gradientCols = 2;
    THOR_THROW_IF_FALSE(realScores.size() == kBatchSize);
    THOR_THROW_IF_FALSE(fakeScores.size() == kBatchSize);
    THOR_THROW_IF_FALSE(sampleGradients.size() == static_cast<size_t>(kBatchSize * gradientRows * gradientCols));

    Api::Network network("wasserstein_gan_gradient_penalty_gradient_numerical");
    Api::NetworkInput realScoresInput = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("real_scores")
                                            .dimensions({1})
                                            .dataType(Api::DataType::FP32)
                                            .build();
    Api::NetworkInput fakeScoresInput = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("fake_scores")
                                            .dimensions({1})
                                            .dataType(Api::DataType::FP32)
                                            .build();
    Api::NetworkInput sampleGradientsInput = Api::NetworkInput::Builder()
                                                 .network(network)
                                                 .name("sample_gradients")
                                                 .dimensions({gradientRows, gradientCols})
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
    Api::GradientRivet sampleGradientsRivet = Api::GradientRivet::Builder()
                                                 .network(network)
                                                 .tensor(sampleGradientsInput.getFeatureOutput().value())
                                                 .build();

    Api::WassersteinGANCriticGradientPenaltyLoss loss = Api::WassersteinGANCriticGradientPenaltyLoss::Builder()
                                                            .network(network)
                                                            .realScores(realScoresRivet.getFeatureOutput().value())
                                                            .fakeScores(fakeScoresRivet.getFeatureOutput().value())
                                                            .sampleGradients(sampleGradientsRivet.getFeatureOutput().value())
                                                            .gradientPenaltyWeight(gradientPenaltyWeight)
                                                            .targetGradientNorm(targetGradientNorm)
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
    shared_ptr<Impl::NetworkInput> physicalSampleGradientsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(sampleGradientsInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalRealScoresInput != nullptr);
    THOR_THROW_IF_FALSE(physicalFakeScoresInput != nullptr);
    THOR_THROW_IF_FALSE(physicalSampleGradientsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    physicalRealScoresInput->forward(makeCpuTensor(realScores, kBatchSize, {1}), false, kBatchSize);
    physicalFakeScoresInput->forward(makeCpuTensor(fakeScores, kBatchSize, {1}), false, kBatchSize);
    physicalSampleGradientsInput->forward(makeCpuTensor(sampleGradients, kBatchSize, {gradientRows, gradientCols}), false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);

    return {tensorToVector(outputLossCpu),
            copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0), physicalRawLoss->getStream()),
            copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1), physicalRawLoss->getStream()),
            copyGpuTensorToVector(physicalRawLoss->getErrorOutput(2), physicalRawLoss->getStream())};
}

vector<float> constantGradient(size_t size, float value) { return vector<float>(size, value * Impl::Loss::getLossScalingFactor()); }

}  // namespace

TEST(WassersteinGANLossApi, CriticNumericalRawLossAndBackwardGradientsMatchReference) {
    const vector<float> realScores = {1.5f, -0.5f, 0.25f, 2.0f, 0.75f, -1.25f};
    const vector<float> fakeScores = {-0.25f, 1.0f, 0.5f, -0.75f, 1.5f, 0.25f};

    CriticRunResult actual = runRawCriticLossNetwork(realScores, fakeScores);
    vector<float> expectedLoss(realScores.size(), 0.0f);
    for (size_t i = 0; i < expectedLoss.size(); ++i)
        expectedLoss[i] = fakeScores[i] - realScores[i];

    expectClose(actual.loss, expectedLoss, 1.0e-6f);
    expectClose(actual.realGradient, constantGradient(realScores.size(), -1.0f), 1.0e-6f);
    expectClose(actual.fakeGradient, constantGradient(fakeScores.size(), 1.0f), 1.0e-6f);
}

TEST(WassersteinGANLossApi, GeneratorNumericalRawLossAndBackwardGradientMatchReference) {
    const vector<float> fakeScores = {-0.25f, 1.0f, 0.5f, -0.75f, 1.5f, 0.25f};

    GeneratorRunResult actual = runRawGeneratorLossNetwork(fakeScores);
    vector<float> expectedLoss(fakeScores.size(), 0.0f);
    for (size_t i = 0; i < expectedLoss.size(); ++i)
        expectedLoss[i] = -fakeScores[i];

    expectClose(actual.loss, expectedLoss, 1.0e-6f);
    expectClose(actual.fakeGradient, constantGradient(fakeScores.size(), -1.0f), 1.0e-6f);
}

TEST(WassersteinGANLossApi, GradientPenaltyNumericalRawLossAndBackwardGradientsMatchReference) {
    constexpr float gradientPenaltyWeight = 3.5f;
    constexpr float targetGradientNorm = 1.25f;
    constexpr float eps = 1.0e-8f;
    const vector<float> realScores = {1.5f, -0.5f};
    const vector<float> fakeScores = {-0.25f, 1.0f};
    const vector<float> sampleGradients = {0.5f, 1.0f, -0.25f, 0.75f, -0.5f, 0.25f, 1.25f, -0.75f};

    GradientPenaltyRunResult actual =
        runRawGradientPenaltyLossNetwork(realScores, fakeScores, sampleGradients, gradientPenaltyWeight, targetGradientNorm, eps);

    vector<float> expectedLoss(kBatchSize, 0.0f);
    vector<float> expectedSampleGradient(sampleGradients.size(), 0.0f);
    for (uint32_t batch = 0; batch < kBatchSize; ++batch) {
        double squaredNorm = 0.0;
        for (uint32_t i = 0; i < 4; ++i) {
            const float value = sampleGradients[static_cast<size_t>(batch) * 4 + i];
            squaredNorm += static_cast<double>(value) * value;
        }
        const double norm = std::sqrt(std::max(squaredNorm, static_cast<double>(eps)));
        const double normDiff = norm - targetGradientNorm;
        expectedLoss[batch] = static_cast<float>(fakeScores[batch] - realScores[batch] + gradientPenaltyWeight * normDiff * normDiff);

        const double penaltyScale = squaredNorm > eps ? (2.0 * gradientPenaltyWeight * normDiff / norm) : 0.0;
        for (uint32_t i = 0; i < 4; ++i) {
            const size_t index = static_cast<size_t>(batch) * 4 + i;
            expectedSampleGradient[index] = static_cast<float>(penaltyScale * sampleGradients[index] * Impl::Loss::getLossScalingFactor());
        }
    }

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.realGradient, constantGradient(kBatchSize, -1.0f), 1.0e-6f);
    expectClose(actual.fakeGradient, constantGradient(kBatchSize, 1.0f), 1.0e-6f);
    expectClose(actual.sampleGradient, expectedSampleGradient, 2.0e-5f);
}
