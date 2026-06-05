#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"
#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

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

enum class LossKind { SMOOTH_L1, HUBER };

float referenceLoss(LossKind lossKind, float diff, float threshold) {
    float absDiff = std::fabs(diff);
    if (lossKind == LossKind::SMOOTH_L1) {
        if (absDiff < threshold)
            return 0.5f * diff * diff / threshold;
        return absDiff - 0.5f * threshold;
    }

    if (absDiff <= threshold)
        return 0.5f * diff * diff;
    return threshold * (absDiff - 0.5f * threshold);
}

vector<float> computeReferenceRawLoss(LossKind lossKind, const vector<float>& predictions, const vector<float>& labels, float threshold) {
    vector<float> reference(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - labels[i];
        reference[i] = referenceLoss(lossKind, diff, threshold);
    }
    return reference;
}

vector<float> runRawLossNetwork(LossKind lossKind, const vector<float>& predictions, const vector<float>& labels, float threshold) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 5;
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * numFeatures));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network(lossKind == LossKind::SMOOTH_L1 ? "smooth_l1_numerical" : "huber_numerical");
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

    Api::Tensor lossTensor;
    if (lossKind == LossKind::SMOOTH_L1) {
        Api::SmoothL1Loss loss = Api::SmoothL1Loss::Builder()
                                     .network(network)
                                     .predictions(predictionsInput.getFeatureOutput().value())
                                     .labels(labelsInput.getFeatureOutput().value())
                                     .beta(threshold)
                                     .lossDataType(Api::DataType::FP32)
                                     .reportsRawLoss()
                                     .build();
        lossTensor = loss.getLoss();
    } else {
        Api::HuberLoss loss = Api::HuberLoss::Builder()
                                  .network(network)
                                  .predictions(predictionsInput.getFeatureOutput().value())
                                  .labels(labelsInput.getFeatureOutput().value())
                                  .delta(threshold)
                                  .lossDataType(Api::DataType::FP32)
                                  .reportsRawLoss()
                                  .build();
        lossTensor = loss.getLoss();
    }

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(lossTensor)
                                        .dataType(Api::DataType::FP32)
                                        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, true, {0}, 1);
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
    THOR_THROW_IF_FALSE(physicalPredictionsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(numFeatures)});
    Impl::Tensor predictionsCpu(cpuPlacement, descriptor);
    Impl::Tensor labelsCpu(cpuPlacement, descriptor);
    std::copy(predictions.begin(), predictions.end(), static_cast<float*>(predictionsCpu.getMemPtr()));
    std::copy(labels.begin(), labels.end(), static_cast<float*>(labelsCpu.getMemPtr()));

    physicalPredictionsInput->forward(predictionsCpu, false, batchSize);
    physicalLabelsInput->forward(labelsCpu, false, batchSize);

    Stream labelsStream = physicalLabelsInput->getStream();
    labelsStream.waitEvent(physicalPredictionsInput->getStream().putEvent());
    labelsStream.waitEvent(physicalLossOutput->getOutputReadyEvent());

    labelsStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    return vector<float>(static_cast<float*>(outputLossCpu.getMemPtr()),
                         static_cast<float*>(outputLossCpu.getMemPtr()) + predictions.size());
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(SmoothL1LossApi, NumericalRawLossMatchesReference) {
    const float beta = 0.75f;
    const vector<float> predictions = {0.0f, 0.25f, 1.5f, -2.0f, 3.0f, -1.0f, 0.75f, 2.25f, -0.5f, 0.125f};
    const vector<float> labels = {0.0f, -0.25f, 0.0f, -0.5f, 0.5f, 0.5f, 0.25f, 1.0f, -1.5f, -0.125f};

    vector<float> reference = computeReferenceRawLoss(LossKind::SMOOTH_L1, predictions, labels, beta);
    vector<float> actual = runRawLossNetwork(LossKind::SMOOTH_L1, predictions, labels, beta);

    expectClose(actual, reference, 1.0e-5f);
}

TEST(HuberLossApi, NumericalRawLossMatchesReference) {
    const float delta = 0.75f;
    const vector<float> predictions = {0.0f, 0.25f, 1.5f, -2.0f, 3.0f, -1.0f, 0.75f, 2.25f, -0.5f, 0.125f};
    const vector<float> labels = {0.0f, -0.25f, 0.0f, -0.5f, 0.5f, 0.5f, 0.25f, 1.0f, -1.5f, -0.125f};

    vector<float> reference = computeReferenceRawLoss(LossKind::HUBER, predictions, labels, delta);
    vector<float> actual = runRawLossNetwork(LossKind::HUBER, predictions, labels, delta);

    expectClose(actual, reference, 1.0e-5f);
}
