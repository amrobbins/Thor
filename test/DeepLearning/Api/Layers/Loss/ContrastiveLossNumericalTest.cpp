#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
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

float referenceContrastiveLoss(float distance, float label, float margin) {
    if (label > 0.5f)
        return distance * distance;

    const float hinge = std::max(margin - distance, 0.0f);
    return hinge * hinge;
}

float numericalContrastiveGradient(float distance, float label, float margin) {
    constexpr float epsilon = 1.0e-3f;
    const float lossPlus = referenceContrastiveLoss(distance + epsilon, label, margin);
    const float lossMinus = referenceContrastiveLoss(distance - epsilon, label, margin);
    return (lossPlus - lossMinus) / (2.0f * epsilon);
}

struct ContrastiveRunResult {
    vector<float> loss;
    vector<float> gradient;
};

shared_ptr<Api::CustomLoss> findRawCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::CustomLoss> customLoss = dynamic_pointer_cast<Api::CustomLoss>(layer);
        if (customLoss != nullptr)
            return customLoss;
    }
    return nullptr;
}

ContrastiveRunResult runRawContrastiveLossNetwork(const vector<float>& predictions, const vector<float>& labels, float margin) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 5;
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * numFeatures));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network("contrastive_numerical");
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

    Api::ContrastiveLoss loss = Api::ContrastiveLoss::Builder()
                                    .network(network)
                                    .predictions(predictionsRivet.getFeatureOutput().value())
                                    .labels(labelsInput.getFeatureOutput().value())
                                    .margin(margin)
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

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(numFeatures)});
    Impl::Tensor predictionsCpu(cpuPlacement, descriptor);
    Impl::Tensor labelsCpu(cpuPlacement, descriptor);
    std::copy(predictions.begin(), predictions.end(), static_cast<float*>(predictionsCpu.getMemPtr()));
    std::copy(labels.begin(), labels.end(), static_cast<float*>(labelsCpu.getMemPtr()));

    physicalPredictionsInput->forward(predictionsCpu, false, batchSize);
    physicalLabelsInput->forward(labelsCpu, false, batchSize);

    Stream labelsStream = physicalLabelsInput->getStream();
    labelsStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    labelsStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + predictions.size());

    Impl::Tensor errorOutputGpu = physicalRawLoss->getErrorOutput().value();
    Impl::Tensor errorOutputCpu(cpuPlacement, errorOutputGpu.getDescriptor());
    Stream rawLossStream = physicalRawLoss->getStream();
    errorOutputCpu.copyFromAsync(errorOutputGpu, rawLossStream);
    rawLossStream.synchronize();
    vector<float> outputGradient(static_cast<float*>(errorOutputCpu.getMemPtr()),
                                 static_cast<float*>(errorOutputCpu.getMemPtr()) + predictions.size());

    return ContrastiveRunResult{outputLoss, outputGradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(ContrastiveLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const float margin = 1.25f;
    const vector<float> predictions = {0.0f, 0.25f, 0.75f, 1.5f, 2.0f, 0.125f, 0.5f, 1.0f, 1.75f, 2.5f};
    const vector<float> labels = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    vector<float> referenceLoss(predictions.size());
    vector<float> referenceGradient(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        referenceLoss[i] = referenceContrastiveLoss(predictions[i], labels[i], margin);
        referenceGradient[i] = numericalContrastiveGradient(predictions[i], labels[i], margin) * Impl::Loss::getLossScalingFactor();
    }

    ContrastiveRunResult actual = runRawContrastiveLossNetwork(predictions, labels, margin);

    expectClose(actual.loss, referenceLoss, 1.0e-5f);
    expectClose(actual.gradient, referenceGradient, 1.0e-3f);
}
