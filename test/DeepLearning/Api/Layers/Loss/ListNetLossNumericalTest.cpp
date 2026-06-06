#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
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
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

vector<float> softmaxRow(const vector<float>& values, size_t rowOffset, uint32_t listSize, float temperature) {
    vector<float> probabilities(listSize, 0.0f);
    float maxScaledValue = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < listSize; ++i)
        maxScaledValue = std::max(maxScaledValue, values[rowOffset + i] / temperature);

    float denominator = 0.0f;
    for (uint32_t i = 0; i < listSize; ++i) {
        probabilities[i] = std::exp((values[rowOffset + i] / temperature) - maxScaledValue);
        denominator += probabilities[i];
    }
    for (float& probability : probabilities)
        probability /= denominator;
    return probabilities;
}

vector<float> logSoftmaxRow(const vector<float>& values, size_t rowOffset, uint32_t listSize, float temperature) {
    vector<float> logProbabilities(listSize, 0.0f);
    float maxScaledValue = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < listSize; ++i)
        maxScaledValue = std::max(maxScaledValue, values[rowOffset + i] / temperature);

    float denominator = 0.0f;
    for (uint32_t i = 0; i < listSize; ++i)
        denominator += std::exp((values[rowOffset + i] / temperature) - maxScaledValue);
    const float logDenominator = std::log(denominator);

    for (uint32_t i = 0; i < listSize; ++i)
        logProbabilities[i] = (values[rowOffset + i] / temperature) - maxScaledValue - logDenominator;
    return logProbabilities;
}

vector<float> referenceListNetRawLoss(const vector<float>& predictions,
                                      const vector<float>& labels,
                                      uint32_t batchSize,
                                      uint32_t listSize,
                                      float scoreTemperature,
                                      float labelTemperature) {
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * listSize));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());
    vector<float> loss(batchSize, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        const size_t rowOffset = static_cast<size_t>(b) * listSize;
        vector<float> targetProbabilities = softmaxRow(labels, rowOffset, listSize, labelTemperature);
        vector<float> logProbabilities = logSoftmaxRow(predictions, rowOffset, listSize, scoreTemperature);
        for (uint32_t i = 0; i < listSize; ++i)
            loss[b] -= targetProbabilities[i] * logProbabilities[i];
    }
    return loss;
}

float totalListNetLoss(const vector<float>& predictions,
                       const vector<float>& labels,
                       uint32_t batchSize,
                       uint32_t listSize,
                       float scoreTemperature,
                       float labelTemperature) {
    vector<float> rawLoss = referenceListNetRawLoss(predictions, labels, batchSize, listSize, scoreTemperature, labelTemperature);
    float total = 0.0f;
    for (float value : rawLoss)
        total += value;
    return total;
}

vector<float> numericalListNetGradient(const vector<float>& predictions,
                                       const vector<float>& labels,
                                       uint32_t batchSize,
                                       uint32_t listSize,
                                       float scoreTemperature,
                                       float labelTemperature) {
    constexpr float epsilon = 1.0e-3f;
    vector<float> gradient(predictions.size(), 0.0f);
    vector<float> perturbed = predictions;
    for (size_t i = 0; i < predictions.size(); ++i) {
        perturbed[i] = predictions[i] + epsilon;
        const float lossPlus = totalListNetLoss(perturbed, labels, batchSize, listSize, scoreTemperature, labelTemperature);
        perturbed[i] = predictions[i] - epsilon;
        const float lossMinus = totalListNetLoss(perturbed, labels, batchSize, listSize, scoreTemperature, labelTemperature);
        perturbed[i] = predictions[i];
        gradient[i] = (lossPlus - lossMinus) / (2.0f * epsilon);
    }
    return gradient;
}

struct ListNetRunResult {
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

ListNetRunResult runRawListNetLossNetwork(const vector<float>& predictions,
                                          const vector<float>& labels,
                                          float scoreTemperature,
                                          float labelTemperature) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t listSize = 5;
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * listSize));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network("list_net_numerical");
    Api::NetworkInput predictionsInput = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("predictions")
                                             .dimensions({listSize})
                                             .dataType(Api::DataType::FP32)
                                             .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({listSize})
                                        .dataType(Api::DataType::FP32)
                                        .build();

    Api::GradientRivet predictionsRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(predictionsInput.getFeatureOutput().value())
                                             .build();

    Api::ListNetLoss loss = Api::ListNetLoss::Builder()
                                .network(network)
                                .predictions(predictionsRivet.getFeatureOutput().value())
                                .labels(labelsInput.getFeatureOutput().value())
                                .scoreTemperature(scoreTemperature)
                                .labelTemperature(labelTemperature)
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
    Impl::TensorDescriptor inputDescriptor(Api::DataType::FP32,
                                           {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(listSize)});
    Impl::Tensor predictionsCpu(cpuPlacement, inputDescriptor);
    Impl::Tensor labelsCpu(cpuPlacement, inputDescriptor);
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
                             static_cast<float*>(outputLossCpu.getMemPtr()) + batchSize);

    Impl::Tensor errorOutputGpu = physicalRawLoss->getErrorOutput().value();
    Impl::Tensor errorOutputCpu(cpuPlacement, errorOutputGpu.getDescriptor());
    Stream rawLossStream = physicalRawLoss->getStream();
    errorOutputCpu.copyFromAsync(errorOutputGpu, rawLossStream);
    rawLossStream.synchronize();
    vector<float> outputGradient(static_cast<float*>(errorOutputCpu.getMemPtr()),
                                 static_cast<float*>(errorOutputCpu.getMemPtr()) + predictions.size());

    return ListNetRunResult{outputLoss, outputGradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(ListNetLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t listSize = 5;
    const float scoreTemperature = 0.8f;
    const float labelTemperature = 0.65f;
    const vector<float> predictions = {0.25f, 1.5f, -0.5f, 0.75f, -1.25f,
                                       1.25f, -0.25f, 0.5f, -1.0f, 0.125f,
                                       -0.75f, 0.125f, 1.75f, 0.375f, -0.5f};
    const vector<float> labels = {0.0f, 3.0f, 1.0f, 2.0f, -0.5f,
                                  2.5f, 0.25f, 1.5f, -1.0f, 0.75f,
                                  -0.25f, 1.0f, 4.0f, 1.5f, 0.5f};

    vector<float> referenceLoss = referenceListNetRawLoss(predictions, labels, batchSize, listSize, scoreTemperature, labelTemperature);
    vector<float> referenceGradient = numericalListNetGradient(predictions, labels, batchSize, listSize, scoreTemperature, labelTemperature);
    for (float& value : referenceGradient)
        value *= Impl::Loss::getLossScalingFactor();

    ListNetRunResult actual = runRawListNetLossNetwork(predictions, labels, scoreTemperature, labelTemperature);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.gradient, referenceGradient, 3.0e-3f);
}
