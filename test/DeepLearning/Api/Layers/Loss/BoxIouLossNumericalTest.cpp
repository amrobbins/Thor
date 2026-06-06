#include "DeepLearning/Api/Layers/Loss/BoxIouLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/BoxIouLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
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

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
};

float positiveLength(float hi, float lo) {
    return std::max(hi - lo, 0.0f);
}

float square(float value) {
    return value * value;
}

float referenceBoxLoss(Box predictions, Box labels, Impl::BoxIouLoss::Kind kind, float eps) {
    const float pW = positiveLength(predictions.x2, predictions.x1);
    const float pH = positiveLength(predictions.y2, predictions.y1);
    const float tW = positiveLength(labels.x2, labels.x1);
    const float tH = positiveLength(labels.y2, labels.y1);
    const float pArea = pW * pH;
    const float tArea = tW * tH;

    const float interW = positiveLength(std::min(predictions.x2, labels.x2), std::max(predictions.x1, labels.x1));
    const float interH = positiveLength(std::min(predictions.y2, labels.y2), std::max(predictions.y1, labels.y1));
    const float interArea = interW * interH;
    const float unionArea = std::max(pArea + tArea - interArea, eps);
    const float iou = interArea / (unionArea + eps);
    float loss = 1.0f - iou;

    const float enclosingW = positiveLength(std::max(predictions.x2, labels.x2), std::min(predictions.x1, labels.x1));
    const float enclosingH = positiveLength(std::max(predictions.y2, labels.y2), std::min(predictions.y1, labels.y1));
    const float enclosingArea = std::max(enclosingW * enclosingH, eps);
    const float enclosingDiagSq = std::max(square(enclosingW) + square(enclosingH) + eps, eps);

    if (kind == Impl::BoxIouLoss::Kind::GIOU) {
        loss += (enclosingArea - unionArea) / (enclosingArea + eps);
    } else if (kind == Impl::BoxIouLoss::Kind::DIOU || kind == Impl::BoxIouLoss::Kind::CIOU) {
        const float pCx2 = predictions.x1 + predictions.x2;
        const float pCy2 = predictions.y1 + predictions.y2;
        const float tCx2 = labels.x1 + labels.x2;
        const float tCy2 = labels.y1 + labels.y2;
        const float centerDistanceSq = (square(pCx2 - tCx2) + square(pCy2 - tCy2)) * 0.25f;
        loss += centerDistanceSq / enclosingDiagSq;

        if (kind == Impl::BoxIouLoss::Kind::CIOU) {
            constexpr float pi = 3.14159265358979323846f;
            const float v = square(std::atan(tW / (tH + eps)) - std::atan(pW / (pH + eps))) * (4.0f / (pi * pi));
            const float alpha = v / ((1.0f - iou) + v + eps);
            loss += alpha * v;
        }
    }

    return loss;
}

Box boxAt(const vector<float>& boxes, size_t index) {
    const size_t offset = index * 4;
    return Box{boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3]};
}

vector<float> referenceRawLoss(const vector<float>& predictions,
                               const vector<float>& labels,
                               uint32_t batchSize,
                               uint32_t boxCount,
                               Impl::BoxIouLoss::Kind kind,
                               float eps) {
    vector<float> loss(batchSize * boxCount, 0.0f);
    for (uint32_t i = 0; i < batchSize * boxCount; ++i)
        loss[i] = referenceBoxLoss(boxAt(predictions, i), boxAt(labels, i), kind, eps);
    return loss;
}

float totalRawLoss(const vector<float>& predictions,
                   const vector<float>& labels,
                   uint32_t batchSize,
                   uint32_t boxCount,
                   Impl::BoxIouLoss::Kind kind,
                   float eps) {
    vector<float> raw = referenceRawLoss(predictions, labels, batchSize, boxCount, kind, eps);
    float total = 0.0f;
    for (float value : raw)
        total += value;
    return total;
}

vector<float> numericalPredictionsGradient(vector<float> predictions,
                                           const vector<float>& labels,
                                           uint32_t batchSize,
                                           uint32_t boxCount,
                                           Impl::BoxIouLoss::Kind kind,
                                           float eps) {
    constexpr float epsilon = 1.0e-3f;
    vector<float> gradient(predictions.size(), 0.0f);
    for (size_t i = 0; i < predictions.size(); ++i) {
        predictions[i] += epsilon;
        const float lossPlus = totalRawLoss(predictions, labels, batchSize, boxCount, kind, eps);
        predictions[i] -= 2.0f * epsilon;
        const float lossMinus = totalRawLoss(predictions, labels, batchSize, boxCount, kind, eps);
        predictions[i] += epsilon;
        gradient[i] = (lossPlus - lossMinus) / (2.0f * epsilon);
    }
    return gradient;
}

struct BoxLossRunResult {
    vector<float> loss;
    vector<float> predictionsGradient;
};

template <typename LossT>
BoxLossRunResult runRawBoxLossNetwork(const vector<float>& predictions,
                                      const vector<float>& labels,
                                      uint32_t batchSize,
                                      uint32_t boxCount,
                                      float eps) {
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(batchSize * boxCount * 4));
    THOR_THROW_IF_FALSE(labels.size() == predictions.size());

    Api::Network network("box_iou_numerical");
    Api::NetworkInput predictionsInput = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("predictions")
                                             .dimensions({boxCount, 4})
                                             .dataType(Api::DataType::FP32)
                                             .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions({boxCount, 4})
                                        .dataType(Api::DataType::FP32)
                                        .build();

    Api::GradientRivet predictionsRivet =
        Api::GradientRivet::Builder().network(network).tensor(predictionsInput.getFeatureOutput().value()).build();

    LossT loss = typename LossT::Builder()
                     .network(network)
                     .predictions(predictionsRivet.getFeatureOutput().value())
                     .labels(labelsInput.getFeatureOutput().value())
                     .eps(eps)
                     .lossDataType(Api::DataType::FP32)
                     .reportsRawLoss()
                     .build();

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
    shared_ptr<Impl::BoxIouLoss> physicalLoss =
        dynamic_pointer_cast<Impl::BoxIouLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(loss.getId()));
    THOR_THROW_IF_FALSE(physicalPredictionsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLabelsInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalLoss != nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32,
                                      {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(boxCount), 4ul});
    Impl::Tensor predictionsCpu(cpuPlacement, descriptor);
    Impl::Tensor labelsCpu(cpuPlacement, descriptor);
    std::copy(predictions.begin(), predictions.end(), static_cast<float*>(predictionsCpu.getMemPtr()));
    std::copy(labels.begin(), labels.end(), static_cast<float*>(labelsCpu.getMemPtr()));

    physicalPredictionsInput->forward(predictionsCpu, false, batchSize);
    physicalLabelsInput->forward(labelsCpu, false, batchSize);

    Stream outputStream = physicalLabelsInput->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + batchSize * boxCount);

    Stream lossStream = physicalLoss->getStream();
    Impl::TensorPlacement cpuGradientPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor gradientCpu(cpuGradientPlacement, physicalLoss->getErrorOutput().value().getDescriptor());
    gradientCpu.copyFromAsync(physicalLoss->getErrorOutput().value(), lossStream);
    lossStream.synchronize();
    vector<float> predictionsGradient(static_cast<float*>(gradientCpu.getMemPtr()),
                                      static_cast<float*>(gradientCpu.getMemPtr()) + predictions.size());

    return BoxLossRunResult{outputLoss, predictionsGradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

template <typename LossT>
void expectForwardAndBackwardMatchReference(Impl::BoxIouLoss::Kind kind) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t boxCount = 2;
    constexpr float eps = 1.0e-7f;

    const vector<float> predictions = {0.10f, 0.20f, 1.40f, 1.80f,
                                       2.00f, 0.50f, 3.50f, 1.90f,
                                       0.30f, 2.10f, 1.70f, 3.40f,
                                       1.20f, 1.30f, 2.80f, 2.90f};
    const vector<float> labels = {0.00f, 0.10f, 1.20f, 1.60f,
                                  1.80f, 0.30f, 3.00f, 2.20f,
                                  0.50f, 1.90f, 1.90f, 3.10f,
                                  1.00f, 1.10f, 3.10f, 3.20f};

    vector<float> expectedLoss = referenceRawLoss(predictions, labels, batchSize, boxCount, kind, eps);
    vector<float> expectedGradient = numericalPredictionsGradient(predictions, labels, batchSize, boxCount, kind, eps);
    for (float& value : expectedGradient)
        value *= Impl::Loss::getLossScalingFactor();

    BoxLossRunResult actual = runRawBoxLossNetwork<LossT>(predictions, labels, batchSize, boxCount, eps);

    expectClose(actual.loss, expectedLoss, 2.5e-5f);
    expectClose(actual.predictionsGradient, expectedGradient, 3.5e-2f);
}

}  // namespace

TEST(BoxIouLossApi, IoULossNumericalRawLossAndBackwardGradientMatchReference) {
    expectForwardAndBackwardMatchReference<Api::IoULoss>(Impl::BoxIouLoss::Kind::IOU);
}

TEST(BoxIouLossApi, GIoULossNumericalRawLossAndBackwardGradientMatchReference) {
    expectForwardAndBackwardMatchReference<Api::GIoULoss>(Impl::BoxIouLoss::Kind::GIOU);
}

TEST(BoxIouLossApi, DIoULossNumericalRawLossAndBackwardGradientMatchReference) {
    expectForwardAndBackwardMatchReference<Api::DIoULoss>(Impl::BoxIouLoss::Kind::DIOU);
}

TEST(BoxIouLossApi, CIoULossNumericalRawLossAndBackwardGradientMatchReference) {
    expectForwardAndBackwardMatchReference<Api::CIoULoss>(Impl::BoxIouLoss::Kind::CIOU);
}
