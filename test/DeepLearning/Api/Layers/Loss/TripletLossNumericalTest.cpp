#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/TripletLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/DeepLearning/Api/Layers/Loss/LossNumericalTestTolerance.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
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

float l2Distance(const vector<float>& lhs, const vector<float>& rhs, size_t row, uint32_t embeddingSize, float eps) {
    const size_t offset = row * embeddingSize;
    float sum = eps;
    for (uint32_t d = 0; d < embeddingSize; ++d) {
        const float diff = lhs[offset + d] - rhs[offset + d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

vector<float> referenceTripletRawLoss(const vector<float>& anchor,
                                      const vector<float>& positive,
                                      const vector<float>& negative,
                                      uint32_t batchSize,
                                      uint32_t embeddingSize,
                                      float margin,
                                      float eps) {
    vector<float> loss(batchSize, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        const float dAp = l2Distance(anchor, positive, b, embeddingSize, eps);
        const float dAn = l2Distance(anchor, negative, b, embeddingSize, eps);
        loss[b] = std::max(dAp - dAn + margin, 0.0f);
    }
    return loss;
}

float totalTripletLoss(const vector<float>& anchor,
                       const vector<float>& positive,
                       const vector<float>& negative,
                       uint32_t batchSize,
                       uint32_t embeddingSize,
                       float margin,
                       float eps) {
    vector<float> raw = referenceTripletRawLoss(anchor, positive, negative, batchSize, embeddingSize, margin, eps);
    float total = 0.0f;
    for (float value : raw)
        total += value;
    return total;
}

vector<float> numericalTripletGradient(vector<float> anchor,
                                       vector<float> positive,
                                       vector<float> negative,
                                       uint32_t batchSize,
                                       uint32_t embeddingSize,
                                       float margin,
                                       float eps,
                                       char wrt) {
    constexpr float epsilon = 1.0e-3f;
    vector<float>* target = nullptr;
    if (wrt == 'a')
        target = &anchor;
    else if (wrt == 'p')
        target = &positive;
    else if (wrt == 'n')
        target = &negative;
    else
        THOR_UNREACHABLE();

    vector<float> gradient(target->size(), 0.0f);
    for (size_t i = 0; i < target->size(); ++i) {
        (*target)[i] += epsilon;
        const float lossPlus = totalTripletLoss(anchor, positive, negative, batchSize, embeddingSize, margin, eps);
        (*target)[i] -= 2.0f * epsilon;
        const float lossMinus = totalTripletLoss(anchor, positive, negative, batchSize, embeddingSize, margin, eps);
        (*target)[i] += epsilon;
        gradient[i] = (lossPlus - lossMinus) / (2.0f * epsilon);
    }
    return gradient;
}

struct TripletRunResult {
    vector<float> loss;
    vector<float> anchorGradient;
    vector<float> positiveGradient;
    vector<float> negativeGradient;
};

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

TripletRunResult runRawTripletLossNetwork(const vector<float>& anchor,
                                          const vector<float>& positive,
                                          const vector<float>& negative,
                                          float margin,
                                          float eps) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t embeddingSize = 3;
    THOR_THROW_IF_FALSE(anchor.size() == static_cast<size_t>(batchSize * embeddingSize));
    THOR_THROW_IF_FALSE(positive.size() == anchor.size());
    THOR_THROW_IF_FALSE(negative.size() == anchor.size());

    Api::Network network("triplet_numerical");
    Api::NetworkInput anchorInput = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("anchor")
                                       .dimensions({embeddingSize})
                                       .dataType(Api::DataType::FP32)
                                       .build();
    Api::NetworkInput positiveInput = Api::NetworkInput::Builder()
                                         .network(network)
                                         .name("positive")
                                         .dimensions({embeddingSize})
                                         .dataType(Api::DataType::FP32)
                                         .build();
    Api::NetworkInput negativeInput = Api::NetworkInput::Builder()
                                         .network(network)
                                         .name("negative")
                                         .dimensions({embeddingSize})
                                         .dataType(Api::DataType::FP32)
                                         .build();

    Api::GradientRivet anchorRivet = Api::GradientRivet::Builder().network(network).tensor(anchorInput.getFeatureOutput().value()).build();
    Api::GradientRivet positiveRivet = Api::GradientRivet::Builder().network(network).tensor(positiveInput.getFeatureOutput().value()).build();
    Api::GradientRivet negativeRivet = Api::GradientRivet::Builder().network(network).tensor(negativeInput.getFeatureOutput().value()).build();

    Api::TripletLoss loss = Api::TripletLoss::Builder()
                                .network(network)
                                .anchor(anchorRivet.getFeatureOutput().value())
                                .positive(positiveRivet.getFeatureOutput().value())
                                .negative(negativeRivet.getFeatureOutput().value())
                                .margin(margin)
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
    shared_ptr<Impl::NetworkInput> physicalAnchorInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(anchorInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalPositiveInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(positiveInput.getId()));
    shared_ptr<Impl::NetworkInput> physicalNegativeInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(negativeInput.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalAnchorInput != nullptr);
    THOR_THROW_IF_FALSE(physicalPositiveInput != nullptr);
    THOR_THROW_IF_FALSE(physicalNegativeInput != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Api::DataType::FP32, {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(embeddingSize)});
    Impl::Tensor anchorCpu(cpuPlacement, descriptor);
    Impl::Tensor positiveCpu(cpuPlacement, descriptor);
    Impl::Tensor negativeCpu(cpuPlacement, descriptor);
    std::copy(anchor.begin(), anchor.end(), static_cast<float*>(anchorCpu.getMemPtr()));
    std::copy(positive.begin(), positive.end(), static_cast<float*>(positiveCpu.getMemPtr()));
    std::copy(negative.begin(), negative.end(), static_cast<float*>(negativeCpu.getMemPtr()));

    physicalAnchorInput->forward(anchorCpu, false, batchSize);
    physicalPositiveInput->forward(positiveCpu, false, batchSize);
    physicalNegativeInput->forward(negativeCpu, false, batchSize);

    Stream outputStream = physicalNegativeInput->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + batchSize);

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> anchorGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    vector<float> positiveGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1).value(), rawLossStream);
    vector<float> negativeGradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(2).value(), rawLossStream);

    return TripletRunResult{outputLoss, anchorGradient, positiveGradient, negativeGradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(TripletLossApi, NumericalRawLossAndAllBackwardGradientsMatchReference) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t embeddingSize = 3;
    const float margin = 0.75f;
    const float eps = 1.0e-6f;

    const vector<float> anchor = {0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  1.0f, -0.5f, 0.25f};
    const vector<float> positive = {0.2f, 0.0f, 0.0f,
                                    0.1f, 0.0f, 0.0f,
                                    0.5f, -0.25f, 0.1f};
    const vector<float> negative = {0.8f, 0.0f, 0.0f,
                                    2.0f, 0.0f, 0.0f,
                                    1.6f, -0.4f, 0.2f};

    vector<float> referenceLoss = referenceTripletRawLoss(anchor, positive, negative, batchSize, embeddingSize, margin, eps);
    vector<float> referenceAnchorGradient = numericalTripletGradient(anchor, positive, negative, batchSize, embeddingSize, margin, eps, 'a');
    vector<float> referencePositiveGradient = numericalTripletGradient(anchor, positive, negative, batchSize, embeddingSize, margin, eps, 'p');
    vector<float> referenceNegativeGradient = numericalTripletGradient(anchor, positive, negative, batchSize, embeddingSize, margin, eps, 'n');
    for (float& value : referenceAnchorGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : referencePositiveGradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : referenceNegativeGradient)
        value *= Impl::Loss::getLossScalingFactor();

    TripletRunResult actual = runRawTripletLossNetwork(anchor, positive, negative, margin, eps);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.anchorGradient, referenceAnchorGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-3f));
    expectClose(actual.positiveGradient, referencePositiveGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-3f));
    expectClose(actual.negativeGradient, referenceNegativeGradient, ThorTest::lossScaleAwareGradientTolerance(3.0e-3f));
}
