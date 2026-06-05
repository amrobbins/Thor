#include "DeepLearning/Api/Layers/Loss/MarginRankingLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

vector<float> referenceMarginRankingRawLoss(const vector<float>& input1,
                                            const vector<float>& input2,
                                            const vector<float>& target,
                                            float margin) {
    THOR_THROW_IF_FALSE(input1.size() == input2.size());
    THOR_THROW_IF_FALSE(input1.size() == target.size());
    vector<float> loss(input1.size(), 0.0f);
    for (size_t i = 0; i < input1.size(); ++i)
        loss[i] = std::max(margin - target[i] * (input1[i] - input2[i]), 0.0f);
    return loss;
}

float totalMarginRankingLoss(const vector<float>& input1, const vector<float>& input2, const vector<float>& target, float margin) {
    vector<float> raw = referenceMarginRankingRawLoss(input1, input2, target, margin);
    float total = 0.0f;
    for (float value : raw)
        total += value;
    return total;
}

vector<float> numericalMarginRankingGradient(
    vector<float> input1, vector<float> input2, const vector<float>& target, float margin, char wrt) {
    constexpr float epsilon = 1.0e-3f;
    vector<float>* tensor = nullptr;
    if (wrt == '1')
        tensor = &input1;
    else if (wrt == '2')
        tensor = &input2;
    else
        THOR_UNREACHABLE();

    vector<float> gradient(tensor->size(), 0.0f);
    for (size_t i = 0; i < tensor->size(); ++i) {
        (*tensor)[i] += epsilon;
        const float lossPlus = totalMarginRankingLoss(input1, input2, target, margin);
        (*tensor)[i] -= 2.0f * epsilon;
        const float lossMinus = totalMarginRankingLoss(input1, input2, target, margin);
        (*tensor)[i] += epsilon;
        gradient[i] = (lossPlus - lossMinus) / (2.0f * epsilon);
    }
    return gradient;
}

struct MarginRankingRunResult {
    vector<float> loss;
    vector<float> input1Gradient;
    vector<float> input2Gradient;
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

MarginRankingRunResult runRawMarginRankingLossNetwork(const vector<float>& input1,
                                                      const vector<float>& input2,
                                                      const vector<float>& target,
                                                      float margin) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t numScores = 3;
    THOR_THROW_IF_FALSE(input1.size() == static_cast<size_t>(batchSize * numScores));
    THOR_THROW_IF_FALSE(input2.size() == input1.size());
    THOR_THROW_IF_FALSE(target.size() == input1.size());

    Api::Network network("margin_ranking_numerical");
    Api::NetworkInput input1Layer =
        Api::NetworkInput::Builder().network(network).name("input1").dimensions({numScores}).dataType(Api::DataType::FP32).build();
    Api::NetworkInput input2Layer =
        Api::NetworkInput::Builder().network(network).name("input2").dimensions({numScores}).dataType(Api::DataType::FP32).build();
    Api::NetworkInput targetLayer =
        Api::NetworkInput::Builder().network(network).name("target").dimensions({numScores}).dataType(Api::DataType::FP32).build();

    Api::GradientRivet input1Rivet = Api::GradientRivet::Builder().network(network).tensor(input1Layer.getFeatureOutput().value()).build();
    Api::GradientRivet input2Rivet = Api::GradientRivet::Builder().network(network).tensor(input2Layer.getFeatureOutput().value()).build();

    Api::MarginRankingLoss loss = Api::MarginRankingLoss::Builder()
                                      .network(network)
                                      .input1(input1Rivet.getFeatureOutput().value())
                                      .input2(input2Rivet.getFeatureOutput().value())
                                      .target(targetLayer.getFeatureOutput().value())
                                      .margin(margin)
                                      .lossDataType(Api::DataType::FP32)
                                      .reportsRawLoss()
                                      .build();
    shared_ptr<Api::MultiInputCustomLoss> rawCustomLoss = findRawMultiInputCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput =
        Api::NetworkOutput::Builder().network(network).name("loss").inputTensor(loss.getLoss()).dataType(Api::DataType::FP32).build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {0}, 1);
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    Stream stream(0);
    for (Event& event : initDoneEvents)
        stream.waitEvent(event);
    stream.synchronize();
    initDoneEvents.clear();

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    shared_ptr<Impl::NetworkInput> physicalInput1Layer =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input1Layer.getId()));
    shared_ptr<Impl::NetworkInput> physicalInput2Layer =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input2Layer.getId()));
    shared_ptr<Impl::NetworkInput> physicalTargetLayer =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(targetLayer.getId()));
    shared_ptr<Impl::NetworkOutput> physicalLossOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(lossOutput.getId()));
    shared_ptr<Impl::MultiInputCustomLoss> physicalRawLoss =
        dynamic_pointer_cast<Impl::MultiInputCustomLoss>(stampedNetwork.getPhysicalLayerFromApiLayer(rawCustomLoss->getId()));
    THOR_THROW_IF_FALSE(physicalInput1Layer != nullptr);
    THOR_THROW_IF_FALSE(physicalInput2Layer != nullptr);
    THOR_THROW_IF_FALSE(physicalTargetLayer != nullptr);
    THOR_THROW_IF_FALSE(physicalLossOutput != nullptr);
    THOR_THROW_IF_FALSE(physicalRawLoss != nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor scoreDescriptor(Api::DataType::FP32,
                                           {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(numScores)});
    Impl::Tensor input1Cpu(cpuPlacement, scoreDescriptor);
    Impl::Tensor input2Cpu(cpuPlacement, scoreDescriptor);
    Impl::Tensor targetCpu(cpuPlacement, scoreDescriptor);
    std::copy(input1.begin(), input1.end(), static_cast<float*>(input1Cpu.getMemPtr()));
    std::copy(input2.begin(), input2.end(), static_cast<float*>(input2Cpu.getMemPtr()));
    std::copy(target.begin(), target.end(), static_cast<float*>(targetCpu.getMemPtr()));

    physicalInput1Layer->forward(input1Cpu, false, batchSize);
    physicalInput2Layer->forward(input2Cpu, false, batchSize);
    physicalTargetLayer->forward(targetCpu, false, batchSize);

    Stream outputStream = physicalTargetLayer->getStream();
    outputStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    outputStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()),
                             static_cast<float*>(outputLossCpu.getMemPtr()) + input1.size());

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> input1Gradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    vector<float> input2Gradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1).value(), rawLossStream);
    THOR_THROW_IF_FALSE(!physicalRawLoss->getErrorOutput(2).has_value());

    return MarginRankingRunResult{outputLoss, input1Gradient, input2Gradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(MarginRankingLossApi, NumericalRawLossAndBackwardGradientsMatchReference) {
    const float margin = 0.4f;

    const vector<float> input1 = {1.2f, -0.1f, 0.3f, 0.5f, 1.1f, -1.3f, -0.7f, 0.8f, 2.0f, 1.5f, -1.0f, 0.2f};
    const vector<float> input2 = {0.4f, 0.2f, 0.8f, 0.7f, 0.1f, -0.5f, -0.9f, 1.4f, 1.0f, 0.2f, -0.4f, -0.1f};
    const vector<float> target = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

    vector<float> referenceLoss = referenceMarginRankingRawLoss(input1, input2, target, margin);
    vector<float> referenceInput1Gradient = numericalMarginRankingGradient(input1, input2, target, margin, '1');
    vector<float> referenceInput2Gradient = numericalMarginRankingGradient(input1, input2, target, margin, '2');
    for (float& value : referenceInput1Gradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : referenceInput2Gradient)
        value *= Impl::Loss::getLossScalingFactor();

    MarginRankingRunResult actual = runRawMarginRankingLossNetwork(input1, input2, target, margin);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.input1Gradient, referenceInput1Gradient, 2.0e-3f);
    expectClose(actual.input2Gradient, referenceInput2Gradient, 2.0e-3f);
}
