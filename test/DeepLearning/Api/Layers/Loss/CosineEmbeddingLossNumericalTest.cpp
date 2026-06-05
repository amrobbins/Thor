#include "DeepLearning/Api/Layers/Loss/CosineEmbeddingLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
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

float cosineForRow(const vector<float>& input1, const vector<float>& input2, size_t row, uint32_t embeddingSize, float eps) {
    const size_t offset = row * embeddingSize;
    float dot = 0.0f;
    float input1Sq = eps;
    float input2Sq = eps;
    for (uint32_t d = 0; d < embeddingSize; ++d) {
        const float a = input1[offset + d];
        const float b = input2[offset + d];
        dot += a * b;
        input1Sq += a * a;
        input2Sq += b * b;
    }
    return dot / std::sqrt(input1Sq * input2Sq);
}

vector<float> referenceCosineEmbeddingRawLoss(const vector<float>& input1,
                                              const vector<float>& input2,
                                              const vector<float>& target,
                                              uint32_t batchSize,
                                              uint32_t embeddingSize,
                                              float margin,
                                              float eps) {
    vector<float> loss(batchSize, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        const float cosine = cosineForRow(input1, input2, b, embeddingSize, eps);
        if (target[b] > 0.0f)
            loss[b] = 1.0f - cosine;
        else
            loss[b] = std::max(cosine - margin, 0.0f);
    }
    return loss;
}

float totalCosineEmbeddingLoss(const vector<float>& input1,
                               const vector<float>& input2,
                               const vector<float>& target,
                               uint32_t batchSize,
                               uint32_t embeddingSize,
                               float margin,
                               float eps) {
    vector<float> raw = referenceCosineEmbeddingRawLoss(input1, input2, target, batchSize, embeddingSize, margin, eps);
    float total = 0.0f;
    for (float value : raw)
        total += value;
    return total;
}

vector<float> numericalCosineEmbeddingGradient(vector<float> input1,
                                               vector<float> input2,
                                               const vector<float>& target,
                                               uint32_t batchSize,
                                               uint32_t embeddingSize,
                                               float margin,
                                               float eps,
                                               char wrt) {
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
        const float lossPlus = totalCosineEmbeddingLoss(input1, input2, target, batchSize, embeddingSize, margin, eps);
        (*tensor)[i] -= 2.0f * epsilon;
        const float lossMinus = totalCosineEmbeddingLoss(input1, input2, target, batchSize, embeddingSize, margin, eps);
        (*tensor)[i] += epsilon;
        gradient[i] = (lossPlus - lossMinus) / (2.0f * epsilon);
    }
    return gradient;
}

struct CosineEmbeddingRunResult {
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

CosineEmbeddingRunResult runRawCosineEmbeddingLossNetwork(
    const vector<float>& input1, const vector<float>& input2, const vector<float>& target, float margin, float eps) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t embeddingSize = 3;
    THOR_THROW_IF_FALSE(input1.size() == static_cast<size_t>(batchSize * embeddingSize));
    THOR_THROW_IF_FALSE(input2.size() == input1.size());
    THOR_THROW_IF_FALSE(target.size() == batchSize);

    Api::Network network("cosine_embedding_numerical");
    Api::NetworkInput input1Layer =
        Api::NetworkInput::Builder().network(network).name("input1").dimensions({embeddingSize}).dataType(Api::DataType::FP32).build();
    Api::NetworkInput input2Layer =
        Api::NetworkInput::Builder().network(network).name("input2").dimensions({embeddingSize}).dataType(Api::DataType::FP32).build();
    Api::NetworkInput targetLayer =
        Api::NetworkInput::Builder().network(network).name("target").dimensions({1}).dataType(Api::DataType::FP32).build();

    Api::GradientRivet input1Rivet = Api::GradientRivet::Builder().network(network).tensor(input1Layer.getFeatureOutput().value()).build();
    Api::GradientRivet input2Rivet = Api::GradientRivet::Builder().network(network).tensor(input2Layer.getFeatureOutput().value()).build();

    Api::CosineEmbeddingLoss loss = Api::CosineEmbeddingLoss::Builder()
                                        .network(network)
                                        .input1(input1Rivet.getFeatureOutput().value())
                                        .input2(input2Rivet.getFeatureOutput().value())
                                        .target(targetLayer.getFeatureOutput().value())
                                        .margin(margin)
                                        .eps(eps)
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
    Impl::TensorDescriptor embeddingDescriptor(Api::DataType::FP32,
                                               {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(embeddingSize)});
    Impl::TensorDescriptor targetDescriptor(Api::DataType::FP32, {static_cast<unsigned long>(batchSize), static_cast<unsigned long>(1)});
    Impl::Tensor input1Cpu(cpuPlacement, embeddingDescriptor);
    Impl::Tensor input2Cpu(cpuPlacement, embeddingDescriptor);
    Impl::Tensor targetCpu(cpuPlacement, targetDescriptor);
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
    vector<float> outputLoss(static_cast<float*>(outputLossCpu.getMemPtr()), static_cast<float*>(outputLossCpu.getMemPtr()) + batchSize);

    Stream rawLossStream = physicalRawLoss->getStream();
    vector<float> input1Gradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(0).value(), rawLossStream);
    vector<float> input2Gradient = copyGpuTensorToVector(physicalRawLoss->getErrorOutput(1).value(), rawLossStream);
    THOR_THROW_IF_FALSE(!physicalRawLoss->getErrorOutput(2).has_value());

    return CosineEmbeddingRunResult{outputLoss, input1Gradient, input2Gradient};
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

}  // namespace

TEST(CosineEmbeddingLossApi, NumericalRawLossAndBackwardGradientsMatchReference) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t embeddingSize = 3;
    const float margin = 0.2f;
    const float eps = 1.0e-6f;

    const vector<float> input1 = {1.0f, 2.0f, -0.5f, 0.5f, -1.0f, 1.5f, -1.0f, 0.25f, 0.75f, 2.0f, 0.5f, -1.5f};
    const vector<float> input2 = {0.75f, 1.5f, -0.25f, 1.0f, 0.25f, -1.0f, 0.25f, -0.5f, 1.25f, -0.75f, 1.0f, 0.5f};
    const vector<float> target = {1.0f, -1.0f, 1.0f, -1.0f};

    vector<float> referenceLoss = referenceCosineEmbeddingRawLoss(input1, input2, target, batchSize, embeddingSize, margin, eps);
    vector<float> referenceInput1Gradient =
        numericalCosineEmbeddingGradient(input1, input2, target, batchSize, embeddingSize, margin, eps, '1');
    vector<float> referenceInput2Gradient =
        numericalCosineEmbeddingGradient(input1, input2, target, batchSize, embeddingSize, margin, eps, '2');
    for (float& value : referenceInput1Gradient)
        value *= Impl::Loss::getLossScalingFactor();
    for (float& value : referenceInput2Gradient)
        value *= Impl::Loss::getLossScalingFactor();

    CosineEmbeddingRunResult actual = runRawCosineEmbeddingLossNetwork(input1, input2, target, margin, eps);

    expectClose(actual.loss, referenceLoss, 2.0e-5f);
    expectClose(actual.input1Gradient, referenceInput1Gradient, 4.0e-3f);
    expectClose(actual.input2Gradient, referenceInput2Gradient, 4.0e-3f);
}
