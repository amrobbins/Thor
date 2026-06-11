#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/DiceLoss.h"
#include "DeepLearning/Api/Layers/Loss/FocalTverskyLoss.h"
#include "DeepLearning/Api/Layers/Loss/TverskyLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint32_t kBatchSize = 2;
constexpr double kFiniteDifferenceEpsilon = 1.0e-3;
constexpr double kOverlapEpsilon = 1.0e-7;

uint64_t featureElementCount(const vector<uint64_t>& featureDims) {
    uint64_t elements = 1;
    for (uint64_t dim : featureDims)
        elements *= dim;
    return elements;
}

vector<unsigned long> tensorDimsWithBatch(uint32_t batchSize, const vector<uint64_t>& featureDims) {
    vector<unsigned long> dims;
    dims.reserve(featureDims.size() + 1);
    dims.push_back(batchSize);
    for (uint64_t dim : featureDims)
        dims.push_back(static_cast<unsigned long>(dim));
    return dims;
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

vector<float> copyGpuTensorToVector(const Impl::Tensor& tensor, Stream stream) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpu(cpuPlacement, tensor.getDescriptor());
    cpu.copyFromAsync(tensor, stream);
    stream.synchronize();
    return tensorToVector(cpu);
}

shared_ptr<Api::CustomLoss> findRawCustomLoss(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::Layer> layer = network.getLayer(i);
        shared_ptr<Api::CustomLoss> customLoss = dynamic_pointer_cast<Api::CustomLoss>(layer);
        if (customLoss != nullptr)
            return customLoss;
    }
    return nullptr;
}

struct RawCustomLossRunResult {
    vector<float> loss;
    vector<float> predictionGradient;
};

RawCustomLossRunResult runRawCustomLossNetwork(const string& networkName,
                                               const vector<uint64_t>& featureDims,
                                               const vector<float>& predictions,
                                               const vector<float>& labels,
                                               const function<Api::Tensor(Api::Network&, Api::Tensor, Api::Tensor)>& buildLoss) {
    THOR_THROW_IF_FALSE(predictions.size() == labels.size());
    THOR_THROW_IF_FALSE(predictions.size() == static_cast<size_t>(kBatchSize * featureElementCount(featureDims)));

    Api::Network network(networkName);
    Api::NetworkInput predictionsInput = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("predictions")
                                             .dimensions(featureDims)
                                             .dataType(Api::DataType::FP32)
                                             .build();
    Api::NetworkInput labelsInput = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("labels")
                                        .dimensions(featureDims)
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::GradientRivet predictionsRivet = Api::GradientRivet::Builder()
                                             .network(network)
                                             .tensor(predictionsInput.getFeatureOutput().value())
                                             .build();

    Api::Tensor lossTensor = buildLoss(network, predictionsRivet.getFeatureOutput().value(), labelsInput.getFeatureOutput().value());
    shared_ptr<Api::CustomLoss> rawCustomLoss = findRawCustomLoss(network);
    THOR_THROW_IF_FALSE(rawCustomLoss != nullptr);

    Api::NetworkOutput lossOutput = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("loss")
                                        .inputTensor(lossTensor)
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

    Impl::Tensor predictionsCpu = makeCpuTensor(predictions, kBatchSize, featureDims);
    Impl::Tensor labelsCpu = makeCpuTensor(labels, kBatchSize, featureDims);
    physicalPredictionsInput->forward(predictionsCpu, false, kBatchSize);
    physicalLabelsInput->forward(labelsCpu, false, kBatchSize);

    Stream lossStream = physicalLossOutput->getStream();
    lossStream.waitEvent(physicalLossOutput->getOutputReadyEvent());
    lossStream.synchronize();

    Impl::Tensor outputLossCpu = physicalLossOutput->getFeatureOutput().value();
    THOR_THROW_IF_FALSE(outputLossCpu.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU);
    vector<float> outputLoss = tensorToVector(outputLossCpu);

    optional<Impl::Tensor> errorOutputGpu = physicalRawLoss->getErrorOutput();
    THOR_THROW_IF_FALSE(errorOutputGpu.has_value());
    vector<float> outputGradient = copyGpuTensorToVector(errorOutputGpu.value(), physicalRawLoss->getStream());
    return {outputLoss, outputGradient};
}

vector<float> finiteDifferenceGradient(const vector<float>& predictions, const function<double(const vector<float>&)>& totalLoss) {
    vector<float> gradient(predictions.size(), 0.0f);
    vector<float> perturbed = predictions;
    for (size_t i = 0; i < predictions.size(); ++i) {
        perturbed[i] = predictions[i] + static_cast<float>(kFiniteDifferenceEpsilon);
        const double lossPlus = totalLoss(perturbed);
        perturbed[i] = predictions[i] - static_cast<float>(kFiniteDifferenceEpsilon);
        const double lossMinus = totalLoss(perturbed);
        perturbed[i] = predictions[i];
        gradient[i] = static_cast<float>((lossPlus - lossMinus) / (2.0 * kFiniteDifferenceEpsilon));
    }
    return gradient;
}

void scaleByLossScalingFactor(vector<float>& values) {
    const float scale = Impl::Loss::getLossScalingFactor();
    for (float& value : values)
        value *= scale;
}

void expectClose(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        EXPECT_LE(diff, tolerance) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

vector<float> referenceBinaryFocalRawLoss(const vector<float>& logits, const vector<float>& labels, float gamma, float alpha) {
    THOR_THROW_IF_FALSE(logits.size() == labels.size());
    vector<float> loss(logits.size(), 0.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        const double logit = logits[i];
        const double label = labels[i];
        const double bce = std::max(logit, 0.0) - (logit * label) + std::log1p(std::exp(-std::fabs(logit)));
        const double pt = std::exp(-bce);
        const double alphaFactor = (label * alpha) + ((1.0 - label) * (1.0 - alpha));
        const double focalWeight = gamma == 0.0f ? 1.0 : std::pow(std::max(1.0 - pt, kOverlapEpsilon), gamma);
        loss[i] = static_cast<float>(alphaFactor * focalWeight * bce);
    }
    return loss;
}

vector<float> referenceCategoricalFocalRawLoss(const vector<float>& logits,
                                               const vector<float>& labels,
                                               uint32_t batchSize,
                                               uint32_t numClasses,
                                               float gamma,
                                               float alpha) {
    THOR_THROW_IF_FALSE(logits.size() == static_cast<size_t>(batchSize * numClasses));
    THOR_THROW_IF_FALSE(labels.size() == logits.size());
    vector<float> loss(logits.size(), 0.0f);
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
        const size_t base = static_cast<size_t>(batch) * numClasses;
        double maxLogit = logits[base];
        for (uint32_t c = 1; c < numClasses; ++c)
            maxLogit = std::max(maxLogit, static_cast<double>(logits[base + c]));
        double denominator = 0.0;
        for (uint32_t c = 0; c < numClasses; ++c)
            denominator += std::exp(static_cast<double>(logits[base + c]) - maxLogit);
        const double logDenominator = std::log(denominator);
        for (uint32_t c = 0; c < numClasses; ++c) {
            const double logProbability = static_cast<double>(logits[base + c]) - maxLogit - logDenominator;
            const double probability = std::exp(logProbability);
            const double focalWeight = gamma == 0.0f ? 1.0 : std::pow(std::max(1.0 - probability, kOverlapEpsilon), gamma);
            loss[base + c] = static_cast<float>(-static_cast<double>(alpha) * labels[base + c] * focalWeight * logProbability);
        }
    }
    return loss;
}

vector<float> referenceDiceRawLoss(const vector<float>& predictions,
                                   const vector<float>& labels,
                                   uint32_t batchSize,
                                   uint32_t channels,
                                   uint32_t spatial,
                                   float smooth) {
    vector<float> loss(static_cast<size_t>(batchSize) * channels, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t c = 0; c < channels; ++c) {
            double intersection = 0.0;
            double predictionSum = 0.0;
            double labelSum = 0.0;
            for (uint32_t s = 0; s < spatial; ++s) {
                const size_t index = (static_cast<size_t>(b) * channels + c) * spatial + s;
                intersection += static_cast<double>(predictions[index]) * labels[index];
                predictionSum += predictions[index];
                labelSum += labels[index];
            }
            const double numerator = (2.0 * intersection) + smooth;
            const double denominator = std::max(predictionSum + labelSum + smooth, kOverlapEpsilon);
            loss[static_cast<size_t>(b) * channels + c] = static_cast<float>(1.0 - (numerator / denominator));
        }
    }
    return loss;
}

vector<float> referenceTverskyRawLoss(const vector<float>& predictions,
                                      const vector<float>& labels,
                                      uint32_t batchSize,
                                      uint32_t channels,
                                      uint32_t spatial,
                                      float alpha,
                                      float beta,
                                      float smooth,
                                      optional<float> focalGamma = nullopt) {
    vector<float> loss(static_cast<size_t>(batchSize) * channels, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t c = 0; c < channels; ++c) {
            double truePositive = 0.0;
            double falsePositive = 0.0;
            double falseNegative = 0.0;
            for (uint32_t s = 0; s < spatial; ++s) {
                const size_t index = (static_cast<size_t>(b) * channels + c) * spatial + s;
                const double prediction = predictions[index];
                const double label = labels[index];
                truePositive += prediction * label;
                falsePositive += prediction * (1.0 - label);
                falseNegative += (1.0 - prediction) * label;
            }
            const double numerator = truePositive + smooth;
            const double denominator = std::max(truePositive + alpha * falsePositive + beta * falseNegative + smooth, kOverlapEpsilon);
            const double tverskyLoss = 1.0 - (numerator / denominator);
            loss[static_cast<size_t>(b) * channels + c] = focalGamma.has_value()
                                                                ? static_cast<float>(std::pow(std::max(tverskyLoss, kOverlapEpsilon), focalGamma.value()))
                                                                : static_cast<float>(tverskyLoss);
        }
    }
    return loss;
}

double sumAsDouble(const vector<float>& values) {
    double total = 0.0;
    for (float value : values)
        total += static_cast<double>(value);
    return total;
}

}  // namespace

TEST(BinaryFocalLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const vector<uint64_t> featureDims = {1};
    const vector<float> predictions = {-1.25f, 0.75f};
    const vector<float> labels = {0.0f, 1.0f};
    const float gamma = 1.5f;
    const float alpha = 0.35f;

    RawCustomLossRunResult actual = runRawCustomLossNetwork(
        "binary_focal_loss_gradient_numerical", featureDims, predictions, labels, [=](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::BinaryFocalLoss loss = Api::BinaryFocalLoss::Builder()
                                            .network(network)
                                            .predictions(predictionsTensor)
                                            .labels(labelsTensor)
                                            .focusingParameter(gamma)
                                            .alpha(alpha)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceBinaryFocalRawLoss(predictions, labels, gamma, alpha);
    vector<float> expectedGradient = finiteDifferenceGradient(
        predictions, [&](const vector<float>& perturbed) { return sumAsDouble(referenceBinaryFocalRawLoss(perturbed, labels, gamma, alpha)); });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, 1.0e-2f);
}

TEST(CategoricalFocalLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const uint32_t numClasses = 3;
    const vector<uint64_t> featureDims = {numClasses};
    const vector<float> predictions = {1.2f, -0.3f, 0.5f, -0.7f, 0.8f, 1.6f};
    const vector<float> labels = {1.0f, 0.0f, 0.0f, 0.0f, 0.25f, 0.75f};
    const float gamma = 1.75f;
    const float alpha = 0.8f;

    RawCustomLossRunResult actual = runRawCustomLossNetwork(
        "categorical_focal_loss_gradient_numerical", featureDims, predictions, labels, [=](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::CategoricalFocalLoss loss = Api::CategoricalFocalLoss::Builder()
                                                 .network(network)
                                                 .predictions(predictionsTensor)
                                                 .labels(labelsTensor)
                                                 .focusingParameter(gamma)
                                                 .alpha(alpha)
                                                 .lossDataType(Api::DataType::FP32)
                                                 .reportsRawLoss()
                                                 .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceCategoricalFocalRawLoss(predictions, labels, kBatchSize, numClasses, gamma, alpha);
    vector<float> expectedGradient = finiteDifferenceGradient(predictions, [&](const vector<float>& perturbed) {
        return sumAsDouble(referenceCategoricalFocalRawLoss(perturbed, labels, kBatchSize, numClasses, gamma, alpha));
    });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, 1.0e-2f);
}

TEST(DiceLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const uint32_t channels = 2;
    const uint32_t spatial = 3;
    const vector<uint64_t> featureDims = {channels, spatial};
    const vector<float> predictions = {0.2f, 0.7f, 0.6f, 0.9f, 0.1f, 0.4f, 0.8f, 0.3f, 0.5f, 0.15f, 0.65f, 0.35f};
    const vector<float> labels = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    const float smooth = 0.7f;

    RawCustomLossRunResult actual = runRawCustomLossNetwork(
        "dice_loss_gradient_numerical", featureDims, predictions, labels, [=](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::DiceLoss loss = Api::DiceLoss::Builder()
                                    .network(network)
                                    .predictions(predictionsTensor)
                                    .labels(labelsTensor)
                                    .smooth(smooth)
                                    .lossDataType(Api::DataType::FP32)
                                    .reportsRawLoss()
                                    .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceDiceRawLoss(predictions, labels, kBatchSize, channels, spatial, smooth);
    vector<float> expectedGradient = finiteDifferenceGradient(predictions, [&](const vector<float>& perturbed) {
        return sumAsDouble(referenceDiceRawLoss(perturbed, labels, kBatchSize, channels, spatial, smooth));
    });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, 1.0e-2f);
}

TEST(TverskyLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const uint32_t channels = 2;
    const uint32_t spatial = 3;
    const vector<uint64_t> featureDims = {channels, spatial};
    const vector<float> predictions = {0.2f, 0.7f, 0.6f, 0.9f, 0.1f, 0.4f, 0.8f, 0.3f, 0.5f, 0.15f, 0.65f, 0.35f};
    const vector<float> labels = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    const float alpha = 0.4f;
    const float beta = 0.7f;
    const float smooth = 0.8f;

    RawCustomLossRunResult actual = runRawCustomLossNetwork(
        "tversky_loss_gradient_numerical", featureDims, predictions, labels, [=](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::TverskyLoss loss = Api::TverskyLoss::Builder()
                                       .network(network)
                                       .predictions(predictionsTensor)
                                       .labels(labelsTensor)
                                       .alpha(alpha)
                                       .beta(beta)
                                       .smooth(smooth)
                                       .lossDataType(Api::DataType::FP32)
                                       .reportsRawLoss()
                                       .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceTverskyRawLoss(predictions, labels, kBatchSize, channels, spatial, alpha, beta, smooth);
    vector<float> expectedGradient = finiteDifferenceGradient(predictions, [&](const vector<float>& perturbed) {
        return sumAsDouble(referenceTverskyRawLoss(perturbed, labels, kBatchSize, channels, spatial, alpha, beta, smooth));
    });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, 1.0e-2f);
}

TEST(FocalTverskyLossApi, NumericalRawLossAndBackwardGradientMatchReference) {
    const uint32_t channels = 2;
    const uint32_t spatial = 3;
    const vector<uint64_t> featureDims = {channels, spatial};
    const vector<float> predictions = {0.2f, 0.7f, 0.6f, 0.9f, 0.1f, 0.4f, 0.8f, 0.3f, 0.5f, 0.15f, 0.65f, 0.35f};
    const vector<float> labels = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    const float alpha = 0.35f;
    const float beta = 0.65f;
    const float gamma = 1.4f;
    const float smooth = 0.6f;

    RawCustomLossRunResult actual = runRawCustomLossNetwork(
        "focal_tversky_loss_gradient_numerical", featureDims, predictions, labels, [=](Api::Network& network, Api::Tensor predictionsTensor, Api::Tensor labelsTensor) {
            Api::FocalTverskyLoss loss = Api::FocalTverskyLoss::Builder()
                                            .network(network)
                                            .predictions(predictionsTensor)
                                            .labels(labelsTensor)
                                            .alpha(alpha)
                                            .beta(beta)
                                            .gamma(gamma)
                                            .smooth(smooth)
                                            .lossDataType(Api::DataType::FP32)
                                            .reportsRawLoss()
                                            .build();
            return loss.getLoss();
        });

    vector<float> expectedLoss = referenceTverskyRawLoss(predictions, labels, kBatchSize, channels, spatial, alpha, beta, smooth, gamma);
    vector<float> expectedGradient = finiteDifferenceGradient(predictions, [&](const vector<float>& perturbed) {
        return sumAsDouble(referenceTverskyRawLoss(perturbed, labels, kBatchSize, channels, spatial, alpha, beta, smooth, gamma));
    });
    scaleByLossScalingFactor(expectedGradient);

    expectClose(actual.loss, expectedLoss, 1.0e-5f);
    expectClose(actual.predictionGradient, expectedGradient, 1.0e-2f);
}
