#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/InstanceNorm.h"
#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/InstanceNorm.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/LayerNorm.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RMSNorm.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/DeepLearning/Implementation/Layers/Helpers/GradientRivet.h"
#include "test/DeepLearning/Implementation/Layers/LayerSynchronizationTestKernels.h"
#include "Utilities/Common/ScopedGpu.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint32_t batchSize = 2;
constexpr uint32_t channelCount = 8;
constexpr uint32_t spatialElementCount = 4;
constexpr uint32_t featureCount = channelCount * spatialElementCount;
constexpr uint32_t totalElementCount = batchSize * featureCount;
constexpr float layerAndInstanceNormEpsilon = 1.0e-5f;
constexpr float batchNormEpsilon = 1.0e-4f;

vector<float> makeInputValues() {
    vector<float> values(totalElementCount);
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
        for (uint32_t channel = 0; channel < channelCount; ++channel) {
            for (uint32_t spatial = 0; spatial < spatialElementCount; ++spatial) {
                const uint32_t index = (batch * channelCount + channel) * spatialElementCount + spatial;
                values[index] = 1.0f + 0.75f * static_cast<float>(batch) + 1.25f * static_cast<float>(channel) +
                                0.5f * static_cast<float>(spatial) +
                                0.125f * static_cast<float>(channel * spatial);
            }
        }
    }
    return values;
}

vector<float> makeIncomingErrorValues() {
    vector<float> values(totalElementCount);
    for (uint32_t index = 0; index < totalElementCount; ++index)
        values[index] = 1.0f + static_cast<float>((index * 7u + index / 3u) % 11u);
    return values;
}

const vector<float> inputValues = makeInputValues();
const vector<float> incomingErrorValues = makeIncomingErrorValues();

enum class NormalizationKind { LAYER_NORM, INSTANCE_NORM, BATCH_NORMALIZATION };

const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

string normalizationKindName(NormalizationKind kind) {
    switch (kind) {
        case NormalizationKind::LAYER_NORM:
            return "LayerNorm";
        case NormalizationKind::INSTANCE_NORM:
            return "InstanceNorm";
        case NormalizationKind::BATCH_NORMALIZATION:
            return "BatchNormalization";
    }
    return "Unknown";
}

vector<uint64_t> inputDimensions(NormalizationKind kind) {
    switch (kind) {
        case NormalizationKind::LAYER_NORM:
            return {featureCount};
        case NormalizationKind::INSTANCE_NORM:
            return {channelCount, spatialElementCount};
        case NormalizationKind::BATCH_NORMALIZATION:
            return {channelCount, spatialElementCount, 1};
    }
    return {};
}

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions())
        numel *= dim;
    return numel;
}

void writeCpuFp32Tensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), Impl::DataType::FP32);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

vector<float> readCpuFp32Tensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), Impl::DataType::FP32);

    vector<float> values(tensorNumel(tensor));
    const float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, const Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    stream.putEvent(/*enableTiming=*/false, /*expectingHostToWaitOnThisOne=*/true).synchronize();
    return cpuTensor;
}

void setTensor(Impl::Tensor tensor, const vector<float>& values, const Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    writeCpuFp32Tensor(cpuTensor, values);
    tensor.copyFromAsync(cpuTensor, stream);
}

void setParameter(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, const Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    setTensor(parameter->getStorage().value(), values, stream);
}

void zeroOptimizerGradient(const shared_ptr<Impl::PhysicalParameter>& parameter, const Stream& gradientStream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->hasOptimizer());
    ASSERT_NE(parameter->getOptimizer(), nullptr);
    ASSERT_TRUE(parameter->getOptimizer()->getWeightsGradient().has_value());

    Impl::Tensor gradient = parameter->getOptimizer()->getWeightsGradient().value();
    setTensor(gradient, vector<float>(tensorNumel(gradient), 0.0f), gradientStream);
}

void expectAllClose(const vector<float>& actual,
                    const vector<float>& expected,
                    float absoluteTolerance,
                    float relativeTolerance,
                    const string& what) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(isfinite(expected[i])) << what << " reference is non-finite at index " << i;
        EXPECT_TRUE(isfinite(actual[i])) << what << " is non-finite at index " << i;
        const float tolerance = absoluteTolerance + relativeTolerance * fabs(expected[i]);
        EXPECT_LE(fabs(actual[i] - expected[i]), tolerance)
            << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

struct BackwardReference {
    vector<float> inputGradient;
    vector<float> weightsGradient;
    vector<float> biasesGradient;
};

void accumulateNormalizedGroup(const vector<float>& input,
                               const vector<float>& incomingError,
                               const vector<uint64_t>& elementIndices,
                               const vector<uint64_t>& parameterIndices,
                               double epsilon,
                               vector<double>& inputGradient,
                               vector<double>& weightsGradient,
                               vector<double>& biasesGradient) {
    ASSERT_EQ(elementIndices.size(), parameterIndices.size());
    ASSERT_FALSE(elementIndices.empty());

    double mean = 0.0;
    for (uint64_t index : elementIndices)
        mean += input.at(index);
    mean /= static_cast<double>(elementIndices.size());

    double variance = 0.0;
    for (uint64_t index : elementIndices) {
        const double centered = static_cast<double>(input.at(index)) - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(elementIndices.size());
    const double inverseStandardDeviation = 1.0 / sqrt(variance + epsilon);

    double sumDy = 0.0;
    double sumDyXhat = 0.0;
    vector<double> normalizedInput(elementIndices.size());
    for (uint64_t i = 0; i < elementIndices.size(); ++i) {
        const uint64_t elementIndex = elementIndices[i];
        normalizedInput[i] = (static_cast<double>(input.at(elementIndex)) - mean) * inverseStandardDeviation;
        sumDy += incomingError.at(elementIndex);
        sumDyXhat += static_cast<double>(incomingError.at(elementIndex)) * normalizedInput[i];
    }

    const double elementCount = static_cast<double>(elementIndices.size());
    for (uint64_t i = 0; i < elementIndices.size(); ++i) {
        const uint64_t elementIndex = elementIndices[i];
        const uint64_t parameterIndex = parameterIndices[i];
        const double dy = incomingError.at(elementIndex);
        inputGradient.at(elementIndex) =
            inverseStandardDeviation / elementCount * (elementCount * dy - sumDy - normalizedInput[i] * sumDyXhat);
        weightsGradient.at(parameterIndex) += dy * normalizedInput[i];
        biasesGradient.at(parameterIndex) += dy;
    }
}

vector<float> toFloatVector(const vector<double>& values) {
    vector<float> result(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        result[i] = static_cast<float>(values[i]);
    return result;
}

BackwardReference computeBackwardReference(NormalizationKind kind) {
    const uint64_t parameterCount = kind == NormalizationKind::LAYER_NORM ? featureCount : channelCount;
    vector<double> inputGradient(inputValues.size(), 0.0);
    vector<double> weightsGradient(parameterCount, 0.0);
    vector<double> biasesGradient(parameterCount, 0.0);

    if (kind == NormalizationKind::LAYER_NORM) {
        for (uint64_t batch = 0; batch < batchSize; ++batch) {
            vector<uint64_t> elementIndices(featureCount);
            vector<uint64_t> parameterIndices(featureCount);
            for (uint64_t feature = 0; feature < featureCount; ++feature) {
                elementIndices[feature] = batch * featureCount + feature;
                parameterIndices[feature] = feature;
            }
            accumulateNormalizedGroup(inputValues,
                                      incomingErrorValues,
                                      elementIndices,
                                      parameterIndices,
                                      layerAndInstanceNormEpsilon,
                                      inputGradient,
                                      weightsGradient,
                                      biasesGradient);
        }
    } else if (kind == NormalizationKind::INSTANCE_NORM) {
        for (uint64_t batch = 0; batch < batchSize; ++batch) {
            for (uint64_t channel = 0; channel < channelCount; ++channel) {
                vector<uint64_t> elementIndices(spatialElementCount);
                vector<uint64_t> parameterIndices(spatialElementCount, channel);
                for (uint64_t spatial = 0; spatial < spatialElementCount; ++spatial) {
                    elementIndices[spatial] =
                        (batch * channelCount + channel) * spatialElementCount + spatial;
                }
                accumulateNormalizedGroup(inputValues,
                                          incomingErrorValues,
                                          elementIndices,
                                          parameterIndices,
                                          layerAndInstanceNormEpsilon,
                                          inputGradient,
                                          weightsGradient,
                                          biasesGradient);
            }
        }
    } else {
        for (uint64_t channel = 0; channel < channelCount; ++channel) {
            vector<uint64_t> elementIndices;
            vector<uint64_t> parameterIndices;
            elementIndices.reserve(batchSize * spatialElementCount);
            parameterIndices.reserve(batchSize * spatialElementCount);
            for (uint64_t batch = 0; batch < batchSize; ++batch) {
                for (uint64_t spatial = 0; spatial < spatialElementCount; ++spatial) {
                    elementIndices.push_back((batch * channelCount + channel) * spatialElementCount + spatial);
                    parameterIndices.push_back(channel);
                }
            }
            accumulateNormalizedGroup(inputValues,
                                      incomingErrorValues,
                                      elementIndices,
                                      parameterIndices,
                                      batchNormEpsilon,
                                      inputGradient,
                                      weightsGradient,
                                      biasesGradient);
        }
    }

    return BackwardReference{
        toFloatVector(inputGradient), toFloatVector(weightsGradient), toFloatVector(biasesGradient)};
}

struct RMSNormBackwardReference {
    vector<float> inputGradient;
    vector<float> weightsGradient;
};

RMSNormBackwardReference computeRMSNormBackwardReference() {
    vector<double> inputGradient(inputValues.size(), 0.0);
    vector<double> weightsGradient(featureCount, 0.0);

    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        double meanSquare = 0.0;
        double sumDyTimesX = 0.0;
        for (uint64_t feature = 0; feature < featureCount; ++feature) {
            const uint64_t index = batch * featureCount + feature;
            const double x = inputValues[index];
            meanSquare += x * x;
            sumDyTimesX += static_cast<double>(incomingErrorValues[index]) * x;
        }
        meanSquare /= static_cast<double>(featureCount);
        const double inverseRms = 1.0 / sqrt(meanSquare + layerAndInstanceNormEpsilon);
        const double inverseRmsCubed = inverseRms * inverseRms * inverseRms;

        for (uint64_t feature = 0; feature < featureCount; ++feature) {
            const uint64_t index = batch * featureCount + feature;
            const double x = inputValues[index];
            const double dy = incomingErrorValues[index];
            inputGradient[index] = inverseRms * dy -
                                   x * inverseRmsCubed * sumDyTimesX / static_cast<double>(featureCount);
            weightsGradient[feature] += dy * x * inverseRms;
        }
    }

    return RMSNormBackwardReference{toFloatVector(inputGradient), toFloatVector(weightsGradient)};
}

struct ApiNormalization {
    uint64_t layerId = 0;
    Api::Tensor output;
    shared_ptr<Api::ParameterSpecification> weights;
    shared_ptr<Api::ParameterSpecification> biases;
};

ApiNormalization addNormalization(NormalizationKind kind,
                                  Api::Network& network,
                                  const Api::Tensor& input,
                                  const shared_ptr<Api::Optimizer>& optimizer) {
    ApiNormalization result;

    switch (kind) {
        case NormalizationKind::LAYER_NORM: {
            Api::LayerNorm::Builder builder;
            builder.network(network).featureInput(input).normalizedShape({featureCount});
            if (optimizer != nullptr)
                builder.weightsOptimizer(optimizer).biasesOptimizer(optimizer);
            Api::LayerNorm layer = builder.build();
            result.layerId = layer.getId();
            result.output = layer.getFeatureOutput().value();
            result.weights = layer.getParameterSpecification("weights");
            result.biases = layer.getParameterSpecification("biases");
            break;
        }
        case NormalizationKind::INSTANCE_NORM: {
            Api::InstanceNorm::Builder builder;
            builder.network(network).featureInput(input);
            if (optimizer != nullptr)
                builder.weightsOptimizer(optimizer).biasesOptimizer(optimizer);
            Api::InstanceNorm layer = builder.build();
            result.layerId = layer.getId();
            result.output = layer.getFeatureOutput().value();
            result.weights = layer.getParameterSpecification("weights");
            result.biases = layer.getParameterSpecification("biases");
            break;
        }
        case NormalizationKind::BATCH_NORMALIZATION: {
            Api::BatchNormalization::Builder builder;
            builder.network(network).featureInput(input);
            if (optimizer != nullptr)
                builder.optimizer(optimizer);
            Api::BatchNormalization layer = builder.build();
            result.layerId = layer.getId();
            result.output = layer.getFeatureOutput().value();
            result.weights = layer.getParameterSpecification("weights");
            result.biases = layer.getParameterSpecification("biases");
            break;
        }
    }

    EXPECT_NE(result.layerId, 0u);
    EXPECT_TRUE(result.output.isInitialized());
    EXPECT_NE(result.weights, nullptr);
    EXPECT_NE(result.biases, nullptr);
    return result;
}

struct NormalizationHarness {
    shared_ptr<Api::Network> network;
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    shared_ptr<Impl::NetworkInput> physicalInput;
    shared_ptr<Impl::TrainableLayer> physicalNormalization;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
};

shared_ptr<Api::Optimizer> makeTestOptimizer() {
    return Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
}

NormalizationHarness buildHarness(NormalizationKind kind,
                                  bool requestInputGradient,
                                  bool attachOptimizer,
                                  bool weightsTrainingEnabled,
                                  bool biasesTrainingEnabled) {
    NormalizationHarness harness;
    harness.network = make_shared<Api::Network>("frozen_normalization_backward_" + normalizationKindName(kind));

    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*harness.network)
                                  .name("input")
                                  .dimensions(inputDimensions(kind))
                                  .dataType(Api::DataType::FP32)
                                  .build();

    Api::Tensor normalizationInput = input.getFeatureOutput().value();
    if (requestInputGradient) {
        Api::GradientRivet upstreamRivet =
            Api::GradientRivet::Builder().network(*harness.network).tensor(normalizationInput).build();
        normalizationInput = upstreamRivet.getFeatureOutput().value();
    }

    shared_ptr<Api::Optimizer> optimizer = attachOptimizer ? makeTestOptimizer() : nullptr;
    ApiNormalization normalization = addNormalization(kind, *harness.network, normalizationInput, optimizer);
    normalization.weights->setTrainingInitiallyEnabled(weightsTrainingEnabled);
    normalization.biases->setTrainingInitiallyEnabled(biasesTrainingEnabled);

    Api::GradientRivet downstreamRivet =
        Api::GradientRivet::Builder().network(*harness.network).tensor(normalization.output).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(*harness.network)
                                    .name("output")
                                    .inputTensor(downstreamRivet.getFeatureOutput().value())
                                    .dataType(Api::DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    harness.placedNetwork = harness.network->place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents)
        event.synchronize();

    if (harness.placedNetwork == nullptr)
        throw runtime_error("Failed to place normalization regression-test network.");
    Impl::StampedNetwork& stampedNetwork = harness.placedNetwork->getStampedNetwork(0);
    harness.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input.getId()));
    harness.physicalNormalization =
        dynamic_pointer_cast<Impl::TrainableLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(normalization.layerId));
    harness.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));

    if (harness.physicalInput == nullptr || harness.physicalNormalization == nullptr || harness.physicalOutput == nullptr)
        throw runtime_error("Failed to resolve physical layers for normalization regression test.");
    return harness;
}

void initializeParametersAndForward(NormalizationHarness& harness, const vector<float>& values) {
    ASSERT_NE(harness.physicalNormalization, nullptr);
    ASSERT_FALSE(harness.physicalNormalization->getStreams().empty());
    Stream dataStream = harness.physicalNormalization->getStreams()[0];

    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    setParameter(weights, vector<float>(tensorNumel(weights->getStorage().value()), 1.0f), dataStream);
    setParameter(biases, vector<float>(tensorNumel(biases->getStorage().value()), 0.0f), dataStream);
    dataStream.synchronize();

    ASSERT_FALSE(harness.physicalNormalization->getFeatureInputs().empty());
    ASSERT_TRUE(harness.physicalNormalization->getFeatureInputs()[0].has_value());
    Impl::Tensor hostInput = harness.physicalNormalization->getFeatureInputs()[0].value().clone(cpuPlacement);
    writeCpuFp32Tensor(hostInput, values);

    harness.physicalInput->forward(hostInput, /*isValidation=*/false, batchSize);
    harness.physicalOutput->getOutputReadyEvent().synchronize();
}

Impl::Tensor seedIncomingError(NormalizationHarness& harness, const vector<float>& errorValues) {
    if (harness.physicalNormalization == nullptr || harness.physicalNormalization->getErrorInputs().empty() ||
        !harness.physicalNormalization->getErrorInputs()[0].has_value()) {
        throw runtime_error("Normalization regression-test graph does not have the expected incoming error tensor.");
    }

    Stream dataStream = harness.physicalNormalization->getStreams()[0];
    Impl::Tensor errorInput = harness.physicalNormalization->getErrorInputs()[0].value();
    setTensor(errorInput, errorValues, dataStream);
    return errorInput;
}

struct RMSNormHarness {
    unique_ptr<Impl::NetworkInput> input;
    unique_ptr<Impl::GradientRivet> upstreamRivet;
    unique_ptr<Impl::RMSNorm> normalization;
    unique_ptr<Impl::GradientRivet> downstreamRivet;
    unique_ptr<Impl::NetworkOutput> output;
};

RMSNormHarness buildRMSNormHarness(bool requestInputGradient, bool attachOptimizer, bool weightsTrainingEnabled) {
    RMSNormHarness harness;
    const Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);
    const Impl::TensorDescriptor descriptor(Impl::DataType::FP32, {batchSize, featureCount});

    harness.input = make_unique<Impl::NetworkInput>(gpuPlacement, Impl::DataType::FP32, descriptor.getDimensions());
    if (requestInputGradient)
        harness.upstreamRivet = make_unique<Impl::GradientRivet>();
    harness.normalization = make_unique<Impl::RMSNorm>(gpuPlacement,
                                                       /*inferenceOnly=*/false,
                                                       vector<uint64_t>{featureCount},
                                                       layerAndInstanceNormEpsilon,
                                                       Impl::DataType::FP32);
    harness.downstreamRivet = make_unique<Impl::GradientRivet>();
    harness.output = make_unique<Impl::NetworkOutput>(cpuPlacement);

    shared_ptr<Impl::PhysicalParameter> weights = harness.normalization->getParameter("weights");
    THOR_THROW_IF_FALSE(weights != nullptr);
    if (attachOptimizer) {
        harness.normalization->setOptimizer(
            "weights", make_shared<Impl::Sgd>(9100, 0.01f, 0.0f, 0.0f, false));
    }
    weights->setTrainingEnabled(weightsTrainingEnabled);

    if (requestInputGradient) {
        harness.input->connectToNextLayer(harness.upstreamRivet.get());
        harness.upstreamRivet->connectToNextLayer(harness.normalization.get());
    } else {
        harness.input->connectToNextLayer(harness.normalization.get());
    }
    harness.normalization->connectToNextLayer(harness.downstreamRivet.get());
    harness.downstreamRivet->connectToNextLayer(harness.output.get());

    harness.input->compile();
    if (harness.upstreamRivet != nullptr)
        harness.upstreamRivet->compile();
    harness.normalization->compile();
    harness.downstreamRivet->compile();
    harness.output->compile();

    harness.input->initialize();
    if (harness.upstreamRivet != nullptr)
        harness.upstreamRivet->initialize();
    harness.normalization->initialize();
    harness.downstreamRivet->initialize();
    harness.output->initialize();
    return harness;
}

void initializeRMSNormAndForward(RMSNormHarness& harness) {
    Stream dataStream = harness.normalization->getStreams()[0];
    shared_ptr<Impl::PhysicalParameter> weights = harness.normalization->getParameter("weights");
    setParameter(weights, vector<float>(tensorNumel(weights->getStorage().value()), 1.0f), dataStream);
    dataStream.synchronize();

    Impl::Tensor hostInput(cpuPlacement,
                           Impl::TensorDescriptor(Impl::DataType::FP32, {batchSize, featureCount}));
    writeCpuFp32Tensor(hostInput, inputValues);
    harness.input->forward(hostInput, /*isValidation=*/false, batchSize);
    harness.output->getOutputReadyEvent().synchronize();
}

Impl::Tensor seedRMSNormIncomingError(RMSNormHarness& harness) {
    THOR_THROW_IF_FALSE(!harness.normalization->getErrorInputs().empty());
    THOR_THROW_IF_FALSE(harness.normalization->getErrorInputs()[0].has_value());
    Stream dataStream = harness.normalization->getStreams()[0];
    Impl::Tensor errorInput = harness.normalization->getErrorInputs()[0].value();
    setTensor(errorInput, incomingErrorValues, dataStream);
    return errorInput;
}

void synchronizeRMSNormHarness(RMSNormHarness& harness) {
    for (Event& event : harness.normalization->getSynchronizeEvents())
        event.synchronize();
}

vector<float> readParameterStorage(const shared_ptr<Impl::PhysicalParameter>& parameter, const Stream& stream) {
    if (parameter == nullptr || !parameter->getStorage().has_value())
        throw runtime_error("Expected a materialized parameter storage tensor.");
    return readCpuFp32Tensor(copyTensorToCpu(parameter->getStorage().value(), stream));
}

vector<float> readOptimizerGradient(const shared_ptr<Impl::PhysicalParameter>& parameter, const Stream& stream) {
    if (parameter == nullptr || !parameter->hasOptimizer() || parameter->getOptimizer() == nullptr ||
        !parameter->getOptimizer()->getWeightsGradient().has_value()) {
        throw runtime_error("Expected a materialized optimizer gradient tensor.");
    }
    return readCpuFp32Tensor(copyTensorToCpu(parameter->getOptimizer()->getWeightsGradient().value(), stream));
}

void expectFrozenParameterHasNoMaterializedGradient(const shared_ptr<Impl::PhysicalParameter>& parameter) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_FALSE(parameter->isTrainingEnabled());
    if (parameter->hasOptimizer()) {
        ASSERT_NE(parameter->getOptimizer(), nullptr);
        EXPECT_FALSE(parameter->getOptimizer()->getWeightsGradient().has_value());
    }
}

class FrozenNormalizationBackwardTest : public testing::TestWithParam<NormalizationKind> {};

TEST_P(FrozenNormalizationBackwardTest, FullyTrainableControlMatchesNumericalReference) {
    NormalizationHarness harness = buildHarness(GetParam(),
                                                /*requestInputGradient=*/true,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/true,
                                                /*biasesTrainingEnabled=*/true);
    const BackwardReference reference = computeBackwardReference(GetParam());

    ASSERT_TRUE(harness.physicalNormalization->getGradientUpdateStream().has_value());
    ASSERT_EQ(harness.physicalNormalization->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(harness.physicalNormalization->getErrorOutputs()[0].has_value());

    Stream gradientStream = harness.physicalNormalization->getGradientUpdateStream().value();
    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    zeroOptimizerGradient(weights, gradientStream);
    zeroOptimizerGradient(biases, gradientStream);
    gradientStream.synchronize();

    initializeParametersAndForward(harness, inputValues);
    Impl::Tensor errorInput = seedIncomingError(harness, incomingErrorValues);
    ASSERT_NO_THROW(harness.physicalNormalization->backward(errorInput, batchSize));
    harness.placedNetwork->synchronize();

    Stream dataStream = harness.physicalNormalization->getStreams()[0];
    const vector<float> actualInputGradient = readCpuFp32Tensor(
        copyTensorToCpu(harness.physicalNormalization->getErrorOutputs()[0].value(), dataStream));
    const vector<float> actualWeightsGradient = readOptimizerGradient(weights, gradientStream);
    const vector<float> actualBiasesGradient = readOptimizerGradient(biases, gradientStream);

    expectAllClose(actualInputGradient,
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " fully-trainable input gradient");
    expectAllClose(actualWeightsGradient,
                   reference.weightsGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " fully-trainable weights gradient");
    expectAllClose(actualBiasesGradient,
                   reference.biasesGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " fully-trainable biases gradient");
}

TEST_P(FrozenNormalizationBackwardTest, FullyFrozenLayerComputesNumericallyCorrectInputGradient) {
    NormalizationHarness harness = buildHarness(GetParam(),
                                                /*requestInputGradient=*/true,
                                                /*attachOptimizer=*/false,
                                                /*weightsTrainingEnabled=*/false,
                                                /*biasesTrainingEnabled=*/false);
    const BackwardReference reference = computeBackwardReference(GetParam());

    ASSERT_FALSE(harness.physicalNormalization->getGradientUpdateStream().has_value());
    ASSERT_EQ(harness.physicalNormalization->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(harness.physicalNormalization->getErrorOutputs()[0].has_value());
    expectFrozenParameterHasNoMaterializedGradient(harness.physicalNormalization->getParameter("weights"));
    expectFrozenParameterHasNoMaterializedGradient(harness.physicalNormalization->getParameter("biases"));

    initializeParametersAndForward(harness, inputValues);
    Impl::Tensor errorInput = seedIncomingError(harness, incomingErrorValues);
    ASSERT_NO_THROW(harness.physicalNormalization->backward(errorInput, batchSize));
    harness.placedNetwork->synchronize();

    Stream dataStream = harness.physicalNormalization->getStreams()[0];
    const vector<float> actualInputGradient = readCpuFp32Tensor(
        copyTensorToCpu(harness.physicalNormalization->getErrorOutputs()[0].value(), dataStream));
    expectAllClose(actualInputGradient,
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " frozen input-only gradient");
}

TEST_P(FrozenNormalizationBackwardTest, TrainableLayerComputesNumericallyCorrectParameterGradientsWhenInputGradientIsPruned) {
    NormalizationHarness harness = buildHarness(GetParam(),
                                                /*requestInputGradient=*/false,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/true,
                                                /*biasesTrainingEnabled=*/true);
    const BackwardReference reference = computeBackwardReference(GetParam());

    ASSERT_TRUE(harness.physicalNormalization->getGradientUpdateStream().has_value());
    ASSERT_EQ(harness.physicalNormalization->getErrorOutputs().size(), 1u);
    ASSERT_FALSE(harness.physicalNormalization->getErrorOutputs()[0].has_value());

    Stream gradientStream = harness.physicalNormalization->getGradientUpdateStream().value();
    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    zeroOptimizerGradient(weights, gradientStream);
    zeroOptimizerGradient(biases, gradientStream);
    gradientStream.synchronize();

    initializeParametersAndForward(harness, inputValues);
    Impl::Tensor errorInput = seedIncomingError(harness, incomingErrorValues);
    ASSERT_NO_THROW(harness.physicalNormalization->backward(errorInput, batchSize));
    harness.placedNetwork->synchronize();

    const vector<float> actualWeightsGradient = readOptimizerGradient(weights, gradientStream);
    const vector<float> actualBiasesGradient = readOptimizerGradient(biases, gradientStream);
    expectAllClose(actualWeightsGradient,
                   reference.weightsGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " parameter-only weights gradient");
    expectAllClose(actualBiasesGradient,
                   reference.biasesGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " parameter-only biases gradient");
}

TEST_P(FrozenNormalizationBackwardTest, FrozenScaleUsesScratchAndComputesBiasAndInputGradients) {
    NormalizationHarness harness = buildHarness(GetParam(),
                                                /*requestInputGradient=*/true,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/false,
                                                /*biasesTrainingEnabled=*/true);
    const BackwardReference reference = computeBackwardReference(GetParam());

    ASSERT_TRUE(harness.physicalNormalization->getGradientUpdateStream().has_value());
    ASSERT_EQ(harness.physicalNormalization->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(harness.physicalNormalization->getErrorOutputs()[0].has_value());

    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    expectFrozenParameterHasNoMaterializedGradient(weights);
    ASSERT_TRUE(biases->isTrainingEnabled());
    ASSERT_TRUE(biases->hasOptimizer());

    Stream gradientStream = harness.physicalNormalization->getGradientUpdateStream().value();
    zeroOptimizerGradient(biases, gradientStream);
    gradientStream.synchronize();

    initializeParametersAndForward(harness, inputValues);
    Impl::Tensor errorInput = seedIncomingError(harness, incomingErrorValues);
    ASSERT_NO_THROW(harness.physicalNormalization->backward(errorInput, batchSize));
    harness.placedNetwork->synchronize();

    Stream dataStream = harness.physicalNormalization->getStreams()[0];
    const vector<float> actualInputGradient = readCpuFp32Tensor(
        copyTensorToCpu(harness.physicalNormalization->getErrorOutputs()[0].value(), dataStream));
    const vector<float> actualBiasesGradient = readOptimizerGradient(biases, gradientStream);
    const vector<float> actualWeights = readParameterStorage(weights, dataStream);

    expectAllClose(actualInputGradient,
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " frozen-scale input gradient");
    expectAllClose(actualBiasesGradient,
                   reference.biasesGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " frozen-scale biases gradient");
    expectAllClose(actualWeights,
                   vector<float>(actualWeights.size(), 1.0f),
                   0.0f,
                   0.0f,
                   normalizationKindName(GetParam()) + " frozen scale storage");
}

TEST_P(FrozenNormalizationBackwardTest, FrozenBiasUsesScratchAndComputesScaleAndInputGradients) {
    NormalizationHarness harness = buildHarness(GetParam(),
                                                /*requestInputGradient=*/true,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/true,
                                                /*biasesTrainingEnabled=*/false);
    const BackwardReference reference = computeBackwardReference(GetParam());

    ASSERT_TRUE(harness.physicalNormalization->getGradientUpdateStream().has_value());
    ASSERT_EQ(harness.physicalNormalization->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(harness.physicalNormalization->getErrorOutputs()[0].has_value());

    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    ASSERT_TRUE(weights->isTrainingEnabled());
    ASSERT_TRUE(weights->hasOptimizer());
    expectFrozenParameterHasNoMaterializedGradient(biases);

    Stream gradientStream = harness.physicalNormalization->getGradientUpdateStream().value();
    zeroOptimizerGradient(weights, gradientStream);
    gradientStream.synchronize();

    initializeParametersAndForward(harness, inputValues);
    Impl::Tensor errorInput = seedIncomingError(harness, incomingErrorValues);
    ASSERT_NO_THROW(harness.physicalNormalization->backward(errorInput, batchSize));
    harness.placedNetwork->synchronize();

    Stream dataStream = harness.physicalNormalization->getStreams()[0];
    const vector<float> actualInputGradient = readCpuFp32Tensor(
        copyTensorToCpu(harness.physicalNormalization->getErrorOutputs()[0].value(), dataStream));
    const vector<float> actualWeightsGradient = readOptimizerGradient(weights, gradientStream);
    const vector<float> actualBiases = readParameterStorage(biases, dataStream);

    expectAllClose(actualInputGradient,
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " frozen-bias input gradient");
    expectAllClose(actualWeightsGradient,
                   reference.weightsGradient,
                   1.0e-3f,
                   1.0e-3f,
                   normalizationKindName(GetParam()) + " frozen-bias weights gradient");
    expectAllClose(actualBiases,
                   vector<float>(actualBiases.size(), 0.0f),
                   0.0f,
                   0.0f,
                   normalizationKindName(GetParam()) + " frozen bias storage");
}


TEST(FrozenRMSNormBackwardTest, FullyTrainableControlMatchesNumericalReference) {
    RMSNormHarness harness = buildRMSNormHarness(/*requestInputGradient=*/true,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/true);
    const RMSNormBackwardReference reference = computeRMSNormBackwardReference();
    ASSERT_TRUE(harness.normalization->getGradientUpdateStream().has_value());
    ASSERT_TRUE(harness.normalization->getErrorOutputs()[0].has_value());

    Stream gradientStream = harness.normalization->getGradientUpdateStream().value();
    shared_ptr<Impl::PhysicalParameter> weights = harness.normalization->getParameter("weights");
    zeroOptimizerGradient(weights, gradientStream);
    gradientStream.synchronize();

    initializeRMSNormAndForward(harness);
    Impl::Tensor errorInput = seedRMSNormIncomingError(harness);
    ASSERT_NO_THROW(harness.normalization->backward(errorInput, batchSize));
    synchronizeRMSNormHarness(harness);

    Stream dataStream = harness.normalization->getStreams()[0];
    expectAllClose(readCpuFp32Tensor(copyTensorToCpu(harness.normalization->getErrorOutputs()[0].value(), dataStream)),
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   "RMSNorm fully-trainable input gradient");
    expectAllClose(readOptimizerGradient(weights, gradientStream),
                   reference.weightsGradient,
                   1.0e-3f,
                   1.0e-3f,
                   "RMSNorm fully-trainable weights gradient");
}

TEST(FrozenRMSNormBackwardTest, FullyFrozenLayerComputesNumericallyCorrectInputGradient) {
    RMSNormHarness harness = buildRMSNormHarness(/*requestInputGradient=*/true,
                                                /*attachOptimizer=*/false,
                                                /*weightsTrainingEnabled=*/false);
    const RMSNormBackwardReference reference = computeRMSNormBackwardReference();
    ASSERT_FALSE(harness.normalization->getGradientUpdateStream().has_value());
    ASSERT_TRUE(harness.normalization->getErrorOutputs()[0].has_value());
    expectFrozenParameterHasNoMaterializedGradient(harness.normalization->getParameter("weights"));

    initializeRMSNormAndForward(harness);
    Impl::Tensor errorInput = seedRMSNormIncomingError(harness);
    ASSERT_NO_THROW(harness.normalization->backward(errorInput, batchSize));
    synchronizeRMSNormHarness(harness);

    Stream dataStream = harness.normalization->getStreams()[0];
    expectAllClose(readCpuFp32Tensor(copyTensorToCpu(harness.normalization->getErrorOutputs()[0].value(), dataStream)),
                   reference.inputGradient,
                   1.0e-3f,
                   1.0e-3f,
                   "RMSNorm frozen input-only gradient");
}

TEST(FrozenRMSNormBackwardTest, ComputesNumericallyCorrectParameterGradientWhenInputGradientIsPruned) {
    RMSNormHarness harness = buildRMSNormHarness(/*requestInputGradient=*/false,
                                                /*attachOptimizer=*/true,
                                                /*weightsTrainingEnabled=*/true);
    const RMSNormBackwardReference reference = computeRMSNormBackwardReference();
    ASSERT_TRUE(harness.normalization->getGradientUpdateStream().has_value());
    ASSERT_FALSE(harness.normalization->getErrorOutputs()[0].has_value());

    Stream gradientStream = harness.normalization->getGradientUpdateStream().value();
    shared_ptr<Impl::PhysicalParameter> weights = harness.normalization->getParameter("weights");
    zeroOptimizerGradient(weights, gradientStream);
    gradientStream.synchronize();

    initializeRMSNormAndForward(harness);
    Impl::Tensor errorInput = seedRMSNormIncomingError(harness);
    ASSERT_NO_THROW(harness.normalization->backward(errorInput, batchSize));
    synchronizeRMSNormHarness(harness);

    expectAllClose(readOptimizerGradient(weights, gradientStream),
                   reference.weightsGradient,
                   1.0e-3f,
                   1.0e-3f,
                   "RMSNorm parameter-only weights gradient");
}

class OrderingCaptureLayer final : public Impl::Layer {
   public:
    OrderingCaptureLayer(const Impl::TensorPlacement& placement, const Stream& stream, const Impl::TensorDescriptor& descriptor)
        : capturedGradient(placement, descriptor) {
        this->stream = stream;
        errorOutput = capturedGradient;
        running = true;
    }

    Impl::Tensor getCapturedGradient() const { return capturedGradient; }
    Event getCaptureCompleteEvent() const { return captureCompleteEvent; }

   protected:
    void infer(optional<Impl::Tensor>, optional<Impl::Tensor>, Stream) override {}

    void backProp(optional<Impl::Tensor>, optional<Impl::Tensor> errorIn, optional<Impl::Tensor>, Stream stream) override {
        THOR_THROW_IF_FALSE(errorIn.has_value());
        capturedGradient.copyFromAsync(errorIn.value(), stream);
        captureCompleteEvent = stream.putEvent(/*enableTiming=*/false,
                                               /*expectingHostToWaitOnThisOne=*/true);
    }

   private:
    Impl::Tensor capturedGradient;
    Event captureCompleteEvent;
};

class OrderingProbeTrainableLayer final : public Impl::TrainableLayer {
   public:
    OrderingProbeTrainableLayer(const Impl::TensorPlacement& placement,
                                const Stream& dataStream,
                                const Stream& gradientStream,
                                OrderingCaptureLayer& previousLayer,
                                const Impl::TensorDescriptor& descriptor)
        : Impl::TrainableLayer(placement, /*inferenceOnly=*/false) {
        featureInputs.emplace_back(Impl::Tensor(placement, descriptor));
        featureOutputs.emplace_back(featureInputs.back()->clone());
        errorInputs.emplace_back(featureInputs.back()->clone());
        errorOutputs.emplace_back(featureInputs.back()->clone());
        streams.push_back(dataStream);
        previousLayers.emplace_back(&previousLayer);
        nextLayers.emplace_back(nullopt);

        uniqueDataStreams.push_back(dataStream);
        gradientUpdateStream = gradientStream;
        numBackwardConnections = 1;
        isStartOfBackward = true;
        running = true;
    }

    Impl::Tensor getIncomingGradient() const { return errorInputs[0].value(); }
    Impl::Tensor getProducedInputGradient() const { return errorOutputs[0].value(); }

    string getLayerType() override { return "OrderingProbeTrainableLayer"; }
    uint64_t flopCountForward() override { return 0; }
    uint64_t flopCountBackward() override { return 0; }

   protected:
    void computeFeatureOut(uint32_t) override {}

    optional<Event> computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                  bool) override {
        THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
        errorOutputs[connectionNumber]->copyFromAsync(errorInputs[connectionNumber].value(),
                                                       gradientUpdateStream.value());
        return gradientUpdateStream->putEvent(/*enableTiming=*/false,
                                              /*expectingHostToWaitOnThisOne=*/false);
    }
};

TEST(FrozenNormalizationBackwardStateTest, PlacedLegacyParameterTrainingToggleIsRejected) {
    NormalizationHarness harness = buildHarness(NormalizationKind::LAYER_NORM,
                                                /*requestInputGradient=*/true,
                                                /*attachOptimizer=*/false,
                                                /*weightsTrainingEnabled=*/false,
                                                /*biasesTrainingEnabled=*/false);

    shared_ptr<Impl::PhysicalParameter> weights = harness.physicalNormalization->getParameter("weights");
    shared_ptr<Impl::PhysicalParameter> biases = harness.physicalNormalization->getParameter("biases");
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(biases, nullptr);
    ASSERT_FALSE(weights->isTrainingEnabled());
    ASSERT_FALSE(biases->isTrainingEnabled());

    EXPECT_THROW(weights->setTrainingEnabled(true), runtime_error);
    EXPECT_THROW(biases->setTrainingEnabled(true), runtime_error);
    EXPECT_FALSE(weights->isTrainingEnabled());
    EXPECT_FALSE(biases->isTrainingEnabled());
}

TEST(FrozenNormalizationBackwardOrderingTest, UpstreamDataStreamWaitsForFusedInputGradient) {
    constexpr float sentinel = 12345.0f;
    const Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);
    const Impl::TensorDescriptor descriptor(Impl::DataType::FP32, {16});
    Stream dataStream(0);
    Stream gradientStream(0);
    ASSERT_NE(dataStream.getId(), gradientStream.getId());

    OrderingCaptureLayer captureLayer(gpuPlacement, dataStream, descriptor);
    OrderingProbeTrainableLayer trainableLayer(gpuPlacement,
                                               dataStream,
                                               gradientStream,
                                               captureLayer,
                                               descriptor);

    const vector<float> incomingValues{1.0f,  2.0f,  3.0f,  4.0f,
                                       5.0f,  6.0f,  7.0f,  8.0f,
                                       9.0f,  10.0f, 11.0f, 12.0f,
                                       13.0f, 14.0f, 15.0f, 16.0f};
    Impl::Tensor incomingGradient = trainableLayer.getIncomingGradient();
    Impl::Tensor producedInputGradient = trainableLayer.getProducedInputGradient();
    setTensor(incomingGradient, incomingValues, dataStream);
    setTensor(producedInputGradient,
              vector<float>(incomingValues.size(), sentinel),
              dataStream);
    Impl::Tensor capturedGradientStorage = captureLayer.getCapturedGradient();
    setTensor(capturedGradientStorage,
              vector<float>(incomingValues.size(), sentinel),
              dataStream);
    dataStream.synchronize();

    // The fused implementation below is intentionally held on the gradient stream. Unlike the earlier cuDNN-based
    // version of this regression, the test implementation performs only asynchronous copies and event recording, so
    // backward() cannot block inside a library call while the gate is closed.
    Impl::Test::DeviceStreamGate gradientGate(gradientStream.getGpuNum());
    gradientGate.enqueue(gradientStream);
    ASSERT_FALSE(gradientGate.isComplete());

    ASSERT_NO_THROW(trainableLayer.backward(incomingGradient, /*batchSize=*/1));
    Event captureCompleteEvent = captureLayer.getCaptureCompleteEvent();
    ASSERT_TRUE(captureCompleteEvent.isInitialized());

    bool upstreamAdvancedBeforeInputGradientWasReady = false;
    {
        ScopedGpu scopedGpu(dataStream.getGpuNum());
        for (uint32_t attempt = 0; attempt < 100; ++attempt) {
            cudaError_t status = cudaEventQuery(captureCompleteEvent.getEvent());
            if (status == cudaSuccess) {
                upstreamAdvancedBeforeInputGradientWasReady = true;
                break;
            }
            ASSERT_EQ(status, cudaErrorNotReady);
            this_thread::sleep_for(chrono::milliseconds(5));
        }
    }

    // Always release before asserting so neither the test nor CUDA teardown can be stranded by an expected failure.
    gradientGate.release();
    gradientStream.synchronize();
    dataStream.synchronize();

    EXPECT_FALSE(upstreamAdvancedBeforeInputGradientWasReady)
        << "TrainableLayer propagated errorOutputs upstream before the fused backward completion event made dx ready.";

    const vector<float> capturedGradient =
        readCpuFp32Tensor(copyTensorToCpu(captureLayer.getCapturedGradient(), dataStream));
    expectAllClose(capturedGradient,
                   incomingValues,
                   0.0f,
                   0.0f,
                   "fused-backward input-gradient stream ordering");
}

INSTANTIATE_TEST_SUITE_P(
    LegacyFusedNormalizationLayers,
    FrozenNormalizationBackwardTest,
    testing::Values(NormalizationKind::LAYER_NORM, NormalizationKind::INSTANCE_NORM, NormalizationKind::BATCH_NORMALIZATION),
    [](const testing::TestParamInfo<NormalizationKind>& info) { return normalizationKindName(info.param); });

}  // namespace
