#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = ThorImplementation::DataType;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

constexpr uint32_t kBatchSize = 3;
constexpr uint64_t kMaxTotalValues = 24;
constexpr uint64_t kMaxValuesPerRow = 8;
constexpr uint64_t kInputFeatures = 6;
constexpr uint64_t kProjectedFeatures = 10;
constexpr uint64_t kTemporalFeatures = 8;
constexpr uint64_t kFilterWidth = 3;
constexpr float kLearningRate = 0.01f;

const vector<uint32_t> kOffsets = {0, 8, 13, 15};
constexpr uint64_t kActiveValues = 15;
constexpr uint64_t kActiveMaxRowLength = 8;

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t count = 1;
    for (uint64_t dim : tensor.getDimensions()) count *= dim;
    return count;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events) event.synchronize();
    events.clear();
}

void writeCpuFp32(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    std::copy(values.begin(), values.end(), tensor.getMemPtr<float>());
}

vector<float> readCpuFp32(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    const float* ptr = tensor.getMemPtr<float>();
    return vector<float>(ptr, ptr + tensorNumel(tensor));
}

vector<uint32_t> readCpuUint32(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::UINT32);
    const uint32_t* ptr = tensor.getMemPtr<uint32_t>();
    return vector<uint32_t>(ptr, ptr + tensorNumel(tensor));
}

Impl::Tensor copyToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpu = tensor.clone(cpuPlacement);
    cpu.copyFromAsync(tensor, stream);
    stream.putEvent().synchronize();
    return cpu;
}

void setParameterConstant(const shared_ptr<Impl::PhysicalParameter>& parameter, float value, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor storage = parameter->getStorage().value();
    ASSERT_EQ(storage.getDataType(), DataType::FP32);
    Impl::Tensor host = storage.clone(cpuPlacement);
    vector<float> values(tensorNumel(storage), value);
    writeCpuFp32(host, values);
    storage.copyFromAsync(host, stream);
}

void setLayerParameters(const shared_ptr<Impl::CustomLayer>& layer, float weightsValue, float biasValue) {
    ASSERT_NE(layer, nullptr);
    Stream stream = layer->getStreams()[0];
    for (const string& parameterName : layer->listParameters()) {
        const float value = parameterName == "biases" ? biasValue : weightsValue;
        setParameterConstant(layer->getParameter(parameterName), value, stream);
    }
    stream.synchronize();
}

struct F2ChainHandles {
    Api::RaggedTensor networkInput;
    Api::RaggedTensor chainInput;
    vector<Api::RaggedTensor> stageOutputs;
    vector<uint64_t> stageChannels;
    uint64_t firstFullyConnectedId = 0;
    uint64_t rmsNormId = 0;
    uint64_t temporalProjectionId = 0;
    uint64_t convDilation1Id = 0;
    Api::RaggedTensor reluAfterDilation1Output;
    uint64_t convDilation7Id = 0;
    Api::RaggedTensor reluAfterDilation7Output;
    uint64_t convDilation28Id = 0;
    uint64_t outputProjectionId = 0;
    std::optional<uint64_t> denseGradientSinkId;
};

F2ChainHandles buildF2PublicChain(Api::Network& network, bool addGradientRivets, bool addRaggedNetworkOutput) {
    F2ChainHandles handles;
    handles.networkInput = Api::RaggedNetworkInput::Builder()
                               .network(network)
                               .name("history")
                               .valuesDataType(DataType::FP32)
                               .offsetsDataType(DataType::UINT32)
                               .trailingDimensions({kInputFeatures})
                               .maxTotalValues(kMaxTotalValues)
                               .maxValuesPerRow(kMaxValuesPerRow)
                               .batchSize(kBatchSize)
                               .build();
    handles.chainInput = handles.networkInput;

    if (addGradientRivets) {
        Api::GradientRivet inputRivet =
            Api::GradientRivet::Builder().network(network).tensor(handles.networkInput.getValues()).build();
        handles.chainInput = Api::RaggedTensor(
            inputRivet.getFeatureOutput().value(), handles.networkInput.getOffsets(), kMaxValuesPerRow);
    }

    Api::FullyConnected projectedHistory = Api::FullyConnected::Builder()
                                              .network(network)
                                              .featureInput(handles.chainInput)
                                              .numOutputFeatures(kProjectedFeatures)
                                              .hasBias(true)
                                              .weightsDataType(DataType::FP32)
                                              .computeDataType(DataType::FP32)
                                              .outputDataType(DataType::FP32)
                                              .noActivation()
                                              .build();
    handles.firstFullyConnectedId = projectedHistory.getId();
    handles.stageOutputs.push_back(projectedHistory.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kProjectedFeatures);

    Api::RMSNorm rmsNorm = Api::RMSNorm::Builder()
                               .network(network)
                               .featureInput(handles.stageOutputs.back())
                               .normalizedShape({kProjectedFeatures})
                               .epsilon(1.0e-5)
                               .parameterDataType(DataType::FP32)
                               .build();
    handles.rmsNormId = rmsNorm.getId();
    handles.stageOutputs.push_back(rmsNorm.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kProjectedFeatures);

    Api::FullyConnected temporalProjection = Api::FullyConnected::Builder()
                                                 .network(network)
                                                 .featureInput(handles.stageOutputs.back())
                                                 .numOutputFeatures(kTemporalFeatures)
                                                 .hasBias(true)
                                                 .weightsDataType(DataType::FP32)
                                                 .computeDataType(DataType::FP32)
                                                 .outputDataType(DataType::FP32)
                                                 .noActivation()
                                                 .build();
    handles.temporalProjectionId = temporalProjection.getId();
    handles.stageOutputs.push_back(temporalProjection.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kTemporalFeatures);

    Api::Convolution1d convDilation1 = Api::Convolution1d::Builder()
                                           .network(network)
                                           .featureInput(handles.stageOutputs.back())
                                           .numOutputChannels(kTemporalFeatures)
                                           .filterWidth(kFilterWidth)
                                           .dilation(1)
                                           .causalPadding()
                                           .hasBias(true)
                                           .noActivation()
                                           .build();
    handles.convDilation1Id = convDilation1.getId();
    handles.stageOutputs.push_back(convDilation1.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kTemporalFeatures);

    shared_ptr<Api::Activation> reluAfterDilation1 = Api::Relu::Builder()
                                                         .network(network)
                                                         .featureInput(handles.stageOutputs.back())
                                                         .build();
    handles.reluAfterDilation1Output = reluAfterDilation1->getRaggedFeatureOutput().value();
    handles.stageOutputs.push_back(handles.reluAfterDilation1Output);
    handles.stageChannels.push_back(kTemporalFeatures);

    Api::Convolution1d convDilation7 = Api::Convolution1d::Builder()
                                           .network(network)
                                           .featureInput(handles.stageOutputs.back())
                                           .numOutputChannels(kTemporalFeatures)
                                           .filterWidth(kFilterWidth)
                                           .dilation(7)
                                           .causalPadding()
                                           .hasBias(true)
                                           .noActivation()
                                           .build();
    handles.convDilation7Id = convDilation7.getId();
    handles.stageOutputs.push_back(convDilation7.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kTemporalFeatures);

    shared_ptr<Api::Activation> reluAfterDilation7 = Api::Relu::Builder()
                                                         .network(network)
                                                         .featureInput(handles.stageOutputs.back())
                                                         .build();
    handles.reluAfterDilation7Output = reluAfterDilation7->getRaggedFeatureOutput().value();
    handles.stageOutputs.push_back(handles.reluAfterDilation7Output);
    handles.stageChannels.push_back(kTemporalFeatures);

    Api::Convolution1d convDilation28 = Api::Convolution1d::Builder()
                                            .network(network)
                                            .featureInput(handles.stageOutputs.back())
                                            .numOutputChannels(kTemporalFeatures)
                                            .filterWidth(kFilterWidth)
                                            .dilation(28)
                                            .causalPadding()
                                            .hasBias(true)
                                            .noActivation()
                                            .build();
    handles.convDilation28Id = convDilation28.getId();
    handles.stageOutputs.push_back(convDilation28.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kTemporalFeatures);

    Api::FullyConnected outputProjection = Api::FullyConnected::Builder()
                                               .network(network)
                                               .featureInput(handles.stageOutputs.back())
                                               .numOutputFeatures(kInputFeatures)
                                               .hasBias(true)
                                               .weightsDataType(DataType::FP32)
                                               .computeDataType(DataType::FP32)
                                               .outputDataType(DataType::FP32)
                                               .noActivation()
                                               .build();
    handles.outputProjectionId = outputProjection.getId();
    handles.stageOutputs.push_back(outputProjection.getRaggedFeatureOutput().value());
    handles.stageChannels.push_back(kInputFeatures);

    if (addRaggedNetworkOutput) {
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("temporal_output")
            .inputTensor(handles.stageOutputs.back())
            .build();
    }

    if (addGradientRivets) {
        Api::GradientRivet outputRivet =
            Api::GradientRivet::Builder().network(network).tensor(handles.stageOutputs.back().getValues()).build();
        Api::NetworkOutput gradientSink = Api::NetworkOutput::Builder()
                                              .network(network)
                                              .name("gradient_sink")
                                              .inputTensor(outputRivet.getFeatureOutput().value())
                                              .dataType(DataType::FP32)
                                              .build();
        handles.denseGradientSinkId = gradientSink.getId();
    }

    return handles;
}

void expectSamePartitionContract(const Api::RaggedTensor& tensor,
                                 const Api::RaggedTensor& root,
                                 uint64_t channels) {
    EXPECT_EQ(tensor.getOffsets(), root.getOffsets());
    EXPECT_EQ(tensor.getBatchSize(), root.getBatchSize());
    EXPECT_EQ(tensor.getMaxTotalValues(), root.getMaxTotalValues());
    ASSERT_TRUE(tensor.hasMaxValuesPerRow());
    EXPECT_EQ(tensor.getMaxValuesPerRow(), root.getMaxValuesPerRow());
    EXPECT_EQ(tensor.getValuesDimensions(), (vector<uint64_t>{root.getMaxTotalValues(), channels}));
}

void expectSameExternalPartitionDescriptor(const Api::RaggedTensor& tensor, const Api::RaggedTensor& root) {
    // RaggedNetworkOutput is a real graph boundary: its internal NetworkOutput
    // layers own distinct values/offsets tensors.  The canonical partition
    // invariant across that boundary is descriptor/content equality, not API
    // Tensor object identity.
    EXPECT_EQ(tensor.getBatchSize(), root.getBatchSize());
    EXPECT_EQ(tensor.getMaxTotalValues(), root.getMaxTotalValues());
    ASSERT_EQ(tensor.hasMaxValuesPerRow(), root.hasMaxValuesPerRow());
    if (root.hasMaxValuesPerRow()) EXPECT_EQ(tensor.getMaxValuesPerRow(), root.getMaxValuesPerRow());
    EXPECT_EQ(tensor.getOffsetsDataType(), root.getOffsetsDataType());
    EXPECT_EQ(tensor.getOffsets().getDimensions(), root.getOffsets().getDimensions());
}

vector<float> makePoisonedInputValues() {
    vector<float> values(kMaxTotalValues * kInputFeatures, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t row = 0; row < kActiveValues; ++row) {
        for (uint64_t channel = 0; channel < kInputFeatures; ++channel) {
            values[row * kInputFeatures + channel] =
                0.05f + 0.01f * static_cast<float>((row * 3 + channel * 5) % 17);
        }
    }
    return values;
}

void feedRaggedInput(Api::PlacedNetwork& placed,
                     const Api::RaggedNetworkInputReference& inputReference,
                     const vector<float>& values) {
    Impl::StampedNetwork& stamped = placed.getStampedNetwork(0);
    auto physicalValuesInput = stamped.getNamedInput(inputReference.valuesInputName);
    auto physicalOffsetsInput = stamped.getNamedInput(inputReference.offsetsInputName);
    ASSERT_NE(physicalValuesInput, nullptr);
    ASSERT_NE(physicalOffsetsInput, nullptr);

    Impl::Tensor offsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {kBatchSize + 1}));
    auto* offsets = offsetsHost.getMemPtr<uint32_t>();
    for (uint32_t i = 0; i <= kBatchSize; ++i) offsets[i] = kOffsets[i];
    physicalOffsetsInput->forwardRowPartitionOffsets(
        offsetsHost,
        false,
        Impl::RowPartitionDescriptor(kBatchSize, kMaxTotalValues, DataType::UINT32, kMaxValuesPerRow),
        kActiveValues,
        kActiveMaxRowLength,
        kBatchSize);

    Impl::Tensor valuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {kMaxTotalValues, kInputFeatures}));
    writeCpuFp32(valuesHost, values);
    physicalValuesInput->forward(valuesHost, false, kBatchSize);
}

vector<float> runRaggedForward(Api::PlacedNetwork& placed,
                               const Api::RaggedNetworkInputReference& inputReference,
                               const Api::RaggedNetworkOutputReference& outputReference,
                               const vector<float>& values) {
    feedRaggedInput(placed, inputReference, values);
    Impl::StampedNetwork& stamped = placed.getStampedNetwork(0);
    auto physicalValuesOutput = stamped.getNamedOutput(outputReference.valuesOutputName);
    EXPECT_NE(physicalValuesOutput, nullptr);
    if (physicalValuesOutput == nullptr) return {};
    physicalValuesOutput->getOutputReadyEvent().synchronize();
    return readCpuFp32(physicalValuesOutput->getFeatureOutput().value());
}

vector<uint32_t> readRaggedOutputOffsets(Api::PlacedNetwork& placed,
                                         const Api::RaggedNetworkOutputReference& outputReference) {
    Impl::StampedNetwork& stamped = placed.getStampedNetwork(0);
    auto physicalOffsetsOutput = stamped.getNamedOutput(outputReference.offsetsOutputName);
    EXPECT_NE(physicalOffsetsOutput, nullptr);
    if (physicalOffsetsOutput == nullptr) return {};
    physicalOffsetsOutput->getOutputReadyEvent().synchronize();
    return readCpuUint32(physicalOffsetsOutput->getFeatureOutput().value());
}

void expectActiveFinite(const vector<float>& values, uint64_t channels) {
    ASSERT_GE(values.size(), kActiveValues * channels);
    for (uint64_t i = 0; i < kActiveValues * channels; ++i) {
        EXPECT_TRUE(std::isfinite(values[i])) << "non-finite active value at index " << i;
    }
}

void expectActiveNear(const vector<float>& actual, const vector<float>& expected, uint64_t channels, float tolerance) {
    ASSERT_GE(actual.size(), kActiveValues * channels);
    ASSERT_GE(expected.size(), kActiveValues * channels);
    for (uint64_t i = 0; i < kActiveValues * channels; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "active value mismatch at index " << i;
    }
}

float sumAbsActive(const vector<float>& values, uint64_t channels) {
    if (values.size() < kActiveValues * channels) {
        ADD_FAILURE() << "active-gradient buffer is smaller than the logical ragged extent";
        return 0.0f;
    }
    float sum = 0.0f;
    for (uint64_t i = 0; i < kActiveValues * channels; ++i) {
        EXPECT_TRUE(std::isfinite(values[i])) << "non-finite active gradient at index " << i;
        sum += std::abs(values[i]);
    }
    return sum;
}

float maxAbsDifference(const vector<float>& a, const vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    const size_t count = std::min(a.size(), b.size());
    float difference = 0.0f;
    for (size_t i = 0; i < count; ++i) difference = std::max(difference, std::abs(a[i] - b[i]));
    return difference;
}

std::filesystem::path makeUniqueArchiveDir(const string& stem) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

vector<shared_ptr<Impl::CustomLayer>> getTrainablePhysicalLayers(Impl::StampedNetwork& stamped,
                                                                  const F2ChainHandles& handles) {
    const vector<uint64_t> ids = {handles.firstFullyConnectedId,
                                  handles.rmsNormId,
                                  handles.temporalProjectionId,
                                  handles.convDilation1Id,
                                  handles.convDilation7Id,
                                  handles.convDilation28Id,
                                  handles.outputProjectionId};
    vector<shared_ptr<Impl::CustomLayer>> layers;
    layers.reserve(ids.size());
    for (uint64_t id : ids) {
        auto layer = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(id));
        if (layer == nullptr) throw std::runtime_error("F2 integration gate expected a ragged CustomLayer-backed trainable stage.");
        layers.push_back(layer);
    }
    return layers;
}

void initializeDeterministicPositiveParameters(Impl::StampedNetwork& stamped, const F2ChainHandles& handles) {
    auto firstFc = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.firstFullyConnectedId));
    auto rms = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.rmsNormId));
    auto temporalFc = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.temporalProjectionId));
    auto conv1 = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation1Id));
    auto conv7 = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation7Id));
    auto conv28 = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation28Id));
    auto outputFc = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.outputProjectionId));
    ASSERT_NE(firstFc, nullptr);
    ASSERT_NE(rms, nullptr);
    ASSERT_NE(temporalFc, nullptr);
    ASSERT_NE(conv1, nullptr);
    ASSERT_NE(conv7, nullptr);
    ASSERT_NE(conv28, nullptr);
    ASSERT_NE(outputFc, nullptr);

    setLayerParameters(firstFc, 0.04f, 0.10f);
    setLayerParameters(rms, 1.0f, 1.0f);
    setLayerParameters(temporalFc, 0.04f, 0.10f);
    setLayerParameters(conv1, 0.03f, 0.08f);
    setLayerParameters(conv7, 0.03f, 0.08f);
    setLayerParameters(conv28, 0.03f, 0.08f);
    setLayerParameters(outputFc, 0.04f, 0.10f);
}

void assertPhysicalRaggedLayerFamilies(Impl::StampedNetwork& stamped, const F2ChainHandles& handles) {
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                  stamped.getPhysicalLayerFromApiLayer(handles.firstFullyConnectedId)),
              nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedRMSNorm>(stamped.getPhysicalLayerFromApiLayer(handles.rmsNormId)),
              nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                  stamped.getPhysicalLayerFromApiLayer(handles.temporalProjectionId)),
              nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
                  stamped.getPhysicalLayerFromApiLayer(handles.convDilation1Id)),
              nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
                  stamped.getPhysicalLayerFromApiLayer(handles.convDilation7Id)),
              nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
                  stamped.getPhysicalLayerFromApiLayer(handles.convDilation28Id)),
              nullptr);

    // Standalone activation builders return the user-facing activation object, but
    // Network::addToNetwork() stores a clone with a fresh API-layer id. Do not use
    // the builder-side ReLU id (or protected tensor-driver maps) to find the stamped
    // layer. The THOR_GTEST accessor is the supported inspection surface used by the
    // activation tests for this exact clone-id behavior. In this F2 chain the three
    // convolutions plus the two standalone ReLUs are all RaggedCustomLayer-backed.
    size_t raggedCustomLayerCount = 0;
    for (const auto& [apiLayerId, physicalLayer] : stamped.getApiLayerToPhysicalLayer()) {
        (void)apiLayerId;
        if (std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(physicalLayer) != nullptr)
            ++raggedCustomLayerCount;
    }
    EXPECT_GE(raggedCustomLayerCount, 5u);
    EXPECT_NE(std::dynamic_pointer_cast<Impl::RaggedFullyConnected>(
                  stamped.getPhysicalLayerFromApiLayer(handles.outputProjectionId)),
              nullptr);
}

struct ConvParameterSnapshot {
    vector<float> weights;
    vector<float> biases;
};

ConvParameterSnapshot snapshotConvParameters(const shared_ptr<Impl::RaggedCustomLayer>& conv) {
    if (conv == nullptr || !conv->getGradientUpdateStream().has_value())
        throw std::runtime_error("F2 integration gate requires a placed trainable ragged convolution.");
    Stream gradientStream = conv->getGradientUpdateStream().value();
    ConvParameterSnapshot snapshot;
    snapshot.weights = readCpuFp32(copyToCpu(conv->getParameter("weights")->getStorage().value(), gradientStream));
    snapshot.biases = readCpuFp32(copyToCpu(conv->getParameter("biases")->getStorage().value(), gradientStream));
    return snapshot;
}

}  // namespace

TEST(RaggedConvolution1dPublicIntegration, F2StyleChainPreservesCanonicalPartitionWithoutDenseFallback) {
    Api::Network network("ragged_conv1d_f2_public_surface");
    F2ChainHandles handles = buildF2PublicChain(network, false, true);

    ASSERT_EQ(handles.stageOutputs.size(), handles.stageChannels.size());
    for (size_t i = 0; i < handles.stageOutputs.size(); ++i) {
        expectSamePartitionContract(handles.stageOutputs[i], handles.networkInput, handles.stageChannels[i]);
    }

    const nlohmann::json architecture = network.architectureJson();
    uint32_t raggedConvCount = 0;
    vector<uint64_t> dilations;
    for (const nlohmann::json& layer : architecture.at("layers")) {
        if (layer.at("layer_type").get<string>() != "convolution_1d") continue;
        ++raggedConvCount;
        ASSERT_TRUE(layer.at("use_ragged").get<bool>());
        EXPECT_EQ(layer.at("padding_mode").get<string>(), "causal");
        EXPECT_EQ(layer.at("stride").get<uint64_t>(), 1u);
        EXPECT_TRUE(layer.contains("ragged_input"));
        EXPECT_TRUE(layer.contains("ragged_output"));
        EXPECT_EQ(layer.at("ragged_input").at("offsets").at("id").get<uint64_t>(),
                  layer.at("ragged_output").at("offsets").at("id").get<uint64_t>());
        EXPECT_EQ(layer.at("ragged_input").at("max_values_per_row").get<uint64_t>(), kMaxValuesPerRow);
        EXPECT_EQ(layer.at("ragged_output").at("max_values_per_row").get<uint64_t>(), kMaxValuesPerRow);
        dilations.push_back(layer.at("dilation").get<uint64_t>());
    }
    EXPECT_EQ(raggedConvCount, 3u);
    EXPECT_EQ(dilations, (vector<uint64_t>{1, 7, 28}));

    const auto externalInputs = network.getExternalRaggedNetworkInputs();
    const auto externalOutputs = network.getExternalRaggedNetworkOutputs();
    ASSERT_EQ(externalInputs.size(), 1u);
    ASSERT_EQ(externalOutputs.size(), 1u);
    expectSameExternalPartitionDescriptor(externalOutputs.front().raggedTensor, externalInputs.front().raggedTensor);
    EXPECT_EQ(externalOutputs.front().raggedTensor.getBatchSize(), kBatchSize);
    EXPECT_EQ(externalOutputs.front().raggedTensor.getMaxTotalValues(), kMaxTotalValues);
    ASSERT_TRUE(externalOutputs.front().raggedTensor.hasMaxValuesPerRow());
    EXPECT_EQ(externalOutputs.front().raggedTensor.getMaxValuesPerRow(), kMaxValuesPerRow);
    EXPECT_EQ(externalOutputs.front().raggedTensor.getValuesDimensions(),
              (vector<uint64_t>{kMaxTotalValues, kInputFeatures}));
}

TEST(RaggedConvolution1dPublicIntegration, F2StyleChainPlacesBackpropagatesAndRoundTripsForTraining) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device required for the ragged Convolution1d F2 public-layer production gate.";

    const vector<float> inputValues = makePoisonedInputValues();

    // Backward gate. GradientRivet is test-only instrumentation around the public
    // chain; every production stage between the rivets is a normal Thor layer.
    {
        Api::Network network("ragged_conv1d_f2_backward_gate");
        F2ChainHandles handles = buildF2PublicChain(network, true, false);
        (void)Api::Sgd::Builder()
            .network(network)
            .initialLearningRate(kLearningRate)
            .decay(0.0f)
            .momentum(0.0f)
            .build();

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        assertPhysicalRaggedLayerFamilies(stamped, handles);
        initializeDeterministicPositiveParameters(stamped, handles);

        auto physicalSink = std::dynamic_pointer_cast<Impl::NetworkOutput>(
            stamped.getPhysicalLayerFromApiLayer(handles.denseGradientSinkId.value()));
        ASSERT_NE(physicalSink, nullptr);
        feedRaggedInput(*placed, network.getExternalRaggedNetworkInputs().front(), inputValues);
        physicalSink->getOutputReadyEvent().synchronize();
        const vector<float> forward = readCpuFp32(physicalSink->getFeatureOutput().value());
        expectActiveFinite(forward, kInputFeatures);

        vector<shared_ptr<Impl::RaggedCustomLayer>> convs = {
            std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation1Id)),
            std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation7Id)),
            std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(handles.convDilation28Id)),
        };
        for (const auto& conv : convs) {
            ASSERT_NE(conv, nullptr);
            ASSERT_TRUE(conv->getGradientUpdateStream().has_value());
            ASSERT_FALSE(conv->getErrorOutputs().empty());
            ASSERT_TRUE(conv->getErrorOutputs()[0].has_value());
        }

        vector<ConvParameterSnapshot> before;
        before.reserve(convs.size());
        for (const auto& conv : convs) before.push_back(snapshotConvParameters(conv));

        auto terminalFc = std::dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(handles.outputProjectionId));
        ASSERT_NE(terminalFc, nullptr);
        ASSERT_FALSE(terminalFc->getErrorInputs().empty());
        ASSERT_TRUE(terminalFc->getErrorInputs()[0].has_value());
        Impl::Tensor errorInput = terminalFc->getErrorInputs()[0].value();
        Impl::Tensor errorHost = errorInput.clone(cpuPlacement);
        vector<float> upstream(kMaxTotalValues * kInputFeatures, std::numeric_limits<float>::quiet_NaN());
        std::fill(upstream.begin(), upstream.begin() + kActiveValues * kInputFeatures, 1.0f);
        writeCpuFp32(errorHost, upstream);
        Stream terminalStream = terminalFc->getStreams()[0];
        errorInput.copyFromAsync(errorHost, terminalStream);
        terminalFc->backward(errorInput, kBatchSize);

        for (const auto& layer : getTrainablePhysicalLayers(stamped, handles)) {
            for (Stream stream : layer->getStreams()) stream.synchronize();
            if (layer->getGradientUpdateStream().has_value()) layer->getGradientUpdateStream()->synchronize();
        }

        for (size_t i = 0; i < convs.size(); ++i) {
            Stream stream = convs[i]->getStreams()[0];
            const vector<float> dx = readCpuFp32(copyToCpu(convs[i]->getErrorOutputs()[0].value(), stream));
            EXPECT_GT(sumAbsActive(dx, kTemporalFeatures), 1.0e-6f) << "conv index " << i << " produced no dX";

            const ConvParameterSnapshot after = snapshotConvParameters(convs[i]);
            EXPECT_GT(maxAbsDifference(after.weights, before[i].weights), 1.0e-7f)
                << "conv index " << i << " produced no dW update";
            EXPECT_GT(maxAbsDifference(after.biases, before[i].biases), 1.0e-7f)
                << "conv index " << i << " produced no dbias update";
        }
    }

    // Clean public-layer save/load gate: no GradientRivet or private expression
    // plumbing is part of the serialized model.
    const string networkName = "ragged_conv1d_f2_save_load_gate";
    const std::filesystem::path archiveDir = makeUniqueArchiveDir(networkName);
    try {
        Api::Network network(networkName);
        F2ChainHandles handles = buildF2PublicChain(network, false, true);
        (void)Api::Sgd::Builder()
            .network(network)
            .initialLearningRate(kLearningRate)
            .decay(0.0f)
            .momentum(0.0f)
            .build();

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = network.place(kBatchSize, initDoneEvents, false, {0}, 1);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        assertPhysicalRaggedLayerFamilies(stamped, handles);
        initializeDeterministicPositiveParameters(stamped, handles);

        const auto sourceInputs = network.getExternalRaggedNetworkInputs();
        const auto sourceOutputs = network.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(sourceInputs.size(), 1u);
        ASSERT_EQ(sourceOutputs.size(), 1u);
        const vector<float> sourceForward = runRaggedForward(*placed, sourceInputs.front(), sourceOutputs.front(), inputValues);
        expectActiveFinite(sourceForward, kInputFeatures);
        EXPECT_EQ(readRaggedOutputOffsets(*placed, sourceOutputs.front()), kOffsets);

        placed->save(archiveDir.string(), true, false);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        const auto loadedInputs = loaded.getExternalRaggedNetworkInputs();
        const auto loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedInputs.size(), 1u);
        ASSERT_EQ(loadedOutputs.size(), 1u);
        expectSameExternalPartitionDescriptor(loadedOutputs.front().raggedTensor, loadedInputs.front().raggedTensor);
        EXPECT_EQ(loadedOutputs.front().raggedTensor.getBatchSize(), kBatchSize);
        EXPECT_EQ(loadedOutputs.front().raggedTensor.getMaxTotalValues(), kMaxTotalValues);
        ASSERT_TRUE(loadedOutputs.front().raggedTensor.hasMaxValuesPerRow());
        EXPECT_EQ(loadedOutputs.front().raggedTensor.getMaxValuesPerRow(), kMaxValuesPerRow);

        uint32_t loadedConvCount = 0;
        vector<uint64_t> loadedDilations;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            shared_ptr<Api::Convolution1d> conv = std::dynamic_pointer_cast<Api::Convolution1d>(loaded.getLayer(i));
            if (conv == nullptr) continue;
            ++loadedConvCount;
            ASSERT_TRUE(conv->getUseRagged());
            ASSERT_TRUE(conv->getRaggedFeatureInput().has_value());
            ASSERT_TRUE(conv->getRaggedFeatureOutput().has_value());
            EXPECT_EQ(conv->getRaggedFeatureInput()->getOffsets(), conv->getRaggedFeatureOutput()->getOffsets());
            ASSERT_TRUE(conv->getRaggedFeatureOutput()->hasMaxValuesPerRow());
            EXPECT_EQ(conv->getRaggedFeatureOutput()->getMaxValuesPerRow(), kMaxValuesPerRow);
            loadedDilations.push_back(conv->getDilation());
        }
        EXPECT_EQ(loadedConvCount, 3u);
        EXPECT_EQ(loadedDilations, (vector<uint64_t>{1, 7, 28}));

        vector<Event> loadedInitDoneEvents;
        shared_ptr<Api::PlacedNetwork> loadedPlaced = loaded.place(kBatchSize, loadedInitDoneEvents, false, {0}, 1);
        synchronizeEvents(loadedInitDoneEvents);
        ASSERT_NE(loadedPlaced, nullptr);
        const vector<float> loadedForward =
            runRaggedForward(*loadedPlaced, loadedInputs.front(), loadedOutputs.front(), inputValues);
        expectActiveFinite(loadedForward, kInputFeatures);
        EXPECT_EQ(readRaggedOutputOffsets(*loadedPlaced, loadedOutputs.front()), kOffsets);
        expectActiveNear(loadedForward, sourceForward, kInputFeatures, 5.0e-4f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
