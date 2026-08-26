#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = ThorImplementation::DataType;
using json = nlohmann::json;
using std::string;
using std::vector;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor &tensor) {
    uint64_t n = 1;
    for (uint64_t dim : tensor.getDimensions())
        n *= dim;
    return n;
}

void synchronizeEvents(vector<Event> &events) {
    for (Event &event : events)
        event.synchronize();
    events.clear();
}

void writeCpuFp32(Impl::Tensor &tensor, const vector<float> &values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    std::copy(values.begin(), values.end(), tensor.getMemPtr<float>());
}

vector<float> readCpuFp32(const Impl::Tensor &tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    const float *ptr = tensor.getMemPtr<float>();
    return vector<float>(ptr, ptr + tensorNumel(tensor));
}

Impl::Tensor copyToCpu(const Impl::Tensor &tensor, Stream &stream) {
    Impl::Tensor cpu = tensor.clone(cpuPlacement);
    cpu.copyFromAsync(tensor, stream);
    stream.putEvent().synchronize();
    return cpu;
}

void setParameterFp32(const std::shared_ptr<Impl::PhysicalParameter> &parameter,
                      const vector<float> &values,
                      Stream &stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor storage = parameter->getStorage().value();
    Impl::Tensor cpu = storage.clone(cpuPlacement);
    writeCpuFp32(cpu, values);
    storage.copyFromAsync(cpu, stream);
}


std::filesystem::path makeUniqueConvolution1dArchiveDir(const std::string &testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

template <typename LayerT>
std::shared_ptr<LayerT> findOnlyConvolution1dTestLayerOfType(Api::Network &network) {
    std::shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        std::shared_ptr<LayerT> candidate = std::dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
    return found;
}

vector<float> runRaggedConvolution1dNetworkForward(
    Api::PlacedNetwork &placed,
    const Api::RaggedNetworkInputReference &inputReference,
    const Api::RaggedNetworkOutputReference &outputReference,
    const vector<uint32_t> &offsets,
    const vector<float> &values,
    uint32_t batchSize,
    uint64_t maxTotalValues,
    uint64_t maxValuesPerRow,
    uint64_t activeValueCount,
    uint64_t activeMaxRowLength,
    uint64_t inputChannels) {
    Impl::StampedNetwork &stamped = placed.getStampedNetwork(0);
    auto physicalValuesInput = stamped.getNamedInput(inputReference.valuesInputName);
    auto physicalOffsetsInput = stamped.getNamedInput(inputReference.offsetsInputName);
    auto physicalValuesOutput = stamped.getNamedOutput(outputReference.valuesOutputName);
    EXPECT_NE(physicalValuesInput, nullptr);
    EXPECT_NE(physicalOffsetsInput, nullptr);
    EXPECT_NE(physicalValuesOutput, nullptr);
    if (physicalValuesInput == nullptr || physicalOffsetsInput == nullptr || physicalValuesOutput == nullptr)
        return {};

    Impl::Tensor offsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    auto *offsetsPtr = offsetsHost.getMemPtr<uint32_t>();
    for (uint32_t i = 0; i <= batchSize; ++i)
        offsetsPtr[i] = offsets[i];
    physicalOffsetsInput->forwardRowPartitionOffsets(offsetsHost,
                                                     false,
                                                     Impl::RowPartitionDescriptor(
                                                         batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow),
                                                     activeValueCount,
                                                     activeMaxRowLength,
                                                     batchSize);

    Impl::Tensor valuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, inputChannels}));
    writeCpuFp32(valuesHost, values);
    physicalValuesInput->forward(valuesHost, false, batchSize);
    physicalValuesOutput->getOutputReadyEvent().synchronize();
    return readCpuFp32(physicalValuesOutput->getFeatureOutput().value());
}

vector<float> causalConvReference(const vector<float> &x,
                                  const vector<uint64_t> &offsets,
                                  const vector<float> &w,
                                  uint64_t maxRows,
                                  uint64_t inputChannels,
                                  uint64_t outputChannels,
                                  uint64_t kernelWidth,
                                  uint64_t dilation,
                                  uint64_t groups) {
    const uint64_t inputChannelsPerGroup = inputChannels / groups;
    const uint64_t outputChannelsPerGroup = outputChannels / groups;
    vector<float> y(maxRows * outputChannels, 0.0f);
    for (uint64_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t timestep = 0; timestep < end - begin; ++timestep) {
            for (uint64_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
                float acc = 0.0f;
                const uint64_t group = outputChannel / outputChannelsPerGroup;
                const uint64_t inputChannelBegin = group * inputChannelsPerGroup;
                for (uint64_t filterPosition = 0; filterPosition < kernelWidth; ++filterPosition) {
                    const uint64_t lag = (kernelWidth - 1 - filterPosition) * dilation;
                    if (timestep < lag)
                        continue;
                    const uint64_t sourceValue = begin + timestep - lag;
                    for (uint64_t inputChannelInGroup = 0; inputChannelInGroup < inputChannelsPerGroup;
                         ++inputChannelInGroup) {
                        const uint64_t inputChannel = inputChannelBegin + inputChannelInGroup;
                        const size_t filterIndex =
                            (outputChannel * inputChannelsPerGroup + inputChannelInGroup) * kernelWidth + filterPosition;
                        acc += x[sourceValue * inputChannels + inputChannel] * w[filterIndex];
                    }
                }
                y[(begin + timestep) * outputChannels + outputChannel] = acc;
            }
        }
    }
    return y;
}

vector<float> causalConvDgradReference(const vector<float> &dy,
                                       const vector<uint64_t> &offsets,
                                       const vector<float> &w,
                                       uint64_t maxRows,
                                       uint64_t inputChannels,
                                       uint64_t outputChannels,
                                       uint64_t kernelWidth,
                                       uint64_t dilation,
                                       uint64_t groups) {
    const uint64_t inputChannelsPerGroup = inputChannels / groups;
    const uint64_t outputChannelsPerGroup = outputChannels / groups;
    vector<float> dx(maxRows * inputChannels, 0.0f);
    for (uint64_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t timestep = 0; timestep < end - begin; ++timestep) {
            for (uint64_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
                const float grad = dy[(begin + timestep) * outputChannels + outputChannel];
                const uint64_t group = outputChannel / outputChannelsPerGroup;
                const uint64_t inputChannelBegin = group * inputChannelsPerGroup;
                for (uint64_t filterPosition = 0; filterPosition < kernelWidth; ++filterPosition) {
                    const uint64_t lag = (kernelWidth - 1 - filterPosition) * dilation;
                    if (timestep < lag)
                        continue;
                    const uint64_t sourceValue = begin + timestep - lag;
                    for (uint64_t inputChannelInGroup = 0; inputChannelInGroup < inputChannelsPerGroup;
                         ++inputChannelInGroup) {
                        const uint64_t inputChannel = inputChannelBegin + inputChannelInGroup;
                        const size_t filterIndex =
                            (outputChannel * inputChannelsPerGroup + inputChannelInGroup) * kernelWidth + filterPosition;
                        dx[sourceValue * inputChannels + inputChannel] += grad * w[filterIndex];
                    }
                }
            }
        }
    }
    return dx;
}

vector<float> causalConvWgradReference(const vector<float> &x,
                                       const vector<float> &dy,
                                       const vector<uint64_t> &offsets,
                                       uint64_t inputChannels,
                                       uint64_t outputChannels,
                                       uint64_t kernelWidth,
                                       uint64_t dilation,
                                       uint64_t groups) {
    const uint64_t inputChannelsPerGroup = inputChannels / groups;
    const uint64_t outputChannelsPerGroup = outputChannels / groups;
    vector<float> dw(outputChannels * inputChannelsPerGroup * kernelWidth, 0.0f);
    for (uint64_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t timestep = 0; timestep < end - begin; ++timestep) {
            for (uint64_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
                const float grad = dy[(begin + timestep) * outputChannels + outputChannel];
                const uint64_t group = outputChannel / outputChannelsPerGroup;
                const uint64_t inputChannelBegin = group * inputChannelsPerGroup;
                for (uint64_t filterPosition = 0; filterPosition < kernelWidth; ++filterPosition) {
                    const uint64_t lag = (kernelWidth - 1 - filterPosition) * dilation;
                    if (timestep < lag)
                        continue;
                    const uint64_t sourceValue = begin + timestep - lag;
                    for (uint64_t inputChannelInGroup = 0; inputChannelInGroup < inputChannelsPerGroup;
                         ++inputChannelInGroup) {
                        const uint64_t inputChannel = inputChannelBegin + inputChannelInGroup;
                        const size_t filterIndex =
                            (outputChannel * inputChannelsPerGroup + inputChannelInGroup) * kernelWidth + filterPosition;
                        dw[filterIndex] += x[sourceValue * inputChannels + inputChannel] * grad;
                    }
                }
            }
        }
    }
    return dw;
}

void expectNearPrefix(const vector<float> &actual, const vector<float> &expected, uint64_t count, float tolerance) {
    ASSERT_GE(actual.size(), count);
    ASSERT_GE(expected.size(), count);
    for (uint64_t i = 0; i < count; ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "mismatch at index " << i;
}

}  // namespace

TEST(Convolution1dApi, DefaultsToValidPaddingAndOwnsRankThreeWeights) {
    Api::Network network("conv1dDefaults");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 16}).dataType(DataType::FP16).build();

    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(5)
                                  .filterWidth(3)
                                  .build();

    ASSERT_TRUE(conv.isInitialized());
    EXPECT_EQ(conv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{5, 14}));
    EXPECT_EQ(conv.getPaddingMode(), Api::Convolution1dPaddingMode::VALID);
    EXPECT_EQ(conv.getPaddingLeft(), 0u);
    EXPECT_EQ(conv.getPaddingRight(), 0u);
    EXPECT_EQ(conv.getComputeDataType(), DataType::FP32);

    const json arch = conv.architectureJson();
    EXPECT_EQ(arch.at("layer_type").get<string>(), "convolution_1d");
    EXPECT_EQ(arch.at("version").get<string>(), "1.0.0");
    EXPECT_EQ(arch.at("data_layout").get<string>(), "NCW");
    EXPECT_EQ(arch.at("padding_mode").get<string>(), "valid");
    EXPECT_EQ(arch.at("groups").get<uint32_t>(), 1u);
    EXPECT_EQ(arch.at("compute_data_type").get<DataType>(), DataType::FP32);
    EXPECT_EQ(arch.at("parameters").at("weights").at("shape"), json::array({5, 3, 3}));
}

TEST(Convolution1dApi, ExplicitTf32ComputeIsUserSelectableAndRequiresFp32Storage) {
    Api::Network network("conv1dTf32");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({4, 16}).dataType(DataType::FP32).build();

    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(8)
                                  .filterWidth(3)
                                  .computeDataType(DataType::TF32)
                                  .noActivation()
                                  .build();
    EXPECT_EQ(conv.getComputeDataType(), DataType::TF32);
    EXPECT_EQ(conv.architectureJson().at("compute_data_type").get<DataType>(), DataType::TF32);

    Api::Network fp16Network("conv1dTf32RejectFp16");
    Api::NetworkInput fp16Input =
        Api::NetworkInput::Builder().network(fp16Network).name("input").dimensions({4, 16}).dataType(DataType::FP16).build();
    EXPECT_THROW((void)Api::Convolution1d::Builder()
                     .network(fp16Network)
                     .featureInput(fp16Input.getFeatureOutput().value())
                     .numOutputChannels(8)
                     .filterWidth(3)
                     .computeDataType(DataType::TF32)
                     .noActivation()
                     .build(),
                 std::invalid_argument);
}

TEST(Convolution1dApi, SameUpperAndCausalResolveStrideAndDilationAwarePadding) {
    Api::Network sameNetwork("conv1dSame");
    Api::NetworkInput sameInput =
        Api::NetworkInput::Builder().network(sameNetwork).name("input").dimensions({2, 8}).dataType(DataType::FP16).build();
    Api::Convolution1d same = Api::Convolution1d::Builder()
                                  .network(sameNetwork)
                                  .featureInput(sameInput.getFeatureOutput().value())
                                  .numOutputChannels(4)
                                  .filterWidth(4)
                                  .stride(2)
                                  .dilation(2)
                                  .samePadding()
                                  .noActivation()
                                  .build();
    EXPECT_EQ(same.getPaddingMode(), Api::Convolution1dPaddingMode::SAME_UPPER);
    EXPECT_EQ(same.getPaddingLeft(), 2u);
    EXPECT_EQ(same.getPaddingRight(), 3u);
    EXPECT_EQ(same.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 4}));

    Api::Network causalNetwork("conv1dCausal");
    Api::NetworkInput causalInput =
        Api::NetworkInput::Builder().network(causalNetwork).name("input").dimensions({2, 8}).dataType(DataType::FP16).build();
    Api::Convolution1d causal = Api::Convolution1d::Builder()
                                    .network(causalNetwork)
                                    .featureInput(causalInput.getFeatureOutput().value())
                                    .numOutputChannels(4)
                                    .filterWidth(4)
                                    .stride(2)
                                    .dilation(2)
                                    .causalPadding()
                                    .noActivation()
                                    .build();
    EXPECT_EQ(causal.getPaddingMode(), Api::Convolution1dPaddingMode::CAUSAL);
    EXPECT_EQ(causal.getPaddingLeft(), 6u);
    EXPECT_EQ(causal.getPaddingRight(), 0u);
    EXPECT_EQ(causal.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 4}));
}

TEST(Convolution1dApi, ExplicitPaddingUsesIndependentLeftAndRight) {
    Api::Network network("conv1dExplicit");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({2, 11}).dataType(DataType::FP16).build();
    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(3)
                                  .filterWidth(3)
                                  .stride(2)
                                  .dilation(2)
                                  .padding(2, 1)
                                  .noActivation()
                                  .build();
    EXPECT_EQ(conv.getPaddingMode(), Api::Convolution1dPaddingMode::EXPLICIT);
    EXPECT_EQ(conv.getPaddingLeft(), 2u);
    EXPECT_EQ(conv.getPaddingRight(), 1u);
    EXPECT_EQ(conv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{3, 5}));
    const json arch = conv.architectureJson();
    EXPECT_EQ(arch.at("padding_mode").get<string>(), "explicit");
    EXPECT_EQ(arch.at("padding_left").get<uint32_t>(), 2u);
    EXPECT_EQ(arch.at("padding_right").get<uint32_t>(), 1u);
}


TEST(Convolution1dApi, GroupedAndDepthwiseWeightsUsePerGroupInputChannels) {
    Api::Network groupedNetwork("conv1dGrouped");
    Api::NetworkInput groupedInput = Api::NetworkInput::Builder()
                                         .network(groupedNetwork)
                                         .name("input")
                                         .dimensions({8, 16})
                                         .dataType(DataType::FP16)
                                         .build();
    Api::Convolution1d grouped = Api::Convolution1d::Builder()
                                     .network(groupedNetwork)
                                     .featureInput(groupedInput.getFeatureOutput().value())
                                     .numOutputChannels(12)
                                     .filterWidth(3)
                                     .groups(4)
                                     .noActivation()
                                     .build();
    EXPECT_EQ(grouped.getGroups(), 4u);
    const json groupedArch = grouped.architectureJson();
    EXPECT_EQ(groupedArch.at("groups").get<uint32_t>(), 4u);
    EXPECT_EQ(groupedArch.at("parameters").at("weights").at("shape"), json::array({12, 2, 3}));

    Api::Network depthwiseNetwork("conv1dDepthwise");
    Api::NetworkInput depthwiseInput = Api::NetworkInput::Builder()
                                           .network(depthwiseNetwork)
                                           .name("input")
                                           .dimensions({8, 16})
                                           .dataType(DataType::FP16)
                                           .build();
    Api::Convolution1d depthwise = Api::Convolution1d::Builder()
                                       .network(depthwiseNetwork)
                                       .featureInput(depthwiseInput.getFeatureOutput().value())
                                       .numOutputChannels(8)
                                       .filterWidth(5)
                                       .groups(8)
                                       .causalPadding()
                                       .noActivation()
                                       .build();
    EXPECT_EQ(depthwise.architectureJson().at("parameters").at("weights").at("shape"), json::array({8, 1, 5}));
}

TEST(Convolution1dApi, RejectsInvalidGroupDivisibility) {
    Api::Network network("conv1dBadGroups");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({6, 16}).dataType(DataType::FP16).build();
    EXPECT_THROW((void)Api::Convolution1d::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numOutputChannels(8)
                     .filterWidth(3)
                     .groups(4)
                     .build(),
                 std::invalid_argument);
}


TEST(Convolution1dApi, RaggedBuilderPreservesCanonicalPartitionAndChangesOnlyChannels) {
    Api::Network network("raggedConv1dBuilder");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({8})
                                  .maxTotalValues(66)
                                  .maxValuesPerRow(17)
                                  .batchSize(3)
                                  .build();

    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .numOutputChannels(12)
                                  .filterWidth(5)
                                  .groups(4)
                                  .dilation(7)
                                  .causalPadding()
                                  .noActivation()
                                  .build();

    ASSERT_TRUE(conv.getUseRagged());
    ASSERT_TRUE(conv.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(conv.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(conv.getRaggedFeatureInput()->getValues(), input.getValues());
    EXPECT_EQ(conv.getRaggedFeatureInput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(conv.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(conv.getRaggedFeatureOutput()->getBatchSize(), input.getBatchSize());
    EXPECT_EQ(conv.getRaggedFeatureOutput()->getMaxTotalValues(), input.getMaxTotalValues());
    ASSERT_TRUE(conv.getRaggedFeatureOutput()->hasMaxValuesPerRow());
    EXPECT_EQ(conv.getRaggedFeatureOutput()->getMaxValuesPerRow(), input.getMaxValuesPerRow());
    EXPECT_EQ(conv.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{66, 12}));
    EXPECT_EQ(conv.getDilation(), 7u);
    EXPECT_EQ(conv.getStride(), 1u);
    EXPECT_EQ(conv.getPaddingMode(), Api::Convolution1dPaddingMode::CAUSAL);
    EXPECT_EQ(conv.getPaddingLeft(), 28u);
    EXPECT_EQ(conv.getPaddingRight(), 0u);

    const json arch = conv.architectureJson();
    EXPECT_EQ(arch.at("version").get<string>(), "1.0.0");
    EXPECT_TRUE(arch.at("use_ragged").get<bool>());
    EXPECT_EQ(arch.at("parameters").at("weights").at("shape"), json::array({12, 2, 5}));
    EXPECT_EQ(arch.at("ragged_input").at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());
    EXPECT_EQ(arch.at("ragged_output").at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());
    EXPECT_EQ(arch.at("ragged_output").at("max_values_per_row").get<uint64_t>(), 17u);

    ASSERT_EQ(conv.getFeatureInputs().size(), 2u);
    EXPECT_EQ(conv.getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(conv.getFeatureInputs()[1], input.getOffsets());
    EXPECT_TRUE(conv.mustConnectAllInputsToDriveOutput());
    EXPECT_EQ(conv.getConnectionType(input.getValues()), 0);
    EXPECT_EQ(conv.getConnectionType(input.getOffsets()), 1);
    EXPECT_EQ(conv.getConnectionType(conv.getRaggedFeatureOutput()->getValues()), 0);

    conv.resetGraphTraversalState();
    conv.informThatInputConnectionMade(input.getValues());
    EXPECT_TRUE(conv.getOutputsFromInput(input.getValues()).empty());
    conv.informThatInputConnectionMade(input.getOffsets());
    const vector<Api::Tensor> ready = conv.getOutputsFromInput(input.getOffsets());
    ASSERT_EQ(ready.size(), 1u);
    EXPECT_EQ(ready[0], conv.getRaggedFeatureOutput()->getValues());
}

TEST(Convolution1dApi, RaggedBuilderRejectsUnsupportedPublicConfigurations) {
    auto makeInput = [](Api::Network& network, bool withMaxValuesPerRow = true) {
        Api::RaggedNetworkInput::Builder builder;
        builder.network(network)
            .name("tokens")
            .valuesDataType(DataType::FP32)
            .offsetsDataType(DataType::UINT32)
            .trailingDimensions({4})
            .maxTotalValues(32)
            .batchSize(2);
        if (withMaxValuesPerRow)
            builder.maxValuesPerRow(16);
        return builder.build();
    };

    {
        Api::Network network("ragged_conv1d_reject_stride");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .stride(2)
                         .causalPadding()
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_valid");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .validPadding()
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_same");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .samePadding()
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_explicit");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .padding(2, 0)
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_missing_capacity");
        Api::RaggedTensor input = makeInput(network, false);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .causalPadding()
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_groups");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(6)
                         .filterWidth(3)
                         .groups(3)
                         .causalPadding()
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_epilogue");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .causalPadding()
                         .epilogue(Api::Convolution1d::epilogueInput().relu())
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_non_ragged_activation");
        Api::RaggedTensor input = makeInput(network);
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .causalPadding()
                         .activation(Api::Softmax::Builder().build())
                         .build(),
                     std::invalid_argument);
    }
    {
        Api::Network network("ragged_conv1d_reject_fp64_storage");
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP64)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({4})
                                      .maxTotalValues(32)
                                      .maxValuesPerRow(16)
                                      .batchSize(2)
                                      .build();
        EXPECT_THROW((void)Api::Convolution1d::Builder()
                         .network(network)
                         .featureInput(input)
                         .numOutputChannels(4)
                         .filterWidth(3)
                         .causalPadding()
                         .build(),
                     std::invalid_argument);
    }
}

TEST(Convolution1dApi, RaggedGroupedAndDepthwiseWeightGeometryUsesTrailingChannels) {
    Api::Network groupedNetwork("raggedConv1dGrouped");
    Api::RaggedTensor groupedInput = Api::RaggedNetworkInput::Builder()
                                         .network(groupedNetwork)
                                         .name("tokens")
                                         .valuesDataType(DataType::FP16)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({8})
                                         .maxTotalValues(40)
                                         .maxValuesPerRow(20)
                                         .batchSize(2)
                                         .build();
    Api::Convolution1d grouped = Api::Convolution1d::Builder()
                                     .network(groupedNetwork)
                                     .featureInput(groupedInput)
                                     .numOutputChannels(12)
                                     .filterWidth(3)
                                     .groups(4)
                                     .causalPadding()
                                     .noActivation()
                                     .build();
    EXPECT_EQ(grouped.architectureJson().at("parameters").at("weights").at("shape"), json::array({12, 2, 3}));
    ASSERT_TRUE(grouped.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(grouped.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{40, 12}));

    Api::Network depthwiseNetwork("raggedConv1dDepthwise");
    Api::RaggedTensor depthwiseInput = Api::RaggedNetworkInput::Builder()
                                           .network(depthwiseNetwork)
                                           .name("tokens")
                                           .valuesDataType(DataType::BF16)
                                           .offsetsDataType(DataType::UINT32)
                                           .trailingDimensions({8})
                                           .maxTotalValues(40)
                                           .maxValuesPerRow(20)
                                           .batchSize(2)
                                           .build();
    Api::Convolution1d depthwise = Api::Convolution1d::Builder()
                                       .network(depthwiseNetwork)
                                       .featureInput(depthwiseInput)
                                       .numOutputChannels(8)
                                       .filterWidth(5)
                                       .groups(8)
                                       .dilation(3)
                                       .causalPadding()
                                       .noActivation()
                                       .build();
    EXPECT_EQ(depthwise.architectureJson().at("parameters").at("weights").at("shape"), json::array({8, 1, 5}));
    ASSERT_TRUE(depthwise.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(depthwise.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{40, 8}));
    EXPECT_EQ(depthwise.getRaggedFeatureOutput()->getOffsets(), depthwiseInput.getOffsets());
    EXPECT_EQ(depthwise.getRaggedFeatureOutput()->getMaxValuesPerRow(), 20u);
}


TEST(Convolution1dApi, RaggedPublicLayerLowersToQualifiedCausalBackendForwardAndBackward) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device required for ragged Convolution1d integration test.";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 12;
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint64_t activeValues = 9;
    constexpr uint64_t inputChannels = 4;
    constexpr uint64_t outputChannels = 4;
    constexpr uint64_t kernelWidth = 3;
    constexpr uint64_t dilation = 2;
    constexpr uint64_t groups = 2;
    constexpr float learningRate = 0.001f;
    const vector<uint64_t> offsets = {0, 4, 7, 9};

    Api::Network network("ragged_public_conv1d_forward_backward");
    Api::RaggedTensor networkInput = Api::RaggedNetworkInput::Builder()
                                         .network(network)
                                         .name("tokens")
                                         .valuesDataType(DataType::FP32)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({inputChannels})
                                         .maxTotalValues(maxTotalValues)
                                         .maxValuesPerRow(maxValuesPerRow)
                                         .batchSize(batchSize)
                                         .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(networkInput.getValues()).build();
    Api::RaggedTensor raggedInput(inputRivet.getFeatureOutput().value(), networkInput.getOffsets(), maxValuesPerRow);

    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(raggedInput)
                                  .numOutputChannels(outputChannels)
                                  .filterWidth(kernelWidth)
                                  .groups(groups)
                                  .dilation(dilation)
                                  .causalPadding()
                                  .hasBias(true)
                                  .activation(Api::Relu::Builder().build())
                                  .build();
    ASSERT_TRUE(conv.getRaggedFeatureOutput().has_value());
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder()
                                         .network(network)
                                         .tensor(conv.getRaggedFeatureOutput()->getValues())
                                         .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .network(network)
                                        .initialLearningRate(learningRate)
                                        .decay(0.0f)
                                        .momentum(0.0f)
                                        .build();
    (void)sgd;

    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork &stamped = placed->getStampedNetwork(0);
    auto physicalValuesInput = stamped.getNamedInput("tokens.values");
    auto physicalOffsetsInput = stamped.getNamedInput("tokens.offsets");
    auto physicalConv = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(conv.getId()));
    auto physicalOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalValuesInput, nullptr);
    ASSERT_NE(physicalOffsetsInput, nullptr);
    ASSERT_NE(physicalConv, nullptr);
    ASSERT_NE(physicalOutput, nullptr);

    vector<float> weights(outputChannels * (inputChannels / groups) * kernelWidth);
    for (uint64_t i = 0; i < weights.size(); ++i)
        weights[i] = 0.04f * static_cast<float>(static_cast<int>(i % 9) - 4);
    const vector<float> biases = {0.15f, -0.10f, 0.05f, 0.20f};
    vector<float> inputValues(maxTotalValues * inputChannels, 0.0f);
    for (uint64_t row = 0; row < activeValues; ++row) {
        for (uint64_t channel = 0; channel < inputChannels; ++channel) {
            inputValues[row * inputChannels + channel] =
                0.08f * static_cast<float>(static_cast<int>((row * 3 + channel) % 11) - 5);
        }
    }

    Stream stream = physicalConv->getStreams()[0];
    ASSERT_TRUE(physicalConv->getGradientUpdateStream().has_value());
    Stream gradientStream = physicalConv->getGradientUpdateStream().value();
    setParameterFp32(physicalConv->getParameter("weights"), weights, stream);
    setParameterFp32(physicalConv->getParameter("biases"), biases, stream);
    stream.synchronize();

    Impl::Tensor offsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    auto *offsetsPtr = offsetsHost.getMemPtr<uint32_t>();
    for (uint32_t i = 0; i <= batchSize; ++i)
        offsetsPtr[i] = static_cast<uint32_t>(offsets[i]);
    physicalOffsetsInput->forwardRowPartitionOffsets(offsetsHost,
                                                     false,
                                                     Impl::RowPartitionDescriptor(
                                                         batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow),
                                                     activeValues,
                                                     maxValuesPerRow,
                                                     batchSize);
    Impl::Tensor valuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, inputChannels}));
    writeCpuFp32(valuesHost, inputValues);
    physicalValuesInput->forward(valuesHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();

    vector<float> preactivation = causalConvReference(inputValues,
                                                      offsets,
                                                      weights,
                                                      maxTotalValues,
                                                      inputChannels,
                                                      outputChannels,
                                                      kernelWidth,
                                                      dilation,
                                                      groups);
    vector<float> expectedForward = preactivation;
    for (uint64_t row = 0; row < activeValues; ++row) {
        for (uint64_t channel = 0; channel < outputChannels; ++channel) {
            const uint64_t index = row * outputChannels + channel;
            preactivation[index] += biases[channel];
            expectedForward[index] = std::max(preactivation[index], 0.0f);
        }
    }
    const vector<float> actualForward = readCpuFp32(physicalOutput->getFeatureOutput().value());
    expectNearPrefix(actualForward, expectedForward, activeValues * outputChannels, 4e-4f);

    ASSERT_FALSE(physicalConv->getErrorInputs().empty());
    ASSERT_TRUE(physicalConv->getErrorInputs()[0].has_value());
    ASSERT_FALSE(physicalConv->getErrorOutputs().empty());
    ASSERT_TRUE(physicalConv->getErrorOutputs()[0].has_value());
    vector<float> upstreamDY(maxTotalValues * outputChannels, 0.0f);
    vector<float> convDY(maxTotalValues * outputChannels, 0.0f);
    for (uint64_t row = 0; row < activeValues; ++row) {
        for (uint64_t channel = 0; channel < outputChannels; ++channel) {
            const uint64_t index = row * outputChannels + channel;
            upstreamDY[index] = 0.11f * static_cast<float>(static_cast<int>((row + 2 * channel) % 7) - 3);
            convDY[index] = preactivation[index] > 0.0f ? upstreamDY[index] : 0.0f;
        }
    }

    Impl::Tensor errorInput = physicalConv->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuFp32(errorInputHost, upstreamDY);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalConv->backward(errorInput, batchSize);
    gradientStream.synchronize();
    stream.synchronize();

    const vector<float> expectedDX = causalConvDgradReference(convDY,
                                                              offsets,
                                                              weights,
                                                              maxTotalValues,
                                                              inputChannels,
                                                              outputChannels,
                                                              kernelWidth,
                                                              dilation,
                                                              groups);
    const vector<float> expectedDW = causalConvWgradReference(inputValues,
                                                              convDY,
                                                              offsets,
                                                              inputChannels,
                                                              outputChannels,
                                                              kernelWidth,
                                                              dilation,
                                                              groups);
    vector<float> expectedDB(outputChannels, 0.0f);
    for (uint64_t row = 0; row < activeValues; ++row)
        for (uint64_t channel = 0; channel < outputChannels; ++channel)
            expectedDB[channel] += convDY[row * outputChannels + channel];

    const vector<float> actualDX =
        readCpuFp32(copyToCpu(physicalConv->getErrorOutputs()[0].value(), stream));
    expectNearPrefix(actualDX, expectedDX, activeValues * inputChannels, 6e-4f);

    const float step = learningRate / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> expectedUpdatedWeights = weights;
    vector<float> expectedUpdatedBiases = biases;
    for (uint64_t i = 0; i < expectedUpdatedWeights.size(); ++i)
        expectedUpdatedWeights[i] -= step * expectedDW[i];
    for (uint64_t i = 0; i < expectedUpdatedBiases.size(); ++i)
        expectedUpdatedBiases[i] -= step * expectedDB[i];

    const vector<float> actualUpdatedWeights =
        readCpuFp32(copyToCpu(physicalConv->getParameter("weights")->getStorage().value(), gradientStream));
    const vector<float> actualUpdatedBiases =
        readCpuFp32(copyToCpu(physicalConv->getParameter("biases")->getStorage().value(), gradientStream));
    expectNearPrefix(actualUpdatedWeights, expectedUpdatedWeights, expectedUpdatedWeights.size(), 8e-4f);
    expectNearPrefix(actualUpdatedBiases, expectedUpdatedBiases, expectedUpdatedBiases.size(), 8e-4f);
}



TEST(Convolution1dApi, RaggedSerializationRoundTripPreservesExactCapacityContract) {
    const std::string networkName = "ragged_conv1d_serialization_contract";
    std::filesystem::path archiveDir = makeUniqueConvolution1dArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({8})
                                      .maxTotalValues(96)
                                      .maxValuesPerRow(48)
                                      .batchSize(3)
                                      .build();
        Api::Convolution1d conv = Api::Convolution1d::Builder()
                                      .network(network)
                                      .featureInput(input)
                                      .numOutputChannels(12)
                                      .filterWidth(5)
                                      .groups(4)
                                      .dilation(7)
                                      .computeDataType(DataType::TF32)
                                      .causalPadding()
                                      .hasBias(true)
                                      .activation(Api::Relu::Builder().build())
                                      .build();
        ASSERT_TRUE(conv.getRaggedFeatureOutput().has_value());
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(conv.getRaggedFeatureOutput().value())
            .build();

        const json before = conv.architectureJson();
        EXPECT_EQ(before.at("version").get<string>(), "1.0.0");
        EXPECT_TRUE(before.at("use_ragged").get<bool>());
        EXPECT_EQ(before.at("compute_data_type").get<DataType>(), DataType::TF32);
        EXPECT_EQ(before.at("ragged_input").at("version").get<string>(), "1.1.0");
        EXPECT_EQ(before.at("ragged_output").at("version").get<string>(), "1.1.0");
        EXPECT_EQ(before.at("ragged_input").at("max_values_per_row").get<uint64_t>(), 48u);
        EXPECT_EQ(before.at("ragged_output").at("max_values_per_row").get<uint64_t>(), 48u);
        EXPECT_EQ(before.at("ragged_input").at("offsets").at("id").get<uint64_t>(),
                  before.at("ragged_output").at("offsets").at("id").get<uint64_t>());

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        std::shared_ptr<Api::Convolution1d> loadedConv =
            findOnlyConvolution1dTestLayerOfType<Api::Convolution1d>(loadedNetwork);
        ASSERT_NE(loadedConv, nullptr);
        ASSERT_TRUE(loadedConv->getUseRagged());
        EXPECT_EQ(loadedConv->getComputeDataType(), DataType::TF32);
        ASSERT_TRUE(loadedConv->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(loadedConv->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedConv->getFilterWidth(), 5u);
        EXPECT_EQ(loadedConv->getDilation(), 7u);
        EXPECT_EQ(loadedConv->getGroups(), 4u);
        EXPECT_EQ(loadedConv->getPaddingMode(), Api::Convolution1dPaddingMode::CAUSAL);
        EXPECT_EQ(loadedConv->getStride(), 1u);
        EXPECT_TRUE(loadedConv->getHasBias());
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getValuesDimensions(), (vector<uint64_t>{96, 8}));
        EXPECT_EQ(loadedConv->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{96, 12}));
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getOffsets(), loadedConv->getRaggedFeatureOutput()->getOffsets());
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getBatchSize(), 3u);
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getMaxTotalValues(), 96u);
        ASSERT_TRUE(loadedConv->getRaggedFeatureInput()->hasMaxValuesPerRow());
        ASSERT_TRUE(loadedConv->getRaggedFeatureOutput()->hasMaxValuesPerRow());
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getMaxValuesPerRow(), 48u);
        EXPECT_EQ(loadedConv->getRaggedFeatureOutput()->getMaxValuesPerRow(), 48u);

        const json after = loadedConv->architectureJson();
        EXPECT_EQ(after.at("version").get<string>(), "1.0.0");
        EXPECT_TRUE(after.at("use_ragged").get<bool>());
        EXPECT_EQ(after.at("ragged_input").at("max_values_per_row").get<uint64_t>(), 48u);
        EXPECT_EQ(after.at("ragged_output").at("max_values_per_row").get<uint64_t>(), 48u);
        EXPECT_EQ(after.at("ragged_input").at("offsets").at("id").get<uint64_t>(),
                  after.at("ragged_output").at("offsets").at("id").get<uint64_t>());
        ASSERT_FALSE(after.at("activation").is_null());
        EXPECT_EQ(after.at("activation").at("layer_type").get<string>(), "relu");
        EXPECT_EQ(after.at("parameters").at("weights").at("shape"), json::array({12, 2, 5}));
        EXPECT_TRUE(after.at("parameters").contains("biases"));
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}


TEST(Convolution1dApi, RaggedSerializationRejectsMissingCapacityMetadata) {
    Api::Network network("ragged_conv1d_reject_missing_serialized_capacity");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(32)
                                  .maxValuesPerRow(16)
                                  .batchSize(2)
                                  .build();
    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .numOutputChannels(6)
                                  .filterWidth(3)
                                  .groups(2)
                                  .causalPadding()
                                  .noActivation()
                                  .build();

    std::shared_ptr<thor_file::TarReader> archiveReader;
    json missingInputCapacity = conv.architectureJson();
    missingInputCapacity.at("ragged_input").erase("max_values_per_row");
    EXPECT_THROW(Api::Convolution1d::deserialize(archiveReader, missingInputCapacity, &network), std::runtime_error);

    json missingOutputCapacity = conv.architectureJson();
    missingOutputCapacity.at("ragged_output").erase("max_values_per_row");
    EXPECT_THROW(Api::Convolution1d::deserialize(archiveReader, missingOutputCapacity, &network), std::runtime_error);

    json oldNestedRaggedMetadata = conv.architectureJson();
    oldNestedRaggedMetadata.at("ragged_input")["version"] = "1.0.0";
    EXPECT_THROW(Api::Convolution1d::deserialize(archiveReader, oldNestedRaggedMetadata, &network), std::runtime_error);
}

TEST(Convolution1dApi, RaggedPlacedSaveLoadRestoresStateAndTrainingExecution) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device required for ragged Convolution1d save/load execution test.";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 16;
    constexpr uint64_t maxValuesPerRow = 6;
    constexpr uint64_t activeValues = 12;
    constexpr uint64_t activeMaxRowLength = 5;
    constexpr uint64_t inputChannels = 4;
    constexpr uint64_t outputChannels = 6;
    constexpr uint64_t kernelWidth = 3;
    constexpr uint64_t dilation = 2;
    constexpr uint64_t groups = 2;
    const vector<uint32_t> offsets = {0, 5, 9, 12};
    const std::string networkName = "ragged_conv1d_state_round_trip";
    std::filesystem::path archiveDir = makeUniqueConvolution1dArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({inputChannels})
                                      .maxTotalValues(maxTotalValues)
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .batchSize(batchSize)
                                      .build();
        Api::Convolution1d conv = Api::Convolution1d::Builder()
                                      .network(network)
                                      .featureInput(input)
                                      .numOutputChannels(outputChannels)
                                      .filterWidth(kernelWidth)
                                      .groups(groups)
                                      .dilation(dilation)
                                      .causalPadding()
                                      .hasBias(true)
                                      .activation(Api::Relu::Builder().build())
                                      .build();
        ASSERT_TRUE(conv.getRaggedFeatureOutput().has_value());
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(conv.getRaggedFeatureOutput().value())
            .build();
        (void)Api::Sgd::Builder()
            .network(network)
            .initialLearningRate(0.01f)
            .decay(0.0f)
            .momentum(0.0f)
            .build();

        vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, false, {0}, 1);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
        Impl::StampedNetwork &stamped = placed->getStampedNetwork(0);
        auto physicalConv = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(conv.getId()));
        ASSERT_NE(physicalConv, nullptr);
        ASSERT_TRUE(physicalConv->getGradientUpdateStream().has_value());

        vector<float> weights(outputChannels * (inputChannels / groups) * kernelWidth);
        for (uint64_t i = 0; i < weights.size(); ++i)
            weights[i] = 0.03f * static_cast<float>(static_cast<int>(i % 11) - 5);
        const vector<float> biases = {0.2f, -0.1f, 0.05f, 0.12f, -0.08f, 0.17f};
        vector<float> inputValues(maxTotalValues * inputChannels, 0.0f);
        for (uint64_t row = 0; row < activeValues; ++row) {
            for (uint64_t channel = 0; channel < inputChannels; ++channel) {
                inputValues[row * inputChannels + channel] =
                    0.07f * static_cast<float>(static_cast<int>((row * 5 + channel * 3) % 13) - 6);
            }
        }

        Stream stream = physicalConv->getStreams()[0];
        setParameterFp32(physicalConv->getParameter("weights"), weights, stream);
        setParameterFp32(physicalConv->getParameter("biases"), biases, stream);
        stream.synchronize();

        const auto sourceInputs = network.getExternalRaggedNetworkInputs();
        const auto sourceOutputs = network.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(sourceInputs.size(), 1u);
        ASSERT_EQ(sourceOutputs.size(), 1u);
        const vector<float> sourceForward = runRaggedConvolution1dNetworkForward(*placed,
                                                                                 sourceInputs.front(),
                                                                                 sourceOutputs.front(),
                                                                                 offsets,
                                                                                 inputValues,
                                                                                 batchSize,
                                                                                 maxTotalValues,
                                                                                 maxValuesPerRow,
                                                                                 activeValues,
                                                                                 activeMaxRowLength,
                                                                                 inputChannels);
        ASSERT_EQ(sourceForward.size(), maxTotalValues * outputChannels);

        placed->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        std::shared_ptr<Api::Convolution1d> loadedConv =
            findOnlyConvolution1dTestLayerOfType<Api::Convolution1d>(loadedNetwork);
        ASSERT_NE(loadedConv, nullptr);
        ASSERT_TRUE(loadedConv->getUseRagged());
        ASSERT_TRUE(loadedConv->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(loadedConv->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getOffsets(), loadedConv->getRaggedFeatureOutput()->getOffsets());
        EXPECT_EQ(loadedConv->getRaggedFeatureInput()->getMaxValuesPerRow(), maxValuesPerRow);
        EXPECT_EQ(loadedConv->getRaggedFeatureOutput()->getMaxValuesPerRow(), maxValuesPerRow);

        vector<Event> loadedInitDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> loadedPlaced = loadedNetwork.place(batchSize, loadedInitDoneEvents, false, {0}, 1);
        synchronizeEvents(loadedInitDoneEvents);
        ASSERT_NE(loadedPlaced, nullptr);
        Impl::StampedNetwork &loadedStamped = loadedPlaced->getStampedNetwork(0);
        auto loadedPhysicalConv =
            std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(loadedStamped.getPhysicalLayerFromApiLayer(loadedConv->getId()));
        ASSERT_NE(loadedPhysicalConv, nullptr);
        ASSERT_TRUE(loadedPhysicalConv->getGradientUpdateStream().has_value());
        Stream loadedStream = loadedPhysicalConv->getStreams()[0];
        const vector<float> loadedWeights =
            readCpuFp32(copyToCpu(loadedPhysicalConv->getParameter("weights")->getStorage().value(), loadedStream));
        const vector<float> loadedBiases =
            readCpuFp32(copyToCpu(loadedPhysicalConv->getParameter("biases")->getStorage().value(), loadedStream));
        expectNearPrefix(loadedWeights, weights, weights.size(), 0.0f);
        expectNearPrefix(loadedBiases, biases, biases.size(), 0.0f);

        const auto loadedInputs = loadedNetwork.getExternalRaggedNetworkInputs();
        const auto loadedOutputs = loadedNetwork.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedInputs.size(), 1u);
        ASSERT_EQ(loadedOutputs.size(), 1u);
        ASSERT_TRUE(loadedInputs.front().raggedTensor.hasMaxValuesPerRow());
        ASSERT_TRUE(loadedOutputs.front().raggedTensor.hasMaxValuesPerRow());
        EXPECT_EQ(loadedInputs.front().raggedTensor.getMaxValuesPerRow(), maxValuesPerRow);
        EXPECT_EQ(loadedOutputs.front().raggedTensor.getMaxValuesPerRow(), maxValuesPerRow);

        const vector<float> loadedForward = runRaggedConvolution1dNetworkForward(*loadedPlaced,
                                                                                 loadedInputs.front(),
                                                                                 loadedOutputs.front(),
                                                                                 offsets,
                                                                                 inputValues,
                                                                                 batchSize,
                                                                                 maxTotalValues,
                                                                                 maxValuesPerRow,
                                                                                 activeValues,
                                                                                 activeMaxRowLength,
                                                                                 inputChannels);
        ASSERT_EQ(loadedForward.size(), sourceForward.size());
        expectNearPrefix(loadedForward, sourceForward, activeValues * outputChannels, 5e-4f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(Convolution1dApi, SerializationContractRejectsPreviousConvolutionVersion) {
    Api::Network network("conv1d_no_legacy_serialization");
    json payload{{"version", "2.0.0"}};
    std::shared_ptr<thor_file::TarReader> archiveReader;
    EXPECT_THROW(Api::Convolution1d::deserialize(archiveReader, payload, &network), std::runtime_error);
}
