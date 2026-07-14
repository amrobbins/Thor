#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
using json = nlohmann::json;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions())
        numel *= d;
    return numel;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void writeCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case DataType::FP16: {
            auto* ptr = static_cast<half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2half(values[i]);
            break;
        }
        case DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2bfloat16(values[i]);
            break;
        }
        case DataType::FP32: {
            auto* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        default:
            FAIL() << "Unsupported tensor dtype in writeCpuTensor.";
    }
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case DataType::FP16: {
            const auto* ptr = static_cast<const half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __half2float(ptr[i]);
            break;
        }
        case DataType::BF16: {
            const auto* ptr = static_cast<const __nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __bfloat162float(ptr[i]);
            break;
        }
        case DataType::FP32: {
            const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported tensor dtype in readCpuTensor.";
            break;
    }
    return values;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

vector<uint8_t> readTensorBytes(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = copyTensorToCpu(tensor, stream);
    vector<uint8_t> bytes(cpuTensor.getArraySizeInBytes());
    std::memcpy(bytes.data(), cpuTensor.getMemPtr(), bytes.size());
    return bytes;
}

void expectAllClose(
    const vector<float>& actual, const vector<float>& expected, float atol = 6e-2f, float rtol = 6e-2f, const string& what = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void setParameterTensor(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor deviceTensor = parameter->getStorage().value();
    Impl::Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

uint32_t convOutputDim(uint32_t input, uint32_t stride, uint32_t filter, uint32_t padding) {
    return 1 + (((input + 2 * padding) - filter) / stride);
}

ConvolutionKernelRequirement makeConvolutionKernelRequirement(uint64_t batchSize,
                                                              uint32_t numInputChannels,
                                                              uint64_t inputH,
                                                              uint64_t inputW,
                                                              uint32_t numOutputChannels,
                                                              uint32_t filterH,
                                                              uint32_t filterW,
                                                              uint32_t strideH,
                                                              uint32_t strideW,
                                                              uint32_t padH,
                                                              uint32_t padW) {
    return ConvolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                        filterW,
                                        filterH,
                                        strideW,
                                        strideH,
                                        padW,
                                        padH,
                                        numInputChannels,
                                        numOutputChannels,
                                        batchSize,
                                        inputW,
                                        inputH);
}

Impl::Tensor makeCpuTensor(DataType dataType, const vector<uint64_t>& dims, const vector<float>& values) {
    Impl::Tensor tensor(cpuPlacement, Impl::TensorDescriptor(dataType, dims));
    writeCpuTensor(tensor, values);
    return tensor;
}

vector<float> conv2dForwardReference(const vector<float>& inputValues,
                                     const vector<float>& weightValues,
                                     const vector<float>& biasValues,
                                     uint64_t batchSize,
                                     uint32_t numInputChannels,
                                     uint64_t inputH,
                                     uint64_t inputW,
                                     uint32_t numOutputChannels,
                                     uint32_t filterH,
                                     uint32_t filterW,
                                     uint32_t strideH,
                                     uint32_t strideW,
                                     uint32_t padH,
                                     uint32_t padW,
                                     bool hasBias) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Impl::Tensor inputTensor = makeCpuTensor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}, inputValues);
    Impl::Tensor weightsTensor = makeCpuTensor(DataType::FP16, {numOutputChannels, numInputChannels, filterH, filterW}, weightValues);
    Impl::Tensor outputTensor(cpuPlacement,
                              Impl::TensorDescriptor(DataType::FP16,
                                                     {batchSize,
                                                      numOutputChannels,
                                                      static_cast<uint64_t>(requirement.getNumOutputRows()),
                                                      static_cast<uint64_t>(requirement.getNumOutputColumns())}));

    std::optional<Impl::Tensor> biasTensor = std::nullopt;
    if (hasBias)
        biasTensor = makeCpuTensor(DataType::FP16, {numOutputChannels}, biasValues);

    Impl::ConvolutionTestHelper::cpuConvolutionForward(inputTensor, weightsTensor, biasTensor, outputTensor, requirement);
    return readCpuTensor(outputTensor);
}

vector<float> conv2dErrorReference(const vector<float>& errorInputValues,
                                   const vector<float>& weightValues,
                                   uint64_t batchSize,
                                   uint32_t numInputChannels,
                                   uint64_t inputH,
                                   uint64_t inputW,
                                   uint32_t numOutputChannels,
                                   uint32_t filterH,
                                   uint32_t filterW,
                                   uint32_t strideH,
                                   uint32_t strideW,
                                   uint32_t padH,
                                   uint32_t padW) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Impl::Tensor errorInputTensor = makeCpuTensor(DataType::FP16,
                                                  {batchSize,
                                                   numOutputChannels,
                                                   static_cast<uint64_t>(requirement.getNumOutputRows()),
                                                   static_cast<uint64_t>(requirement.getNumOutputColumns())},
                                                  errorInputValues);
    Impl::Tensor weightsTensor = makeCpuTensor(DataType::FP16, {numOutputChannels, numInputChannels, filterH, filterW}, weightValues);
    Impl::Tensor errorOutputTensor(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}));

    Impl::ConvolutionTestHelper::cpuConvolutionBackwardData(errorInputTensor, weightsTensor, errorOutputTensor, requirement);
    return readCpuTensor(errorOutputTensor);
}

vector<float> conv2dWeightGradReference(const vector<float>& inputValues,
                                        const vector<float>& errorInputValues,
                                        uint64_t batchSize,
                                        uint32_t numInputChannels,
                                        uint64_t inputH,
                                        uint64_t inputW,
                                        uint32_t numOutputChannels,
                                        uint32_t filterH,
                                        uint32_t filterW,
                                        uint32_t strideH,
                                        uint32_t strideW,
                                        uint32_t padH,
                                        uint32_t padW) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Impl::Tensor inputTensor = makeCpuTensor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}, inputValues);
    Impl::Tensor errorInputTensor = makeCpuTensor(DataType::FP16,
                                                  {batchSize,
                                                   numOutputChannels,
                                                   static_cast<uint64_t>(requirement.getNumOutputRows()),
                                                   static_cast<uint64_t>(requirement.getNumOutputColumns())},
                                                  errorInputValues);
    Impl::Tensor weightsGradTensor(cpuPlacement,
                                   Impl::TensorDescriptor(DataType::FP32, {numOutputChannels, numInputChannels, filterH, filterW}));

    Impl::ConvolutionTestHelper::cpuConvolutionBackwardFilter(inputTensor, errorInputTensor, weightsGradTensor, requirement, false);
    return readCpuTensor(weightsGradTensor);
}

vector<float> conv2dBiasGradReference(
    const vector<float>& errorInputValues, uint64_t batchSize, uint32_t numOutputChannels, uint64_t outputH, uint64_t outputW) {
    Impl::Tensor errorInputTensor = makeCpuTensor(DataType::FP16, {batchSize, numOutputChannels, outputH, outputW}, errorInputValues);
    Impl::Tensor biasGradTensor(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {numOutputChannels}));
    Impl::ConvolutionTestHelper::cpuConvolutionBackwardBias(errorInputTensor, biasGradTensor, false);
    return readCpuTensor(biasGradTensor);
}

vector<float> sgdUpdatedReference(const vector<float>& initial, const vector<float>& rawGradient, uint64_t batchSize, float lr) {
    const float step = lr / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> updated(initial.size());
    for (uint64_t i = 0; i < initial.size(); ++i)
        updated[i] = initial[i] - step * rawGradient[i];
    return updated;
}

struct PlacedConvolution2dFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    Impl::StampedNetwork* stampedNetwork = nullptr;
    shared_ptr<Impl::NetworkInput> physicalInput;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
    shared_ptr<Impl::CustomLayer> physicalConvolution;
};

PlacedConvolution2dFixture placeSingleConvolution2dNetwork(Api::Network& network,
                                                           const Api::NetworkInput& apiInput,
                                                           const Api::NetworkOutput& apiOutput,
                                                           const Api::Convolution2d& apiConvolution,
                                                           uint32_t batchSize,
                                                           bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedConvolution2dFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);

    fixture.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalConvolution =
        dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiConvolution.getId()));

    EXPECT_NE(fixture.physicalInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalConvolution, nullptr);
    return fixture;
}

vector<float> runForward(Impl::NetworkInput& physicalInput,
                         Impl::NetworkOutput& physicalOutput,
                         Impl::Tensor& featureInHost,
                         uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

std::filesystem::path makeUniqueTestArchiveDir(const std::string& testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

template <typename LayerT>
std::shared_ptr<LayerT> findOnlyLayerOfType(Api::Network& network) {
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

}  // namespace


TEST(Convolution2dApi, Bf16PlacedFilterMutationChangesExecutionWithoutRebinding) {
    constexpr uint32_t batchSize = 1;
    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f};

    Api::Network network("conv2d_bf16_live_filter_identity");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({1, 2, 2})
                                  .dataType(DataType::BF16)
                                  .build();
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(1)
                                  .filterHeight(1)
                                  .filterWidth(1)
                                  .verticalPadding(0)
                                  .horizontalPadding(0)
                                  .hasBias(false)
                                  .noActivation()
                                  .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(conv.getFeatureOutput().value())
                                    .dataType(DataType::BF16)
                                    .build();

    PlacedConvolution2dFixture fixture = placeSingleConvolution2dNetwork(network, input, output, conv, batchSize, true);
    Stream stream = fixture.physicalConvolution->getStreams()[0];
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::BF16, {batchSize, 1, 2, 2}));
    writeCpuTensor(featureInHost, inputValues);

    setParameterTensor(fixture.physicalConvolution->getParameter("weights"), {2.0f}, stream);
    stream.synchronize();
    const vector<float> first = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    expectAllClose(first, {2.0f, -4.0f, 1.0f, 6.0f}, 2e-2f, 2e-2f, "first BF16 filter");

    setParameterTensor(fixture.physicalConvolution->getParameter("weights"), {-3.0f}, stream);
    stream.synchronize();
    const vector<float> second = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    expectAllClose(second, {-3.0f, 6.0f, -1.5f, -9.0f}, 2e-2f, 2e-2f, "mutated BF16 filter");
    EXPECT_NE(first, second);
}

TEST(Convolution2dApi, PlacedSaveLoadRoundTripRestoresBf16FilterBytesAndExecution) {
    constexpr uint32_t batchSize = 2;
    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 4.0f, 2.0f, -1.0f};
    const vector<float> weightValues = {1.75f};
    const vector<float> biasValues = {-0.5f};
    const string networkName = "conv2d_bf16_parameter_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("input")
                                      .dimensions({1, 2, 2})
                                      .dataType(DataType::BF16)
                                      .build();
        Api::Convolution2d conv = Api::Convolution2d::Builder()
                                      .network(network)
                                      .featureInput(input.getFeatureOutput().value())
                                      .numOutputChannels(1)
                                      .filterHeight(1)
                                      .filterWidth(1)
                                      .verticalPadding(0)
                                      .horizontalPadding(0)
                                      .hasBias(true)
                                      .noActivation()
                                      .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(conv.getFeatureOutput().value())
                                        .dataType(DataType::BF16)
                                        .build();

        PlacedConvolution2dFixture source = placeSingleConvolution2dNetwork(network, input, output, conv, batchSize, true);
        Stream sourceStream = source.physicalConvolution->getStreams()[0];
        auto sourceWeights = source.physicalConvolution->getParameter("weights");
        auto sourceBiases = source.physicalConvolution->getParameter("biases");
        setParameterTensor(sourceWeights, weightValues, sourceStream);
        setParameterTensor(sourceBiases, biasValues, sourceStream);
        sourceStream.synchronize();
        ASSERT_EQ(sourceWeights->getStorage()->getDataType(), DataType::BF16);
        ASSERT_EQ(sourceBiases->getStorage()->getDataType(), DataType::BF16);
        const vector<uint8_t> sourceWeightBytes = readTensorBytes(sourceWeights->getStorage().value(), sourceStream);
        const vector<uint8_t> sourceBiasBytes = readTensorBytes(sourceBiases->getStorage().value(), sourceStream);

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::BF16, {batchSize, 1, 2, 2}));
        writeCpuTensor(featureInHost, inputValues);
        const vector<float> sourceOutput = runForward(*source.physicalInput, *source.physicalOutput, featureInHost, batchSize);
        source.placedNetwork->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        auto loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        auto loadedConv = findOnlyLayerOfType<Api::Convolution2d>(loadedNetwork);
        auto loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedConv, nullptr);
        ASSERT_NE(loadedOutput, nullptr);
        const json loadedArchitecture = loadedConv->architectureJson();
        EXPECT_EQ(loadedArchitecture.at("parameters").at("weights").at("dtype").get<DataType>(), DataType::BF16);
        EXPECT_EQ(loadedArchitecture.at("parameters").at("biases").at("dtype").get<DataType>(), DataType::BF16);

        PlacedConvolution2dFixture loaded =
            placeSingleConvolution2dNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedConv, batchSize, true);
        Stream loadedStream = loaded.physicalConvolution->getStreams()[0];
        auto loadedWeights = loaded.physicalConvolution->getParameter("weights");
        auto loadedBiases = loaded.physicalConvolution->getParameter("biases");
        ASSERT_EQ(loadedWeights->getStorage()->getDataType(), DataType::BF16);
        ASSERT_EQ(loadedBiases->getStorage()->getDataType(), DataType::BF16);
        EXPECT_EQ(readTensorBytes(loadedWeights->getStorage().value(), loadedStream), sourceWeightBytes);
        EXPECT_EQ(readTensorBytes(loadedBiases->getStorage().value(), loadedStream), sourceBiasBytes);

        const vector<float> loadedOutputValues =
            runForward(*loaded.physicalInput, *loaded.physicalOutput, featureInHost, batchSize);
        expectAllClose(loadedOutputValues, sourceOutput, 2e-2f, 2e-2f, "loaded BF16 convolution2d");
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}

TEST(Convolution2dApi, DefaultsToSamePaddingGeluActivationAndCanDisableActivation) {
    Api::Network defaultNetwork("conv2dDefaults");
    Api::NetworkInput defaultInput =
        Api::NetworkInput::Builder().network(defaultNetwork).name("input").dimensions({3, 8, 10}).dataType(DataType::FP16).build();
    Api::Convolution2d defaultConv = Api::Convolution2d::Builder()
                                         .network(defaultNetwork)
                                         .featureInput(defaultInput.getFeatureOutput().value())
                                         .numOutputChannels(4)
                                         .filterHeight(3)
                                         .filterWidth(5)
                                         .build();

    ASSERT_TRUE(defaultConv.isInitialized());
    EXPECT_EQ(defaultConv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 8, 10}));
    const json defaultJson = defaultConv.architectureJson();
    ASSERT_TRUE(defaultJson.contains("activation"));
    ASSERT_FALSE(defaultJson.at("activation").is_null());
    EXPECT_EQ(defaultJson.at("activation").at("layer_type").get<string>(), "gelu");
    EXPECT_EQ(defaultJson.at("vertical_padding").get<uint32_t>(), 1u);
    EXPECT_EQ(defaultJson.at("horizontal_padding").get<uint32_t>(), 2u);

    Api::Network explicitNetwork("conv2dExplicit");
    Api::NetworkInput explicitInput =
        Api::NetworkInput::Builder().network(explicitNetwork).name("input").dimensions({3, 7, 8}).dataType(DataType::FP16).build();
    Api::Convolution2d explicitConv = Api::Convolution2d::Builder()
                                          .network(explicitNetwork)
                                          .featureInput(explicitInput.getFeatureOutput().value())
                                          .numOutputChannels(5)
                                          .filterHeight(3)
                                          .filterWidth(2)
                                          .verticalStride(2)
                                          .horizontalStride(2)
                                          .verticalPadding(1)
                                          .horizontalPadding(0)
                                          .hasBias(true)
                                          .noActivation()
                                          .build();

    EXPECT_EQ(explicitConv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{5, 4, 4}));
    const json explicitJson = explicitConv.architectureJson();
    ASSERT_TRUE(explicitJson.contains("activation"));
    EXPECT_TRUE(explicitJson.at("activation").is_null());
    EXPECT_TRUE(explicitJson.at("has_bias").get<bool>());
}

TEST(Convolution2dApi, MultiInputEpilogueSerializesAuxiliaryBindings) {
    Api::Network network("conv2dMultiInputEpilogue");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 8, 8}).dataType(DataType::FP16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({4, 8, 8}).dataType(DataType::FP16).build();

    Impl::Expression convOutput = Api::Convolution2d::epilogueInput(DataType::FP32, DataType::FP16);
    Impl::Expression residualInput = Api::Convolution2d::epilogueAuxInput("residual", DataType::FP32, DataType::FP16);
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(4)
                                  .filterHeight(3)
                                  .filterWidth(3)
                                  .verticalPadding(1)
                                  .horizontalPadding(1)
                                  .hasBias(false)
                                  .noActivation()
                                  .epilogueInput("residual", residual.getFeatureOutput().value())
                                  .epilogue(convOutput + residualInput)
                                  .build();

    ASSERT_TRUE(conv.isInitialized());
    EXPECT_EQ(conv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 8, 8}));

    const json j = conv.architectureJson();
    ASSERT_TRUE(j.contains("epilogue"));
    ASSERT_FALSE(j.at("epilogue").is_null());
    ASSERT_TRUE(j.at("epilogue").contains("expected_input_names"));
    const vector<string> expectedInputNames = j.at("epilogue").at("expected_input_names").get<vector<string>>();
    EXPECT_EQ(set<string>(expectedInputNames.begin(), expectedInputNames.end()),
              (set<string>{Api::Convolution2d::epilogueInputName(), "residual"}));
    ASSERT_TRUE(j.contains("epilogue_inputs"));
    ASSERT_EQ(j.at("epilogue_inputs").size(), 1u);
    EXPECT_EQ(j.at("epilogue_inputs").at(0).at("name").get<string>(), "residual");
    EXPECT_EQ(j.at("epilogue_inputs").at(0).at("tensor").at("id").get<uint64_t>(),
              residual.getFeatureOutput().value().getOriginalId());
    ASSERT_EQ(j.at("inputs").size(), 1u) << "primary convolution inputs stay in inputs; aux bindings serialize separately.";
}


TEST(Convolution2dApi, MultiInputEpilogueRunsForwardResidualAdd) {
    constexpr uint32_t batchSize = 1;
    const DataType dataType = DataType::FP16;

    Api::Network network("conv2dMultiInputEpilogueForward");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({1, 3, 3}).dataType(dataType).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({1, 3, 3}).dataType(dataType).build();

    Impl::Expression convOutput = Api::Convolution2d::epilogueInput(DataType::FP32, dataType);
    Impl::Expression residualInput = Api::Convolution2d::epilogueAuxInput("residual", DataType::FP32, dataType);
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(1)
                                  .filterHeight(1)
                                  .filterWidth(1)
                                  .verticalPadding(0)
                                  .horizontalPadding(0)
                                  .hasBias(false)
                                  .noActivation()
                                  .epilogueInput("residual", residual.getFeatureOutput().value())
                                  .epilogue(convOutput + residualInput)
                                  .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(conv.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    // This test only drives forward inference after manually setting the
    // convolution weights.  Placing for training requires every trainable
    // layer to have an optimizer, which is intentionally irrelevant here.
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    synchronizeEvents(initDoneEvents);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input.getId()));
    auto physicalResidual = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalConv = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(conv.getId()));
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_NE(physicalResidual, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalConv, nullptr);

    Stream stream = physicalConv->getStreams()[0];
    setParameterTensor(physicalConv->getParameter("weights"), {2.0f}, stream);
    stream.synchronize();

    Impl::Tensor inputHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, 1, 3, 3}));
    Impl::Tensor residualHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, 1, 3, 3}));
    const vector<float> inputValues = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f, 0.5f, 1.5f, 2.5f};
    const vector<float> residualValues = {0.25f, 0.50f, 0.75f, 1.0f, 1.25f, 1.50f, -0.25f, -0.50f, -0.75f};
    writeCpuTensor(inputHost, inputValues);
    writeCpuTensor(residualHost, residualValues);

    physicalInput->forward(inputHost, false, batchSize);
    physicalResidual->forward(residualHost, false, batchSize);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    vector<float> expected(inputValues.size());
    for (uint32_t i = 0; i < inputValues.size(); ++i) {
        expected[i] = inputValues[i] * 2.0f + residualValues[i];
    }
    expectAllClose(readCpuTensor(physicalOutput->getFeatureOutput().value()), expected, 7e-2f, 7e-2f, "conv2d residual epilogue");
}


TEST(Convolution2dApi, MultiInputEpilogueRunsForwardBackwardResidualAddAndUpdatesWeights) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t C = 1;
    constexpr uint32_t H = 3;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 1;
    constexpr uint32_t R = 1;
    constexpr uint32_t S = 1;
    constexpr uint32_t strideH = 1;
    constexpr uint32_t strideW = 1;
    constexpr uint32_t padH = 0;
    constexpr uint32_t padW = 0;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {
        1.0f, 2.0f, -1.0f,
        0.5f, -0.5f, 1.5f,
        2.5f, -2.0f, 0.25f,
        -1.5f, 0.75f, 1.25f,
        0.0f, -0.25f, 2.0f,
        1.0f, -1.0f, 0.5f,
    };
    const vector<float> residualValues = {
        0.25f, -0.50f, 0.75f,
        1.0f, -1.25f, 1.50f,
        -0.25f, 0.50f, -0.75f,
        1.25f, -1.50f, 0.25f,
        -0.50f, 0.75f, -1.0f,
        0.5f, -0.25f, 1.0f,
    };
    const vector<float> upstreamErrors = {
        0.5f, -1.0f, 1.5f,
        -0.25f, 0.75f, -1.25f,
        1.0f, 0.25f, -0.5f,
        -1.5f, 0.5f, 0.25f,
        1.25f, -0.75f, 1.0f,
        0.5f, 1.5f, -1.0f,
    };
    const vector<float> initialWeights = {2.0f};

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network("conv2dMultiInputEpilogueForwardBackward");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({C, H, W}).dataType(dataType).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({K, H, W}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::GradientRivet residualRivet =
        Api::GradientRivet::Builder().network(network).tensor(residual.getFeatureOutput().value()).build();

    Impl::Expression convOutput = Api::Convolution2d::epilogueInput(DataType::FP32, dataType);
    Impl::Expression residualInput = Api::Convolution2d::epilogueAuxInput("residual", DataType::FP32, dataType);
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(inputRivet.getFeatureOutput().value())
                                  .numOutputChannels(K)
                                  .filterHeight(R)
                                  .filterWidth(S)
                                  .verticalStride(strideH)
                                  .horizontalStride(strideW)
                                  .verticalPadding(padH)
                                  .horizontalPadding(padW)
                                  .hasBias(false)
                                  .weightsOptimizer(weightsSgd)
                                  .noActivation()
                                  .epilogueInput("residual", residualRivet.getFeatureOutput().value())
                                  .epilogue(convOutput + residualInput)
                                  .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(conv.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input.getId()));
    auto physicalResidual = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalConv = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(conv.getId()));
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_NE(physicalResidual, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalConv, nullptr);
    ASSERT_TRUE(physicalConv->getGradientUpdateStream().has_value());

    Stream stream = physicalConv->getStreams()[0];
    Stream gradientStream = physicalConv->getGradientUpdateStream().value();
    setParameterTensor(physicalConv->getParameter("weights"), initialWeights, stream);
    stream.synchronize();

    Impl::Tensor inputHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, C, H, W}));
    Impl::Tensor residualHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, K, H, W}));
    writeCpuTensor(inputHost, inputValues);
    writeCpuTensor(residualHost, residualValues);

    physicalInput->forward(inputHost, false, batchSize);
    physicalResidual->forward(residualHost, false, batchSize);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    vector<float> expectedForward(inputValues.size());
    for (uint64_t i = 0; i < inputValues.size(); ++i) {
        expectedForward[i] = inputValues[i] * initialWeights[0] + residualValues[i];
    }
    expectAllClose(readCpuTensor(physicalOutput->getFeatureOutput().value()), expectedForward, 7e-2f, 7e-2f,
                   "conv2d residual epilogue forward/backward feature out");

    ASSERT_EQ(physicalConv->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalConv->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalConv->getErrorOutputs().size(), 2u)
        << "Multi-input epilogue backward must produce gradients for the primary convolution input and auxiliary residual input.";
    ASSERT_TRUE(physicalConv->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalConv->getErrorOutputs()[1].has_value());

    Impl::Tensor errorInput = physicalConv->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstreamErrors);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalConv->backward(errorInput, batchSize);

    Impl::Tensor primaryErrorOutputHost = copyTensorToCpu(physicalConv->getErrorOutputs()[0].value(), stream);
    Impl::Tensor residualErrorOutputHost = copyTensorToCpu(physicalConv->getErrorOutputs()[1].value(), stream);
    Impl::Tensor weightsAfterHost = copyTensorToCpu(physicalConv->getParameter("weights")->getStorage().value(), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedPrimaryError =
        conv2dErrorReference(upstreamErrors, initialWeights, batchSize, C, H, W, K, R, S, strideH, strideW, padH, padW);
    const vector<float> expectedWeightsGrad =
        conv2dWeightGradReference(inputValues, upstreamErrors, batchSize, C, H, W, K, R, S, strideH, strideW, padH, padW);
    const vector<float> expectedWeightsAfter = sgdUpdatedReference(initialWeights, expectedWeightsGrad, batchSize, learningRate);

    expectAllClose(readCpuTensor(primaryErrorOutputHost), expectedPrimaryError, 8e-2f, 8e-2f,
                   "conv2d residual epilogue primary error out");
    expectAllClose(readCpuTensor(residualErrorOutputHost), upstreamErrors, 8e-2f, 8e-2f,
                   "conv2d residual epilogue auxiliary residual error out");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 8e-2f, 8e-2f,
                   "conv2d residual epilogue weights after");
}

TEST(Convolution2dApi, MultiInputEpilogueRejectsMissingOrInvalidAuxiliaryBindings) {
    Api::Network network("conv2dMultiInputEpilogueRejects");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 8, 8}).dataType(DataType::FP16).build();
    Api::NetworkInput wrongResidual =
        Api::NetworkInput::Builder().network(network).name("wrong_residual").dimensions({5, 8, 8}).dataType(DataType::FP16).build();

    Impl::Expression convOutput = Api::Convolution2d::epilogueInput(DataType::FP32, DataType::FP16);
    Impl::Expression residualInput = Api::Convolution2d::epilogueAuxInput("residual", DataType::FP32, DataType::FP16);

    EXPECT_THROW(static_cast<void>(Api::Convolution2d::epilogueAuxInput("__reserved")), invalid_argument);
    EXPECT_THROW(static_cast<void>(Api::Convolution2d::epilogueAuxInput("weights")), invalid_argument);

    EXPECT_THROW(Api::Convolution2d::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numOutputChannels(4)
                     .filterHeight(3)
                     .filterWidth(3)
                     .verticalPadding(1)
                     .horizontalPadding(1)
                     .hasBias(false)
                     .noActivation()
                     .epilogue(convOutput + residualInput)
                     .build(),
                 invalid_argument);

    EXPECT_THROW(Api::Convolution2d::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numOutputChannels(4)
                     .filterHeight(3)
                     .filterWidth(3)
                     .verticalPadding(1)
                     .horizontalPadding(1)
                     .hasBias(false)
                     .noActivation()
                     .epilogueInput("residual", wrongResidual.getFeatureOutput().value())
                     .epilogue(convOutput + residualInput)
                     .build(),
                 exception);
}

TEST(Convolution2dApi, StampsAsPhysicalCustomLayerAllocatesParametersAndSerializesOptimizers) {
    constexpr uint32_t batchSize = 2;
    Api::Network network("conv2dStamp");

    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 6, 6}).dataType(DataType::FP16).build();
    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
    shared_ptr<Api::Sgd> biasesSgd = Api::Sgd::Builder().initialLearningRate(0.02f).decay(0.0f).momentum(0.0f).build();
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(2)
                                  .filterHeight(3)
                                  .filterWidth(3)
                                  .verticalPadding(0)
                                  .horizontalPadding(0)
                                  .hasBias(true)
                                  .weightsOptimizer(weightsSgd)
                                  .biasesOptimizer(biasesSgd)
                                  .noActivation()
                                  .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(conv.getFeatureOutput().value())
                                    .dataType(DataType::FP16)
                                    .build();

    const json j = conv.architectureJson();
    ASSERT_TRUE(j.contains("parameters"));
    ASSERT_TRUE(j.at("parameters").contains("weights"));
    ASSERT_TRUE(j.at("parameters").contains("biases"));
    ASSERT_TRUE(j.at("parameters").at("weights").contains("optimizer_override"));
    ASSERT_TRUE(j.at("parameters").at("biases").contains("optimizer_override"));

    PlacedConvolution2dFixture fixture = placeSingleConvolution2dNetwork(network, input, output, conv, batchSize, false);
    ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
    EXPECT_EQ(fixture.physicalConvolution->getLayerType(), "CustomLayer<Convolution2d>");
    EXPECT_EQ(fixture.physicalConvolution->listParameters(), (vector<string>{"weights", "biases"}));

    Impl::Tensor weights = fixture.physicalConvolution->getParameter("weights")->getStorage().value();
    Impl::Tensor biases = fixture.physicalConvolution->getParameter("biases")->getStorage().value();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{2, 3, 3, 3}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{2}));
    EXPECT_EQ(weights.getDataType(), DataType::FP16);
    EXPECT_EQ(biases.getDataType(), DataType::FP16);
    EXPECT_TRUE(fixture.physicalConvolution->getParameter("weights")->hasOptimizer());
    EXPECT_TRUE(fixture.physicalConvolution->getParameter("biases")->hasOptimizer());
}

TEST(Convolution2dApi, ThreePassForwardBackwardWithSgdUpdatesWeightsAndBiases) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t C = 1;
    constexpr uint32_t H = 3;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 1;
    constexpr uint32_t R = 2;
    constexpr uint32_t S = 2;
    constexpr uint32_t strideH = 1;
    constexpr uint32_t strideW = 1;
    constexpr uint32_t padH = 0;
    constexpr uint32_t padW = 0;
    constexpr uint32_t OH = 2;
    constexpr uint32_t OW = 2;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP16;

    vector<float> currentWeights = {0.25f, -0.50f, 0.75f, 0.50f};
    vector<float> currentBiases = {0.125f};

    const vector<vector<float>> inputsByPass = {
        {1.0f, 0.5f, -0.5f, 0.25f, -1.0f, 0.75f, 1.5f, -0.25f, 0.0f, -0.5f, 1.0f, 0.25f, -1.25f, 0.5f, 1.25f, 0.75f, -0.75f, 1.5f},
        {0.5f, -1.5f, 1.0f, 1.25f, 0.0f, -0.25f, -1.0f, 0.75f, 0.5f, 1.5f, -0.5f, 0.25f, 0.0f, 1.0f, -1.25f, 0.5f, 0.25f, -0.75f},
        {-1.0f, 0.25f, 1.25f, 0.75f, -0.5f, 0.5f, 1.0f, -1.5f, 0.0f, 0.25f, 1.5f, -0.75f, -0.5f, 0.75f, 1.0f, -1.25f, 0.5f, 0.0f},
    };
    const vector<vector<float>> errorsByPass = {
        {0.5f, -1.0f, 1.5f, -0.25f, 0.75f, -1.25f, 1.0f, 0.25f},
        {-1.5f, 0.5f, 0.25f, 1.25f, -0.75f, 1.0f, 0.5f, 1.5f},
        {0.75f, 1.25f, -0.25f, -1.0f, 0.5f, 1.75f, 1.5f, -1.25f},
    };

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
    shared_ptr<Api::Sgd> biasesSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network("conv2dThreePasses");
    Api::NetworkInput input = Api::NetworkInput::Builder().network(network).name("input").dimensions({C, H, W}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Convolution2d conv = Api::Convolution2d::Builder()
                                  .network(network)
                                  .featureInput(inputRivet.getFeatureOutput().value())
                                  .numOutputChannels(K)
                                  .filterHeight(R)
                                  .filterWidth(S)
                                  .verticalStride(strideH)
                                  .horizontalStride(strideW)
                                  .verticalPadding(padH)
                                  .horizontalPadding(padW)
                                  .hasBias(true)
                                  .weightsOptimizer(weightsSgd)
                                  .biasesOptimizer(biasesSgd)
                                  .noActivation()
                                  .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(conv.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    PlacedConvolution2dFixture fixture = placeSingleConvolution2dNetwork(network, input, output, conv, batchSize, false);
    ASSERT_TRUE(fixture.physicalConvolution->getGradientUpdateStream().has_value());
    Stream stream = fixture.physicalConvolution->getStreams()[0];
    Stream gradientStream = fixture.physicalConvolution->getGradientUpdateStream().value();

    setParameterTensor(fixture.physicalConvolution->getParameter("weights"), currentWeights, stream);
    setParameterTensor(fixture.physicalConvolution->getParameter("biases"), currentBiases, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, C, H, W}));

    for (uint32_t pass = 0; pass < inputsByPass.size(); ++pass) {
        writeCpuTensor(featureInHost, inputsByPass[pass]);
        const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        const vector<float> expectedForward = conv2dForwardReference(
            inputsByPass[pass], currentWeights, currentBiases, batchSize, C, H, W, K, R, S, strideH, strideW, padH, padW, true);
        expectAllClose(actualForward, expectedForward, 7e-2f, 7e-2f, "pass " + to_string(pass) + " feature out");

        ASSERT_GT(fixture.physicalConvolution->getErrorInputs().size(), 0u);
        ASSERT_TRUE(fixture.physicalConvolution->getErrorInputs()[0].has_value());
        ASSERT_GT(fixture.physicalConvolution->getErrorOutputs().size(), 0u);
        ASSERT_TRUE(fixture.physicalConvolution->getErrorOutputs()[0].has_value());

        Impl::Tensor errorInput = fixture.physicalConvolution->getErrorInputs()[0].value();
        Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
        writeCpuTensor(errorInputHost, errorsByPass[pass]);
        errorInput.copyFromAsync(errorInputHost, stream);
        fixture.physicalConvolution->backward(errorInput, batchSize);

        Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalConvolution->getErrorOutputs()[0].value(), stream);
        EXPECT_FALSE(fixture.physicalConvolution->getParameter("weights")->getOptimizer()->getWeightsGradient().has_value())
            << "Fused Convolution2d CustomLayer update should not allocate a dense weights gradient tensor.";
        EXPECT_FALSE(fixture.physicalConvolution->getParameter("biases")->getOptimizer()->getWeightsGradient().has_value())
            << "Fused Convolution2d CustomLayer update should not allocate a dense biases gradient tensor.";
        Impl::Tensor weightsAfterHost =
            copyTensorToCpu(fixture.physicalConvolution->getParameter("weights")->getStorage().value(), gradientStream);
        Impl::Tensor biasesAfterHost =
            copyTensorToCpu(fixture.physicalConvolution->getParameter("biases")->getStorage().value(), gradientStream);
        stream.synchronize();
        gradientStream.synchronize();

        const vector<float> expectedErrorOut =
            conv2dErrorReference(errorsByPass[pass], currentWeights, batchSize, C, H, W, K, R, S, strideH, strideW, padH, padW);
        const vector<float> expectedWeightsGrad =
            conv2dWeightGradReference(inputsByPass[pass], errorsByPass[pass], batchSize, C, H, W, K, R, S, strideH, strideW, padH, padW);
        const vector<float> expectedBiasesGrad = conv2dBiasGradReference(errorsByPass[pass], batchSize, K, OH, OW);
        currentWeights = sgdUpdatedReference(currentWeights, expectedWeightsGrad, batchSize, learningRate);
        currentBiases = sgdUpdatedReference(currentBiases, expectedBiasesGrad, batchSize, learningRate);

        expectAllClose(readCpuTensor(errorOutputHost), expectedErrorOut, 8e-2f, 8e-2f, "pass " + to_string(pass) + " error out");
        expectAllClose(readCpuTensor(weightsAfterHost), currentWeights, 8e-2f, 8e-2f, "pass " + to_string(pass) + " weights after");
        expectAllClose(readCpuTensor(biasesAfterHost), currentBiases, 8e-2f, 8e-2f, "pass " + to_string(pass) + " biases after");
    }
}

TEST(Convolution2dApi, ArchitectureSaveLoadRoundTripPreservesConfigurationAndDeserializedLayerRunsForward) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t C = 1;
    constexpr uint32_t H = 3;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 1;
    constexpr uint32_t R = 2;
    constexpr uint32_t S = 2;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {
        1.0f, 0.5f, -0.5f, 0.25f, -1.0f, 0.75f, 1.5f, -0.25f, 0.0f, -0.5f, 1.0f, 0.25f, -1.25f, 0.5f, 1.25f, 0.75f, -0.75f, 1.5f};
    const vector<float> weightValues = {0.25f, -0.50f, 0.75f, 0.50f};
    const vector<float> biasValues = {0.125f};

    const string networkName = "conv2d_arch_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({C, H, W}).dataType(dataType).build();
        Api::Convolution2d conv = Api::Convolution2d::Builder()
                                      .network(network)
                                      .featureInput(input.getFeatureOutput().value())
                                      .numOutputChannels(K)
                                      .filterHeight(R)
                                      .filterWidth(S)
                                      .verticalPadding(0)
                                      .horizontalPadding(0)
                                      .hasBias(true)
                                      .noActivation()
                                      .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(conv.getFeatureOutput().value())
                                        .dataType(dataType)
                                        .build();

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        shared_ptr<Api::Convolution2d> loadedConv = findOnlyLayerOfType<Api::Convolution2d>(loadedNetwork);
        shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedConv, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        const json j = loadedConv->architectureJson();
        EXPECT_EQ(j.at("layer_type").get<string>(), "convolution_2d");
        EXPECT_EQ(j.at("filter_height").get<uint32_t>(), R);
        EXPECT_EQ(j.at("filter_width").get<uint32_t>(), S);
        EXPECT_EQ(j.at("num_output_channels").get<uint32_t>(), K);
        EXPECT_TRUE(j.at("has_bias").get<bool>());
        EXPECT_TRUE(j.at("activation").is_null());

        PlacedConvolution2dFixture fixture =
            placeSingleConvolution2dNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedConv, batchSize, true);
        Stream stream = fixture.physicalConvolution->getStreams()[0];
        setParameterTensor(fixture.physicalConvolution->getParameter("weights"), weightValues, stream);
        setParameterTensor(fixture.physicalConvolution->getParameter("biases"), biasValues, stream);
        stream.synchronize();

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, C, H, W}));
        writeCpuTensor(featureInHost, inputValues);

        const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        const vector<float> expected =
            conv2dForwardReference(inputValues, weightValues, biasValues, batchSize, C, H, W, K, R, S, 1, 1, 0, 0, true);
        expectAllClose(actual, expected, 7e-2f, 7e-2f, "loaded conv2d output");
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}
