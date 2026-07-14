#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Adam.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include "cuda_bf16.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint32_t kBatchSize = 2;
constexpr uint32_t kNumInputFeatures = 2;
constexpr uint32_t kNumOutputFeatures = 3;
constexpr Impl::DataType kDataType = Impl::DataType::FP32;

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

void synchronizeEvents(std::vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
    events.clear();
}

void writeCpuTensor(Impl::Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case Impl::DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) {
                ptr[i] = __float2bfloat16(values[i]);
            }
            break;
        }
        case Impl::DataType::FP32: {
            float* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) {
                ptr[i] = values[i];
            }
            break;
        }
        default:
            FAIL() << "Unsupported tensor dtype in PlacedNetworkArtifactHandoffTest::writeCpuTensor.";
    }
}

std::vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    std::vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case Impl::DataType::BF16: {
            const auto* ptr = static_cast<const __nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) {
                values[i] = __bfloat162float(ptr[i]);
            }
            break;
        }
        case Impl::DataType::FP32: {
            const float* ptr = static_cast<const float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) {
                values[i] = ptr[i];
            }
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported tensor dtype in PlacedNetworkArtifactHandoffTest::readCpuTensor.";
    }
    return values;
}

void setDeviceTensor(Impl::Tensor destination, const std::vector<float>& values, Stream& stream) {
    Impl::Tensor cpuTensor = destination.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    destination.copyFromAsync(cpuTensor, stream);
}

std::vector<float> readDeviceTensor(Impl::Tensor source, Stream& stream) {
    Impl::Tensor cpuTensor = source.clone(cpuPlacement);
    cpuTensor.copyFromAsync(source, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return readCpuTensor(cpuTensor);
}

std::vector<uint8_t> readDeviceTensorBytes(Impl::Tensor source, Stream& stream) {
    Impl::Tensor cpuTensor = source.clone(cpuPlacement);
    cpuTensor.copyFromAsync(source, stream);
    Event copied = stream.putEvent();
    copied.synchronize();

    std::vector<uint8_t> bytes(cpuTensor.getArraySizeInBytes());
    std::memcpy(bytes.data(), cpuTensor.getMemPtr(), bytes.size());
    return bytes;
}

void expectAllClose(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    const std::string& context,
                    float absoluteTolerance = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size()) << context;
    for (uint64_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], absoluteTolerance) << context << " mismatch at index " << i;
    }
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

std::shared_ptr<Api::Network> buildFullyConnectedPhaseNetwork(const std::string& networkName,
                                                              bool attachAdamOptimizers = true,
                                                              Impl::DataType inputDataType = kDataType,
                                                              Impl::DataType weightsDataType = kDataType,
                                                              Impl::DataType computeDataType = kDataType) {
    auto network = std::make_shared<Api::Network>(networkName);
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*network)
                                  .name("features")
                                  .dimensions({kNumInputFeatures})
                                  .dataType(inputDataType)
                                  .build();

    Api::FullyConnected::Builder fcBuilder = Api::FullyConnected::Builder()
                                                 .network(*network)
                                                 .featureInput(input.getFeatureOutput().value())
                                                 .numOutputFeatures(kNumOutputFeatures)
                                                 .hasBias(true)
                                                 .weightsDataType(weightsDataType)
                                                 .computeDataType(computeDataType)
                                                 .outputDataType(kDataType)
                                                 .weightsInitializer(Api::UniformRandom::Builder().minValue(-0.01f).maxValue(0.01f).build())
                                                 .biasInitializer(Api::UniformRandom::Builder().minValue(-0.01f).maxValue(0.01f).build())
                                                 .noActivation();
    if (attachAdamOptimizers) {
        fcBuilder.weightsOptimizer(Api::Adam::Builder().alpha(0.002f).build());
        fcBuilder.biasesOptimizer(Api::Adam::Builder().alpha(0.003f).build());
    }

    Api::FullyConnected fc = fcBuilder.build();
    Api::NetworkOutput::Builder()
        .network(*network)
        .name("prediction")
        .inputTensor(fc.getFeatureOutput().value())
        .dataType(kDataType)
        .build();
    return network;
}

struct PlacedFcGraph {
    std::shared_ptr<Api::Network> network;
    std::shared_ptr<Api::PlacedNetwork> placed;
    std::shared_ptr<Api::FullyConnected> apiFc;
    std::shared_ptr<Impl::NetworkInput> physicalInput;
    std::shared_ptr<Impl::NetworkOutput> physicalOutput;
    std::shared_ptr<Impl::CustomLayer> physicalFc;
};

PlacedFcGraph placeNetworkWithSingleFc(std::shared_ptr<Api::Network> network) {
    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network->place(kBatchSize,
                                                                initDoneEvents,
                                                                /*inferenceOnly=*/false,
                                                                std::vector<int32_t>{0},
                                                                /*forcedNumStampsPerGpu=*/1);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(placed, nullptr);

    std::shared_ptr<Api::FullyConnected> apiFc = findOnlyLayerOfType<Api::FullyConnected>(*network);
    EXPECT_NE(apiFc, nullptr);
    std::shared_ptr<Api::NetworkInput> apiInput = findOnlyLayerOfType<Api::NetworkInput>(*network);
    std::shared_ptr<Api::NetworkOutput> apiOutput = findOnlyLayerOfType<Api::NetworkOutput>(*network);
    EXPECT_NE(apiInput, nullptr);
    EXPECT_NE(apiOutput, nullptr);
    std::shared_ptr<Impl::NetworkInput> physicalInput =
        std::dynamic_pointer_cast<Impl::NetworkInput>(placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiInput->getId()));
    std::shared_ptr<Impl::NetworkOutput> physicalOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiOutput->getId()));
    std::shared_ptr<Impl::CustomLayer> physicalFc =
        std::dynamic_pointer_cast<Impl::CustomLayer>(placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiFc->getId()));
    EXPECT_NE(physicalInput, nullptr);
    EXPECT_NE(physicalOutput, nullptr);
    EXPECT_NE(physicalFc, nullptr);
    return PlacedFcGraph{std::move(network),
                         std::move(placed),
                         std::move(apiFc),
                         std::move(physicalInput),
                         std::move(physicalOutput),
                         std::move(physicalFc)};
}

PlacedFcGraph placeComposedFcGraph(const std::string& composedNetworkName, const std::shared_ptr<Api::Network>& phaseNetwork) {
    Api::PhaseGraphComposeOptions options;
    options.networkName = composedNetworkName;
    options.inferenceOnly = false;
    options.exposePhaseOutputsAsNetworkOutputs = true;
    Api::ComposedPhaseGraph graph = Api::buildComposedPhaseGraphByName(
        std::vector<Api::PhaseGraphNetworkSpec>{{"forecast_phase", phaseNetwork, true}}, options);
    return placeNetworkWithSingleFc(graph.network);
}


std::shared_ptr<Api::Network> buildBf16Convolution2dPhaseNetwork(const std::string& networkName) {
    auto network = std::make_shared<Api::Network>(networkName);
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*network)
                                  .name("features")
                                  .dimensions({1, 2, 2})
                                  .dataType(Impl::DataType::BF16)
                                  .build();
    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(*network)
                                         .featureInput(input.getFeatureOutput().value())
                                         .numOutputChannels(1)
                                         .filterHeight(1)
                                         .filterWidth(1)
                                         .verticalPadding(0)
                                         .horizontalPadding(0)
                                         .hasBias(true)
                                         .weightsOptimizer(Api::Adam::Builder().alpha(0.002f).build())
                                         .biasesOptimizer(Api::Adam::Builder().alpha(0.003f).build())
                                         .noActivation()
                                         .build();
    Api::NetworkOutput::Builder()
        .network(*network)
        .name("prediction")
        .inputTensor(convolution.getFeatureOutput().value())
        .dataType(Impl::DataType::BF16)
        .build();
    return network;
}

struct PlacedConvolution2dGraph {
    std::shared_ptr<Api::Network> network;
    std::shared_ptr<Api::PlacedNetwork> placed;
    std::shared_ptr<Impl::NetworkInput> physicalInput;
    std::shared_ptr<Impl::NetworkOutput> physicalOutput;
    std::shared_ptr<Impl::CustomLayer> physicalConvolution;
};

PlacedConvolution2dGraph placeComposedConvolution2dGraph(const std::string& composedNetworkName,
                                                         const std::shared_ptr<Api::Network>& phaseNetwork) {
    Api::PhaseGraphComposeOptions options;
    options.networkName = composedNetworkName;
    options.inferenceOnly = false;
    options.exposePhaseOutputsAsNetworkOutputs = true;
    Api::ComposedPhaseGraph graph = Api::buildComposedPhaseGraphByName(
        std::vector<Api::PhaseGraphNetworkSpec>{{"forecast_phase", phaseNetwork, true}}, options);

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = graph.network->place(kBatchSize,
                                                                      initDoneEvents,
                                                                      /*inferenceOnly=*/false,
                                                                      std::vector<int32_t>{0},
                                                                      /*forcedNumStampsPerGpu=*/1);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(placed, nullptr);

    auto apiInput = findOnlyLayerOfType<Api::NetworkInput>(*graph.network);
    auto apiOutput = findOnlyLayerOfType<Api::NetworkOutput>(*graph.network);
    auto apiConvolution = findOnlyLayerOfType<Api::Convolution2d>(*graph.network);
    EXPECT_NE(apiInput, nullptr);
    EXPECT_NE(apiOutput, nullptr);
    EXPECT_NE(apiConvolution, nullptr);

    auto physicalInput = std::dynamic_pointer_cast<Impl::NetworkInput>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiInput->getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiOutput->getId()));
    auto physicalConvolution = std::dynamic_pointer_cast<Impl::CustomLayer>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiConvolution->getId()));
    EXPECT_NE(physicalInput, nullptr);
    EXPECT_NE(physicalOutput, nullptr);
    EXPECT_NE(physicalConvolution, nullptr);

    return PlacedConvolution2dGraph{
        std::move(graph.network), std::move(placed), std::move(physicalInput), std::move(physicalOutput), std::move(physicalConvolution)};
}

std::vector<float> runForward(PlacedConvolution2dGraph& graph, const std::vector<float>& inputValues) {
    Impl::Tensor input(cpuPlacement, Impl::TensorDescriptor(Impl::DataType::BF16, {kBatchSize, 1, 2, 2}));
    writeCpuTensor(input, inputValues);
    graph.physicalInput->forward(input, false, kBatchSize);
    Event ready = graph.physicalOutput->getOutputReadyEvent();
    ready.synchronize();
    return readCpuTensor(graph.physicalOutput->getFeatureOutput().value());
}

void expectRuntimeErrorContains(const std::function<void()>& fn, const std::string& expectedFragment) {
    try {
        fn();
        FAIL() << "Expected std::runtime_error containing: " << expectedFragment;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expectedFragment), std::string::npos) << "Actual error: " << error.what();
    } catch (...) {
        FAIL() << "Expected std::runtime_error containing: " << expectedFragment;
    }
}

std::vector<float> fullyConnectedReference(const std::vector<float>& input,
                                           const std::vector<float>& weights,
                                           const std::vector<float>& biases) {
    std::vector<float> output(kBatchSize * kNumOutputFeatures, 0.0f);
    for (uint64_t batch = 0; batch < kBatchSize; ++batch) {
        for (uint64_t out = 0; out < kNumOutputFeatures; ++out) {
            float value = biases[out];
            for (uint64_t in = 0; in < kNumInputFeatures; ++in) {
                value += input[batch * kNumInputFeatures + in] * weights[in * kNumOutputFeatures + out];
            }
            output[batch * kNumOutputFeatures + out] = value;
        }
    }
    return output;
}

std::vector<float> runForward(PlacedFcGraph& graph, const std::vector<float>& inputValues) {
    if (!graph.physicalInput->getFeatureOutput().has_value()) {
        throw std::runtime_error("Placed fully connected graph is missing the physical network input tensor.");
    }
    const Impl::DataType inputDataType = graph.physicalInput->getFeatureOutput()->getDataType();
    Impl::Tensor input(cpuPlacement, Impl::TensorDescriptor(inputDataType, {kBatchSize, kNumInputFeatures}));
    writeCpuTensor(input, inputValues);
    graph.physicalInput->forward(input, false, kBatchSize);
    Event ready = graph.physicalOutput->getOutputReadyEvent();
    ready.synchronize();
    return readCpuTensor(graph.physicalOutput->getFeatureOutput().value());
}

}  // namespace

TEST(PlacedNetworkArtifactHandoff, DirectArtifactLoadRestoresCloneSourceMatchedParameterAndOptimizerState) {
    const std::vector<float> weightValues = {0.25f, -0.5f, 1.25f, 0.75f, -1.5f, 2.0f};
    const std::vector<float> biasValues = {0.5f, -1.0f, 1.5f};
    const std::vector<float> weightsMValues = {0.01f, 0.02f, -0.03f, 0.04f, -0.05f, 0.06f};
    const std::vector<float> weightsVValues = {0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f};

    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_clone_handoff_positive");
    try {
        std::shared_ptr<Api::Network> phaseNetwork = buildFullyConnectedPhaseNetwork("direct_clone_handoff_phase");
        PlacedFcGraph source = placeComposedFcGraph("direct_clone_handoff_source", phaseNetwork);
        Stream sourceStream = source.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> sourceWeights = source.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> sourceBiases = source.physicalFc->getParameter("biases");
        ASSERT_NE(sourceWeights, nullptr);
        ASSERT_NE(sourceBiases, nullptr);
        ASSERT_TRUE(sourceWeights->getStorage().has_value());
        ASSERT_TRUE(sourceBiases->getStorage().has_value());
        ASSERT_NE(sourceWeights->getOptimizer(), nullptr);
        ASSERT_TRUE(sourceWeights->getOptimizer()->isCompiled());
        std::shared_ptr<Impl::Adam> sourceWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(sourceWeights->getOptimizer());
        ASSERT_NE(sourceWeightsAdam, nullptr);
        sourceWeightsAdam->setT(17.0f);

        setDeviceTensor(sourceWeights->getStorage().value(), weightValues, sourceStream);
        setDeviceTensor(sourceBiases->getStorage().value(), biasValues, sourceStream);
        setDeviceTensor(sourceWeights->getOptimizer()->getOptimizerParameterTensor("m"), weightsMValues, sourceStream);
        setDeviceTensor(sourceWeights->getOptimizer()->getOptimizerParameterTensor("v"), weightsVValues, sourceStream);
        sourceStream.synchronize();
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);

        PlacedFcGraph destination = placeComposedFcGraph("direct_clone_handoff_destination", phaseNetwork);
        destination.placed->loadMatchingTrainingStateFromArtifact(archiveDir.string(), source.network->getNetworkName());
        Stream destinationStream = destination.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> destinationWeights = destination.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> destinationBiases = destination.physicalFc->getParameter("biases");
        ASSERT_NE(destinationWeights, nullptr);
        ASSERT_NE(destinationBiases, nullptr);
        ASSERT_TRUE(destinationWeights->getStorage().has_value());
        ASSERT_TRUE(destinationBiases->getStorage().has_value());
        ASSERT_NE(destinationWeights->getOptimizer(), nullptr);
        ASSERT_TRUE(destinationWeights->getOptimizer()->isCompiled());
        std::shared_ptr<Impl::Adam> destinationWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(destinationWeights->getOptimizer());
        ASSERT_NE(destinationWeightsAdam, nullptr);
        EXPECT_FLOAT_EQ(destinationWeightsAdam->getT(), 17.0f);

        expectAllClose(readDeviceTensor(destinationWeights->getStorage().value(), destinationStream), weightValues, "weights");
        expectAllClose(readDeviceTensor(destinationBiases->getStorage().value(), destinationStream), biasValues, "biases");
        expectAllClose(readDeviceTensor(destinationWeights->getOptimizer()->getOptimizerParameterTensor("m"), destinationStream),
                       weightsMValues,
                       "weights Adam m");
        expectAllClose(readDeviceTensor(destinationWeights->getOptimizer()->getOptimizerParameterTensor("v"), destinationStream),
                       weightsVValues,
                       "weights Adam v");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(PlacedNetworkArtifactHandoff, MatchingArtifactLoadRestoresBf16WeightBytesAndExecution) {
    const std::vector<float> inputValues = {1.0f, -2.0f, 3.0f, 0.5f};
    const std::vector<float> weightValues = {0.5f, -1.0f, 0.25f, 2.0f, -0.5f, 0.75f};
    const std::vector<float> biasValues = {0.25f, -0.5f, 1.0f};
    const std::vector<float> expected = fullyConnectedReference(inputValues, weightValues, biasValues);

    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_clone_handoff_bf16");
    try {
        std::shared_ptr<Api::Network> phaseNetwork = buildFullyConnectedPhaseNetwork(
            "direct_clone_handoff_bf16_phase",
            /*attachAdamOptimizers=*/true,
            /*inputDataType=*/Impl::DataType::BF16,
            /*weightsDataType=*/Impl::DataType::BF16,
            /*computeDataType=*/Impl::DataType::BF16);
        PlacedFcGraph source = placeComposedFcGraph("direct_clone_handoff_bf16_source", phaseNetwork);
        ASSERT_NE(source.apiFc, nullptr);
        EXPECT_EQ(source.apiFc->getWeightsDataType(), Impl::DataType::BF16);
        EXPECT_EQ(source.apiFc->getComputeDataType(), Impl::DataType::BF16);
        EXPECT_EQ(source.apiFc->getOutputDataType(), Impl::DataType::FP32);
        Stream sourceStream = source.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> sourceWeights = source.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> sourceBiases = source.physicalFc->getParameter("biases");
        ASSERT_NE(sourceWeights, nullptr);
        ASSERT_NE(sourceBiases, nullptr);
        ASSERT_TRUE(sourceWeights->getStorage().has_value());
        ASSERT_TRUE(sourceBiases->getStorage().has_value());
        ASSERT_EQ(sourceWeights->getStorage()->getDataType(), Impl::DataType::BF16);
        ASSERT_EQ(sourceBiases->getStorage()->getDataType(), Impl::DataType::FP32);

        setDeviceTensor(sourceWeights->getStorage().value(), weightValues, sourceStream);
        setDeviceTensor(sourceBiases->getStorage().value(), biasValues, sourceStream);
        sourceStream.synchronize();
        const std::vector<uint8_t> sourceWeightBytes = readDeviceTensorBytes(sourceWeights->getStorage().value(), sourceStream);
        const std::vector<uint8_t> sourceBiasBytes = readDeviceTensorBytes(sourceBiases->getStorage().value(), sourceStream);
        const std::vector<float> sourceOutput = runForward(source, inputValues);
        expectAllClose(sourceOutput, expected, "source BF16 output", 6e-2f);
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        PlacedFcGraph destination = placeComposedFcGraph("direct_clone_handoff_bf16_destination", phaseNetwork);
        destination.placed->loadMatchingTrainingStateFromArtifact(archiveDir.string(), source.network->getNetworkName());
        Stream destinationStream = destination.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> destinationWeights = destination.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> destinationBiases = destination.physicalFc->getParameter("biases");
        ASSERT_NE(destinationWeights, nullptr);
        ASSERT_NE(destinationBiases, nullptr);
        ASSERT_TRUE(destinationWeights->getStorage().has_value());
        ASSERT_TRUE(destinationBiases->getStorage().has_value());
        EXPECT_EQ(readDeviceTensorBytes(destinationWeights->getStorage().value(), destinationStream), sourceWeightBytes);
        EXPECT_EQ(readDeviceTensorBytes(destinationBiases->getStorage().value(), destinationStream), sourceBiasBytes);

        const std::vector<float> actual = runForward(destination, inputValues);
        expectAllClose(actual, sourceOutput, "matching-state BF16 round-trip output", 6e-2f);
        expectAllClose(actual, expected, "matching-state BF16 output", 6e-2f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}


TEST(PlacedNetworkArtifactHandoff, MatchingArtifactLoadRestoresBf16ConvolutionFilterBytesAndExecution) {
    const std::vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 4.0f, 2.0f, -1.0f};
    const std::vector<float> weightValues = {1.5f};
    const std::vector<float> biasValues = {-0.25f};
    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_clone_handoff_bf16_convolution2d");

    try {
        std::shared_ptr<Api::Network> phaseNetwork =
            buildBf16Convolution2dPhaseNetwork("direct_clone_handoff_bf16_convolution2d_phase");
        PlacedConvolution2dGraph source =
            placeComposedConvolution2dGraph("direct_clone_handoff_bf16_convolution2d_source", phaseNetwork);
        Stream sourceStream = source.physicalConvolution->getStreams()[0];
        auto sourceWeights = source.physicalConvolution->getParameter("weights");
        auto sourceBiases = source.physicalConvolution->getParameter("biases");
        ASSERT_NE(sourceWeights, nullptr);
        ASSERT_NE(sourceBiases, nullptr);
        ASSERT_TRUE(sourceWeights->getStorage().has_value());
        ASSERT_TRUE(sourceBiases->getStorage().has_value());
        ASSERT_EQ(sourceWeights->getStorage()->getDataType(), Impl::DataType::BF16);
        ASSERT_EQ(sourceBiases->getStorage()->getDataType(), Impl::DataType::BF16);

        setDeviceTensor(sourceWeights->getStorage().value(), weightValues, sourceStream);
        setDeviceTensor(sourceBiases->getStorage().value(), biasValues, sourceStream);
        sourceStream.synchronize();
        const std::vector<uint8_t> sourceWeightBytes = readDeviceTensorBytes(sourceWeights->getStorage().value(), sourceStream);
        const std::vector<uint8_t> sourceBiasBytes = readDeviceTensorBytes(sourceBiases->getStorage().value(), sourceStream);
        const std::vector<float> sourceOutput = runForward(source, inputValues);
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        PlacedConvolution2dGraph destination =
            placeComposedConvolution2dGraph("direct_clone_handoff_bf16_convolution2d_destination", phaseNetwork);
        destination.placed->loadMatchingTrainingStateFromArtifact(archiveDir.string(), source.network->getNetworkName());
        Stream destinationStream = destination.physicalConvolution->getStreams()[0];
        auto destinationWeights = destination.physicalConvolution->getParameter("weights");
        auto destinationBiases = destination.physicalConvolution->getParameter("biases");
        ASSERT_NE(destinationWeights, nullptr);
        ASSERT_NE(destinationBiases, nullptr);
        ASSERT_TRUE(destinationWeights->getStorage().has_value());
        ASSERT_TRUE(destinationBiases->getStorage().has_value());
        EXPECT_EQ(readDeviceTensorBytes(destinationWeights->getStorage().value(), destinationStream), sourceWeightBytes);
        EXPECT_EQ(readDeviceTensorBytes(destinationBiases->getStorage().value(), destinationStream), sourceBiasBytes);

        const std::vector<float> actual = runForward(destination, inputValues);
        expectAllClose(actual, sourceOutput, "matching-state BF16 convolution2d round-trip output", 3e-2f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(PlacedNetworkArtifactHandoff, CopyMatchingTrainingStateRestoresOptimizerNonTensorState) {
    const std::vector<float> weightValues = {0.75f, -0.25f, 1.5f, -1.0f, 0.5f, -0.5f};
    const std::vector<float> weightsMValues = {0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f};

    std::shared_ptr<Api::Network> phaseNetwork = buildFullyConnectedPhaseNetwork("copy_matching_non_tensor_phase");
    PlacedFcGraph source = placeComposedFcGraph("copy_matching_non_tensor_source", phaseNetwork);
    PlacedFcGraph destination = placeComposedFcGraph("copy_matching_non_tensor_destination", phaseNetwork);

    Stream sourceStream = source.physicalFc->getStreams()[0];
    std::shared_ptr<Impl::PhysicalParameter> sourceWeights = source.physicalFc->getParameter("weights");
    ASSERT_NE(sourceWeights, nullptr);
    ASSERT_TRUE(sourceWeights->getStorage().has_value());
    ASSERT_NE(sourceWeights->getOptimizer(), nullptr);
    ASSERT_TRUE(sourceWeights->getOptimizer()->isCompiled());

    std::shared_ptr<Impl::Adam> sourceWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(sourceWeights->getOptimizer());
    ASSERT_NE(sourceWeightsAdam, nullptr);
    sourceWeightsAdam->setT(41.0f);
    setDeviceTensor(sourceWeights->getStorage().value(), weightValues, sourceStream);
    setDeviceTensor(sourceWeights->getOptimizer()->getOptimizerParameterTensor("m"), weightsMValues, sourceStream);
    sourceStream.synchronize();

    destination.placed->copyMatchingTrainingStateFrom(*source.placed);

    Stream destinationStream = destination.physicalFc->getStreams()[0];
    std::shared_ptr<Impl::PhysicalParameter> destinationWeights = destination.physicalFc->getParameter("weights");
    ASSERT_NE(destinationWeights, nullptr);
    ASSERT_TRUE(destinationWeights->getStorage().has_value());
    ASSERT_NE(destinationWeights->getOptimizer(), nullptr);
    ASSERT_TRUE(destinationWeights->getOptimizer()->isCompiled());

    std::shared_ptr<Impl::Adam> destinationWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(destinationWeights->getOptimizer());
    ASSERT_NE(destinationWeightsAdam, nullptr);
    EXPECT_FLOAT_EQ(destinationWeightsAdam->getT(), 41.0f);
    expectAllClose(readDeviceTensor(destinationWeights->getStorage().value(), destinationStream), weightValues, "copied weights");
    expectAllClose(readDeviceTensor(destinationWeights->getOptimizer()->getOptimizerParameterTensor("m"), destinationStream),
                   weightsMValues,
                   "copied weights Adam m");
}

TEST(PlacedNetworkArtifactHandoff, DirectSameNetworkArtifactLoadRestoresApiLayerMatchedParameterAndOptimizerState) {
    const std::vector<float> weightValues = {-0.75f, 0.125f, 0.5f, 1.0f, -1.25f, 1.75f};
    const std::vector<float> biasValues = {-0.25f, 0.75f, 1.25f};
    const std::vector<float> weightsMValues = {0.21f, -0.22f, 0.23f, -0.24f, 0.25f, -0.26f};

    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_same_network_handoff_positive");
    try {
        std::shared_ptr<Api::Network> network = buildFullyConnectedPhaseNetwork("direct_same_network_handoff");
        PlacedFcGraph source = placeNetworkWithSingleFc(network);
        Stream sourceStream = source.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> sourceWeights = source.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> sourceBiases = source.physicalFc->getParameter("biases");
        ASSERT_NE(sourceWeights, nullptr);
        ASSERT_NE(sourceBiases, nullptr);
        ASSERT_TRUE(sourceWeights->getStorage().has_value());
        ASSERT_TRUE(sourceBiases->getStorage().has_value());
        ASSERT_NE(sourceWeights->getOptimizer(), nullptr);
        ASSERT_TRUE(sourceWeights->getOptimizer()->isCompiled());
        std::shared_ptr<Impl::Adam> sourceWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(sourceWeights->getOptimizer());
        ASSERT_NE(sourceWeightsAdam, nullptr);
        sourceWeightsAdam->setT(23.0f);

        setDeviceTensor(sourceWeights->getStorage().value(), weightValues, sourceStream);
        setDeviceTensor(sourceBiases->getStorage().value(), biasValues, sourceStream);
        setDeviceTensor(sourceWeights->getOptimizer()->getOptimizerParameterTensor("m"), weightsMValues, sourceStream);
        sourceStream.synchronize();
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);

        PlacedFcGraph destination = placeNetworkWithSingleFc(network);
        destination.placed->loadTrainingStateFromSameNetworkArtifact(archiveDir.string(), source.network->getNetworkName());
        Stream destinationStream = destination.physicalFc->getStreams()[0];

        std::shared_ptr<Impl::PhysicalParameter> destinationWeights = destination.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> destinationBiases = destination.physicalFc->getParameter("biases");
        ASSERT_NE(destinationWeights, nullptr);
        ASSERT_NE(destinationBiases, nullptr);
        ASSERT_TRUE(destinationWeights->getStorage().has_value());
        ASSERT_TRUE(destinationBiases->getStorage().has_value());
        ASSERT_NE(destinationWeights->getOptimizer(), nullptr);
        ASSERT_TRUE(destinationWeights->getOptimizer()->isCompiled());
        std::shared_ptr<Impl::Adam> destinationWeightsAdam = std::dynamic_pointer_cast<Impl::Adam>(destinationWeights->getOptimizer());
        ASSERT_NE(destinationWeightsAdam, nullptr);
        EXPECT_FLOAT_EQ(destinationWeightsAdam->getT(), 23.0f);

        expectAllClose(readDeviceTensor(destinationWeights->getStorage().value(), destinationStream), weightValues, "same-network weights");
        expectAllClose(readDeviceTensor(destinationBiases->getStorage().value(), destinationStream), biasValues, "same-network biases");
        expectAllClose(readDeviceTensor(destinationWeights->getOptimizer()->getOptimizerParameterTensor("m"), destinationStream),
                       weightsMValues,
                       "same-network weights Adam m");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(PlacedNetworkArtifactHandoff, DirectSameNetworkArtifactLoadRejectsDifferentApiLayerIdentity) {
    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_same_network_handoff_wrong_identity");
    try {
        PlacedFcGraph source = placeNetworkWithSingleFc(buildFullyConnectedPhaseNetwork("same_network_source"));
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        PlacedFcGraph differentDestination =
            placeNetworkWithSingleFc(buildFullyConnectedPhaseNetwork("same_network_different_destination"));
        expectRuntimeErrorContains(
            [&]() {
                differentDestination.placed->loadTrainingStateFromSameNetworkArtifact(archiveDir.string(),
                                                                                      source.network->getNetworkName());
            },
            "artifact has no saved state for exact API-layer parameter key");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(PlacedNetworkArtifactHandoff, DirectArtifactLoadRejectsLegacyArtifactWithoutCloneSourceKeys) {
    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_clone_handoff_no_source_keys");
    try {
        PlacedFcGraph legacySource = placeNetworkWithSingleFc(buildFullyConnectedPhaseNetwork("legacy_source_without_clone_keys"));
        legacySource.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);

        PlacedFcGraph legacyDestination = placeNetworkWithSingleFc(buildFullyConnectedPhaseNetwork("legacy_destination_without_clone_keys"));
        expectRuntimeErrorContains(
            [&]() {
                legacyDestination.placed->loadMatchingTrainingStateFromArtifact(archiveDir.string(), legacySource.network->getNetworkName());
            },
            "artifact is missing clone_source_keys");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(PlacedNetworkArtifactHandoff, DirectArtifactLoadRejectsDestinationWithoutCloneSourceKeys) {
    const std::filesystem::path archiveDir = makeUniqueTestArchiveDir("direct_clone_handoff_no_destination_keys");
    try {
        std::shared_ptr<Api::Network> phaseNetwork = buildFullyConnectedPhaseNetwork("destination_key_source_phase");
        PlacedFcGraph source = placeComposedFcGraph("destination_key_source_composed", phaseNetwork);
        source.placed->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        PlacedFcGraph destinationWithoutCloneKeys =
            placeNetworkWithSingleFc(buildFullyConnectedPhaseNetwork("destination_without_clone_keys"));
        expectRuntimeErrorContains(
            [&]() {
                destinationWithoutCloneKeys.placed->loadMatchingTrainingStateFromArtifact(archiveDir.string(),
                                                                                         source.network->getNetworkName());
            },
            "no destination parameters matched clone-source keyed artifact parameters");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
