#include "DeepLearning/Api/Initializers/UniformRandom.h"
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
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
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
    ASSERT_EQ(tensor.getDataType(), kDataType);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    float* ptr = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
}

std::vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), kDataType);

    std::vector<float> values(tensorNumel(tensor));
    const float* ptr = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
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

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& context) {
    ASSERT_EQ(actual.size(), expected.size()) << context;
    for (uint64_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << context << " mismatch at index " << i;
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

std::shared_ptr<Api::Network> buildFullyConnectedPhaseNetwork(const std::string& networkName, bool attachAdamOptimizers = true) {
    auto network = std::make_shared<Api::Network>(networkName);
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*network)
                                  .name("features")
                                  .dimensions({kNumInputFeatures})
                                  .dataType(kDataType)
                                  .build();

    Api::FullyConnected::Builder fcBuilder = Api::FullyConnected::Builder()
                                                 .network(*network)
                                                 .featureInput(input.getFeatureOutput().value())
                                                 .numOutputFeatures(kNumOutputFeatures)
                                                 .hasBias(true)
                                                 .weightsDataType(kDataType)
                                                 .computeDataType(kDataType)
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
    std::shared_ptr<Impl::CustomLayer> physicalFc =
        std::dynamic_pointer_cast<Impl::CustomLayer>(placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(apiFc->getId()));
    EXPECT_NE(physicalFc, nullptr);
    return PlacedFcGraph{std::move(network), std::move(placed), std::move(apiFc), std::move(physicalFc)};
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
        destination.placed->synchronizeDevices();
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
    destination.placed->synchronizeDevices();

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
        destination.placed->synchronizeDevices();
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
