#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dimension : tensor.getDimensions())
        numel *= dimension;
    return numel;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void writeCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    auto* data = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        data[i] = values[i];
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    vector<float> values(tensorNumel(tensor));
    const auto* data = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = data[i];
    return values;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor host = tensor.clone(cpuPlacement);
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    return host;
}

void expectEqual(const vector<float>& actual, const vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i)
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
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

vector<float> runSingleInputNetwork(Api::Network& network,
                                    const Api::NetworkInput& apiInput,
                                    const Api::NetworkOutput& apiOutput,
                                    uint32_t batchSize,
                                    const vector<uint64_t>& logicalInputDimensions,
                                    const vector<float>& values) {
    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(placed, nullptr);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalInput = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(apiInput.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(apiOutput.getId()));
    EXPECT_NE(physicalInput, nullptr);
    EXPECT_NE(physicalOutput, nullptr);

    vector<uint64_t> physicalInputDimensions{batchSize};
    physicalInputDimensions.insert(physicalInputDimensions.end(), logicalInputDimensions.begin(), logicalInputDimensions.end());
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, physicalInputDimensions));
    writeCpuTensor(featureInHost, values);
    physicalInput->forward(featureInHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();
    return readCpuTensor(physicalOutput->getFeatureOutput().value());
}

struct SingleSliceNetwork {
    std::shared_ptr<Api::Network> network;
    Api::NetworkInput input;
    Api::Slice slice;
    Api::NetworkOutput output;
};

SingleSliceNetwork makeSingleSliceNetwork(const std::string& name) {
    auto network = std::make_shared<Api::Network>(name);
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*network)
                                  .name("sequence")
                                  .dimensions({6, 2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::Slice slice = Api::Slice::Builder()
                           .network(*network)
                           .featureInput(input.getFeatureOutput().value())
                           .axis(0)
                           .start(-3)
                           .length(3)
                           .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(*network)
                                    .name("tail")
                                    .inputTensor(slice.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    return {network, input, slice, output};
}

vector<float> sequentialValues(uint64_t count) {
    vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i)
        values[i] = static_cast<float>(i);
    return values;
}

vector<float> expectedTail(const vector<float>& inputValues, uint32_t batchSize) {
    vector<float> expected;
    expected.reserve(batchSize * 3 * 2);
    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        const uint64_t base = batch * 6 * 2;
        for (uint64_t i = 3 * 2; i < 6 * 2; ++i)
            expected.push_back(inputValues[base + i]);
    }
    return expected;
}

}  // namespace

TEST(SliceApi, LogicalAxisExcludesBatchAndNegativeStartSelectsTail) {
    Api::Network network("slice_logical_shape");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("sequence")
                                  .dimensions({6, 2})
                                  .dataType(DataType::FP32)
                                  .build();

    Api::Slice timeSlice = Api::Slice::Builder()
                               .network(network)
                               .featureInput(input.getFeatureOutput().value())
                               .axis(0)
                               .start(-3)
                               .length(3)
                               .build();
    EXPECT_EQ(timeSlice.getFeatureOutput()->getDimensions(), (vector<uint64_t>{3, 2}));
    EXPECT_EQ(timeSlice.getAxis(), 0u);
    EXPECT_EQ(timeSlice.getStart(), -3);
    EXPECT_EQ(timeSlice.getLength(), 3u);

    Api::Slice channelSlice = Api::Slice::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .axis(1)
                                  .start(-1)
                                  .length(1)
                                  .build();
    EXPECT_EQ(channelSlice.getFeatureOutput()->getDimensions(), (vector<uint64_t>{6, 1}));

    const nlohmann::json architecture = timeSlice.architectureJson();
    EXPECT_EQ(architecture.at("layer_type"), "slice");
    EXPECT_EQ(architecture.at("axis"), 0);
    EXPECT_EQ(architecture.at("start"), -3);
    EXPECT_EQ(architecture.at("length"), 3);
}

TEST(SliceApi, DirectPlacementUsesRuntimeBatchInsteadOfApiInferenceBatch) {
    constexpr uint32_t batchSize = 4;
    SingleSliceNetwork fixture = makeSingleSliceNetwork("slice_runtime_batch");
    const vector<float> values = sequentialValues(batchSize * 6 * 2);
    const vector<float> actual = runSingleInputNetwork(*fixture.network, fixture.input, fixture.output, batchSize, {6, 2}, values);
    expectEqual(actual, expectedTail(values, batchSize));
}

TEST(SliceApi, BackwardScattersGradientIntoTheSelectedLogicalWindow) {
    constexpr uint32_t batchSize = 4;
    Api::Network network("slice_backward");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("sequence")
                                  .dimensions({6, 2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Slice slice = Api::Slice::Builder()
                           .network(network)
                           .featureInput(inputRivet.getFeatureOutput().value())
                           .axis(0)
                           .start(-3)
                           .length(3)
                           .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(slice.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("tail")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalInput = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(input.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalSlice = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(slice.getId()));
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalSlice, nullptr);

    const vector<float> inputValues = sequentialValues(batchSize * 6 * 2);
    Impl::Tensor inputHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 6, 2}));
    writeCpuTensor(inputHost, inputValues);
    physicalInput->forward(inputHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();

    ASSERT_EQ(physicalSlice->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalSlice->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalSlice->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(physicalSlice->getErrorOutputs()[0].has_value());

    vector<float> upstream(batchSize * 3 * 2);
    for (uint64_t i = 0; i < upstream.size(); ++i)
        upstream[i] = static_cast<float>(i + 1);
    Stream stream = physicalSlice->getStreams()[0];
    Impl::Tensor errorInput = physicalSlice->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstream);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalSlice->backward(errorInput, batchSize);

    Impl::Tensor errorOutputHost = copyTensorToCpu(physicalSlice->getErrorOutputs()[0].value(), stream);
    vector<float> expected(batchSize * 6 * 2, 0.0f);
    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        const uint64_t sourceOffset = batch * 3 * 2;
        const uint64_t destinationOffset = batch * 6 * 2 + 3 * 2;
        for (uint64_t i = 0; i < 3 * 2; ++i)
            expected[destinationOffset + i] = upstream[sourceOffset + i];
    }
    expectEqual(readCpuTensor(errorOutputHost), expected);
}

TEST(SliceApi, SaveLoadPlacesAtDifferentBatchSizeWithoutPythonCallback) {
    constexpr uint32_t runtimeBatchSize = 4;
    const std::string networkName = "slice_save_load_different_batch";
    SingleSliceNetwork fixture = makeSingleSliceNetwork(networkName);
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        fixture.network->save(archiveDir.string(), true);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loaded);
        std::shared_ptr<Api::Slice> loadedSlice = findOnlyLayerOfType<Api::Slice>(loaded);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loaded);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedSlice, nullptr);
        ASSERT_NE(loadedOutput, nullptr);
        EXPECT_EQ(loadedSlice->getFeatureOutput()->getDimensions(), (vector<uint64_t>{3, 2}));

        const vector<float> values = sequentialValues(runtimeBatchSize * 6 * 2);
        const vector<float> actual =
            runSingleInputNetwork(loaded, *loadedInput, *loadedOutput, runtimeBatchSize, {6, 2}, values);
        expectEqual(actual, expectedTail(values, runtimeBatchSize));
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(SliceApi, PhaseCompositionPreservesDeclarativeSliceAtRuntimeBatchFour) {
    constexpr uint32_t batchSize = 4;
    SingleSliceNetwork phase = makeSingleSliceNetwork("slice_phase_source");
    Api::ComposedPhaseGraph composed = Api::buildComposedPhaseGraphByName(
        {{"slice_phase", phase.network, true}}, Api::PhaseGraphComposeOptions{"slice_phase_composed"});

    ASSERT_NE(composed.network, nullptr);
    std::shared_ptr<Api::NetworkInput> input = findOnlyLayerOfType<Api::NetworkInput>(*composed.network);
    std::shared_ptr<Api::Slice> slice = findOnlyLayerOfType<Api::Slice>(*composed.network);
    std::shared_ptr<Api::NetworkOutput> output = findOnlyLayerOfType<Api::NetworkOutput>(*composed.network);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(slice, nullptr);
    ASSERT_NE(output, nullptr);

    const vector<float> values = sequentialValues(batchSize * 6 * 2);
    const vector<float> actual = runSingleInputNetwork(*composed.network, *input, *output, batchSize, {6, 2}, values);
    expectEqual(actual, expectedTail(values, batchSize));
}

TEST(SliceApi, DenseSliceOutputFeedsConcatenate) {
    constexpr uint32_t batchSize = 4;
    Api::Network network("slice_to_concatenate");
    Api::NetworkInput sequence = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("sequence")
                                     .dimensions({6, 2})
                                     .dataType(DataType::FP32)
                                     .build();
    Api::NetworkInput extra = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("extra")
                                  .dimensions({3, 1})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::Slice slice = Api::Slice::Builder()
                           .network(network)
                           .featureInput(sequence.getFeatureOutput().value())
                           .axis(0)
                           .start(-3)
                           .length(3)
                           .build();
    Api::Concatenate concatenate = Api::Concatenate::Builder()
                                       .network(network)
                                       .featureInput(slice.getFeatureOutput().value())
                                       .featureInput(extra.getFeatureOutput().value())
                                       .concatenationAxis(1)
                                       .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("joined")
                                    .inputTensor(concatenate.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    EXPECT_EQ(concatenate.getFeatureOutput()->getDimensions(), (vector<uint64_t>{3, 3}));

    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalSequence = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(sequence.getId()));
    auto physicalExtra = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(extra.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalSequence, nullptr);
    ASSERT_NE(physicalExtra, nullptr);
    ASSERT_NE(physicalOutput, nullptr);

    const vector<float> sequenceValues = sequentialValues(batchSize * 6 * 2);
    vector<float> extraValues(batchSize * 3);
    for (uint64_t i = 0; i < extraValues.size(); ++i)
        extraValues[i] = 1000.0f + static_cast<float>(i);
    Impl::Tensor sequenceHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 6, 2}));
    Impl::Tensor extraHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 3, 1}));
    writeCpuTensor(sequenceHost, sequenceValues);
    writeCpuTensor(extraHost, extraValues);
    physicalSequence->forward(sequenceHost, false, batchSize);
    physicalExtra->forward(extraHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();

    vector<float> expected;
    expected.reserve(batchSize * 3 * 3);
    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        for (uint64_t step = 0; step < 3; ++step) {
            const uint64_t sequenceBase = batch * 12 + (step + 3) * 2;
            expected.push_back(sequenceValues[sequenceBase]);
            expected.push_back(sequenceValues[sequenceBase + 1]);
            expected.push_back(extraValues[batch * 3 + step]);
        }
    }
    expectEqual(readCpuTensor(physicalOutput->getFeatureOutput().value()), expected);
}

TEST(SliceApi, DenseSliceOutputFeedsFullyConnectedWithPreservedPrefixDimensions) {
    constexpr uint32_t batchSize = 4;
    Api::Network network("slice_to_fully_connected");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("sequence")
                                  .dimensions({6, 2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::Slice slice = Api::Slice::Builder()
                           .network(network)
                           .featureInput(input.getFeatureOutput().value())
                           .axis(0)
                           .start(-3)
                           .length(3)
                           .build();
    Api::FullyConnected fullyConnected = Api::FullyConnected::Builder()
                                              .network(network)
                                              .featureInput(slice.getFeatureOutput().value())
                                              .numOutputFeatures(4)
                                              .hasBias(true)
                                              .preserveInputPrefixDimensions(true)
                                              .noActivation()
                                              .weightsDataType(DataType::FP32)
                                              .computeDataType(DataType::FP32)
                                              .outputDataType(DataType::FP32)
                                              .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("projection")
                                    .inputTensor(fullyConnected.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    EXPECT_EQ(fullyConnected.getFeatureOutput()->getDimensions(), (vector<uint64_t>{3, 4}));

    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalOutput->getFeatureOutput().has_value());
    EXPECT_EQ(physicalOutput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{batchSize, 3, 4}));
}
