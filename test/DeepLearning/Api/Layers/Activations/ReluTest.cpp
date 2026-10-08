#include <optional>
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "cuda_fp16.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <filesystem>
#include <map>
#include <variant>
#include <stdexcept>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const ThorImplementation::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions())
        numel *= dim;
    return numel;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void writeCpuTensor(ThorImplementation::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case ThorImplementation::DataType::FP16: {
            auto* ptr = static_cast<half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2half(values[i]);
            break;
        }
        case ThorImplementation::DataType::FP32: {
            auto* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        default:
            FAIL() << "Unsupported tensor dtype in writeCpuTensor.";
    }
}

vector<float> readCpuTensor(const ThorImplementation::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case ThorImplementation::DataType::FP16: {
            const auto* ptr = static_cast<const half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __half2float(ptr[i]);
            break;
        }
        case ThorImplementation::DataType::FP32: {
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

ThorImplementation::Tensor copyTensorToCpu(const ThorImplementation::Tensor& tensor, Stream& stream) {
    ThorImplementation::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
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


class TestableRelu : public Relu {
   public:
    void initializeNeutralForTest(RaggedTensor input) {
        initializeStandaloneActivation(input);
        initialized = true;
    }
    uint64_t memoryBytes(uint32_t batchSize) const { return getExpressionBackedActivationMemRequirementInBytes(batchSize); }
};

shared_ptr<Relu> findOnlyRelu(Network& network) {
    shared_ptr<Relu> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<Relu>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1U);
    return found;
}

Batch makeNeutralReluBatch(const string& inputName, const RaggedTensor& logicalInput, uint32_t batchSize) {
    const ThorImplementation::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    const uint64_t elementsPerValue = 2;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (vector<uint64_t>{elementsPerValue}));

    vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    uint64_t activeValues = 0;
    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint64_t rowLength = row == 0 ? 0 : 1 + ((row - 1) % 2);
        activeValues += rowLength;
        hostOffsets[row + 1] = activeValues;
    }
    THOR_THROW_IF_FALSE(activeValues <= capacity);

    ThorImplementation::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t value = 0; value < capacity; ++value) {
        for (uint64_t element = 0; element < elementsPerValue; ++element) {
            const uint64_t i = value * elementsPerValue + element;
            valuesPtr[i] = value < activeValues
                ? ((i % 3 == 0) ? -static_cast<float>(i + 1) : static_cast<float>(i + 1))
                : -1000.0F;
        }
    }

    ThorImplementation::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == DataType::UINT32);
    uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

    ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, ThorImplementation::RaggedTensor(values, partition));
    return batch;
}

shared_ptr<ThorImplementation::CustomLayer> findCustomLayerByType(ThorImplementation::StampedNetwork& stampedNetwork,
                                                                  const string& layerType) {
    for (uint64_t i = 0; i < stampedNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::CustomLayer> candidate =
            dynamic_pointer_cast<ThorImplementation::CustomLayer>(stampedNetwork.getTrainableLayer(i));
        if (candidate != nullptr && candidate->getLayerType() == layerType) {
            return candidate;
        }
    }
    return nullptr;
}

}  // namespace

TEST(Activations, ReluBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Relu::Builder reluBuilder;
    reluBuilder.network(network);
    reluBuilder.featureInput(featureInput);
    shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(reluBuilder.build());

    ASSERT_TRUE(relu->isInitialized());

    std::optional<Tensor> actualInput = relu->getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = relu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = relu->clone();
    Relu *clone = dynamic_cast<Relu *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), dataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.has_value());
    ASSERT_EQ(cloneOutput.value().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);

    ASSERT_NE(relu->getId(), clone->getId());
    ASSERT_GT(relu->getId(), 1u);
}

TEST(Activations, ReluSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Relu::Builder reluBuilder = Relu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(reluBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(relu->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(relu->isInitialized());

    Tensor featureInput = relu->getFeatureInput().value();
    Tensor featureOutput = relu->getFeatureOutput().value();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(relu->getFeatureOutput().has_value());
    ASSERT_EQ(relu->getFeatureOutput().value(), featureOutput);

    ASSERT_TRUE(relu->getFeatureInput().has_value());
    assert(relu->getFeatureInput().value() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(initialPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel");

    json reluJ = relu->serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = relu.get();
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(reluJ, fromLayerJ);

    ASSERT_EQ(reluJ["factory"], "activation");
    ASSERT_EQ(reluJ["version"], "1.0.0");
    ASSERT_EQ(reluJ["layer_type"], "relu");

    EXPECT_TRUE(reluJ.contains("feature_input"));
    EXPECT_TRUE(reluJ.contains("feature_output"));

    const auto &input = reluJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = reluJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", reluJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork("newNetwork");

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Relu::deserialize(reluJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(newStamp.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedRelu = dynamic_pointer_cast<ThorImplementation::CustomLayer>(newStamp.getTrainableLayer(0));
    ASSERT_NE(stampedRelu, nullptr);
    ASSERT_EQ(stampedRelu->getLayerType(), "CustomLayer<Relu>");

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedRelu->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value(), stampedRelu->getFeatureInput().value());
    ASSERT_EQ(stampedRelu->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    filesystem::remove("/tmp/testModel.thor.tar");
}

TEST(Activations, ReluEpilogueRunsForwardBackwardResidualAdd) {
    constexpr uint32_t batchSize = 2;
    const DataType dataType = DataType::FP16;
    const vector<uint64_t> dims = {1, 2, 3};

    Network network("reluEpilogueResidualAdd");
    NetworkInput mainInput = NetworkInput::Builder().network(network).name("main").dimensions(dims).dataType(dataType).build();
    NetworkInput shortcutInput = NetworkInput::Builder().network(network).name("shortcut").dimensions(dims).dataType(dataType).build();
    GradientRivet mainRivet = GradientRivet::Builder().network(network).tensor(mainInput.getFeatureOutput().value()).build();
    GradientRivet shortcutRivet = GradientRivet::Builder().network(network).tensor(shortcutInput.getFeatureOutput().value()).build();

    ThorImplementation::Expression mainExpr = Activation::epilogueInput(ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP16);
    ThorImplementation::Expression shortcutExpr =
        Activation::epilogueAuxInput("shortcut", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP16);
    Relu reluTemplate;
    ThorImplementation::Expression reluResidual = reluTemplate.toExpression(mainExpr + shortcutExpr)
                                                   .withDTypes(ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP16);

    shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(Relu::Builder()
                                                           .network(network)
                                                           .featureInput(mainRivet.getFeatureOutput().value())
                                                           .epilogueInput("shortcut", shortcutRivet.getFeatureOutput().value())
                                                           .epilogue(reluResidual)
                                                           .build());
    ASSERT_NE(relu, nullptr);
    GradientRivet outputRivet = GradientRivet::Builder().network(network).tensor(relu->getFeatureOutput().value()).build();
    NetworkOutput output = NetworkOutput::Builder()
                               .network(network)
                               .name("output")
                               .inputTensor(outputRivet.getFeatureOutput().value())
                               .dataType(dataType)
                               .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);

    ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalMain = dynamic_pointer_cast<ThorImplementation::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(mainInput.getId()));
    auto physicalShortcut =
        dynamic_pointer_cast<ThorImplementation::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(shortcutInput.getId()));
    auto physicalOutput =
        dynamic_pointer_cast<ThorImplementation::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    // Standalone activation builders return the user-facing activation object, but
    // activation clones intentionally receive fresh API ids when inserted into the
    // network so that a template activation can be reused by many layers.  Find the
    // stamped ReLU by its physical custom-layer type rather than by the returned
    // builder object's id.
    auto physicalRelu = findCustomLayerByType(stampedNetwork, "CustomLayer<Relu>");
    ASSERT_NE(physicalMain, nullptr);
    ASSERT_NE(physicalShortcut, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalRelu, nullptr);

    const vector<float> mainValues = {
        -1.0f, 0.25f, 2.0f,
        -2.0f, 1.5f, -0.5f,
        0.5f, -1.25f, 3.0f,
        -0.75f, -0.25f, 1.0f,
    };
    const vector<float> shortcutValues = {
        0.5f, 0.5f, -1.0f,
        3.0f, -2.0f, 1.0f,
        0.25f, 1.5f, -4.0f,
        1.0f, -0.5f, 0.75f,
    };
    const vector<float> upstreamErrors = {
        0.5f, -1.0f, 1.5f,
        -0.25f, 0.75f, -1.25f,
        1.0f, 0.25f, -0.5f,
        -1.5f, 0.5f, 0.25f,
    };

    ThorImplementation::Tensor mainHost(cpuPlacement, ThorImplementation::TensorDescriptor(dataType, {batchSize, 1, 2, 3}));
    ThorImplementation::Tensor shortcutHost(cpuPlacement, ThorImplementation::TensorDescriptor(dataType, {batchSize, 1, 2, 3}));
    writeCpuTensor(mainHost, mainValues);
    writeCpuTensor(shortcutHost, shortcutValues);

    physicalMain->forward(mainHost, false, batchSize);
    physicalShortcut->forward(shortcutHost, false, batchSize);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    vector<float> expectedForward(mainValues.size());
    vector<float> expectedBackward(mainValues.size());
    for (uint64_t i = 0; i < mainValues.size(); ++i) {
        const float residualSum = mainValues[i] + shortcutValues[i];
        expectedForward[i] = std::max(0.0f, residualSum);
        expectedBackward[i] = residualSum > 0.0f ? upstreamErrors[i] : 0.0f;
    }
    expectAllClose(readCpuTensor(physicalOutput->getFeatureOutput().value()), expectedForward, 7e-2f, 7e-2f,
                   "relu residual epilogue forward");

    ASSERT_EQ(physicalRelu->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalRelu->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalRelu->getErrorOutputs().size(), 2u)
        << "Activation epilogue backward must produce gradients for the primary input and auxiliary shortcut input.";
    ASSERT_TRUE(physicalRelu->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalRelu->getErrorOutputs()[1].has_value());

    Stream stream = physicalRelu->getStreams()[0];
    ThorImplementation::Tensor errorInput = physicalRelu->getErrorInputs()[0].value();
    ThorImplementation::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstreamErrors);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalRelu->backward(errorInput, batchSize);

    ThorImplementation::Tensor mainErrorHost = copyTensorToCpu(physicalRelu->getErrorOutputs()[0].value(), stream);
    ThorImplementation::Tensor shortcutErrorHost = copyTensorToCpu(physicalRelu->getErrorOutputs()[1].value(), stream);
    stream.synchronize();

    expectAllClose(readCpuTensor(mainErrorHost), expectedBackward, 8e-2f, 8e-2f, "relu residual epilogue primary error out");
    expectAllClose(readCpuTensor(shortcutErrorHost), expectedBackward, 8e-2f, 8e-2f, "relu residual epilogue shortcut error out");

    const json reluJ = relu->architectureJson();
    ASSERT_TRUE(reluJ.contains("epilogue"));
    EXPECT_FALSE(reluJ.at("epilogue").is_null());
    ASSERT_TRUE(reluJ.contains("epilogue_inputs"));
    ASSERT_EQ(reluJ.at("epilogue_inputs").size(), 1u);
    EXPECT_EQ(reluJ.at("epilogue_inputs").at(0).at("name").get<string>(), "shortcut");
}

TEST(Activations, ReluRegistered) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Relu::Builder reluBuilder = Relu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(reluBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(relu->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(relu->isInitialized());

    thor_file::TarWriter archiveWriter("testModel");
    Stream stream(0);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json reluJ = relu->serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork("newNetwork");
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(reluJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedRelu = dynamic_pointer_cast<ThorImplementation::CustomLayer>(stampedNetwork.getTrainableLayer(0));
    ASSERT_NE(stampedRelu, nullptr);
    ASSERT_EQ(stampedRelu->getLayerType(), "CustomLayer<Relu>");

    filesystem::remove("/tmp/testModel.thor.tar");
}


TEST(Activations, PlacementNeutralReluSerializesSemanticGeometryAndRejectsMismatches) {
    constexpr uint64_t maxValuesPerRow = 4;
    Network network("placement_neutral_relu_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();
    ASSERT_TRUE(input.isPlacementNeutral());

    auto relu = dynamic_pointer_cast<Relu>(Relu::Builder().network(network).featureInput(input).build());
    ASSERT_NE(relu, nullptr);
    ASSERT_TRUE(relu->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(relu->getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(relu->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 2}));

    TestableRelu testable;
    testable.initializeNeutralForTest(input);
    const uint64_t logicalValuesBytes = maxValuesPerRow * 2U * sizeof(float);
    EXPECT_EQ(testable.memoryBytes(2), 2U * logicalValuesBytes * 2U + sizeof(uint32_t));
    EXPECT_EQ(testable.memoryBytes(7), 7U * logicalValuesBytes * 2U + sizeof(uint32_t));

    const json architecture = relu->architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
    EXPECT_EQ(inputJson.at("layer_ragged_reference_version").get<string>(), "1.0.0");
    EXPECT_EQ(inputJson.at("geometry_mode").get<string>(), "placement_neutral");
    EXPECT_EQ(inputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(inputJson.contains("batch_size"));
    EXPECT_FALSE(inputJson.contains("max_total_values"));
    EXPECT_FALSE(outputJson.contains("batch_size"));
    EXPECT_FALSE(outputJson.contains("max_total_values"));
    EXPECT_EQ(inputJson.at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_EQ(inputJson.at("offsets").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
    EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), inputJson.at("offsets").at("id").get<uint64_t>());
    EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(),
              architecture.at("feature_output").at("id").get<uint64_t>());

    json wrongVersion = architecture;
    wrongVersion["version"] = "1.0.0";
    EXPECT_THROW(Activation::deserialize(wrongVersion, &network), runtime_error);

    json wrongGeometry = architecture;
    wrongGeometry["ragged_feature_input"]["geometry_mode"] = "fixed";
    EXPECT_THROW(Activation::deserialize(wrongGeometry, &network), runtime_error);

    json wrongPartition = architecture;
    wrongPartition["ragged_feature_output"]["offsets"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongPartition, &network), runtime_error);

    json wrongOutputValues = architecture;
    wrongOutputValues["ragged_feature_output"]["values"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongOutputValues, &network), runtime_error);
}

TEST(Activations, PlacementNeutralReluSaveLoadReplacesPlacementGeometryPerBatchAndExecutesEmptyRows) {
    constexpr uint64_t maxValuesPerRow = 4;
    const string networkName = "placement_neutral_relu_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_relu_neutral_" + to_string(nonce));
    filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .build();
        auto relu = dynamic_pointer_cast<Relu>(Relu::Builder().network(network).featureInput(input).build());
        ASSERT_NE(relu, nullptr);
        ASSERT_TRUE(relu->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(relu->getRaggedFeatureOutput().value())
            .build();

        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = relu->getFeatureOutput()->getOriginalId();
        const json layerJson = relu->architectureJson();
        EXPECT_EQ(layerJson.at("version").get<string>(), "1.1.0");
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("max_total_values"));
        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto inputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 1U);
        const RaggedTensor loadedInput = inputs.front().raggedTensor;
        ASSERT_TRUE(loadedInput.isPlacementNeutral());
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionOriginalId);

        shared_ptr<Relu> loadedRelu = findOnlyRelu(loaded);
        ASSERT_NE(loadedRelu, nullptr);
        ASSERT_TRUE(loadedRelu->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedRelu->getFeatureInput().value(), loadedInput.getValues());
        EXPECT_EQ(loadedRelu->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedRelu->getRaggedFeatureOutput()->getRowPartitionToken(), loadedInput.getRowPartitionToken());
        EXPECT_EQ(loadedRelu->getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{maxValuesPerRow, 2}));

        const json architectureBeforePlacement = loaded.architectureJson();
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        synchronizeEvents(initDoneEvents);

        auto& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_TRUE(ThorImplementation::hasRaggedPartitionRequirement(
            requirements.value(), ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT));
        auto activeCount = stamped.getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(activeCount, nullptr);
        ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
        EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId()), nullptr);

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        auto physicalInput = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalInput, nullptr);
        ASSERT_TRUE(physicalInput->getFeatureOutput().has_value());
        EXPECT_EQ(physicalInput->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, 2}));
        auto physicalRelu = dynamic_pointer_cast<ThorImplementation::RaggedCustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(loadedRelu->getId()));
        ASSERT_NE(physicalRelu, nullptr);
        EXPECT_EQ(physicalRelu->getLayerType(), "RaggedCustomLayer");
        ASSERT_TRUE(physicalRelu->getFeatureOutput().has_value());
        EXPECT_EQ(physicalRelu->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const map<string, InferenceOutputValue> outputs =
            placed->inferLogical(makeNeutralReluBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("activated"));
        ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputs.at("activated")));
        const ThorImplementation::RaggedTensor activated =
            get<ThorImplementation::RaggedTensor>(outputs.at("activated"));
        EXPECT_EQ(activated.getValues().getDimensions(), (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(activated.getOffsets().getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
        const auto hostOffsets = activated.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        ASSERT_GE(hostOffsets->size(), 2U);
        EXPECT_EQ(hostOffsets->at(0), 0U);
        EXPECT_EQ(hostOffsets->at(1), 0U) << "first row must remain empty";

        const uint64_t activeValues = hostOffsets->back();
        const float* values = activated.getValues().getMemPtr<float>();
        for (uint64_t value = 0; value < activeValues; ++value) {
            for (uint64_t element = 0; element < 2; ++element) {
                const uint64_t i = value * 2 + element;
                const float original = (i % 3 == 0) ? -static_cast<float>(i + 1) : static_cast<float>(i + 1);
                EXPECT_FLOAT_EQ(values[i], std::max(0.0F, original));
            }
        }
    };

    try {
        loadPlaceAndRun(2);
        loadPlaceAndRun(7);
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

TEST(Activations, FixedRaggedReluRemainsVersionOneAndLoads) {
    const string networkName = "fixed_ragged_relu_compatibility";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_relu_fixed_" + to_string(nonce));
    filesystem::remove_all(archive);

    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .batchSize(3)
                                 .maxTotalValues(9)
                                 .maxValuesPerRow(4)
                                 .build();
        auto relu = dynamic_pointer_cast<Relu>(Relu::Builder().network(network).featureInput(input).build());
        ASSERT_NE(relu, nullptr);
        ASSERT_TRUE(relu->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(relu->getRaggedFeatureOutput().value())
            .build();
        const json architecture = relu->architectureJson();
        EXPECT_EQ(architecture.at("version").get<string>(), "1.0.0");
        EXPECT_EQ(architecture.at("ragged_feature_input").at("batch_size").get<uint64_t>(), 3U);
        EXPECT_EQ(architecture.at("ragged_feature_input").at("max_total_values").get<uint64_t>(), 9U);
        EXPECT_FALSE(architecture.at("ragged_feature_input").contains("layer_ragged_reference_version"));
        network.save(archive.string(), true);
    }

    try {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        shared_ptr<Relu> relu = findOnlyRelu(loaded);
        ASSERT_NE(relu, nullptr);
        ASSERT_TRUE(relu->getRaggedFeatureInput().has_value());
        EXPECT_FALSE(relu->getRaggedFeatureInput()->isPlacementNeutral());
        EXPECT_EQ(relu->architectureJson().at("version").get<string>(), "1.0.0");
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}
