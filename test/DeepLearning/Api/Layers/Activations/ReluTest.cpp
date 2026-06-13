#include <optional>
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
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
