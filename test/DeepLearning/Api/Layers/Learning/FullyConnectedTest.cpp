#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using std::shared_ptr;
using json = nlohmann::json;

using namespace Thor;
using namespace std;

TEST(FullyConnectedSingleFeatureInput, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    UniformRandom::Builder uniformRandomInitializerBuilder;
    Tanh::Builder tanhBuilder;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
    double epsilon = (1 + (rand() % 1000)) / 1000.0f;

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(featureInput)
                                        .numOutputFeatures(numOutputFeatures)
                                        .hasBias(hasBias)
                                        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                        .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                        .activationBuilder(tanhBuilder)
                                        .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                        .dropOut(dropProportion)
                                        .build();

    ASSERT_TRUE(fullyConnected.isInitialized());

    Optional<Tensor> actualInput = fullyConnected.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions = {numOutputFeatures};
    Optional<Tensor> actualOutput = fullyConnected.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    shared_ptr<Layer> cloneLayer = fullyConnected.clone();
    FullyConnected *clone = dynamic_cast<FullyConnected *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(fullyConnected.getId(), clone->getId());
    ASSERT_GT(fullyConnected.getId(), 1u);

    ASSERT_TRUE(fullyConnected == *clone);
    ASSERT_FALSE(fullyConnected != *clone);
    ASSERT_FALSE(fullyConnected > *clone);
    ASSERT_FALSE(fullyConnected < *clone);
}

TEST(FullyConnectedMultipleFeatureInputs, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput0(dataType, dimensions);
    Tensor featureInput1(dataType, dimensions);

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(featureInput0)
                                        .featureInput(featureInput1)
                                        .numOutputFeatures(numOutputFeatures)
                                        .hasBias(hasBias)
                                        .dropOut(dropProportion)
                                        .build();

    ASSERT_TRUE(fullyConnected.isInitialized());

    vector<uint64_t> outputDimensions = {numOutputFeatures};
    vector<Tensor> featureInputs = fullyConnected.getFeatureInputs();
    vector<Tensor> featureOutputs = fullyConnected.getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(fullyConnected.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(fullyConnected.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(fullyConnected.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(fullyConnected.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[0].getDimensions(), outputDimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[1].getDimensions(), outputDimensions);

    shared_ptr<Layer> cloneLayer = fullyConnected.clone();
    FullyConnected *clone = dynamic_cast<FullyConnected *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    featureInputs.clear();
    featureOutputs.clear();
    featureInputs = clone->getFeatureInputs();
    featureOutputs = clone->getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(clone->getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(clone->getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(clone->getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(clone->getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[0].getDimensions(), outputDimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[1].getDimensions(), outputDimensions);

    ASSERT_EQ(fullyConnected.getId(), clone->getId());
    ASSERT_GT(fullyConnected.getId(), 1u);

    ASSERT_TRUE(fullyConnected == *clone);
    ASSERT_FALSE(fullyConnected != *clone);
    ASSERT_FALSE(fullyConnected > *clone);
    ASSERT_FALSE(fullyConnected < *clone);
}

TEST(FullyConnected, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;

    bool use_batch_norm = rand() % 2;

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    FullyConnected::Builder fullyConnectedBuilder = FullyConnected::Builder()
                                                        .network(initialNetwork)
                                                        .featureInput(networkInput.getFeatureOutput())
                                                        .numOutputFeatures(numOutputFeatures)
                                                        .hasBias(hasBias)
                                                        .dropOut(dropProportion);
    if (use_batch_norm) {
        fullyConnectedBuilder.batchNormalization();
    }
    FullyConnected fullyConnected = fullyConnectedBuilder.build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(fullyConnected.getFeatureOutputs()[0])
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(fullyConnected.isInitialized());

    vector<uint64_t> outputDimensions = {numOutputFeatures};
    vector<Tensor> featureInputs = fullyConnected.getFeatureInputs();
    vector<Tensor> featureOutputs = fullyConnected.getFeatureOutputs();
    assert(featureInputs[0] == networkInput.getFeatureOutput());

    ASSERT_EQ(fullyConnected.getFeatureOutput(networkInput.getFeatureOutput()), featureOutputs[0]);

    assert(fullyConnected.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), outputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the fully connected layer from the network and write to its weights
    ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);
    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), use_batch_norm ? 2UL : 1UL);
    shared_ptr<ThorImplementation::FullyConnected> physicalFCLayer =
        dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
    if (use_batch_norm) {
        if (physicalFCLayer == nullptr)
            physicalFCLayer = dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(1));
    }
    ASSERT_TRUE(physicalFCLayer != nullptr);
    ThorImplementation::Tensor weights = physicalFCLayer->getWeights();
    ThorImplementation::Tensor weightsCpu = weights.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    half *weightsCpuMem = (half *)weightsCpu.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        weightsCpuMem[i] = i;
    }
    weights.copyFromAsync(weightsCpu, stream);

    ThorImplementation::Tensor biases;
    ThorImplementation::Tensor biasesCpu;
    if (hasBias) {
        biases = physicalFCLayer->getBiases();
        biasesCpu = biases.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
        half *biasesCpuMem = (half *)biasesCpu.getMemPtr();
        for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
            biasesCpuMem[i] = i * i + 6;
        }
        biases.copyFromAsync(biasesCpu, stream);
    }

    json fullyConnectedJ = fullyConnected.serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &fullyConnected;
    json fromLayerJ = layer->serialize("/tmp/", stream);
    ASSERT_EQ(fullyConnectedJ, fromLayerJ);

    ASSERT_EQ(fullyConnectedJ["version"], "1.0.0");
    ASSERT_EQ(fullyConnectedJ["layer_type"], "fully_connected");

    EXPECT_TRUE(fullyConnectedJ.contains("num_output_features"));
    EXPECT_TRUE(fullyConnectedJ.contains("has_bias"));
    EXPECT_FALSE(fullyConnectedJ.contains("activation"));
    EXPECT_FALSE(fullyConnectedJ.contains("drop_out"));
    EXPECT_FALSE(fullyConnectedJ.contains("batch_normalization"));
    EXPECT_FALSE(fullyConnectedJ.contains("activation"));
    EXPECT_EQ(fullyConnectedJ.contains("biases_tensor"), hasBias);
    EXPECT_TRUE(fullyConnectedJ.contains("weights_tensor"));
    EXPECT_TRUE(fullyConnectedJ.contains("inputs"));
    EXPECT_TRUE(fullyConnectedJ.contains("outputs"));

    ASSERT_TRUE(fullyConnectedJ.at("num_output_features").is_number_integer());
    ASSERT_TRUE(fullyConnectedJ.at("has_bias").is_boolean());
    ASSERT_TRUE(fullyConnectedJ.at("weights_tensor").is_string());
    ASSERT_TRUE(fullyConnectedJ.at("inputs").is_array());
    ASSERT_TRUE(fullyConnectedJ.at("outputs").is_array());

    EXPECT_EQ(fullyConnectedJ.at("num_output_features").get<uint32_t>(), numOutputFeatures);
    EXPECT_EQ(fullyConnectedJ.at("has_bias").get<bool>(), hasBias);

    const auto &inputs = fullyConnectedJ.at("inputs");
    ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
    const auto &in0 = inputs.at(0);
    ASSERT_TRUE(in0.is_object());
    ASSERT_TRUE(in0.at("data_type").is_string());
    EXPECT_EQ(in0.at("data_type").get<string>(), "fp16");

    ASSERT_TRUE(in0.at("dimensions").is_array());
    ASSERT_EQ(in0.at("dimensions").size(), 1U);
    EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);

    ASSERT_TRUE(in0.at("id").is_number_integer());

    const auto &outputs = fullyConnectedJ.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<string>(), "fp16");

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), 1U);
    EXPECT_TRUE(out0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(out0.at("dimensions").at(0).get<uint32_t>(), numOutputFeatures);

    ASSERT_TRUE(out0.at("id").is_number_integer());

    string file_prefix = "/tmp/layer" + to_string(fullyConnected.getId());
    EXPECT_FALSE(fullyConnectedJ.at("weights_tensor").get<string>().empty());
    EXPECT_EQ(fullyConnectedJ.at("weights_tensor").get<string>(), file_prefix + "_weights.gds");
    if (hasBias) {
        EXPECT_FALSE(fullyConnectedJ.at("biases_tensor").get<string>().empty());
        EXPECT_EQ(fullyConnectedJ.at("biases_tensor").get<string>(), file_prefix + "_biases.gds");
    }

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", fullyConnectedJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    // FIXME: Why does this pass when there is a batch norm? Its output tensor which is FC's input tensor should not be found then.
    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(fullyConnectedJ, &newNetwork);
    Layer::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    stampedNetwork = newNetwork.getStampedNetwork(0);
    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::FullyConnected> physicalFCLayerDes =
        dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
    ASSERT_TRUE(physicalFCLayerDes != nullptr);

    ThorImplementation::Tensor weightsDes = physicalFCLayerDes->getWeights();
    ThorImplementation::Tensor weightsCpuDes = weightsDes.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    weightsCpuDes.copyFromAsync(weightsDes, stream);

    ThorImplementation::Tensor biasesDes;
    ThorImplementation::Tensor biasesCpuDes;
    if (hasBias) {
        biasesDes = physicalFCLayerDes->getBiases();
        biasesCpuDes = biasesDes.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
        biasesCpuDes.copyFromAsync(biasesDes, stream);
    }

    stream.synchronize();

    ASSERT_NE(weightsDes, weights);
    ASSERT_EQ(weightsDes.getDimensions(), weights.getDimensions());
    ASSERT_EQ(weightsDes.getDataType(), weights.getDataType());
    ASSERT_TRUE(weightsDes.getPlacement() == weights.getPlacement());

    half *weightsCpuMemDes = (half *)weightsCpuDes.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        ASSERT_EQ(weightsCpuMemDes[i], half(i));
    }

    if (hasBias) {
        ASSERT_NE(biasesDes, biases);
        ASSERT_EQ(biasesDes.getDimensions(), biases.getDimensions());
        ASSERT_EQ(biasesDes.getDataType(), biases.getDataType());
        ASSERT_TRUE(biasesDes.getPlacement() == biases.getPlacement());

        half *biasesCpuMemDes = (half *)biasesCpuDes.getMemPtr();
        for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
            ASSERT_EQ(biasesCpuMemDes[i], half(i * i + 6));
        }
    }
}
