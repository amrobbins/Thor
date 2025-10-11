#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

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

TEST(FullyConnected, Serializes) {
    srand(time(nullptr));

    Network network;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;

    bool use_batch_norm = rand() % 2;

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    FullyConnected::Builder fullyConnectedBuilder = FullyConnected::Builder()
                                                        .network(network)
                                                        .featureInput(networkInput.getFeatureOutput())
                                                        .numOutputFeatures(numOutputFeatures)
                                                        .hasBias(hasBias)
                                                        .dropOut(dropProportion);
    if (use_batch_norm) {
        fullyConnectedBuilder.batchNormalization();
    }
    FullyConnected fullyConnected = fullyConnectedBuilder.build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(network)
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

    // Now stamp the nework and test serialization
    uint32_t batchSize = 1 + (rand() % 16);
    network.place(batchSize);
    Stream stream(0);
    json j = fullyConnected.serialize(stream);

    ASSERT_EQ(j["version"], "1.0.0");
    ASSERT_EQ(j["layer_type"], "fully_connected");

    EXPECT_TRUE(j.contains("num_output_features"));
    EXPECT_TRUE(j.contains("has_bias"));
    EXPECT_FALSE(j.contains("activation"));
    EXPECT_FALSE(j.contains("drop_out"));
    EXPECT_FALSE(j.contains("batch_normalization"));
    EXPECT_FALSE(j.contains("activation"));
    EXPECT_EQ(j.contains("biases_tensor"), hasBias);
    EXPECT_TRUE(j.contains("weights_tensor"));
    EXPECT_TRUE(j.contains("inputs"));
    EXPECT_TRUE(j.contains("outputs"));

    ASSERT_TRUE(j.at("num_output_features").is_number_integer());
    ASSERT_TRUE(j.at("has_bias").is_boolean());
    ASSERT_TRUE(j.at("weights_tensor").is_string());
    ASSERT_TRUE(j.at("inputs").is_array());
    ASSERT_TRUE(j.at("outputs").is_array());

    EXPECT_EQ(j.at("num_output_features").get<uint32_t>(), numOutputFeatures);
    EXPECT_EQ(j.at("has_bias").get<bool>(), hasBias);

    const auto &inputs = j.at("inputs");
    ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
    const auto &in0 = inputs.at(0);
    ASSERT_TRUE(in0.is_object());
    ASSERT_TRUE(in0.at("data_type").is_string());
    EXPECT_EQ(in0.at("data_type").get<std::string>(), "fp16");

    ASSERT_TRUE(in0.at("dimensions").is_array());
    ASSERT_EQ(in0.at("dimensions").size(), 1U);
    EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);

    ASSERT_TRUE(in0.at("id").is_number_integer());

    const auto &outputs = j.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<std::string>(), "fp16");

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), 1U);
    EXPECT_TRUE(out0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(out0.at("dimensions").at(0).get<uint32_t>(), numOutputFeatures);

    ASSERT_TRUE(out0.at("id").is_number_integer());

    EXPECT_FALSE(j.at("weights_tensor").get<std::string>().empty());
    if (hasBias) {
        EXPECT_FALSE(j.at("biases_tensor").get<std::string>().empty());
    }

    printf("%s\n", j.dump(4).c_str());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
