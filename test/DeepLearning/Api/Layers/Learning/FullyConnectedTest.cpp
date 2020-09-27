#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

/*
    virtual FullyConnected::Builder &network(Network &_network) {
    virtual FullyConnected::Builder featureInput(Tensor _featureInput) {
    virtual FullyConnected::Builder numOutputFeatures(uint32_t _numOutputFeatures) {
    virtual FullyConnected::Builder hasBias(bool _hasBias) {
    virtual FullyConnected::Builder weightsInitializer(Initializer _weightsInitializer) {
    virtual FullyConnected::Builder biasInitializer(Initializer _biasInitializer) {
    virtual FullyConnected::Builder activation(Optional<Activation> _activation) {
    virtual FullyConnected::Builder batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
   Optional<double> epsilon = Optional<double>::empty()) { virtual FullyConnected::Builder dropOut(float _dropProportion) {
*/

TEST(FullyConnectedSingleFeatureInput, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    UniformRandomInitializer uniformRandomInitializer;
    XavierInitializer xavierInitializer;
    Tanh::Builder tanhBuilder;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
    double epsilon = (1 + (rand() % 1000)) / 1000.0f;

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(featureInput)
                                        .numOutputFeatures(numOutputFeatures)
                                        .hasBias(hasBias)
                                        .weightsInitializer(uniformRandomInitializer)
                                        .biasInitializer(xavierInitializer)
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
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
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
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(fullyConnected.getId(), clone->getId());
    ASSERT_GT(fullyConnected.getId(), 1u);

    ASSERT_TRUE(fullyConnected == *clone);
    ASSERT_FALSE(fullyConnected != *clone);
    ASSERT_FALSE(fullyConnected > *clone);
    ASSERT_FALSE(fullyConnected < *clone);
}

/*

TEST(FullyConnectedMultipleFeatureInputs, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions0 = 1 + rand() % 6;
    for (int i = 0; i < numDimensions0; ++i)
        dimensions.push_back(1 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor featureInput0(dataType, dimensions);
    Tensor featureInput1(dataType, dimensions);

    double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;

    double epsilon = (1 + (rand() % 100)) / 100000.0f;

    FullyConnected batchNormalization = FullyConnected::Builder()
                                                .network(network)
                                                .featureInput(featureInput0)
                                                .featureInput(featureInput1)
                                                .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
                                                .epsilon(epsilon)
                                                .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
    vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
    ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double actualEpsilon = batchNormalization.getEpsilon();
    ASSERT_EQ(actualEpsilon, epsilon);

    shared_ptr<Layer> cloneLayer = batchNormalization.clone();
    FullyConnected *clone = dynamic_cast<FullyConnected *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    featureInputs.clear();
    featureOutputs.clear();
    featureInputs = clone->getFeatureInputs();
    featureOutputs = clone->getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
    ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double cloneEpsilon = clone->getEpsilon();
    ASSERT_EQ(cloneEpsilon, epsilon);

    ASSERT_EQ(batchNormalization.getId(), clone->getId());
    ASSERT_GT(batchNormalization.getId(), 1u);

    ASSERT_TRUE(batchNormalization == *clone);
    ASSERT_FALSE(batchNormalization != *clone);
    ASSERT_FALSE(batchNormalization > *clone);
    ASSERT_FALSE(batchNormalization < *clone);
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
