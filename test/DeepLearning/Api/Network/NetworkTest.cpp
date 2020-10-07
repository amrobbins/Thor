#include "Thor.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using std::set;
using std::vector;

using namespace Thor;

// This version of alexnet is modified from the original paper by stacking the two convolution paths channelwise
Network buildAlexNet() {
    Network alexNet;

    Tensor latestOutputTensor;

    vector<uint64_t> expectedDimensions, actualDimensions;

    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    latestOutputTensor =
        NetworkInput::Builder().network(alexNet).dimensions({3, 224, 224}).dataType(Tensor::DataType::UINT8).build().getFeatureOutput();

    // For Convolution and FullyConnected layers, batchNormalization, dropOut and activation may be applied,
    // this is specified using builder parameters.
    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(96)
                             .filterHeight(11)
                             .filterWidth(11)
                             .verticalStride(4)
                             .horizontalStride(4)
                             .noPadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = Pooling::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(2)
                             .windowWidth(2)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {96, 27, 27};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(256)
                             .filterHeight(5)
                             .filterWidth(5)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .samePadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = Pooling::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(2)
                             .windowWidth(2)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {256, 13, 13};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(384)
                             .filterHeight(3)
                             .filterWidth(3)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .samePadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(384)
                             .filterHeight(3)
                             .filterWidth(3)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .samePadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(256)
                             .filterHeight(3)
                             .filterWidth(3)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .samePadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = Pooling::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(2)
                             .windowWidth(2)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {256, 6, 6};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .dropOut(0.5)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {4096};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .dropOut(0.5)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(1000)
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {1000};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor =
        NetworkInput::Builder().network(alexNet).dimensions({1000}).dataType(Tensor::DataType::FP32).build().getFeatureOutput();

    CategoricalCrossEntropyLoss lossLayer =
        CategoricalCrossEntropyLoss::Builder().network(alexNet).featureInput(latestOutputTensor).labels(labelsTensor).build();

    latestOutputTensor = lossLayer.getFeatureInput();
    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions =
        NetworkOutput::Builder().network(alexNet).inputTensor(lossLayer.getPredictions()).dataType(Tensor::DataType::FP32).build();
    NetworkOutput loss =
        NetworkOutput::Builder().network(alexNet).inputTensor(lossLayer.getLoss()).dataType(Tensor::DataType::FP32).build();

    return alexNet;
}

void checkSimplyConnectedNetwork(vector<Layer> network, ThorImplementation::StampedNetwork stampedNetwork) {
    for (uint32_t i = 0; i < network.size(); ++i) {
    }
}

TEST(Network, SimplestNetworkProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput networkInput = NetworkInput::Builder().network(network).dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    latestOutputTensor = networkInput.getFeatureOutput();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(latestOutputTensor)
                                        .numOutputFeatures(500)
                                        .hasBias(true)
                                        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                        .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                        .noActivation()
                                        .build();
    latestOutputTensor = fullyConnected.getFeatureOutput();

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).inputTensor(latestOutputTensor).dataType(Tensor::DataType::FP16).build();
    Tensor networkOutputTensor = networkOutput.getFeatureOutput();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkInput.getId()]->getFeatureOutput().get());
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 1u);
    ThorImplementation::FullyConnected *fc =
        dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.apiLayerToPhysicalLayer[fullyConnected.getId()]);
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs()[0].get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs()[0].get(), fc->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(fc->getFeatureOutputs()[0].get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.outputs[0]->getFeatureInput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkOutput.getId()]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 0u);

    stampedNetwork.clear();
}

TEST(Network, SimpleNetworkWithCompoundLayerProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    latestOutputTensor =
        NetworkInput::Builder().network(network).dimensions({1024}).dataType(Tensor::DataType::UINT8).build().getFeatureOutput();
    latestOutputTensor = FullyConnected::Builder()
                             .network(network)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(500)
                             .hasBias(true)
                             .activationBuilder(Relu::Builder())
                             .dropOut(0.5)
                             .batchNormalization()
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .build()
                             .getFeatureOutput();
    latestOutputTensor =
        DropOut::Builder().network(network).featureInput(latestOutputTensor).dropProportion(0.25).build().getFeatureOutput();
    Tensor networkOutputTensor = NetworkOutput::Builder()
                                     .network(network)
                                     .inputTensor(latestOutputTensor)
                                     .dataType(Tensor::DataType::UINT8)
                                     .build()
                                     .getFeatureOutput();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // FIXME: Verify

    stampedNetwork.clear();
}

TEST(Network, AlexnetIsProperlyFormed) {
    ThorImplementation::StampedNetwork stampedNetwork;

    Network alexNet = buildAlexNet();
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = alexNet.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // FIXME: Verify

    stampedNetwork.clear();
}

// FIXME: Create a network with branches, multiple inputs and multiple outputs

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
