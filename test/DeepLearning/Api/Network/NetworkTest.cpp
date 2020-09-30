#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

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

    // FIXME: initializer

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
                             .weightsInitializer(uniformRandomInitializer)
                             .biasInitializer(uniformRandomInitializer)
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

    assert(latestOutputTensor.getDimensions() == {96, 27, 27});

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(256)
                             .filterHeight(5)
                             .filterWidth(5)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .noPadding()
                             .hasBias(true)
                             .weightsInitializer(uniformRandomInitializer)
                             .biasInitializer(uniformRandomInitializer)
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

    assert(latestOutputTensor.getDimensions() == {256, 13, 13});

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
                             .weightsInitializer(uniformRandomInitializer)
                             .biasInitializer(uniformRandomInitializer)
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
                             .weightsInitializer(uniformRandomInitializer)
                             .biasInitializer(uniformRandomInitializer)
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
                             .weightsInitializer(uniformRandomInitializer)
                             .biasInitializer(uniformRandomInitializer)
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

    assert(latestOutputTensor.getDimensions() == {256, 6, 6});

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .dropOut(0.5)
                             .build()
                             .getFeatureOutput();

    assert(latestOutputTensor.getDimensions() == {4096});

    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .dropOut(0.5)
                             .build()
                             .getFeatureOutput();

    latestOutputTensor = FullyConnected::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(1000)
                             .hasBias(true)
                             .build()
                             .getFeatureOutput();

    assert(latestOutputTensor.getDimensions() == {1000});

    Tensor labelsTensor =
        NetworkInput::Builder().network(alexNet).dimensions({1000}).dataType(Tensor::DataType::FP32).build().getFeatureOutput();

    CategoricalCrossEntropyLoss lossLayer =
        CategoricalCrossEntropyLoss::Builder().network(alexNet).featureInput(latestOutputTensor).labels(labelsTensor).build();

    lossLayer.getFeatureInput(latestOutputTensor);
    lossLayer.getLabels(labelsTensor);

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(alexNet).inputTensor(lossLayer->getPredictions()).dataType(Tensor::DataType::FP32).build();
    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(alexNet).inputTensor(lossLayer->getLoss()).dataType(Tensor::DataType::FP32).build();

    return alexNet;
}

TEST(Network, AlexnetIsProperlyFormed) { Network alexNet = buildAlexnet(); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
