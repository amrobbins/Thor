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

    // Note that while alexnet is a straight forward example, this framework allows arbitrarily complex networks.
    // For example the output tensor from one layer may be connected as the input tensor to any number of other layers,
    // in this case the work for each branch will be scheduled in parallel.

    // Utilitiy layers like concatenate, reshape, typeConversion, etc are provided.

    // Layers are not required to learn or back propagate error. For example an GPU based FFT layer may be useful to put inline with the
    // neural network to pre-process audio input signals.
}

int main() {
    srand(time(NULL));

    Network alexNet = buildAlexnet();

    FileSystemLoader fileSystemLoader = FileSystemLoader::Buider()
                                            .trainingData("/home/andrew/ImageNet/train")
                                            .validationData("/home/andrew/ImageNet/validation")
                                            .testData("/home/andrew/ImageNet/test")
                                            .build();

    HyperparameterController learningRateController =
        HyperparameterController::Builder().learningRateStart(0.01).learningRateMin(0.0001).uniformReductionPerEpoch();

    // Also want to create dashboards for viewing in a browser as other types of visualizers
    Visualizer consoleVisualizer = ConsoleVisualizer::Builder().build();

    LocalExecutor executor = LocalExecutor::Builder()
                                 .network(alexNet)
                                 .loader(fileSystemLoader)
                                 .hyperparameterController(learningRateController)
                                 .visualizer(consoleVisualizer)
                                 .build();

    executor.trainEpochs(50);
    executor.createSnapshot("/home/andrew/ImageNet/trainedNetworks");

    // Also need to offer:
    //
    // AwsExecutor executor = AwsExecutor::Builder()...
    //
    // For training business-sized data sets.

    // The user can create a custom version of anything that is constructed using a builder interface.
    // For example some new type of neural network layer, or a custom visualizer, or a training parameter controller.
    // This is meant to promote encapsulation, ease of use, and IP creation for reuse across the organization.

    return 0;
}
