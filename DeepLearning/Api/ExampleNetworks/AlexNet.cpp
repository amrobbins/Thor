#include "DeepLearning/Api/ExampleNetworks/AlexNet.h"

using namespace Thor;

Tensor buildAlexnetConvolutionalPath(Network &alexNet, NetworkInput imagesInput) {
    Tensor latestOutputTensor;

    vector<uint64_t> expectedDimensions;
    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputChannels(48)
                             .filterHeight(11)
                             .filterWidth(11)
                             .verticalStride(4)
                             .horizontalStride(4)
                             .horizontalPadding(2)
                             .verticalPadding(2)
                             .hasBias(true)
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .biasInitializerBuilder(uniformRandomInitializerBuilder)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {48, 55, 55};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(128)
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

    expectedDimensions = {128, 55, 55};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

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

    expectedDimensions = {128, 27, 27};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(192)
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

    expectedDimensions = {192, 27, 27};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

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

    expectedDimensions = {192, 13, 13};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(192)
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

    expectedDimensions = {192, 13, 13};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(alexNet)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(128)
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

    expectedDimensions = {128, 13, 13};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

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

    expectedDimensions = {128, 6, 6};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    return latestOutputTensor;
}

Network buildAlexNet() {
    Network alexNet;

    vector<uint64_t> expectedDimensions;

    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput imagesInput =
        NetworkInput::Builder().network(alexNet).name("examples").dimensions({3, 224, 224}).dataType(Tensor::DataType::UINT8).build();

    Tensor topPathTensor = buildAlexnetConvolutionalPath(alexNet, imagesInput);
    Tensor bottomPathTensor = buildAlexnetConvolutionalPath(alexNet, imagesInput);
    Tensor latestOutputTensor = Concatenate::Builder()
                                    .network(alexNet)
                                    .featureInput(topPathTensor)
                                    .featureInput(bottomPathTensor)
                                    .concatenationAxis(0)
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
                             .noActivation()
                             .build()
                             .getFeatureOutput();

    expectedDimensions = {1000};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(alexNet)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(Tensor::DataType::FP16)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropyLoss lossLayer =
        CategoricalCrossEntropyLoss::Builder().network(alexNet).featureInput(latestOutputTensor).labels(labelsTensor).build();

    latestOutputTensor = lossLayer.getFeatureInput();
    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(alexNet)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss =
        NetworkOutput::Builder().network(alexNet).name("loss").inputTensor(lossLayer.getLoss()).dataType(Tensor::DataType::FP32).build();

    return alexNet;
}
