#include "DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.h"

using namespace Thor;
using namespace std;

Network buildSingleLayerConvolution2d() {
    Network singleLayerConvolution2d;
    singleLayerConvolution2d.setNetworkName("singleLayerConvolution2d");

    vector<uint64_t> expectedDimensions;

    shared_ptr<Initializer> glorot = Glorot::Builder().build();

    Tensor latestOutputTensor;
    latestOutputTensor = NetworkInput::Builder()
                             .network(singleLayerConvolution2d)
                             .name("examples")
                             .dimensions({1, 28, 28})
                             .dataType(Tensor::DataType::FP32)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {1, 28, 28};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(128)
                             .filterHeight(25)
                             .filterWidth(25)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .noPadding()
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .activationBuilder(Relu::Builder())
                             .batchNormalization()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128, 4, 4};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(4)
                             .windowWidth(4)
                             .noPadding()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128, 1, 1};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(10)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .noActivation()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {10};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(singleLayerConvolution2d)
                              .name("labels")
                              .dimensions({10})
                              .dataType(Tensor::DataType::UINT8)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(singleLayerConvolution2d)
                                            .predictions(latestOutputTensor)
                                            .labels(labelsTensor)
                                            .reportsBatchLoss()
                                            .receivesOneHotLabels()
                                            .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(singleLayerConvolution2d)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(singleLayerConvolution2d)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer = CategoricalAccuracy::Builder()
                                            .network(singleLayerConvolution2d)
                                            .predictions(lossLayer.getPredictions())
                                            .labels(labelsTensor)
                                            .receivesOneHotLabels()
                                            .build();

    NetworkOutput accuracyOutput = NetworkOutput::Builder()
                                       .network(singleLayerConvolution2d)
                                       .name("accuracy")
                                       .inputTensor(accuracyLayer.getMetric())
                                       .dataType(Tensor::DataType::FP32)
                                       .build();

    // Return the assembled network
    return singleLayerConvolution2d;
}
