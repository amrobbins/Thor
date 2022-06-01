#include "DeepLearning/Api/ExampleNetworks/SingleLayerConvolution2d.h"

using namespace Thor;

Network buildSingleLayerConvolution2d() {
    Network singleLayerConvolution2d;
    singleLayerConvolution2d.setNetworkName("singleLayerConvolution2d");

    vector<uint64_t> expectedDimensions;

    Glorot::Builder glorot = Glorot::Builder();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(singleLayerConvolution2d)
                                   .name("examples")
                                   .dimensions({1, 28, 28})
                                   .dataType(Tensor::DataType::FP32)
                                   .build();

    Tensor latestOutputTensor;
    latestOutputTensor = Convolution2d::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputChannels(128)
                             .filterHeight(28)
                             .filterWidth(28)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .noPadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128, 1, 1};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(10)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
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

    CategoricalCrossEntropyLoss lossLayer = CategoricalCrossEntropyLoss::Builder()
                                                .network(singleLayerConvolution2d)
                                                .featureInput(latestOutputTensor)
                                                .labels(labelsTensor)
                                                .lossType(ThorImplementation::Loss::ConnectionType::BATCH_LOSS)
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
                                            .build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(singleLayerConvolution2d)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(Tensor::DataType::FP32)
                                 .build();

    // Return the assembled network
    return singleLayerConvolution2d;
}
