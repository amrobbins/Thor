#include "DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.h"

using namespace Thor;

Network buildSingleLayerFullyConnected() {
    Network singleLayerFullyConnected;
    singleLayerFullyConnected.setNetworkName("SingleLayerFullyConnected");

    vector<uint64_t> expectedDimensions;

    Glorot::Builder glorot = Glorot::Builder();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(singleLayerFullyConnected)
                                   .name("examples")
                                   .dimensions({28 * 28})
                                   .dataType(Tensor::DataType::FP32)
                                   .build();

    Tensor latestOutputTensor;

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerFullyConnected)
                             .featureInput(imagesInput.getFeatureOutput())
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
                              .network(singleLayerFullyConnected)
                              .name("labels")
                              .dimensions({10})
                              .dataType(Tensor::DataType::UINT8)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropyLoss lossLayer = CategoricalCrossEntropyLoss::Builder()
                                                .network(singleLayerFullyConnected)
                                                .featureInput(latestOutputTensor)
                                                .labels(labelsTensor)
                                                .lossType(ThorImplementation::Loss::ConnectionType::BATCH_LOSS)
                                                .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(singleLayerFullyConnected)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(singleLayerFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer = CategoricalAccuracy::Builder()
                                            .network(singleLayerFullyConnected)
                                            .predictions(lossLayer.getPredictions())
                                            .labels(labelsTensor)
                                            .build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(singleLayerFullyConnected)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(Tensor::DataType::FP32)
                                 .build();

    // Return the assembled network
    return singleLayerFullyConnected;
}
