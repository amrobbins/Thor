#include "DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.h"

using namespace Thor;
using namespace std;

Network buildFewLayerFullyConnected() {
    Network fewLayerFullyConnected;
    fewLayerFullyConnected.setNetworkName("FewLayerFullyConnected");

    vector<uint64_t> expectedDimensions;

    Glorot::Builder glorot = Glorot::Builder();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(fewLayerFullyConnected)
                                   .name("examples")
                                   .dimensions({3, 224, 224})
                                   .dataType(Tensor::DataType::UINT8)
                                   .build();

    Tensor latestOutputTensor;

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(4096)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(1000)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {1000};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(fewLayerFullyConnected)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(Tensor::DataType::FP16)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(fewLayerFullyConnected)
                                            .predictions(latestOutputTensor)
                                            .labels(labelsTensor)
                                            .reportsBatchLoss()
                                            .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(fewLayerFullyConnected)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(fewLayerFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer =
        CategoricalAccuracy::Builder().network(fewLayerFullyConnected).predictions(lossLayer.getPredictions()).labels(labelsTensor).build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(fewLayerFullyConnected)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(Tensor::DataType::FP32)
                                 .build();

    return fewLayerFullyConnected;
}
