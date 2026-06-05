#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/ExampleNetworks/FewLayerFullyConnected.h"

using namespace Thor;
using namespace std;

Network buildFewLayerFullyConnected() {
    Network fewLayerFullyConnected("FewLayerFullyConnected");

    vector<uint64_t> expectedDimensions;

    shared_ptr<Initializer> glorot = Glorot::Builder().build();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(fewLayerFullyConnected)
                                   .name("examples")
                                   .dimensions({3, 224, 224})
                                   .dataType(DataType::UINT8)
                                   .build();

    Tensor latestOutputTensor;

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(imagesInput.getFeatureOutput().value())
                             .numOutputFeatures(128)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {128};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(128)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {128};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(fewLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(1000)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {1000};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(fewLayerFullyConnected)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(DataType::FP16)
                              .build()
                              .getFeatureOutput().value();

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
                                    .dataType(DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(fewLayerFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer = CategoricalAccuracy::Builder()
                                            .network(fewLayerFullyConnected)
                                            .predictions(lossLayer.getPredictions())
                                            .labels(labelsTensor)
                                            .receivesOneHotLabels()
                                            .build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(fewLayerFullyConnected)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(DataType::FP32)
                                 .build();

    return fewLayerFullyConnected;
}
