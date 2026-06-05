#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/ExampleNetworks/SingleLayerFullyConnected.h"

using namespace Thor;
using namespace std;

Network buildSingleLayerFullyConnected() {
    Network singleLayerFullyConnected("SingleLayerFullyConnected");

    vector<uint64_t> expectedDimensions;

    shared_ptr<Activation> relu = Relu::Builder().build();
    shared_ptr<Initializer> glorot = Glorot::Builder().build();

    Tensor latestOutputTensor = NetworkInput::Builder()
                                    .network(singleLayerFullyConnected)
                                    .name("examples")
                                    .dimensions({28 * 28})
                                    .dataType(DataType::FP32)
                                    .build()
                                    .getFeatureOutput().value();

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(128)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .activation(relu)
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {128};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerFullyConnected)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(10)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .noActivation()
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {10};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(singleLayerFullyConnected)
                              .name("labels")
                              .dimensions({10})
                              .dataType(DataType::UINT8)
                              .build()
                              .getFeatureOutput().value();

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(singleLayerFullyConnected)
                                            .predictions(latestOutputTensor)
                                            .labels(labelsTensor)
                                            .reportsBatchLoss()
                                            .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(singleLayerFullyConnected)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(singleLayerFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer = CategoricalAccuracy::Builder()
                                            .network(singleLayerFullyConnected)
                                            .predictions(lossLayer.getPredictions())
                                            .labels(labelsTensor)
                                            .receivesOneHotLabels()
                                            .build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(singleLayerFullyConnected)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(DataType::FP32)
                                 .build();

    // Return the assembled network
    return singleLayerFullyConnected;
}
