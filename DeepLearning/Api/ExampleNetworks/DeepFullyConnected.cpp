#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/ExampleNetworks/DeepFullyConnected.h"

using namespace Thor;
using namespace std;

Network buildDeepFullyConnected() {
    Network deepFullyConnected("DeepFullyConnected");

    vector<uint64_t> expectedDimensions;

    shared_ptr<Initializer> glorot = Glorot::Builder().build();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(deepFullyConnected)
                                   .name("examples")
                                   .dimensions({3, 224, 224})
                                   .dataType(DataType::UINT8)
                                   .build();

    Tensor latestOutputTensor;

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(deepFullyConnected)
                             .featureInput(imagesInput.getFeatureOutput().value())
                             .numOutputFeatures(512)
                             .hasBias(true)
                             .weightsInitializer(glorot)
                             .biasInitializer(glorot)
                             .build()
                             .getFeatureOutput().value();
    expectedDimensions = {512};
    THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);

    for (uint32_t i = 0; i < 50; ++i) {
        latestOutputTensor = FullyConnected::Builder()
                                 .network(deepFullyConnected)
                                 .featureInput(latestOutputTensor)
                                 .numOutputFeatures(8192)
                                 .hasBias(true)
                                 .weightsInitializer(glorot)
                                 .biasInitializer(glorot)
                                 .build()
                                 .getFeatureOutput().value();
        expectedDimensions = {8192};
        THOR_THROW_IF_FALSE(latestOutputTensor.getDimensions() == expectedDimensions);
    }

    latestOutputTensor = FullyConnected::Builder()
                             .network(deepFullyConnected)
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
                              .network(deepFullyConnected)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(DataType::FP16)
                              .build()
                              .getFeatureOutput().value();

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(deepFullyConnected)
                                            .predictions(latestOutputTensor)
                                            .labels(labelsTensor)
                                            .reportsBatchLoss()
                                            .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(deepFullyConnected)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(deepFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(DataType::FP32)
                             .build();

    return deepFullyConnected;
}
