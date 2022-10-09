#include "DeepLearning/Api/ExampleNetworks/DeepFullyConnected.h"

using namespace Thor;

Network buildDeepFullyConnected() {
    Network deepFullyConnected;
    deepFullyConnected.setNetworkName("DeepFullyConnected");

    vector<uint64_t> expectedDimensions;

    Glorot::Builder glorot = Glorot::Builder();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(deepFullyConnected)
                                   .name("examples")
                                   .dimensions({3, 224, 224})
                                   .dataType(Tensor::DataType::UINT8)
                                   .build();

    Tensor latestOutputTensor;

    // Input tensor is automatically flattened when sent to a fully connected layer.
    latestOutputTensor = FullyConnected::Builder()
                             .network(deepFullyConnected)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputFeatures(512)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {512};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    for (uint32_t i = 0; i < 50; ++i) {
        latestOutputTensor = FullyConnected::Builder()
                                 .network(deepFullyConnected)
                                 .featureInput(latestOutputTensor)
                                 .numOutputFeatures(8192)
                                 .hasBias(true)
                                 .weightsInitializerBuilder(glorot)
                                 .biasInitializerBuilder(glorot)
                                 .build()
                                 .getFeatureOutput();
        expectedDimensions = {8192};
        assert(latestOutputTensor.getDimensions() == expectedDimensions);
    }

    latestOutputTensor = FullyConnected::Builder()
                             .network(deepFullyConnected)
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
                              .network(deepFullyConnected)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(Tensor::DataType::FP16)
                              .build()
                              .getFeatureOutput();

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
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(deepFullyConnected)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    return deepFullyConnected;
}
