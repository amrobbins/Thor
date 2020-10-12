#include "DeepLearning/Api/Layers/Learning/Inception.h"

// FIXME: build this raw in implementation layer as a performance optimization (no need for concatenate layer and associated memory)

using namespace Thor;

void Inception::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    Convolution2d::Builder convolution1x1Builder;
    convolution1x1Builder.network(*network)
        .numOutputChannels(outputChannels1x1)
        .filterHeight(1)
        .filterWidth(1)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());

    Convolution2d::Builder convolution3x3ReduceBuilder;
    convolution3x3ReduceBuilder.network(*network)
        .numOutputChannels(inputChannels3x3)
        .filterHeight(1)
        .filterWidth(1)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());
    Convolution2d::Builder convolution3x3Builder;
    convolution3x3Builder.network(*network)
        .numOutputChannels(outputChannels3x3)
        .filterHeight(3)
        .filterWidth(3)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());

    Convolution2d::Builder convolution5x5ReduceBuilder;
    convolution5x5ReduceBuilder.network(*network)
        .numOutputChannels(inputChannels5x5)
        .filterHeight(1)
        .filterWidth(1)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());
    Convolution2d::Builder convolution5x5Builder;
    convolution5x5Builder.network(*network)
        .numOutputChannels(outputChannels5x5)
        .filterHeight(3)
        .filterWidth(3)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());

    Convolution2d::Builder convolutionPoolingReduceBuilder;
    convolutionPoolingReduceBuilder.network(*network)
        .numOutputChannels(outputChannelsPooling)
        .filterHeight(1)
        .filterWidth(1)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
        .biasInitializerBuilder(uniformRandomInitializerBuilder)
        .activationBuilder(Relu::Builder());

    Concatenate::Builder concatenateBuilder;
    concatenateBuilder.network(*network).concatenationAxis(1);

    vector<Pooling> poolingLayers;

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        convolution1x1Builder.featureInput(featureInputs[i]);
        convolution3x3ReduceBuilder.featureInput(featureInputs[i]);
        convolution5x5ReduceBuilder.featureInput(featureInputs[i]);

        Pooling::Builder pooling3x3Builder;
        Pooling pooling = Pooling::Builder()
                              .network(*network)
                              .featureInput(featureInputs[i])
                              .type(Pooling::Type::MAX)
                              .windowHeight(3)
                              .windowWidth(3)
                              .verticalStride(1)
                              .horizontalStride(1)
                              .samePadding()
                              .build();
        poolingLayers.push_back(pooling);
    }

    Convolution2d convolution1x1 = convolution1x1Builder.build();
    Convolution2d convolution3x3Reduce = convolution3x3ReduceBuilder.build();
    Convolution2d convolution5x5Reduce = convolution5x5ReduceBuilder.build();

    Convolution2d convolution3x3;
    Convolution2d convolution5x5;
    Convolution2d convolutionPoolingReduce;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        convolution3x3 = convolution3x3Builder.featureInput(convolution3x3Reduce.getFeatureOutputs()[i]).build();
        convolution5x5 = convolution5x5Builder.featureInput(convolution5x5Reduce.getFeatureOutputs()[i]).build();
        convolutionPoolingReduce = convolutionPoolingReduceBuilder.featureInput(poolingLayers[i].getFeatureOutput()).build();
    }

    vector<Concatenate> concatenateLayers;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        Concatenate concatenate = concatenateBuilder.featureInput(convolution1x1.getFeatureOutputs()[i])
                                      .featureInput(convolution3x3.getFeatureOutputs()[i])
                                      .featureInput(convolution5x5.getFeatureOutputs()[i])
                                      .featureInput(convolutionPoolingReduce.getFeatureOutputs()[i])
                                      .build();
        concatenateLayers.push_back(concatenate);
    }

    // Replace the outputs on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual outputs of the compound layer,
    // these are not necessarily the outputs of the stand-alone convolution layer.
    // Network uses single layers, user uses compound layer.
    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        featureOutputs[i] = concatenateLayers[i].getFeatureOutput();
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}
