#include "DeepLearning/Api/Layers/Learning/Inception.h"

// FIXME: build this raw in implementation layer as a performance optimization (no need for concatenate layer and associated memory)
// FIXME: Instead of that, optimize concatenate to rewrite its input tensors memory locations, so that concatenate is a no op.

using namespace Thor;
using namespace std;

void Inception::buildSupportLayersAndAddToNetwork() {
    vector<Tensor> currentFeatureInputs;

    // Force the input tensor to this type of layer to be FP16
    if (featureInputs.front().getDataType() != Tensor::DataType::FP16) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            TypeConverter typeConverter = TypeConverter::Builder()
                                              .network(*network)
                                              .featureInput(currentFeatureInputs[i])
                                              .newDataType(Tensor::DataType::FP16)
                                              .build();
            currentFeatureInputs[i] = typeConverter.getFeatureOutput();
        }
    }

    Convolution2d::Builder convolution1x1Builder;
    convolution1x1Builder.network(*network)
        .numOutputChannels(outputChannels1x1)
        .filterHeight(1)
        .filterWidth(1)
        .verticalStride(1)
        .horizontalStride(1)
        .samePadding()
        .hasBias(true)
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
        .activationBuilder(Relu::Builder());

    Concatenate::Builder concatenateBuilder;
    concatenateBuilder.network(*network).concatenationAxis(0);

    vector<Pooling> poolingLayers;

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        convolution1x1Builder.featureInput(currentFeatureInputs[i]);
        convolution3x3ReduceBuilder.featureInput(currentFeatureInputs[i]);
        convolution5x5ReduceBuilder.featureInput(currentFeatureInputs[i]);

        Pooling::Builder pooling3x3Builder;
        Pooling pooling = Pooling::Builder()
                              .network(*network)
                              .featureInput(currentFeatureInputs[i])
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
        featureOutputs.push_back(concatenateLayers[i].getFeatureOutput());
        outputTensorFromInputTensor[currentFeatureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = currentFeatureInputs[i];
    }
}
