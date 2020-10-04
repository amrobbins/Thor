#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"

using namespace Thor;

void Convolution2d::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    BatchNormalization::Builder batchNormBuilder;
    if (useBatchNormalization) {
        batchNormBuilder.network(*network);
        if (batchNormExponentialRunningAverageFactor.isPresent())
            batchNormBuilder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor.get());
        if (batchNormEpsilon.isPresent())
            batchNormBuilder.epsilon(batchNormEpsilon.get());
    }

    Convolution2d::Builder convolution2dBuilder;
    convolution2dBuilder.network(*network)
        .numOutputChannels(numOutputChannels)
        .filterHeight(filterHeight)
        .filterWidth(filterWidth)
        .verticalStride(verticalStride)
        .horizontalStride(horizontalStride)
        .verticalPadding(verticalPadding)
        .horizontalPadding(horizontalPadding)
        .hasBias(hasBias)
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
        .noActivation();

    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    if (useBatchNormalization) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            batchNormBuilder.featureInput(currentFeatureInputs[i]);
        }
        BatchNormalization batchNormalization = batchNormBuilder.build();
        for (uint32_t i = 0; i < featureInputs.size(); ++i)
            currentFeatureInputs[i] = batchNormalization.getFeatureOutputs()[i];
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            DropOut dropOut =
                DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput();
        }
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        convolution2dBuilder.featureInput(currentFeatureInputs[i]);
    Convolution2d convolution2d = convolution2dBuilder.build();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs[i] = convolution2d.getFeatureOutputs()[i];

    if (activationBuilder) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            activationBuilder->network(*network);
            activationBuilder->featureInput(currentFeatureInputs[i]);
            shared_ptr<Layer> activation = activationBuilder->build();
            currentFeatureInputs[i] = activation->getFeatureOutput();
        }
    }

    // Replace the outputs on the compound layer to be the outputs of the last stage
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        convolution2d.featureOutputs[i] = currentFeatureInputs[i];
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}
