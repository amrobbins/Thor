#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"

using namespace Thor;

void Convolution2d::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        Tensor currentFeatureInput = featureInputs[i];

        if (useBatchNormalization) {
            BatchNormalization::Builder builder;
            builder.network(*network);
            if (batchNormExponentialRunningAverageFactor.isPresent())
                builder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor.get());
            if (batchNormEpsilon.isPresent())
                builder.epsilon(batchNormEpsilon.get());
            BatchNormalization batchNormalization = builder.featureInput(currentFeatureInput).build();
            currentFeatureInput = batchNormalization.getFeatureOutput();
        }

        if (dropProportion > 0.0f) {
            DropOut dropOut = DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInput).build();
            currentFeatureInput = dropOut.getFeatureOutput();
        }

        Convolution2d onlyConvolution = Builder()
                                            .network(*network)
                                            .featureInput(currentFeatureInput)
                                            .numOutputChannels(numOutputChannels)
                                            .filterHeight(filterHeight)
                                            .filterWidth(filterWidth)
                                            .verticalStride(verticalStride)
                                            .horizontalStride(horizontalStride)
                                            .verticalPadding(verticalPadding)
                                            .horizontalPadding(horizontalPadding)
                                            .hasBias(hasBias)
                                            .weightsInitializer(weightsInitializer)
                                            .biasInitializer(biasInitializer)
                                            .activationBuilder(Optional<Activation::Builder>::empty())
                                            .build();
        currentFeatureInput = onlyConvolution.getFeatureOutput();
        /*
                if (activationBuilder.isPresent()) {
                    activationBuilder.get().network(*network);
                    activationBuilder.get().featureInput(currentFeatureInputs[i]);
                    shared_ptr<Layer> activation = activationBuilder->build();
                    currentFeatureInput = activation->getFeatureOutput();
                }
        */

        Tensor finalFeatureOutput = currentFeatureInput;
        featureOutputs.push_back(finalFeatureOutput);
        outputTensorFromInputTensor[featureInputs[i]] = finalFeatureOutput;
        inputTensorFromOutputTensor[finalFeatureOutput] = featureInputs[i];
    }
}
