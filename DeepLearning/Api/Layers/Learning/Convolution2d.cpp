#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"

void Convolution2d::toSingleLayers(vector<shared_ptr<Layer>> &singleLayers) const {
    if (isMultiLayer()) {
        Tensor currentFeatureInput = getFeatureInput();

        if (useBatchNormalization) {
            BatchNormalization::Builder builder;
            if (batchNormExponentialRunningAverageFactor.isPresent())
                builder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor.get());
            if (batchNormEpsilon.isPresent())
                builder.epsilon(batchNormEpsilon.get());
            BatchNormalization batchNormalization = builder.featureInput(currentFeatureInput).build();
            currentFeatureInput = batchNormalization.getFeatureOutput();
            singleLayers.push_back(batchNormalization.clone());
        }

        if (dropProportion > 0.0f) {
            DropOut dropOut = DropOut::Builder().dropProportion(dropProportion).featureInput(currentFeatureInput).build();
            currentFeatureInput = dropOut.getFeatureOutput();
            singleLayers.push_back(dropOut.clone());
        }

        Convolution2d onlyConvolution = Builder()
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
                                            .build();
        currentFeatureInput = onlyConvolution.getFeatureOutput();
        singleLayers.push_back(onlyConvolution.clone());

        if (activation.isPresent()) {
            singleLayers.push_back(activation.get().cloneWithReconnect(currentFeatureInput));
        }
    } else {
        singleLayers.push_back(clone());
    }
}
