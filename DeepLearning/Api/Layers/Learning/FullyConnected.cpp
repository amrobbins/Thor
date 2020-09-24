#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

void FullyConnected::toSingleLayers(vector<shared_ptr<Layer>> &singleLayers) const {
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

        FullyConnected onlyFullyConnected = Builder()
                                                .featureInput(currentFeatureInput)
                                                .numOutputFeatures(numOutputFeatures)
                                                .hasBias(hasBias)
                                                .weightsInitializer(weightsInitializer)
                                                .biasInitializer(biasInitializer)
                                                .build();
        currentFeatureInput = onlyFullyConnected.getFeatureOutput();
        singleLayers.push_back(onlyFullyConnected.clone());

        if (activation.isPresent()) {
            singleLayers.push_back(activation.get().cloneWithReconnect(currentFeatureInput));
        }
    } else {
        singleLayers.push_back(clone());
    }
}
