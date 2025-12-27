#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class GradientRivet : public Layer {
   public:
    virtual ~GradientRivet() {}

    GradientRivet() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        featureOutput = featureInput.get().clone();
        return featureOutput;
    }

    virtual void compile() {}

    virtual void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) {}

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(!compiled);

        assert(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<Tensor>::empty();

        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, true, loaderConnectionType);

        if (errorInput.isPresent() && featureOutput.isPresent()) {
            assert(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            assert(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {}

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {}
};

}  // namespace ThorImplementation
