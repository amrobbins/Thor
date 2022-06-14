#pragma once

#include "Thor.h"

/**
 * This layer is useful in unit testing to prevent pruning dangling backwards path tensors.
 * This layer is not meant to be used outside of unit testing.
 */
class NoOpLayer : public ThorImplementation::Layer {
   public:
    NoOpLayer() {}

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(!compiled);

        assert(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<ThorImplementation::Tensor>::empty();

        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, shouldConnectToBackPropErrorIn(), loaderConnectionType);

        if (errorInput.isPresent() && featureOutput.isPresent()) {
            assert(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            assert(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual void infer(Optional<ThorImplementation::Tensor> inputTensor, Optional<ThorImplementation::Tensor> outputTensor, Stream stream) {
        if (outputTensor.isPresent()) {
            assert(inputTensor.isPresent());
            outputTensor.get().copyFromAsync(inputTensor, stream);
        }
    }

    virtual void backProp(Optional<ThorImplementation::Tensor> dataIn,
                          Optional<ThorImplementation::Tensor> errorIn,
                          Optional<ThorImplementation::Tensor> errorOut,
                          Stream stream) {
        if (errorOut.isPresent()) {
            assert(errorIn.isPresent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }
};
