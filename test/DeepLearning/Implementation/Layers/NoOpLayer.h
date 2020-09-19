#pragma once

#include "Thor.h"

class NoOpLayer : public ThorImplementation::Layer {
   public:
    NoOpLayer() {}

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
