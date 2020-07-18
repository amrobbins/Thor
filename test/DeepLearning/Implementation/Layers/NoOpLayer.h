#pragma once

#include "MLDev.h"

class NoOpLayer : public Layer {
   public:
    NoOpLayer() {}

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        if (outputTensor.isPresent())
            assert(inputTensor.isPresent());

        if (outputTensor.isPresent())
            outputTensor.get().copyFromAsync(inputTensor, stream);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isPresent())
            assert(errorIn.isPresent());

        if (errorOut.isPresent()) {
            assert(errorIn.isPresent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }
};
