#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Reshape : public Layer {
   public:
    virtual ~Reshape() {}

    Reshape(vector<unsigned long> outputDimensions) : outputDimensions(outputDimensions) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        // They share the same memory that stores the elements but their elements are organized into different dimensions
        Tensor outputTensor = featureInput;
        outputTensor.reshape(outputDimensions);
        return outputTensor;
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // No op
    }

    // FIXME: How to avoid the unnecessary copy
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        errorOut.get().copyFromAsync(errorIn.get(), stream);
    }

   private:
    vector<unsigned long> outputDimensions;
};

}  // namespace ThorImplementation
