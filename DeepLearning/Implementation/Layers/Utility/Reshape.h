#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Reshape : public Layer {
   public:
    virtual ~Reshape() {}

    Reshape(std::vector<unsigned long> outputDimensions) : outputDimensions(outputDimensions) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());

        // When the first dimension is 0, this is a flag to use the batch size from the incoming featureInput
        if (outputDimensions[0] == 0)
            outputDimensions[0] = featureInput.get().getDimensions()[0];

        // They share the same memory that stores the elements but their elements are organized into different dimensions
        Tensor outputTensor = featureInput;
        outputTensor.reshape(outputDimensions);
        return outputTensor;
    }

    virtual void postCompile() {
        // ErrorInput to the previous layer is the errorInput coming to this layer,
        // then backProp is a no op
        if (errorInput.isPresent() && errorOutput.isPresent() && previousLayer.isPresent()) {
            previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
        }
        errorOutput = errorInput;
        Layer::postCompile();
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // No op
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        // No op
    }

   private:
    std::vector<unsigned long> outputDimensions;
};

}  // namespace ThorImplementation
