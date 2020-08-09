#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

class Reshape : public Layer {
   public:
    virtual ~Reshape() {}

    Reshape(vector<unsigned long> newDimensions) : newDimensions(newDimensions) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        // They share the same memory that stores the elements but their elements are organized into different dimensions
        Tensor outputTensor = featureInput;
        outputTensor.reshape(newDimensions);
        return outputTensor;
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // No op
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        // No Op
    }

   private:
    vector<unsigned long> newDimensions;
};
