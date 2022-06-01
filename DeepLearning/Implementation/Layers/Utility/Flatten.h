#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Flatten : public Layer {
   public:
    virtual ~Flatten() {}

    Flatten(unsigned int toNumDimensions) { this->toNumDimensions = toNumDimensions; }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());

        vector<unsigned long> originalDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(toNumDimensions < originalDimensions.size());
        unsigned int d = 0;
        vector<unsigned long> dimensions;
        for (; d < toNumDimensions - 1; ++d) {
            dimensions.push_back(originalDimensions[d]);
        }
        unsigned long lastDimensionSize = 1;
        for (; d < originalDimensions.size(); ++d) {
            lastDimensionSize *= originalDimensions[d];
        }
        dimensions.push_back(lastDimensionSize);

        // They share the same memory that stores the elements but their elements are organized into different dimensions
        Tensor outputTensor = featureInput;
        outputTensor.reshape(dimensions);
        return outputTensor;
    }

    // FIXME: how to avoid needing a new tensor here?

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // No Op, the output tensor is the same memory as the input tensor, but has a different tensor descriptor representing a flattened
        // output tensor
    }

    // FIXME: How to avoid the unnecessary copy
    // FIXME: Reshape is broken still
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        errorOut.get().copyFromAsync(errorIn.get(), stream);
    }

   private:
    bool uninitialized;

    unsigned int toNumDimensions;
};

}  // namespace ThorImplementation
