#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Flatten : public Layer {
   public:
    ~Flatten() override {}

    Flatten(unsigned int toNumDimensions) { this->toNumDimensions = toNumDimensions; }

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());

        std::vector<unsigned long> originalDimensions = featureInput.get().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(toNumDimensions < originalDimensions.size());
        unsigned int d = 0;
        std::vector<unsigned long> dimensions;
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

    void postCompile() override {
        // ErrorInput to the previous layer is the errorInput coming to this layer,
        // then backProp is a no op
        if (errorInput.isPresent() && errorOutput.isPresent() && previousLayer.isPresent()) {
            previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
        }
        errorOutput = errorInput;
        Layer::postCompile();
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        // No Op, the output tensor is the same memory as the input tensor, but has a different tensor descriptor representing a flattened
        // output tensor
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        // No Op
    }

   private:
    bool uninitialized;

    unsigned int toNumDimensions;
};

}  // namespace ThorImplementation
