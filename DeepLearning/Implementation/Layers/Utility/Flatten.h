#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Flatten : public Layer {
   public:
    ~Flatten() override {}

    Flatten(unsigned int toNumDimensions) { this->toNumDimensions = toNumDimensions; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());

        std::vector<unsigned long> originalDimensions = featureInput.value().getDescriptor().getDimensions();
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
        Tensor outputTensor = featureInput.value();
        outputTensor.reshape(dimensions);
        return outputTensor;
    }

    void postCompile() override {
        // ErrorInput to the previous layer is the errorInput coming to this layer,
        // then backProp is a no op
        if (errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
        }
        errorOutput = errorInput;
        Layer::postCompile();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        // No Op, the output tensor is the same memory as the input tensor, but has a different tensor descriptor representing a flattened
        // output tensor
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        // No Op
    }

   private:
    bool uninitialized;

    unsigned int toNumDimensions;
};

}  // namespace ThorImplementation
