#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Reshape : public Layer {
   public:
    ~Reshape() override {}

    Reshape(std::vector<unsigned long> outputDimensions) : outputDimensions(outputDimensions) {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());

        // When the first dimension is 0, this is a flag to use the batch size from the incoming featureInput
        if (outputDimensions[0] == 0)
            outputDimensions[0] = featureInput.value().getDimensions()[0];

        // They share the same memory that stores the elements but their elements are organized into different dimensions
        Tensor outputTensor = featureInput.value();
        outputTensor.reshape(outputDimensions);
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
        // No op
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        // No op
    }

   private:
    std::vector<unsigned long> outputDimensions;
};

}  // namespace ThorImplementation
