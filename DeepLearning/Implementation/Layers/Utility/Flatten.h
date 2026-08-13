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

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        Layer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
        fuseBackwardAliasThroughMetadataOnlyReshape();
    }

    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        if (errorInput.has_value()) {
            THOR_THROW_IF_FALSE(oldErrorInput.value() == errorInput.value());
        }

        // Metadata-only reshape/flatten layers must preserve the original input
        // descriptor on the gradient passed upstream. TensorFanout may replace
        // this layer's downstream error tensor during compile after the initial
        // backward alias has already been fused. Re-applying the generic Layer
        // replacement would forward the downstream (reshaped) descriptor upstream
        // and make expression-backed activations stamp backward with the wrong
        // physical rank.
        if (!newErrorInput.has_value()) {
            if (errorOutput.has_value() && previousLayer.has_value()) {
                previousLayer.value()->replaceErrorInput(errorOutput, std::nullopt);
            }
            errorInput.reset();
            errorOutput.reset();
            return;
        }

        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(newErrorInput.value().getDescriptor() == featureOutput.value().getDescriptor());

        Tensor reshapedErrorOutput = newErrorInput.value();
        reshapedErrorOutput.reshape(featureInput.value().getDimensions());

        if (errorOutput.has_value() && previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, reshapedErrorOutput);
        }
        errorInput = newErrorInput;
        errorOutput = reshapedErrorOutput;
    }

    void postCompile() override {
        // Backward alias fusion must happen during connection, before upstream
        // CustomLayer compileImpl() snapshots its expected incoming error tensor
        // ids.  Keep postCompile() intentionally empty except for the base flag.
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
    void fuseBackwardAliasThroughMetadataOnlyReshape() {
        // errorInput is the downstream gradient tensor whose descriptor matches
        // this layer's feature output.  errorOutput is the tensor the upstream
        // layer will receive.  For metadata-only reshape/flatten, both should
        // alias the same storage, but upstream must see the original feature
        // input descriptor.
        if (!errorInput.has_value() || !errorOutput.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureInput.has_value());

        Tensor reshapedErrorOutput = errorInput.value();
        reshapedErrorOutput.reshape(featureInput.value().getDimensions());

        if (previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, reshapedErrorOutput);
        }
        errorOutput = reshapedErrorOutput;
    }

    bool uninitialized;

    unsigned int toNumDimensions;
};

}  // namespace ThorImplementation
