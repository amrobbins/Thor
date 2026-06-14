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

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        Layer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
        fuseBackwardAliasThroughMetadataOnlyReshape();
    }

    void postCompile() override {
        // Backward alias fusion must happen during connection, before upstream
        // CustomLayer compileImpl() snapshots its expected incoming error tensor
        // ids.  Keep postCompile() intentionally empty except for the base flag.
        Layer::postCompile();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        // No op
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        // No op
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

    std::vector<unsigned long> outputDimensions;
};

}  // namespace ThorImplementation
