#pragma once
#include <optional>

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class GradientRivet : public Layer {
   public:
    virtual ~GradientRivet() {}

    GradientRivet() {}

    virtual std::optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.has_value());
        featureOutput = featureInput.value().clone();
        return featureOutput;
    }

    virtual void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) {}

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(!compiled);

        assert(!this->nextLayer.has_value());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = std::nullopt;

        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, true, loaderConnectionType);

        if (errorInput.has_value() && featureOutput.has_value()) {
            assert(errorInput.value().getDescriptor() == featureOutput.value().getDescriptor());
            assert(errorInput.value().getPlacement() == featureOutput.value().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.has_value());
        assert(outputTensor.has_value());
        outputTensor.value().copyFromAsync(inputTensor.value(), stream);
    }

    virtual void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) {
        assert(errorIn.has_value());
        if (errorOut.has_value())
            errorOut.value().copyFromAsync(errorIn.value(), stream);
    }
};

}  // namespace ThorImplementation
