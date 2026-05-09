#pragma once
#include <optional>

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

/**
 * This layer is useful in unit testing to prevent pruning dangling backwards path tensors.
 * This layer is not meant to be used outside of unit testing.
 */
class NoOpLayer : public ThorImplementation::Layer {
   public:
    NoOpLayer() = default;

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        assert(!compiled);

        assert(!this->nextLayer.has_value());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = std::nullopt;

        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, shouldConnectToBackPropErrorIn(), loaderConnectionType);

        if (errorInput.has_value() && featureOutput.has_value()) {
            assert(errorInput.value().getDescriptor() == featureOutput.value().getDescriptor());
            assert(errorInput.value().getPlacement() == featureOutput.value().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    void infer(std::optional<ThorImplementation::Tensor> inputTensor,
               std::optional<ThorImplementation::Tensor> outputTensor,
               Stream stream) override {
        if (outputTensor.has_value()) {
            assert(inputTensor.has_value());
            outputTensor.value().copyFromAsync(inputTensor.value(), stream);
        }
    }

    void backProp(std::optional<ThorImplementation::Tensor> dataIn,
                  std::optional<ThorImplementation::Tensor> errorIn,
                  std::optional<ThorImplementation::Tensor> errorOut,
                  Stream stream) override {
        if (errorOut.has_value()) {
            assert(errorIn.has_value());
            errorOut.value().copyFromAsync(errorIn.value(), stream);
        }
    }
};
