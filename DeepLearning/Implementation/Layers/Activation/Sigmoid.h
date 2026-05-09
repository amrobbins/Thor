#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Sigmoid.h"

namespace ThorImplementation {

class Sigmoid : public Activation {
   public:
    Sigmoid() { this->backwardComputedExternally = false; }
    Sigmoid(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

    ~Sigmoid() override {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    void postCompile() override {
        if (backwardComputedExternally) {
            // ErrorInput to the previous layer is the errorInput coming to this layer,
            // then backProp is a no op
            if (errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value()) {
                previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
            }
            errorOutput = errorInput;
        }
        Layer::postCompile();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        TensorPlacement placement = inputTensor.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchSigmoid((half*)outputTensor.value().getMemPtr(),
                      (half*)inputTensor.value().getMemPtr(),
                      inputTensor.value().getDescriptor().getTotalNumElements(),
                      stream);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(dataIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorOut.has_value());
        TensorPlacement placement = errorOut.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (backwardComputedExternally)
            return;

        launchSigmoidBackward((half*)errorOut.value().getMemPtr(),
                              (half*)dataIn.value().getMemPtr(),
                              (half*)errorIn.value().getMemPtr(),
                              errorOut.value().getDescriptor().getTotalNumElements(),
                              stream);
    }

    std::string getType() override { return "Sigmoid"; }

   protected:
    bool backwardComputedExternally;
};

}  // namespace ThorImplementation
