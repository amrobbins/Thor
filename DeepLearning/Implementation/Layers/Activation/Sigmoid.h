#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Sigmoid.h"

namespace ThorImplementation {

class Sigmoid : public Activation {
   public:
    Sigmoid() { this->backwardComputedExternally = false; }
    Sigmoid(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

    virtual ~Sigmoid() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone();
    }

    virtual void compile() {
        if (backwardComputedExternally) {
            // ErrorInput to the previous layer is the errorInput coming to this layer,
            // then backProp is a no op
            if (errorInput.isPresent() && errorOutput.isPresent() && previousLayer.isPresent()) {
                previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
            }
            errorOutput = errorInput;
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        TensorPlacement placement = inputTensor.get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchSigmoid((half*)outputTensor.get().getMemPtr(),
                      (half*)inputTensor.get().getMemPtr(),
                      inputTensor.get().getDescriptor().getTotalNumElements(),
                      stream);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        assert(dataIn.isPresent());
        assert(errorIn.isPresent());
        assert(errorOut.isPresent());
        TensorPlacement placement = errorOut.get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (backwardComputedExternally)
            return;

        launchSigmoidBackward((half*)errorOut.get().getMemPtr(),
                              (half*)dataIn.get().getMemPtr(),
                              (half*)errorIn.get().getMemPtr(),
                              errorOut.get().getDescriptor().getTotalNumElements(),
                              stream);
    }

   protected:
    bool backwardComputedExternally;
};

}  // namespace ThorImplementation
