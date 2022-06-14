#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Tanh.h"

namespace ThorImplementation {

class Tanh : public Activation {
   public:
    virtual ~Tanh() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone();
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        TensorPlacement placement = inputTensor.get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchTanh((half*)outputTensor.get().getMemPtr(),
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
        launchTanhBackward((half*)errorOut.get().getMemPtr(),
                           (half*)dataIn.get().getMemPtr(),
                           (half*)errorIn.get().getMemPtr(),
                           errorOut.get().getDescriptor().getTotalNumElements(),
                           stream);
    }

    virtual uint64_t floatingPointOperationsPerExampleForward() {
        // https://stackoverflow.com/questions/41251698/how-many-flops-does-tanh-need#:~:text=for%20very%20small%20x%20%2C%20(let's,floating%20point%20operations%20are%20needed.
        return 8 * Activation::floatingPointOperationsPerExampleForward();
    }

    virtual uint64_t floatingPointOperationsPerExampleBackward() { return 11 * Activation::floatingPointOperationsPerExampleForward(); }

   private:
    bool uninitialized;

    TensorDescriptor::DataType dataType;
};

}  // namespace ThorImplementation
