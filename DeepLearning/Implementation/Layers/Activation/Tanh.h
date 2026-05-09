#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Tanh.h"

namespace ThorImplementation {

class Tanh : public Activation {
   public:
    ~Tanh() override {}

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        return featureInput.get().clone();
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputTensor.isPresent());
        TensorPlacement placement = inputTensor.get().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchTanh((half*)outputTensor.get().getMemPtr(),
                   (half*)inputTensor.get().getMemPtr(),
                   inputTensor.get().getDescriptor().getTotalNumElements(),
                   stream);
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(dataIn.isPresent());
        THOR_THROW_IF_FALSE(errorIn.isPresent());
        THOR_THROW_IF_FALSE(errorOut.isPresent());
        TensorPlacement placement = errorOut.get().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchTanhBackward((half*)errorOut.get().getMemPtr(),
                           (half*)dataIn.get().getMemPtr(),
                           (half*)errorIn.get().getMemPtr(),
                           errorOut.get().getDescriptor().getTotalNumElements(),
                           stream);
    }

    uint64_t floatingPointOperationsPerExampleForward() override {
        // https://stackoverflow.com/questions/41251698/how-many-flops-does-tanh-need#:~:text=for%20very%20small%20x%20%2C%20(let's,floating%20point%20operations%20are%20needed.
        return 8 * Activation::floatingPointOperationsPerExampleForward();
    }

    uint64_t floatingPointOperationsPerExampleBackward() override { return 11 * Activation::floatingPointOperationsPerExampleForward(); }

   private:
    bool uninitialized;

    TensorDescriptor::DataType dataType;
};

}  // namespace ThorImplementation
