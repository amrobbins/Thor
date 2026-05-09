#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/SoftSign.h"

namespace ThorImplementation {

class SoftSign : public Activation {
   public:
    ~SoftSign() override {}

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        return featureInput.get().clone();
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputTensor.isPresent());
        TensorPlacement placement = inputTensor.get().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchSoftSign((half*)outputTensor.get().getMemPtr(),
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
        launchSoftSignBackward((half*)errorOut.get().getMemPtr(),
                               (half*)dataIn.get().getMemPtr(),
                               (half*)errorIn.get().getMemPtr(),
                               errorOut.get().getDescriptor().getTotalNumElements(),
                               stream);
    }
};

}  // namespace ThorImplementation
