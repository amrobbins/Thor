#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/SoftPlus.h"

namespace ThorImplementation {

class SoftPlus : public Activation {
   public:
    virtual ~SoftPlus() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone();
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        TensorPlacement placement = inputTensor.get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchSoftPlus((half*)outputTensor.get().getMemPtr(),
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
        launchSoftPlusBackward((half*)errorOut.get().getMemPtr(),
                               (half*)dataIn.get().getMemPtr(),
                               (half*)errorIn.get().getMemPtr(),
                               errorOut.get().getDescriptor().getTotalNumElements(),
                               stream);
    }
};

}  // namespace ThorImplementation
