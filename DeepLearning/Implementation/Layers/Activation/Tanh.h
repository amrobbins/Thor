#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Arithmetic/Tanh.h"

namespace ThorImplementation {

class Tanh : public Layer {
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

   private:
    bool uninitialized;

    TensorDescriptor::DataType dataType;
};

}  // namespace ThorImplementation
