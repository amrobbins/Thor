#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class TypeConversion : public Layer {
   public:
    virtual ~TypeConversion() {}

    TypeConversion() { uninitialized = true; }

    TypeConversion(TensorDescriptor::DataType newDataType) {
        uninitialized = false;
        dataType = newDataType;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        assert(featureInput.get().getDescriptor().getDataType() != dataType);
        return featureInput.get().clone(dataType);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(outputTensor.isPresent());
        outputTensor.get().copyFromAsync(inputTensor, stream);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isPresent()) {
            assert(errorIn.isPresent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }

   private:
    bool uninitialized;

    TensorDescriptor::DataType dataType;
};

}  // namespace ThorImplementation
