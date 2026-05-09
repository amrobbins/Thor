#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class TypeConversion : public Layer {
   public:
    ~TypeConversion() override {}

    TypeConversion() { uninitialized = true; }

    TypeConversion(TensorDescriptor::DataType newDataType) {
        uninitialized = false;
        dataType = newDataType;
    }

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() != dataType);
        return featureInput.get().clone(dataType);
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(outputTensor.isPresent());
        outputTensor.get().copyFromAsync(inputTensor, stream);
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        if (errorOut.isPresent()) {
            THOR_THROW_IF_FALSE(errorIn.isPresent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }

   private:
    bool uninitialized;

    TensorDescriptor::DataType dataType;
};

}  // namespace ThorImplementation
