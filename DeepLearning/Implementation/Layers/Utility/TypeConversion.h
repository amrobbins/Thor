#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class TypeConversion : public Layer {
   public:
    ~TypeConversion() override {}

    TypeConversion() { uninitialized = true; }

    TypeConversion(DataType newDataType) {
        uninitialized = false;
        dataType = newDataType;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() != dataType);
        return featureInput.value().clone(dataType);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        outputTensor.value().copyFromAsync(inputTensor.value(), stream);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        if (errorOut.has_value()) {
            THOR_THROW_IF_FALSE(errorIn.has_value());
            errorOut.value().copyFromAsync(errorIn.value(), stream);
        }
    }

   private:
    bool uninitialized;

    DataType dataType;
};

}  // namespace ThorImplementation
