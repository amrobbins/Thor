#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Elu.h"

namespace ThorImplementation {

class Elu : public Activation {
   public:
    Elu(float alpha = 1.0f) : alpha(alpha) {}

    ~Elu() override {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        TensorPlacement placement = inputTensor.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchElu((half*)outputTensor.value().getMemPtr(),
                  (half*)inputTensor.value().getMemPtr(),
                  inputTensor.value().getDescriptor().getTotalNumElements(),
                  alpha,
                  stream);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(dataIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorOut.has_value());
        TensorPlacement placement = errorOut.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchEluBackward((half*)errorOut.value().getMemPtr(),
                          (half*)dataIn.value().getMemPtr(),
                          (half*)errorIn.value().getMemPtr(),
                          errorOut.value().getDescriptor().getTotalNumElements(),
                          alpha,
                          stream);
    }

    float getAlpha() { return alpha; }

   protected:
    const float alpha;
};

}  // namespace ThorImplementation
