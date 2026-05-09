#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Activation/Tanh.h"

namespace ThorImplementation {

class Tanh : public Activation {
   public:
    ~Tanh() override {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        TensorPlacement placement = inputTensor.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchTanh((half*)outputTensor.value().getMemPtr(),
                   (half*)inputTensor.value().getMemPtr(),
                   inputTensor.value().getDescriptor().getTotalNumElements(),
                   stream);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(dataIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorOut.has_value());
        TensorPlacement placement = errorOut.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        launchTanhBackward((half*)errorOut.value().getMemPtr(),
                           (half*)dataIn.value().getMemPtr(),
                           (half*)errorIn.value().getMemPtr(),
                           errorOut.value().getDescriptor().getTotalNumElements(),
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
