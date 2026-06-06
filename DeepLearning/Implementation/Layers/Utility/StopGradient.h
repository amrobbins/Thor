#pragma once
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

#include <optional>

namespace ThorImplementation {

class StopGradient : public Layer {
   public:
    StopGradient() = default;
    ~StopGradient() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value();
    }

    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override {
        (void)backPropagateError;
        return std::nullopt;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
        // Forward is an identity alias.  The output tensor shares storage with the input tensor.
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
        // Backward intentionally produces no error output.  Gradient propagation stops here.
    }

    std::string getType() override { return "StopGradient"; }
};

}  // namespace ThorImplementation
