#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Utility/ScaleGradientKernel.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cmath>
#include <optional>

namespace ThorImplementation {

class ScaleGradient : public Layer {
   public:
    explicit ScaleGradient(float scale) : scale(scale) { THOR_THROW_IF_FALSE(std::isfinite(scale)); }
    ~ScaleGradient() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(isSupportedDataType(featureInput.value().getDataType()));
        return featureInput.value();
    }

    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override {
        if (!backPropagateError || isInferenceOnly())
            return std::nullopt;
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(isSupportedDataType(featureInput.value().getDataType()));
        return featureInput.value().clone();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
        // Forward is an identity alias. The output tensor shares storage with the input tensor.
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        (void)dataIn;
        if (!errorOut.has_value())
            return;

        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.value().getDescriptor() == errorOut.value().getDescriptor());
        launchScaleGradient(errorIn.value().getMemPtr(),
                            errorOut.value().getMemPtr(),
                            errorIn.value().getDataType(),
                            scale,
                            errorIn.value().getTotalNumElements(),
                            stream);
    }

    float getScale() const { return scale; }
    std::string getType() override { return "ScaleGradient"; }

   private:
    static bool isSupportedDataType(DataType dataType) {
        switch (dataType) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
            case DataType::FP64:
                return true;
            default:
                return false;
        }
    }

    float scale;
};

}  // namespace ThorImplementation
