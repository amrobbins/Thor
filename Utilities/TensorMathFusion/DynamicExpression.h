#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

class DynamicExpression {
   public:
    using TensorMap = std::unordered_map<std::string, Tensor>;
    using BuilderFn = std::function<StampedExecutionPlan(const TensorMap& inputs, Stream& stream)>;

    explicit DynamicExpression(BuilderFn builder) : builder_(std::move(builder)) {
        if (!builder_) {
            throw std::invalid_argument("DynamicExpression requires a non-empty builder.");
        }
    }

    [[nodiscard]] StampedExecutionPlan stamp(const TensorMap& inputs, Stream& stream) const {
        validateInputs(inputs, stream);
        return builder_(inputs, stream);
    }

   private:
    static void validateInputs(const TensorMap& inputs, Stream& stream) {
        if (inputs.empty()) {
            throw std::invalid_argument("DynamicExpression requires at least one input tensor.");
        }

        const auto firstPlacement = inputs.begin()->second.getPlacement();
        const bool onGpu = firstPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU;
        if (!onGpu) {
            throw std::invalid_argument("DynamicExpression currently requires GPU input tensors.");
        }

        const int32_t gpuNum = firstPlacement.getDeviceNum();

        for (const auto& [name, tensor] : inputs) {
            if (!tensor.isInitialized()) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is not initialized.");
            }

            const auto placement = tensor.getPlacement();
            if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is not on GPU.");
            }
            if (placement.getDeviceNum() != gpuNum) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is on a different GPU than the other inputs.");
            }
        }

        if (stream.getGpuNum() != gpuNum) {
            throw std::runtime_error("DynamicExpression stream GPU does not match input tensor GPU.");
        }
    }

    BuilderFn builder_;
};

}  // namespace ThorImplementation
