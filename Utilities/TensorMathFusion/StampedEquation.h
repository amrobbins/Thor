#pragma once

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorMathFusion/CompiledEquation.h"

namespace ThorImplementation {
struct BuiltReduction;

class StampedEquation {
   public:
    StampedEquation(std::shared_ptr<CompiledEquation> compiledEquation,
                    const std::vector<Tensor>& inputs,
                    const Tensor& output,
                    const Stream& stream,
                    Optional<Tensor> deviceBroadcastInfo = Optional<Tensor>::empty())
        : compiledEquation(std::move(compiledEquation)),
          inputs(inputs),
          output(output),
          stream(stream),
          deviceBroadcastInfo(deviceBroadcastInfo) {}

    void run();
    Tensor getOutputTensor() const { return output; }

   private:
    std::shared_ptr<CompiledEquation> compiledEquation;
    std::vector<Tensor> inputs;
    Tensor output;
    Stream stream;
    Optional<Tensor> deviceBroadcastInfo = Optional<Tensor>::empty();
};

class StampedReduction {
   public:
    void run();
    Tensor getOutputTensor() const { return output; }

    StampedReduction(
        std::shared_ptr<BuiltReduction> built, const Tensor& input, const Tensor& output, const Stream& stream, Optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor input;
    Tensor output;
    const Optional<Tensor> workspace;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_1 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_1;
};

struct StampedExecutionStage {
    enum class Kind { FusedKernel, Reduction };
    const Kind kind;

    const std::shared_ptr<StampedEquation> fused = nullptr;
    const std::shared_ptr<StampedReduction> reduction = nullptr;

    explicit StampedExecutionStage(const std::shared_ptr<StampedEquation>& fused) : kind(Kind::FusedKernel), fused(fused) {}
    explicit StampedExecutionStage(const std::shared_ptr<StampedReduction>& reduction) : kind(Kind::Reduction), reduction(reduction) {}
};

class StampedExecutionPlan {
   public:
    StampedExecutionPlan(std::vector<StampedExecutionStage> steps) : steps(std::move(steps)) {}

    void run() {
        for (const StampedExecutionStage& step : steps) {
            if (step.kind == StampedExecutionStage::Kind::FusedKernel) {
                assert(step.fused != nullptr);
                step.fused->run();
            } else if (step.kind == StampedExecutionStage::Kind::Reduction) {
                assert(step.reduction != nullptr);
                step.reduction->run();
            } else {
                throw std::runtime_error("Unknown StampedExecutionStep kind: " + std::to_string((int)step.kind));
            }
        }
    }

   private:
    const std::vector<StampedExecutionStage> steps;
};
}  // namespace ThorImplementation
