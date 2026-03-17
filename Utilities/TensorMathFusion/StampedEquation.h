#pragma once

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorMathFusion/CompiledEquation.h"

namespace ThorImplementation {

struct ReductionCacheKey {
    const ExprOp op;
    const std::vector<uint64_t> input_dims;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const TensorDescriptor::DataType inout_dtype;
    const TensorDescriptor::DataType compute_dtype;
    const int device_num;

    bool operator==(const ReductionCacheKey& other) const = default;

    ReductionCacheKey(ExprOp op,
                      std::vector<uint64_t> input_dims,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      TensorDescriptor::DataType inout_dtype,
                      TensorDescriptor::DataType compute_dtype,
                      int device_num)
        : op(op),
          input_dims(std::move(input_dims)),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          inout_dtype(inout_dtype),
          compute_dtype(compute_dtype),
          device_num(device_num) {
        if (this->reduction_axes.empty()) {
            this->reduction_axes.resize(this->input_dims.size());
            std::iota(this->reduction_axes.begin(), this->reduction_axes.end(), 0);
        } else {
            std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
            this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());
        }
        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

struct BuiltReduction {
    ReductionCacheKey key;

    cudnnTensorDescriptor_t a_desc = nullptr;
    cudnnTensorDescriptor_t c_desc = nullptr;
    cudnnReduceTensorDescriptor_t reduce_desc = nullptr;

    size_t workspace_bytes = 0;

    explicit BuiltReduction(ReductionCacheKey key) : key(std::move(key)) {}

    ~BuiltReduction() {
        if (a_desc)
            cudnnDestroyTensorDescriptor(a_desc);
        if (c_desc)
            cudnnDestroyTensorDescriptor(c_desc);
        if (reduce_desc)
            cudnnDestroyReduceTensorDescriptor(reduce_desc);
    }

    BuiltReduction(const BuiltReduction&) = delete;
    BuiltReduction& operator=(const BuiltReduction&) = delete;
};

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

    static std::vector<uint64_t> computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                            const std::vector<uint64_t>& reduction_axes,
                                                            const std::vector<uint64_t>& squeeze_axes);

    static std::shared_ptr<BuiltReduction> buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                          const Tensor& input,
                                                          int device_num);

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
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};

struct StampedExecutionStage {
    enum class Kind { FusedKernel, Reduction };
    const Kind kind;

    const std::shared_ptr<StampedEquation> kernel = nullptr;
    const std::shared_ptr<StampedReduction> reduction = nullptr;

    explicit StampedExecutionStage(const std::shared_ptr<StampedEquation>& fused) : kind(Kind::FusedKernel), kernel(fused) {
        outputTensor = fused->getOutputTensor();
    }
    explicit StampedExecutionStage(const std::shared_ptr<StampedReduction>& reduction) : kind(Kind::Reduction), reduction(reduction) {
        outputTensor = reduction->getOutputTensor();
    }

    Tensor getOutputTensor() { return outputTensor; }

   private:
    Tensor outputTensor;
};

class StampedExecutionPlan {
   public:
    StampedExecutionPlan(std::vector<StampedExecutionStage> steps) : steps(std::move(steps)) {}

    void run() {
        for (const StampedExecutionStage& step : steps) {
            if (step.kind == StampedExecutionStage::Kind::FusedKernel) {
                assert(step.kernel != nullptr);
                step.kernel->run();
            } else if (step.kind == StampedExecutionStage::Kind::Reduction) {
                assert(step.reduction != nullptr);
                step.reduction->run();
            } else {
                throw std::runtime_error("Unknown StampedExecutionStep kind: " + std::to_string((int)step.kind));
            }
        }
    }

    Tensor getOutputTensor() const {
        if (steps.empty()) {
            throw std::runtime_error("StampedExecutionPlan has no execution stages.");
        }

        const StampedExecutionStage& last = steps.back();
        if (last.kind == StampedExecutionStage::Kind::FusedKernel) {
            if (!last.kernel) {
                throw std::runtime_error("Final fused stage is null.");
            }
            return last.kernel->getOutputTensor();
        } else if (last.kind == StampedExecutionStage::Kind::Reduction) {
            if (!last.reduction) {
                throw std::runtime_error("Final reduction stage is null.");
            }
            return last.reduction->getOutputTensor();
        }

        throw std::runtime_error("Unknown final execution stage kind.");
    }

   private:
    const std::vector<StampedExecutionStage> steps;
};
}  // namespace ThorImplementation

namespace std {
template <>
struct hash<ThorImplementation::ReductionCacheKey> {
    size_t operator()(const ThorImplementation::ReductionCacheKey& k) const noexcept {
        size_t h = hash<ThorImplementation::ExprOp>{}(k.op);

        hashCombine(h, hash<size_t>{}(k.input_dims.size()));
        for (uint64_t d : k.input_dims)
            hashCombine(h, hash<uint64_t>{}(d));
        hashCombine(h, hash<size_t>{}(k.reduction_axes.size()));
        for (uint64_t axis : k.reduction_axes)
            hashCombine(h, hash<uint64_t>{}(axis));
        hashCombine(h, hash<size_t>{}(k.squeeze_axes.size()));
        for (uint64_t axis : k.squeeze_axes)
            hashCombine(h, hash<uint64_t>{}(axis));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.inout_dtype));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.compute_dtype));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};
}  // namespace std
