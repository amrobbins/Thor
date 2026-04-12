#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/Cache/LruCache.h"
#include "Utilities/TensorMathFusion/CompiledEquation.h"
#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

namespace ThorImplementation {

struct ReductionCacheKey {
    const ExprOp op;
    const std::vector<uint64_t> input_dims;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const TensorDescriptor::DataType input_dtype;
    const TensorDescriptor::DataType compute_dtype;
    const TensorDescriptor::DataType output_dtype;
    const bool output_indices;
    const int device_num;

    bool operator==(const ReductionCacheKey& other) const = default;

    ReductionCacheKey(ExprOp op,
                      std::vector<uint64_t> input_dims,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      TensorDescriptor::DataType input_dtype,
                      TensorDescriptor::DataType output_dtype,
                      TensorDescriptor::DataType compute_dtype,
                      bool output_indices,
                      int device_num)
        : op(op),
          input_dims(std::move(input_dims)),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          compute_dtype(compute_dtype),
          output_dtype(output_dtype),
          output_indices(output_indices),
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
    size_t indices_bytes = 0;

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

struct MatmulCacheKey {
    const ExprOp op;
    const int32_t a_rows;
    const int32_t a_cols;
    const int32_t b_rows;
    const int32_t b_cols;
    const int32_t ld_a;
    const int32_t ld_b;
    const int32_t ld_c;
    const int32_t ld_d;
    const bool transpose_a;
    const bool transpose_b;
    const bool transpose_c;
    const TensorDescriptor::DataType data_dtype;
    const int device_num;

    bool operator==(const MatmulCacheKey& other) const = default;

    MatmulCacheKey(ExprOp op,
                   int32_t a_rows,
                   int32_t a_cols,
                   int32_t b_rows,
                   int32_t b_cols,
                   int32_t ld_a,
                   int32_t ld_b,
                   int32_t ld_c,
                   int32_t ld_d,
                   bool transpose_a,
                   bool transpose_b,
                   bool transpose_c,
                   TensorDescriptor::DataType data_dtype,
                   int device_num)
        : op(op),
          a_rows(a_rows),
          a_cols(a_cols),
          b_rows(b_rows),
          b_cols(b_cols),
          ld_a(ld_a),
          ld_b(ld_b),
          ld_c(ld_c),
          ld_d(ld_d),
          transpose_a(transpose_a),
          transpose_b(transpose_b),
          transpose_c(transpose_c),
          data_dtype(data_dtype),
          device_num(device_num) {}
};

struct BuiltMatmul {
    MatmulCacheKey key;
    size_t workspace_bytes = 0;

    explicit BuiltMatmul(MatmulCacheKey key) : key(std::move(key)) {}

    BuiltMatmul(const BuiltMatmul&) = delete;
    BuiltMatmul& operator=(const BuiltMatmul&) = delete;
};

struct BuiltConvolution {
    Optional<ConvolutionKernelRequirement> requirement = Optional<ConvolutionKernelRequirement>::empty();
    size_t workspace_bytes = 0;
};

class StampedEquation {
   public:
    StampedEquation(std::shared_ptr<CompiledEquation> compiledEquation,
                    const std::vector<RuntimeInputValue>& inputs,
                    const std::vector<Tensor>& outputs,
                    const Stream& stream)
        : compiledEquation(std::move(compiledEquation)), inputs(inputs), outputs(outputs), stream(stream) {}

    void run();
    void run(const std::unordered_map<std::string, float>& runtime_scalars);
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;

    uint32_t gpuNum() const {
        if (!outputs.empty()) {
            return outputs[0].getPlacement().getDeviceNum();
        }
        for (const RuntimeInputValue& input : inputs) {
            if (std::holds_alternative<Tensor>(input)) {
                return std::get<Tensor>(input).getPlacement().getDeviceNum();
            }
        }
        throw std::runtime_error("StampedEquation::gpuNum() requires at least one input or output tensor.");
    }

    Tensor getOutputTensor() const {
        if (outputs.size() != 1)
            throw std::runtime_error("getOutputTensor called but there are " + std::to_string(outputs.size()) +
                                     "outputs. This function is only valid for single output equations.");
        return outputs[0];
    }

    const std::vector<Tensor>& getOutputTensors() const { return outputs; }

    static std::vector<uint64_t> computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                            const std::vector<uint64_t>& reduction_axes,
                                                            const std::vector<uint64_t>& squeeze_axes);

    static std::shared_ptr<BuiltReduction> buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                          const Tensor& input,
                                                          int device_num);

    static std::shared_ptr<BuiltReduction> buildReduction(ExprOp op,
                                                          const std::vector<uint64_t>& reduction_axes,
                                                          const std::vector<uint64_t>& squeeze_axes,
                                                          TensorDescriptor::DataType input_dtype,
                                                          TensorDescriptor::DataType output_dtype,
                                                          TensorDescriptor::DataType compute_dtype,
                                                          bool output_indices,
                                                          const Tensor& input,
                                                          int device_num);

    static std::shared_ptr<BuiltMatmul> buildMatmul(const std::shared_ptr<CompiledMatmul>& compiled_matmul,
                                                    const Tensor& lhs,
                                                    const Tensor& rhs,
                                                    const Optional<Tensor>& addend,
                                                    const Tensor& output,
                                                    int device_num);

    static std::shared_ptr<BuiltConvolution> buildConvolution(const std::shared_ptr<CompiledConvolution>& compiled_convolution,
                                                              const Tensor& input,
                                                              const Tensor& filter,
                                                              const Tensor& output,
                                                              const Stream& stream,
                                                              int device_num);

    [[nodiscard]] bool requiresRuntimeScalars() const;
    [[nodiscard]] std::unordered_set<std::string> runtimeScalarNames() const;

   private:
    std::shared_ptr<CompiledEquation> compiledEquation;
    std::vector<RuntimeInputValue> inputs;
    std::vector<Tensor> outputs;
    Stream stream;
};

class StampedReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

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

class StampedArgMinMax {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedArgMinMax(std::shared_ptr<BuiltReduction> built,
                     const Tensor& input,
                     const Tensor& output,
                     const Tensor& reduction_value_output,
                     const Stream& stream,
                     Optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor input;
    Tensor output;
    const Tensor reduction_value_output;
    const Optional<Tensor> workspace;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};

class StampedMatmul {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedMatmul(std::shared_ptr<CompiledMatmul> compiled,
                  std::shared_ptr<BuiltMatmul> built,
                  const Tensor& lhs,
                  const Tensor& rhs,
                  const Optional<Tensor>& addend,
                  const Tensor& output,
                  const Stream& stream,
                  Optional<Tensor> workspace,
                  Optional<RuntimeInputValue> alpha_input,
                  Optional<RuntimeInputValue> beta_input,
                  std::optional<std::string> alpha_runtime_name,
                  std::optional<std::string> beta_runtime_name,
                  Optional<Tensor> alpha_device_scratch,
                  Optional<Tensor> beta_device_scratch,
                  Optional<Tensor> alpha_host_scratch,
                  Optional<Tensor> beta_host_scratch);

    [[nodiscard]] std::optional<std::string> alphaRuntimeName() const { return alpha_runtime_name; }
    [[nodiscard]] std::optional<std::string> betaRuntimeName() const { return beta_runtime_name; }

   private:
    const std::shared_ptr<CompiledMatmul> compiled_matmul;
    const std::shared_ptr<BuiltMatmul> built_matmul;
    const Tensor lhs;
    const Tensor rhs;
    const Optional<Tensor> addend;
    Tensor output;
    Stream stream;
    const Optional<Tensor> workspace;
    const Optional<RuntimeInputValue> alpha_input;
    const Optional<RuntimeInputValue> beta_input;
    const std::optional<std::string> alpha_runtime_name;
    const std::optional<std::string> beta_runtime_name;
    const Optional<Tensor> alpha_device_scratch;
    const Optional<Tensor> beta_device_scratch;
    const Optional<Tensor> alpha_host_scratch;
    const Optional<Tensor> beta_host_scratch;
};

class StampedConvolution {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedConvolution(std::shared_ptr<CompiledConvolution> compiled,
                       std::shared_ptr<BuiltConvolution> built,
                       const Tensor& input,
                       const Tensor& filter,
                       const Tensor& output,
                       const Stream& stream,
                       Optional<Tensor> workspace);

   private:
    const std::shared_ptr<CompiledConvolution> compiled_convolution;
    const std::shared_ptr<BuiltConvolution> built_convolution;
    const Tensor input;
    const Tensor filter;
    Tensor output;
    Stream stream;
    const Optional<Tensor> workspace;
};

class StampedReduceMinMaxBackward {
   public:
    void run();
    void runOn(Stream& run_stream);

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedReduceMinMaxBackward(std::shared_ptr<BuiltReduction> built,
                                const Tensor& input,
                                const Tensor& grad_output,
                                const Tensor& output,
                                const Tensor& indices,
                                const Tensor& reduction_value_output,
                                const Stream& stream,
                                Optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor input;
    const Tensor grad_output;
    Tensor output;
    const Tensor indices;
    const Tensor reduction_value_output;
    const Optional<Tensor> workspace;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};

struct StampedExecutionStage {
    enum class Kind { FusedKernel, Reduction, ArgMinMax, Matmul, Convolution, ReduceMinMaxBackward };
    const Kind kind;

    const std::vector<uint32_t> dependency_stage_indices;
    const uint32_t gpu_num;

    const std::shared_ptr<StampedEquation> kernel = nullptr;
    const std::shared_ptr<StampedReduction> reduction = nullptr;
    const std::shared_ptr<StampedArgMinMax> arg_minmax = nullptr;
    const std::shared_ptr<StampedMatmul> matmul = nullptr;
    const std::shared_ptr<StampedConvolution> convolution = nullptr;
    const std::shared_ptr<StampedReduceMinMaxBackward> reduce_minmax_backward = nullptr;

    explicit StampedExecutionStage(const std::shared_ptr<StampedEquation>& fused, std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::FusedKernel), dependency_stage_indices(std::move(dependency_stage_indices)), gpu_num(fused->gpuNum()), kernel(fused) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedReduction>& reduction, std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::Reduction),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(reduction->gpuNum()),
          reduction(reduction) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedArgMinMax>& arg_minmax, std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::ArgMinMax),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(arg_minmax->gpuNum()),
          arg_minmax(arg_minmax) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedMatmul>& matmul, std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::Matmul), dependency_stage_indices(std::move(dependency_stage_indices)), gpu_num(matmul->gpuNum()), matmul(matmul) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedConvolution>& convolution,
                                   std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::Convolution),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(convolution->gpuNum()),
          convolution(convolution) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedReduceMinMaxBackward>& reduce_minmax_backward,
                                   std::vector<uint32_t> dependency_stage_indices = {})
        : kind(Kind::ReduceMinMaxBackward),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(reduce_minmax_backward->gpuNum()),
          reduce_minmax_backward(reduce_minmax_backward) {}

    void runOn(Stream& run_stream) const { runOn(run_stream, {}); }

    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
        if (kind == Kind::FusedKernel) {
            assert(kernel != nullptr);
            if (runtime_scalars.empty())
                kernel->runOn(run_stream);
            else
                kernel->runOn(run_stream, runtime_scalars);
        } else if (kind == Kind::Reduction) {
            assert(reduction != nullptr);
            reduction->runOn(run_stream);
        } else if (kind == Kind::ArgMinMax) {
            assert(arg_minmax != nullptr);
            arg_minmax->runOn(run_stream);
        } else if (kind == Kind::Matmul) {
            assert(matmul != nullptr);
            if (runtime_scalars.empty())
                matmul->runOn(run_stream);
            else
                matmul->runOn(run_stream, runtime_scalars);
        } else if (kind == Kind::Convolution) {
            assert(convolution != nullptr);
            convolution->runOn(run_stream);
        } else if (kind == Kind::ReduceMinMaxBackward) {
            assert(reduce_minmax_backward != nullptr);
            reduce_minmax_backward->runOn(run_stream);
        } else {
            throw std::runtime_error("Unknown StampedExecutionStage kind: " + std::to_string((int)kind));
        }
    }
};

class StampedExecutionPlan {
   public:
    StampedExecutionPlan(std::vector<StampedExecutionStage> steps,
                         std::unordered_map<std::string, Tensor> final_outputs,
                         const Stream& stream)
        : steps(std::move(steps)), final_outputs(std::move(final_outputs)), stream(stream) {}

    void run();
    void run(const std::unordered_map<std::string, float>& runtime_scalars);

    Tensor output(const std::string& name) const {
        auto it = final_outputs.find(name);
        if (it == final_outputs.end()) {
            throw std::runtime_error("No such output in stamped execution plan: " + name);
        }
        return it->second;
    }

    Tensor output() const {
        if (final_outputs.size() != 1)
            throw std::runtime_error("StampedEquation.output() called to return the single output tensor, but there are " +
                                     std::to_string(final_outputs.size()) + "output tensors.");
        return final_outputs.begin()->second;
    }

    std::vector<std::string> outputNames() const {
        std::vector<std::string> output_names;
        output_names.reserve(final_outputs.size());

        for (const auto& [key, value] : final_outputs) {
            output_names.push_back(key);
        }
        return output_names;
    }

    std::unordered_map<std::string, Tensor> getFinalOutputs() const { return final_outputs; }

   private:
    const std::vector<StampedExecutionStage> steps;
    std::unordered_map<std::string, Tensor> final_outputs;
    Stream stream;
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
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.input_dtype));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.compute_dtype));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.output_dtype));
        hashCombine(h, hash<bool>{}(k.output_indices));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};

template <>
struct hash<ThorImplementation::MatmulCacheKey> {
    size_t operator()(const ThorImplementation::MatmulCacheKey& k) const noexcept {
        size_t h = hash<ThorImplementation::ExprOp>{}(k.op);
        hashCombine(h, hash<int32_t>{}(k.a_rows));
        hashCombine(h, hash<int32_t>{}(k.a_cols));
        hashCombine(h, hash<int32_t>{}(k.b_rows));
        hashCombine(h, hash<int32_t>{}(k.b_cols));
        hashCombine(h, hash<int32_t>{}(k.ld_a));
        hashCombine(h, hash<int32_t>{}(k.ld_b));
        hashCombine(h, hash<int32_t>{}(k.ld_c));
        hashCombine(h, hash<int32_t>{}(k.ld_d));
        hashCombine(h, hash<bool>{}(k.transpose_a));
        hashCombine(h, hash<bool>{}(k.transpose_b));
        hashCombine(h, hash<bool>{}(k.transpose_c));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.data_dtype));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};
}  // namespace std
