#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>
#include <variant>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"

#include "Utilities/Cache/LruCache.h"
#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/InPlaceRopeKernel.h"
#include "Utilities/Expression/FlatScatterAddKernel.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

namespace cudnn_frontend {
namespace graph {
class Graph;
}
}

namespace ThorImplementation {

class StampedExecutionPlan;

struct ReductionCacheKey {
    const ExprOp op;
    const std::vector<uint64_t> input_dims;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const DataType input_dtype;
    const DataType compute_dtype;
    const DataType output_dtype;
    const bool output_indices;
    const int device_num;

    bool operator==(const ReductionCacheKey& other) const = default;

    ReductionCacheKey(ExprOp op,
                      std::vector<uint64_t> input_dims,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      DataType input_dtype,
                      DataType output_dtype,
                      DataType compute_dtype,
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
    bool identity_reduction = false;

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

struct SoftmaxCacheKey {
    const std::vector<uint64_t> input_dims;
    const DataType input_dtype;
    const DataType output_dtype;
    const cudnnSoftmaxAlgorithm_t algorithm;
    const cudnnSoftmaxMode_t mode;
    const int device_num;

    bool operator==(const SoftmaxCacheKey& other) const = default;
};

struct BuiltSoftmax {
    SoftmaxCacheKey key;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;

    explicit BuiltSoftmax(SoftmaxCacheKey key) : key(std::move(key)) {}

    ~BuiltSoftmax() {
        if (x_desc)
            cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc)
            cudnnDestroyTensorDescriptor(y_desc);
    }

    BuiltSoftmax(const BuiltSoftmax&) = delete;
    BuiltSoftmax& operator=(const BuiltSoftmax&) = delete;
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
    const bool bias_epilogue;
    const MatmulEpilogue epilogue;
    const MatmulBackwardEpilogue backward_epilogue;
    const bool bgrad_epilogue;
    const DataType a_dtype;
    const DataType b_dtype;
    const DataType c_dtype;
    const DataType d_dtype;
    const DataType compute_dtype;
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
                   bool bias_epilogue,
                   MatmulEpilogue epilogue,
                   MatmulBackwardEpilogue backward_epilogue,
                   bool bgrad_epilogue,
                   DataType a_dtype,
                   DataType b_dtype,
                   DataType c_dtype,
                   DataType d_dtype,
                   DataType compute_dtype,
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
          bias_epilogue(bias_epilogue),
          epilogue(epilogue),
          backward_epilogue(backward_epilogue),
          bgrad_epilogue(bgrad_epilogue),
          a_dtype(a_dtype),
          b_dtype(b_dtype),
          c_dtype(c_dtype),
          d_dtype(d_dtype),
          compute_dtype(compute_dtype),
          device_num(device_num) {}
};

struct BuiltMatmul {
    MatmulCacheKey key;
    size_t workspace_bytes = 0;
    std::optional<CublasKernel> cublas_kernel;
    std::shared_ptr<CublasMatrixMultiply::LtMatmulPlan> epilogue_plan;
    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> epilogue_algorithm;

    explicit BuiltMatmul(MatmulCacheKey key) : key(std::move(key)) {}

    BuiltMatmul(const BuiltMatmul&) = delete;
    BuiltMatmul& operator=(const BuiltMatmul&) = delete;
};


struct CompiledRmsNorm {
    uint64_t normalized_feature_count = 0;
    double epsilon = 1.0e-5;
    DataType input_dtype = DataType::FP16;
    DataType scale_dtype = DataType::FP32;
    DataType output_dtype = DataType::FP16;
    DataType compute_dtype = DataType::FP32;
    CudnnRmsNormFusedActivation fused_activation = CudnnRmsNormFusedActivation::NONE;
    std::string debug_name = "thor_expr_rms_norm";

    [[nodiscard]] CudnnRmsNormDescriptor descriptorFor(const Tensor& input, const Tensor& scale, const Tensor& output) const;
};

struct CompiledAttention {
    AttentionTensorLayout q_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout k_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout v_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout o_layout = AttentionTensorLayout::BHSD;
    AttentionMaskKind mask_kind = AttentionMaskKind::None;
    int64_t diagonal_left_bound = 0;
    int64_t diagonal_right_bound = 0;
    std::optional<float> attention_scale = std::nullopt;
    bool use_alibi_mask = false;
    bool use_bias = false;
    bool use_padding_mask = false;
    bool use_ragged_offsets = false;
    bool use_paged_kv_cache = false;
    int64_t paged_kv_max_sequence_length = 0;
    float dropout_probability = 0.0f;
    bool use_fp8_forward_scaling = false;
    DataType compute_dtype = DataType::FP32;
    DataType output_dtype = DataType::FP16;
    std::string debug_name = "thor_expr_attention";

    CudnnAttentionDescriptor descriptorFor(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o) const;
};


struct CompiledEmbeddingLookup {
    bool has_padding_index = false;
    uint64_t padding_index = 0;
    DataType index_dtype = DataType::UINT32;
    DataType weights_dtype = DataType::FP32;
    DataType output_dtype = DataType::FP32;
    std::string debug_name = "thor_expr_embedding_lookup";
    EmbeddingForwardEpilogue epilogue;
};

struct CompiledCudaKernel {
    std::string cache_key;
    std::string kernel_name;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    int device_num = 0;

    CompiledCudaKernel() = default;
    CompiledCudaKernel(const CompiledCudaKernel&) = delete;
    CompiledCudaKernel& operator=(const CompiledCudaKernel&) = delete;
    CompiledCudaKernel(CompiledCudaKernel&&) = default;
    CompiledCudaKernel& operator=(CompiledCudaKernel&&) = default;

    ~CompiledCudaKernel() {
        if (module != nullptr) {
            try {
                CU_CHECK(cuModuleUnload(module));
            } catch (...) {
            }
        }
    }
};

struct CudaKernelLaunchConfig {
    dim3 grid{1, 1, 1};
    dim3 block{1, 1, 1};
    uint32_t dynamic_shared_bytes = 0;
};

using CudaKernelScalarValue = std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double>;

struct StampedCudaKernelParam {
    enum class Kind : uint8_t { TensorInput, TensorRuntimeScalar, HostRuntimeScalar, TensorOutput, Scalar };

    Kind kind = Kind::TensorInput;
    std::string name;
    size_t tensor_index = 0;
    CudaKernelScalarValue scalar_value = int32_t{0};
};

struct CompiledAttentionBackward {
    AttentionTensorLayout q_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout k_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout v_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout o_layout = AttentionTensorLayout::BHSD;
    AttentionMaskKind mask_kind = AttentionMaskKind::None;
    int64_t diagonal_left_bound = 0;
    int64_t diagonal_right_bound = 0;
    std::optional<float> attention_scale = std::nullopt;
    bool use_alibi_mask = false;
    bool deterministic_backward = false;
    bool use_bias = false;
    bool use_padding_mask = false;
    bool use_ragged_offsets = false;
    bool use_paged_kv_cache = false;
    int64_t paged_kv_max_sequence_length = 0;
    float dropout_probability = 0.0f;
    DataType compute_dtype = DataType::FP32;
    DataType dQ_dtype = DataType::FP16;
    DataType dK_dtype = DataType::FP16;
    DataType dV_dtype = DataType::FP16;
    std::string debug_name = "thor_expr_attention_backward";

    CudnnAttentionDescriptor descriptorFor(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o) const;
    [[nodiscard]] DataType outputDTypeFor(ExprOp op) const;
};

struct BuiltConvolution {
    bool use_cudnn_frontend = false;
    std::shared_ptr<cudnn_frontend::graph::Graph> frontend_graph;
    // Placement-time autotuning saves the measured execution plan and exact workspace size here.
    // Runtime execution must use this saved plan directly rather than re-running cuDNN heuristics.
    int64_t selected_plan_index = -1;
    size_t workspace_bytes = 0;

    BuiltConvolution() = default;
    BuiltConvolution(const BuiltConvolution&) = delete;
    BuiltConvolution& operator=(const BuiltConvolution&) = delete;
    ~BuiltConvolution() = default;
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

    [[nodiscard]] bool requiresRuntimeScalars() const;
    [[nodiscard]] std::unordered_set<std::string> runtimeScalarNames() const;
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
                                                          DataType input_dtype,
                                                          DataType output_dtype,
                                                          DataType compute_dtype,
                                                          bool output_indices,
                                                          const Tensor& input,
                                                          int device_num);

    static std::shared_ptr<BuiltSoftmax> buildSoftmax(const std::shared_ptr<CompiledSoftmax>& compiled_softmax,
                                                      const Tensor& input,
                                                      const Tensor& output,
                                                      int device_num);

    static std::shared_ptr<BuiltMatmul> buildMatmul(const std::shared_ptr<CompiledMatmul>& compiled_matmul,
                                                    const Tensor& lhs,
                                                    const Tensor& rhs,
                                                    const std::optional<Tensor>& addend,
                                                    const Tensor& output,
                                                    int device_num,
                                                    const std::optional<Tensor>& epilogue_aux = std::nullopt,
                                                    const std::optional<Tensor>& bgrad_output = std::nullopt);

    static std::shared_ptr<BuiltConvolution> buildConvolution(const std::shared_ptr<CompiledConvolution>& compiled_convolution,
                                                              const Tensor& input,
                                                              const Tensor& filter,
                                                              const Tensor& output,
                                                              const Stream& stream,
                                                              int device_num);
    static std::shared_ptr<BuiltConvolution> buildConvolutionBackward(
        const std::shared_ptr<CompiledConvolutionBackward>& compiled_convolution_backward,
        const Tensor& input,
        const Tensor& grad_output,
        const Tensor& output,
        const Stream& stream,
        int device_num);

   private:
    std::shared_ptr<CompiledEquation> compiledEquation;
    std::vector<RuntimeInputValue> inputs;
    std::vector<Tensor> outputs;
    Stream stream;
};

class StampedCudaKernel {
   public:
    StampedCudaKernel(std::shared_ptr<CompiledCudaKernel> compiled,
                      std::vector<Tensor> inputs,
                      std::vector<TensorScalarBinding> tensor_runtime_scalars,
                      std::vector<Tensor> outputs,
                      std::vector<StampedCudaKernelParam> params,
                      CudaKernelLaunchConfig launch_config,
                      const Stream& stream);

    void run();
    void runOn(Stream& run_stream) const;
    void run(const std::unordered_map<std::string, float>& runtime_scalars);
    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;

    [[nodiscard]] bool requiresRuntimeScalars() const;
    [[nodiscard]] std::unordered_set<std::string> runtimeScalarNames() const;
    [[nodiscard]] uint32_t gpuNum() const;
    [[nodiscard]] const std::vector<Tensor>& getOutputTensors() const { return outputs; }
    [[nodiscard]] Tensor getOutputTensor() const;

   private:
    std::shared_ptr<CompiledCudaKernel> compiled;
    std::vector<Tensor> inputs;
    std::vector<TensorScalarBinding> tensor_runtime_scalars;
    std::vector<Tensor> outputs;
    std::vector<StampedCudaKernelParam> params;
    CudaKernelLaunchConfig launch_config;
    Stream stream;
};

class StampedReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedReduction(std::shared_ptr<BuiltReduction> built,
                     const Tensor& source_input,
                     const Tensor& input,
                     const Tensor& output,
                     const Stream& stream,
                     std::optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor source_input;
    mutable Tensor input;
    Tensor output;
    const std::optional<Tensor> workspace;
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
                     const Tensor& source_input,
                     const Tensor& input,
                     const Tensor& output,
                     const Tensor& reduction_value_output,
                     const Stream& stream,
                     std::optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor source_input;
    mutable Tensor input;
    Tensor output;
    const Tensor reduction_value_output;
    const std::optional<Tensor> workspace;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};

class StampedSegmentedReduction {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedSegmentedReduction(std::shared_ptr<CompiledSegmentedReduction> compiled,
                              const Tensor& input,
                              const Tensor& output,
                              const Tensor& segment_offsets,
                              const Stream& stream);

   private:
    const std::shared_ptr<CompiledSegmentedReduction> compiled_segmented_reduction;
    const Tensor input;
    mutable Tensor output;
    const Tensor segment_offsets;
    Tensor temp_storage;
    Stream stream;
    CubDeviceSegmentedReduceSumPlan sum_plan;
    CubDeviceSegmentedReduceMinPlan min_plan;
    CubDeviceSegmentedReduceMaxPlan max_plan;
};

class StampedScan {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }
    Tensor getValueOutputTensor() const { return value_output; }

    StampedScan(std::shared_ptr<CompiledScan> compiled,
                const Tensor& input,
                const Tensor& output,
                const Stream& stream,
                std::optional<Tensor> segment_offsets = std::nullopt,
                std::optional<Tensor> value_output = std::nullopt);

   private:
    const std::shared_ptr<CompiledScan> compiled_scan;
    const Tensor input;
    mutable Tensor output;
    mutable Tensor value_output;
    const std::optional<Tensor> segment_offsets;
    bool has_value_output = false;
    Tensor temp_storage;
    Stream stream;
    bool uniform_segmented = false;
    bool ragged_segmented = false;
    CubDeviceScanPlan scan_plan;
    CubDeviceArgScanPlan arg_scan_plan;
    CubDeviceSegmentedUniformScanPlan segmented_scan_plan;
    CubDeviceSegmentedUniformArgScanPlan segmented_arg_scan_plan;
    CubDeviceSegmentedScanPlan ragged_segmented_scan_plan;
    CubDeviceSegmentedArgScanPlan ragged_segmented_arg_scan_plan;
};

class StampedSoftmax {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedSoftmax(std::shared_ptr<CompiledSoftmax> compiled,
                   std::shared_ptr<BuiltSoftmax> built,
                   const Tensor& source_input,
                   const Tensor& input,
                   const Tensor& output,
                   const Stream& stream);

   private:
    const std::shared_ptr<CompiledSoftmax> compiled_softmax;
    const std::shared_ptr<BuiltSoftmax> built_softmax;
    const Tensor source_input;
    mutable Tensor input;
    Tensor output;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};


class StampedRmsNorm {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedRmsNorm(std::shared_ptr<CompiledRmsNorm> compiled, const Tensor& input, const Tensor& scale, const Tensor& output, const Stream& stream);

   private:
    const std::shared_ptr<CompiledRmsNorm> compiled_rms_norm;
    const Tensor input;
    const Tensor scale;
    Tensor output;
    Stream stream;
};


class StampedEmbeddingLookup {
   public:
    void run() { runOn(stream); }
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedEmbeddingLookup(std::shared_ptr<CompiledEmbeddingLookup> compiled,
                           const Tensor& indices,
                           const Tensor& weights,
                           const Tensor& output,
                           const Stream& stream,
                           std::vector<Tensor> epilogue_inputs = {});

   private:
    const std::shared_ptr<CompiledEmbeddingLookup> compiled_embedding_lookup;
    const Tensor indices;
    const Tensor weights;
    mutable Tensor output;
    Stream stream;
    std::vector<Tensor> epilogue_inputs;
    std::shared_ptr<PreparedEmbeddingForward> prepared_forward;
};

class StampedMatmul {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }
    std::optional<Tensor> getBiasGradientTensor() const { return bgrad_output; }

    StampedMatmul(std::shared_ptr<CompiledMatmul> compiled,
                  std::shared_ptr<BuiltMatmul> built,
                  const Tensor& lhs,
                  const Tensor& rhs,
                  const std::optional<Tensor>& addend,
                  const Tensor& output,
                  const Stream& stream,
                  std::optional<Tensor> workspace,
                  std::optional<RuntimeInputValue> alpha_input,
                  std::optional<RuntimeInputValue> beta_input,
                  std::optional<std::string> alpha_runtime_name,
                  std::optional<std::string> beta_runtime_name,
                  std::optional<Tensor> alpha_device_scratch,
                  std::optional<Tensor> beta_device_scratch,
                  std::optional<Tensor> alpha_host_scratch,
                  std::optional<Tensor> beta_host_scratch,
                  std::optional<Tensor> epilogue_aux = std::nullopt,
                  std::optional<Tensor> bgrad_output = std::nullopt);

    [[nodiscard]] std::optional<std::string> alphaRuntimeName() const { return alpha_runtime_name; }
    [[nodiscard]] std::optional<std::string> betaRuntimeName() const { return beta_runtime_name; }

   private:
    const std::shared_ptr<CompiledMatmul> compiled_matmul;
    const std::shared_ptr<BuiltMatmul> built_matmul;
    const Tensor lhs;
    const Tensor rhs;
    const std::optional<Tensor> addend;
    Tensor output;
    const std::optional<Tensor> epilogue_aux;
    const std::optional<Tensor> bgrad_output;
    Stream stream;
    const std::optional<Tensor> workspace;
    const std::optional<RuntimeInputValue> alpha_input;
    const std::optional<RuntimeInputValue> beta_input;
    const std::optional<std::string> alpha_runtime_name;
    const std::optional<std::string> beta_runtime_name;
    const std::optional<Tensor> alpha_device_scratch;
    const std::optional<Tensor> beta_device_scratch;
    const std::optional<Tensor> alpha_host_scratch;
    const std::optional<Tensor> beta_host_scratch;
};

struct AttentionForwardState {
    Tensor output;
    Tensor stats;
    bool retain_for_backward = false;
    bool has_valid_stats = false;
};

class StampedAttention {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }
    std::shared_ptr<AttentionForwardState> getForwardState() const { return forward_state; }

    bool canProvideForwardStateFor(const CompiledAttentionBackward& backward,
                                   const Tensor& q_tensor,
                                   const Tensor& k_tensor,
                                   const Tensor& v_tensor,
                                   const std::optional<Tensor>& bias_tensor,
                                   const std::optional<Tensor>& seq_len_q_tensor,
                                   const std::optional<Tensor>& seq_len_kv_tensor,
                                   const std::optional<Tensor>& q_ragged_offsets_tensor,
                                   const std::optional<Tensor>& kv_ragged_offsets_tensor,
                                   const std::optional<Tensor>& dropout_seed_tensor,
                                   const std::optional<Tensor>& dropout_offset_tensor,
                                   const Tensor& dO_tensor) const;

    StampedAttention(std::shared_ptr<CompiledAttention> compiled,
                     const Tensor& q,
                     const Tensor& k,
                     const Tensor& v,
                     const std::optional<Tensor>& bias,
                     const std::optional<Tensor>& seq_len_q,
                     const std::optional<Tensor>& seq_len_kv,
                     const std::optional<Tensor>& q_ragged_offsets,
                     const std::optional<Tensor>& kv_ragged_offsets,
                     const std::optional<Tensor>& page_table_k,
                     const std::optional<Tensor>& page_table_v,
                     const std::optional<Tensor>& dropout_seed,
                     const std::optional<Tensor>& dropout_offset,
                     const std::optional<Tensor>& descale_q,
                     const std::optional<Tensor>& descale_k,
                     const std::optional<Tensor>& descale_v,
                     const std::optional<Tensor>& descale_s,
                     const std::optional<Tensor>& scale_s,
                     const std::optional<Tensor>& scale_o,
                     const std::optional<Tensor>& amax_s,
                     const std::optional<Tensor>& amax_o,
                     const Tensor& output,
                     const Stream& stream,
                     std::shared_ptr<AttentionForwardState> forward_state = nullptr);

   private:
    const std::shared_ptr<CompiledAttention> compiled_attention;
    const Tensor q;
    const Tensor k;
    const Tensor v;
    const std::optional<Tensor> bias;
    const std::optional<Tensor> seq_len_q;
    const std::optional<Tensor> seq_len_kv;
    const std::optional<Tensor> q_ragged_offsets;
    const std::optional<Tensor> kv_ragged_offsets;
    const std::optional<Tensor> page_table_k;
    const std::optional<Tensor> page_table_v;
    const std::optional<Tensor> dropout_seed;
    const std::optional<Tensor> dropout_offset;
    const std::optional<Tensor> descale_q;
    const std::optional<Tensor> descale_k;
    const std::optional<Tensor> descale_v;
    const std::optional<Tensor> descale_s;
    const std::optional<Tensor> scale_s;
    const std::optional<Tensor> scale_o;
    const std::optional<Tensor> amax_s;
    const std::optional<Tensor> amax_o;
    Tensor output;
    Stream stream;
    std::shared_ptr<AttentionForwardState> forward_state;
};

class StampedAttentionBackward {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return dQ.getPlacement().getDeviceNum(); }

    const std::vector<Tensor>& getOutputTensors() const { return outputs; }

    StampedAttentionBackward(std::shared_ptr<CompiledAttentionBackward> compiled,
                             const Tensor& q,
                             const Tensor& k,
                             const Tensor& v,
                             const std::optional<Tensor>& bias,
                             const std::optional<Tensor>& seq_len_q,
                             const std::optional<Tensor>& seq_len_kv,
                             const std::optional<Tensor>& q_ragged_offsets,
                             const std::optional<Tensor>& kv_ragged_offsets,
                             const std::optional<Tensor>& dropout_seed,
                             const std::optional<Tensor>& dropout_offset,
                             const Tensor& dO,
                             const Tensor& dQ,
                             const Tensor& dK,
                             const Tensor& dV,
                             const Tensor& oScratch,
                             const Tensor& stats,
                             const std::optional<Tensor>& dBiasScratch,
                             const Stream& stream,
                             std::shared_ptr<AttentionForwardState> saved_forward_state = nullptr);

   private:
    const std::shared_ptr<CompiledAttentionBackward> compiled_attention_backward;
    const Tensor q;
    const Tensor k;
    const Tensor v;
    const std::optional<Tensor> bias;
    const std::optional<Tensor> seq_len_q;
    const std::optional<Tensor> seq_len_kv;
    const std::optional<Tensor> q_ragged_offsets;
    const std::optional<Tensor> kv_ragged_offsets;
    const std::optional<Tensor> dropout_seed;
    const std::optional<Tensor> dropout_offset;
    const Tensor dO;
    Tensor dQ;
    Tensor dK;
    Tensor dV;
    Tensor oScratch;
    Tensor stats;
    std::optional<Tensor> dBiasScratch;
    Stream stream;
    std::shared_ptr<AttentionForwardState> saved_forward_state;
    std::vector<Tensor> outputs;
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
                       std::optional<Tensor> workspace);

   private:
    const std::shared_ptr<CompiledConvolution> compiled_convolution;
    const std::shared_ptr<BuiltConvolution> built_convolution;
    const Tensor input;
    const Tensor filter;
    Tensor output;
    Stream stream;
    const std::optional<Tensor> workspace;
};

class StampedConvolutionBackward {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedConvolutionBackward(std::shared_ptr<CompiledConvolutionBackward> compiled,
                               std::shared_ptr<BuiltConvolution> built,
                               const Tensor& input,
                               const Tensor& grad_output,
                               const Tensor& output,
                               const Stream& stream,
                               std::optional<Tensor> workspace);

   private:
    const std::shared_ptr<CompiledConvolutionBackward> compiled_convolution_backward;
    const std::shared_ptr<BuiltConvolution> built_convolution;
    const Tensor input;
    const Tensor grad_output;
    Tensor output;
    Stream stream;
    const std::optional<Tensor> workspace;
};


class StampedScanMinMaxBackward {
   public:
    void run();
    void runOn(Stream& run_stream);

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedScanMinMaxBackward(std::shared_ptr<CompiledScanMinMaxBackward> compiled,
                              std::shared_ptr<StampedScan> arg_scan,
                              std::shared_ptr<BuiltFlatScatterAdd> scatter_add,
                              const Tensor& input,
                              const Tensor& grad_output,
                              const Tensor& output,
                              const Tensor& indices,
                              const Stream& stream);

   private:
    const std::shared_ptr<CompiledScanMinMaxBackward> compiled_scan_minmax_backward;
    const std::shared_ptr<StampedScan> arg_scan;
    const std::shared_ptr<BuiltFlatScatterAdd> scatter_add;
    const Tensor input;
    const Tensor grad_output;
    Tensor output;
    Tensor indices;
    Stream stream;
};

class StampedReduceMinMaxBackward {
   public:
    void run();
    void runOn(Stream& run_stream);

    uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }

    Tensor getOutputTensor() const { return output; }

    StampedReduceMinMaxBackward(std::shared_ptr<BuiltReduction> built,
                                const Tensor& source_input,
                                const Tensor& input,
                                const Tensor& grad_output,
                                const Tensor& output,
                                const Tensor& indices,
                                const Tensor& reduction_value_output,
                                const Stream& stream,
                                std::optional<Tensor> workspace);

   private:
    const std::shared_ptr<BuiltReduction> built_reduction;
    const Tensor source_input;
    mutable Tensor input;
    const Tensor grad_output;
    Tensor output;
    const Tensor indices;
    const Tensor reduction_value_output;
    const std::optional<Tensor> workspace;
    Stream stream;

    const float alpha_1 = 1.0f;
    const float beta_0 = 0.0f;
    const void* alpha = &alpha_1;
    const void* beta = &beta_0;
};


class StampedInPlaceRope {
   public:
    StampedInPlaceRope(std::shared_ptr<CompiledInPlaceRope> compiled, std::vector<Tensor> tensors, const Stream& stream)
        : compiled(std::move(compiled)), tensors(std::move(tensors)), stream(stream) {
        if (!this->compiled) {
            throw std::runtime_error("StampedInPlaceRope requires a compiled plan.");
        }
        if (this->compiled->tensors.size() != this->tensors.size()) {
            throw std::runtime_error("StampedInPlaceRope tensor count mismatch.");
        }
    }

    [[nodiscard]] uint32_t gpuNum() const {
        if (tensors.empty()) {
            throw std::runtime_error("StampedInPlaceRope has no tensors.");
        }
        return static_cast<uint32_t>(tensors.front().getPlacement().getDeviceNum());
    }

    [[nodiscard]] const Tensor& outputTensor(size_t idx) const {
        if (idx >= tensors.size()) {
            throw std::runtime_error("StampedInPlaceRope output index out of range.");
        }
        return tensors[idx];
    }

    void runOn(Stream& run_stream) const {
        std::vector<RotaryPositionEmbeddingOptions> options;
        options.reserve(compiled->tensors.size());
        for (const CompiledInPlaceRopeTensor& tensor : compiled->tensors) {
            options.push_back(tensor.options);
        }
        std::vector<Tensor> mutable_tensors = tensors;
        runGroupedInPlaceRotaryPositionEmbedding(mutable_tensors, options, run_stream);
    }

   private:
    std::shared_ptr<CompiledInPlaceRope> compiled;
    std::vector<Tensor> tensors;
    Stream stream;
};

class StampedConditional {
   public:
    StampedConditional(std::shared_ptr<StampedExecutionPlan> predicate_plan,
                       std::shared_ptr<StampedExecutionPlan> then_plan,
                       std::shared_ptr<StampedExecutionPlan> else_plan,
                       std::vector<std::string> output_names,
                       const Stream& stream);

    void run();
    void run(const std::unordered_map<std::string, float>& runtime_scalars);
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;
    [[nodiscard]] bool requiresRuntimeScalars() const;
    [[nodiscard]] std::unordered_set<std::string> runtimeScalarNames() const;
    uint32_t gpuNum() const;

   private:
    std::shared_ptr<StampedExecutionPlan> predicate_plan;
    std::shared_ptr<StampedExecutionPlan> then_plan;
    std::shared_ptr<StampedExecutionPlan> else_plan;
    std::vector<std::string> output_names;
    Stream stream;
    CudaGraphExecutable conditional_graph;
};

struct StampedExecutionStage {
    enum class Kind { FusedKernel, CudaKernel, Reduction, ArgMinMax, SegmentedReduction, Scan, Softmax, RmsNorm, EmbeddingLookup, Matmul, InPlaceRope, Attention, AttentionBackward, Convolution, ConvolutionBackward, ReduceMinMaxBackward, ScanMinMaxBackward, Conditional };
    static std::string kindToString(const Kind kind) {
        switch (kind) {
            case Kind::FusedKernel:
                return "FusedKernel";
            case Kind::CudaKernel:
                return "CudaKernel";
            case Kind::Reduction:
                return "Reduction";
            case Kind::ArgMinMax:
                return "ArgMinMax";
            case Kind::SegmentedReduction:
                return "SegmentedReduction";
            case Kind::Scan:
                return "Scan";
            case Kind::Softmax:
                return "Softmax";
            case Kind::RmsNorm:
                return "RmsNorm";
            case Kind::EmbeddingLookup:
                return "EmbeddingLookup";
            case Kind::Matmul:
                return "Matmul";
            case Kind::InPlaceRope:
                return "InPlaceRope";
            case Kind::Attention:
                return "Attention";
            case Kind::AttentionBackward:
                return "AttentionBackward";
            case Kind::Convolution:
                return "Convolution";
            case Kind::ConvolutionBackward:
                return "ConvolutionBackward";
            case Kind::ReduceMinMaxBackward:
                return "ReduceMinMaxBackward";
            case Kind::ScanMinMaxBackward:
                return "ScanMinMaxBackward";
            case Kind::Conditional:
                return "Conditional";
        }
        return "<unknown>";
    }

    const Kind kind;

    const std::vector<uint32_t> dependency_stage_indices;
    const uint32_t gpu_num;
    const uint64_t flop_count = 0;

    const std::shared_ptr<StampedEquation> kernel = nullptr;
    const std::shared_ptr<StampedCudaKernel> cuda_kernel = nullptr;
    const std::shared_ptr<StampedReduction> reduction = nullptr;
    const std::shared_ptr<StampedArgMinMax> arg_minmax = nullptr;
    const std::shared_ptr<StampedSegmentedReduction> segmented_reduction = nullptr;
    const std::shared_ptr<StampedScan> scan = nullptr;
    const std::shared_ptr<StampedSoftmax> softmax = nullptr;
    const std::shared_ptr<StampedRmsNorm> rms_norm = nullptr;
    const std::shared_ptr<StampedEmbeddingLookup> embedding_lookup = nullptr;
    const std::shared_ptr<StampedMatmul> matmul = nullptr;
    const std::shared_ptr<StampedInPlaceRope> in_place_rope = nullptr;
    const std::shared_ptr<StampedAttention> attention = nullptr;
    const std::shared_ptr<StampedAttentionBackward> attention_backward = nullptr;
    const std::shared_ptr<StampedConvolution> convolution = nullptr;
    const std::shared_ptr<StampedConvolutionBackward> convolution_backward = nullptr;
    const std::shared_ptr<StampedReduceMinMaxBackward> reduce_minmax_backward = nullptr;
    const std::shared_ptr<StampedScanMinMaxBackward> scan_minmax_backward = nullptr;
    const std::shared_ptr<StampedConditional> conditional = nullptr;

    explicit StampedExecutionStage(const std::shared_ptr<StampedEquation>& fused,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::FusedKernel),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(fused->gpuNum()),
          flop_count(flop_count),
          kernel(fused) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedCudaKernel>& cuda_kernel,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::CudaKernel),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(cuda_kernel->gpuNum()),
          flop_count(flop_count),
          cuda_kernel(cuda_kernel) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedReduction>& reduction,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Reduction),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(reduction->gpuNum()),
          flop_count(flop_count),
          reduction(reduction) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedArgMinMax>& arg_minmax,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::ArgMinMax),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(arg_minmax->gpuNum()),
          flop_count(flop_count),
          arg_minmax(arg_minmax) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedSegmentedReduction>& segmented_reduction,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::SegmentedReduction),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(segmented_reduction->gpuNum()),
          flop_count(flop_count),
          segmented_reduction(segmented_reduction) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedScan>& scan,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Scan),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(scan->gpuNum()),
          flop_count(flop_count),
          scan(scan) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedSoftmax>& softmax,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Softmax),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(softmax->gpuNum()),
          flop_count(flop_count),
          softmax(softmax) {}


    explicit StampedExecutionStage(const std::shared_ptr<StampedRmsNorm>& rms_norm,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::RmsNorm),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(rms_norm->gpuNum()),
          flop_count(flop_count),
          rms_norm(rms_norm) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedEmbeddingLookup>& embedding_lookup,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::EmbeddingLookup),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(embedding_lookup->gpuNum()),
          flop_count(flop_count),
          embedding_lookup(embedding_lookup) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedMatmul>& matmul,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Matmul),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(matmul->gpuNum()),
          flop_count(flop_count),
          matmul(matmul) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedInPlaceRope>& in_place_rope,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::InPlaceRope),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(in_place_rope->gpuNum()),
          flop_count(flop_count),
          in_place_rope(in_place_rope) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedAttention>& attention,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Attention),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(attention->gpuNum()),
          flop_count(flop_count),
          attention(attention) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedAttentionBackward>& attention_backward,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::AttentionBackward),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(attention_backward->gpuNum()),
          flop_count(flop_count),
          attention_backward(attention_backward) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedConvolution>& convolution,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Convolution),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(convolution->gpuNum()),
          flop_count(flop_count),
          convolution(convolution) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedConvolutionBackward>& convolution_backward,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::ConvolutionBackward),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(convolution_backward->gpuNum()),
          flop_count(flop_count),
          convolution_backward(convolution_backward) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedReduceMinMaxBackward>& reduce_minmax_backward,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::ReduceMinMaxBackward),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(reduce_minmax_backward->gpuNum()),
          flop_count(flop_count),
          reduce_minmax_backward(reduce_minmax_backward) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedScanMinMaxBackward>& scan_minmax_backward,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::ScanMinMaxBackward),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(scan_minmax_backward->gpuNum()),
          flop_count(flop_count),
          scan_minmax_backward(scan_minmax_backward) {}

    explicit StampedExecutionStage(const std::shared_ptr<StampedConditional>& conditional,
                                   std::vector<uint32_t> dependency_stage_indices = {},
                                   uint64_t flop_count = 0)
        : kind(Kind::Conditional),
          dependency_stage_indices(std::move(dependency_stage_indices)),
          gpu_num(conditional->gpuNum()),
          flop_count(flop_count),
          conditional(conditional) {}

    [[nodiscard]] uint64_t flopCount() const { return flop_count; }

    void runOn(Stream& run_stream) const { runOn(run_stream, {}); }

    void runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
        if (kind == Kind::FusedKernel) {
            THOR_THROW_IF_FALSE(kernel != nullptr);
            if (runtime_scalars.empty())
                kernel->runOn(run_stream);
            else
                kernel->runOn(run_stream, runtime_scalars);
        } else if (kind == Kind::CudaKernel) {
            THOR_THROW_IF_FALSE(cuda_kernel != nullptr);
            if (runtime_scalars.empty())
                cuda_kernel->runOn(run_stream);
            else
                cuda_kernel->runOn(run_stream, runtime_scalars);
        } else if (kind == Kind::Reduction) {
            THOR_THROW_IF_FALSE(reduction != nullptr);
            reduction->runOn(run_stream);
        } else if (kind == Kind::ArgMinMax) {
            THOR_THROW_IF_FALSE(arg_minmax != nullptr);
            arg_minmax->runOn(run_stream);
        } else if (kind == Kind::SegmentedReduction) {
            THOR_THROW_IF_FALSE(segmented_reduction != nullptr);
            segmented_reduction->runOn(run_stream);
        } else if (kind == Kind::Scan) {
            THOR_THROW_IF_FALSE(scan != nullptr);
            scan->runOn(run_stream);
        } else if (kind == Kind::Softmax) {
            THOR_THROW_IF_FALSE(softmax != nullptr);
            softmax->runOn(run_stream);
        } else if (kind == Kind::RmsNorm) {
            THOR_THROW_IF_FALSE(rms_norm != nullptr);
            rms_norm->runOn(run_stream);
        } else if (kind == Kind::EmbeddingLookup) {
            THOR_THROW_IF_FALSE(embedding_lookup != nullptr);
            embedding_lookup->runOn(run_stream);
        } else if (kind == Kind::Matmul) {
            THOR_THROW_IF_FALSE(matmul != nullptr);
            if (runtime_scalars.empty())
                matmul->runOn(run_stream);
            else
                matmul->runOn(run_stream, runtime_scalars);
        } else if (kind == Kind::InPlaceRope) {
            THOR_THROW_IF_FALSE(in_place_rope != nullptr);
            in_place_rope->runOn(run_stream);
        } else if (kind == Kind::Attention) {
            THOR_THROW_IF_FALSE(attention != nullptr);
            attention->runOn(run_stream);
        } else if (kind == Kind::AttentionBackward) {
            THOR_THROW_IF_FALSE(attention_backward != nullptr);
            attention_backward->runOn(run_stream);
        } else if (kind == Kind::Convolution) {
            THOR_THROW_IF_FALSE(convolution != nullptr);
            convolution->runOn(run_stream);
        } else if (kind == Kind::ConvolutionBackward) {
            THOR_THROW_IF_FALSE(convolution_backward != nullptr);
            convolution_backward->runOn(run_stream);
        } else if (kind == Kind::ReduceMinMaxBackward) {
            THOR_THROW_IF_FALSE(reduce_minmax_backward != nullptr);
            reduce_minmax_backward->runOn(run_stream);
        } else if (kind == Kind::ScanMinMaxBackward) {
            THOR_THROW_IF_FALSE(scan_minmax_backward != nullptr);
            scan_minmax_backward->runOn(run_stream);
        } else if (kind == Kind::Conditional) {
            THOR_THROW_IF_FALSE(conditional != nullptr);
            if (runtime_scalars.empty())
                conditional->runOn(run_stream);
            else
                conditional->runOn(run_stream, runtime_scalars);
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
    void runSequentialOn(Stream& run_stream) const;
    void runSequentialOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const;

    [[nodiscard]] bool requiresRuntimeScalars() const;
    [[nodiscard]] std::unordered_set<std::string> runtimeScalarNames() const;

    [[nodiscard]] uint64_t flopCount() const {
        uint64_t total = 0;
        for (const StampedExecutionStage& step : steps) {
            const uint64_t f = step.flopCount();
            if (std::numeric_limits<uint64_t>::max() - total < f) {
                throw std::runtime_error("StampedExecutionPlan::flopCount overflow.");
            }
            total += f;
        }
        return total;
    }

    [[nodiscard]] std::vector<uint64_t> stageFlopCounts() const {
        std::vector<uint64_t> out;
        out.reserve(steps.size());
        for (const StampedExecutionStage& step : steps) {
            out.push_back(step.flopCount());
        }
        return out;
    }

    [[nodiscard]] std::vector<std::string> stageKindNames() const {
        std::vector<std::string> out;
        out.reserve(steps.size());
        for (const StampedExecutionStage& step : steps) {
            out.push_back(StampedExecutionStage::kindToString(step.kind));
        }
        return out;
    }

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
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.input_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.compute_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.output_dtype));
        hashCombine(h, hash<bool>{}(k.output_indices));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};

template <>
struct hash<ThorImplementation::SoftmaxCacheKey> {
    size_t operator()(const ThorImplementation::SoftmaxCacheKey& k) const noexcept {
        size_t h = 0;
        hashCombine(h, hash<size_t>{}(k.input_dims.size()));
        for (uint64_t d : k.input_dims)
            hashCombine(h, hash<uint64_t>{}(d));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.input_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.output_dtype));
        hashCombine(h, hash<int>{}(static_cast<int>(k.algorithm)));
        hashCombine(h, hash<int>{}(static_cast<int>(k.mode)));
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
        hashCombine(h, hash<bool>{}(k.bias_epilogue));
        hashCombine(h, hash<int>{}(static_cast<int>(k.epilogue)));
        hashCombine(h, hash<int>{}(static_cast<int>(k.backward_epilogue)));
        hashCombine(h, hash<bool>{}(k.bgrad_epilogue));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.a_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.b_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.c_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.d_dtype));
        hashCombine(h, hash<ThorImplementation::DataType>{}(k.compute_dtype));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};
}  // namespace std
