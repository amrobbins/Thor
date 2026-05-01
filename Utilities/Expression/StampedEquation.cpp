#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationRunner.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/MatmulScalarKernel.h"
#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

void StampedEquation::run() { runOn(stream); }

void StampedEquation::runOn(Stream& run_stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("StampedEquation::runOn called with null compiled equation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("StampedEquation::runOn called with no output tensors.");
    }

    for (size_t i = 0; i < compiledEquation->input_kinds.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            throw std::runtime_error("StampedEquation::runOn requires runtime scalar values. Call run(runtime_scalars).");
        }
    }

    EquationRunner::run(compiledEquation, inputs, outputs, run_stream);
}

void StampedEquation::run(const std::unordered_map<std::string, float>& runtime_scalars) { runOn(stream, runtime_scalars); }

void StampedEquation::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (!compiledEquation) {
        throw std::runtime_error("StampedEquation::runOn called with null compiled equation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("StampedEquation::runOn called with no output tensors.");
    }

    if (runtime_scalars.empty()) {
        runOn(run_stream);
        return;
    }

    std::vector<RuntimeInputValue> overridden_inputs = inputs;
    std::unordered_set<std::string> consumed_names;

    for (size_t i = 0; i < compiledEquation->input_names.size(); ++i) {
        if (compiledEquation->input_kinds[i] != NamedInput::Kind::RuntimeScalarFp32) {
            continue;
        }

        const std::string& name = compiledEquation->input_names[i];
        auto it = runtime_scalars.find(name);
        if (it == runtime_scalars.end()) {
            throw std::runtime_error("Missing value for runtime scalar: " + name +
                                     "  - if it was meant to be constant, use a constant scalar instead.");
        }

        overridden_inputs[i] = it->second;
        consumed_names.insert(name);
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped equation: " + name);
        }
    }

    EquationRunner::run(compiledEquation, overridden_inputs, outputs, run_stream);
}

StampedReduction::StampedReduction(
    std::shared_ptr<BuiltReduction> built, const Tensor& input, const Tensor& output, const Stream& stream, Optional<Tensor> workspace)
    : built_reduction(built), input(input), output(output), workspace(workspace), stream(stream) {
    if (built_reduction->workspace_bytes != 0) {
        assert(workspace.isPresent());
        assert(workspace.get().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
    assert(input.getDataType() == built_reduction->key.input_dtype);
    assert(output.getDataType() == built_reduction->key.output_dtype);
}

void StampedReduction::run() { runOn(stream); }

void StampedReduction::runOn(Stream& run_stream) const {
    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        assert(workspace.isPresent());
        workspace_ptr = workspace.get().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  nullptr,
                                  0,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)output.getMemPtr()));
}

StampedArgMinMax::StampedArgMinMax(std::shared_ptr<BuiltReduction> built,
                                   const Tensor& input,
                                   const Tensor& output,
                                   const Tensor& reduction_value_output,
                                   const Stream& stream,
                                   Optional<Tensor> workspace)
    : built_reduction(built),
      input(input),
      output(output),
      reduction_value_output(reduction_value_output),
      workspace(workspace),
      stream(stream) {
    if (!built_reduction->key.output_indices) {
        throw std::runtime_error("StampedArgMinMax requires a BuiltReduction configured for indices.");
    }
    if (built_reduction->workspace_bytes != 0) {
        assert(workspace.isPresent());
        assert(workspace.get().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
}

void StampedArgMinMax::run() { runOn(stream); }

void StampedArgMinMax::runOn(Stream& run_stream) const {
    // std::cerr << "[REDUCE_MINMAX_BW] input dtype=" << TensorDescriptor::getElementTypeName(input.getDataType())
    //           << " built.input_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.input_dtype)
    //           << " built.output_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.output_dtype)
    //           << " built.compute_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.compute_dtype)
    //           << " reduction_value_output dtype=" << TensorDescriptor::getElementTypeName(reduction_value_output.getDataType())
    //           << std::endl;

    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        assert(workspace.isPresent());
        workspace_ptr = workspace.get().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  (void*)output.getMemPtr(),
                                  built_reduction->indices_bytes,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)reduction_value_output.getMemPtr()));
}

StampedConvolution::StampedConvolution(std::shared_ptr<CompiledConvolution> compiled,
                                       std::shared_ptr<BuiltConvolution> built,
                                       const Tensor& input,
                                       const Tensor& filter,
                                       const Tensor& output,
                                       const Stream& stream,
                                       Optional<Tensor> workspace)
    : compiled_convolution(std::move(compiled)),
      built_convolution(std::move(built)),
      input(input),
      filter(filter),
      output(output),
      stream(stream),
      workspace(std::move(workspace)) {}

void StampedConvolution::run() { runOn(stream); }

void StampedConvolution::runOn(Stream& run_stream) const {
    if (!built_convolution || !built_convolution->requirement.isPresent()) {
        throw std::runtime_error("StampedConvolution missing built convolution requirement.");
    }

    GpuConvolution::instance().convolutionForward(
        built_convolution->requirement.get(), input, filter, Optional<Tensor>::empty(), output, workspace, run_stream);
}

StampedConvolutionBackward::StampedConvolutionBackward(std::shared_ptr<CompiledConvolutionBackward> compiled,
                                                       std::shared_ptr<BuiltConvolution> built,
                                                       const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& output,
                                                       const Stream& stream,
                                                       Optional<Tensor> workspace)
    : compiled_convolution_backward(std::move(compiled)),
      built_convolution(std::move(built)),
      input(input),
      grad_output(grad_output),
      output(output),
      stream(stream),
      workspace(std::move(workspace)) {}

void StampedConvolutionBackward::run() { runOn(stream); }

void StampedConvolutionBackward::runOn(Stream& run_stream) const {
    if (!built_convolution || !built_convolution->requirement.isPresent()) {
        throw std::runtime_error("StampedConvolutionBackward missing built convolution requirement.");
    }
    if (!compiled_convolution_backward) {
        throw std::runtime_error("StampedConvolutionBackward missing compiled convolution payload.");
    }

    if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_DATA) {
        GpuConvolution::instance().convolutionBackwardData(
            built_convolution->requirement.get(), grad_output, input, output, workspace, run_stream);
    } else if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_FILTER) {
        GpuConvolution::instance().convolutionBackwardFilter(
            built_convolution->requirement.get(), input, grad_output, output, workspace, run_stream, false);
    } else {
        throw std::runtime_error("StampedConvolutionBackward received unsupported convolution backward op.");
    }
}

StampedMatmul::StampedMatmul(std::shared_ptr<CompiledMatmul> compiled,
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
                             Optional<Tensor> beta_host_scratch)
    : compiled_matmul(std::move(compiled)),
      built_matmul(std::move(built)),
      lhs(lhs),
      rhs(rhs),
      addend(addend),
      output(output),
      stream(stream),
      workspace(workspace),
      alpha_input(alpha_input),
      beta_input(beta_input),
      alpha_runtime_name(std::move(alpha_runtime_name)),
      beta_runtime_name(std::move(beta_runtime_name)),
      alpha_device_scratch(alpha_device_scratch),
      beta_device_scratch(beta_device_scratch),
      alpha_host_scratch(alpha_host_scratch),
      beta_host_scratch(beta_host_scratch) {
    if (!compiled_matmul) {
        throw std::runtime_error("StampedMatmul requires non-null compiled payload.");
    }
    if (!built_matmul) {
        throw std::runtime_error("StampedMatmul requires non-null built matmul payload.");
    }
    if (built_matmul->workspace_bytes != 0) {
        if (!workspace.isPresent()) {
            throw std::runtime_error("StampedMatmul requires workspace for the chosen optimal kernel.");
        }
        assert(workspace.get().getArraySizeInBytes() >= built_matmul->workspace_bytes);
    }
}

struct ResolvedMatmulScale {
    float host_value = 1.0f;
    const float* ptr = nullptr;
    bool is_device_pointer = false;
    Optional<Tensor> device_scratch = Optional<Tensor>::empty();
    Optional<Tensor> host_scratch = Optional<Tensor>::empty();

    explicit ResolvedMatmulScale(Optional<Tensor> device_scratch = Optional<Tensor>::empty(),
                                 Optional<Tensor> host_scratch = Optional<Tensor>::empty())
        : ptr(&host_value), device_scratch(device_scratch), host_scratch(host_scratch) {}

    void refreshHostPointer() {
        if (!is_device_pointer) {
            ptr = &host_value;
        }
    }

    void setDevicePointer(const float* device_ptr) {
        ptr = device_ptr;
        is_device_pointer = true;
    }

    void copyHostValueToDevice(Stream& run_stream) {
        if (!device_scratch.isPresent()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (host_scratch.isPresent()) {
            std::memcpy(host_scratch.get().getMemPtr(), &host_value, sizeof(float));
            device_scratch.get().copyFromAsync(host_scratch.get(), run_stream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(device_scratch.get().getMemPtr(), &host_value, sizeof(float), cudaMemcpyHostToDevice, run_stream));
        }
        ptr = reinterpret_cast<const float*>(device_scratch.get().getMemPtr());
        is_device_pointer = true;
    }

    void scaleTensorDeviceValueIntoScratch(const TensorScalarBinding& binding, Stream& run_stream) {
        if (!device_scratch.isPresent()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (binding.sourceDType != TensorDescriptor::DataType::FP32) {
            throw std::runtime_error("Dynamic GEMM tensor-backed alpha/beta currently require FP32 source dtype.");
        }
        const char* device_ptr = static_cast<const char*>(binding.buffer.getMemPtr());
        const float* source_ptr = reinterpret_cast<const float*>(device_ptr + binding.byteOffset);
        launchScaleFp32DeviceScalar(source_ptr, static_cast<float*>(device_scratch.get().getMemPtr()), host_value, run_stream);
        ptr = reinterpret_cast<const float*>(device_scratch.get().getMemPtr());
        is_device_pointer = true;
    }

    void copyTensorValueToScratch(const Tensor& tensor, Stream& run_stream) {
        if (!device_scratch.isPresent()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        device_scratch.get().copyFromAsync(tensor, run_stream);
        ptr = reinterpret_cast<const float*>(device_scratch.get().getMemPtr());
        is_device_pointer = true;
    }

    void scaleTensorValueIntoScratch(const Tensor& tensor, Stream& run_stream) {
        if (!device_scratch.isPresent()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (tensor.getDataType() == TensorDescriptor::DataType::FP32) {
            launchScaleFp32DeviceScalar(static_cast<const float*>(tensor.getMemPtr()),
                                        static_cast<float*>(device_scratch.get().getMemPtr()),
                                        host_value,
                                        run_stream);
        } else {
            device_scratch.get().copyFromAsync(tensor, run_stream);
            launchScaleFp32DeviceScalar(static_cast<const float*>(device_scratch.get().getMemPtr()),
                                        static_cast<float*>(device_scratch.get().getMemPtr()),
                                        host_value,
                                        run_stream);
        }
        ptr = reinterpret_cast<const float*>(device_scratch.get().getMemPtr());
        is_device_pointer = true;
    }
};

struct ResolvedMatmulScales {
    ResolvedMatmulScale alpha;
    ResolvedMatmulScale beta;
    CublasScalarPointerMode pointer_mode = CublasScalarPointerMode::Host;
};

static const float* getTensorRuntimeScalarDevicePtr(const TensorScalarBinding& binding) {
    if (binding.sourceDType != TensorDescriptor::DataType::FP32) {
        throw std::runtime_error("Dynamic GEMM tensor-backed alpha/beta currently require FP32 source dtype.");
    }
    const char* device_ptr = static_cast<const char*>(binding.buffer.getMemPtr());
    return reinterpret_cast<const float*>(device_ptr + binding.byteOffset);
}

static bool tensorResolvesToSingleElement(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions()) {
        numel *= d;
    }
    return numel == 1;
}

static ResolvedMatmulScale resolveMatmulRuntimeScale(const Optional<RuntimeInputValue>& bound_input,
                                                     const std::optional<std::string>& runtime_name,
                                                     double base_scale,
                                                     const std::unordered_map<std::string, float>& runtime_scalars,
                                                     const Optional<Tensor>& device_scratch,
                                                     const Optional<Tensor>& host_scratch,
                                                     Stream& run_stream) {
    ResolvedMatmulScale resolved(device_scratch, host_scratch);
    resolved.host_value = static_cast<float>(base_scale);
    resolved.ptr = &resolved.host_value;

    bool used_runtime_override = false;
    if (runtime_name.has_value()) {
        auto it = runtime_scalars.find(*runtime_name);
        if (it != runtime_scalars.end()) {
            resolved.host_value *= it->second;
            used_runtime_override = true;
        }
    }
    if (!bound_input.isPresent()) {
        return resolved;
    }

    const RuntimeInputValue& value = bound_input.get();
    if (std::holds_alternative<float>(value)) {
        if (!used_runtime_override) {
            resolved.host_value *= std::get<float>(value);
        }
        return resolved;
    }
    if (std::holds_alternative<Tensor>(value)) {
        const Tensor& tensor = std::get<Tensor>(value);
        if (!tensorResolvesToSingleElement(tensor)) {
            throw std::runtime_error("Dynamic GEMM alpha/beta expression must resolve to a single element.");
        }
        if (tensor.getDataType() == TensorDescriptor::DataType::FP32 && resolved.host_value == 1.0f) {
            resolved.setDevicePointer(static_cast<const float*>(tensor.getMemPtr()));
            return resolved;
        }
        if (resolved.host_value == 1.0f) {
            resolved.copyTensorValueToScratch(tensor, run_stream);
            return resolved;
        }
        resolved.scaleTensorValueIntoScratch(tensor, run_stream);
        return resolved;
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        const TensorScalarBinding& binding = std::get<TensorScalarBinding>(value);
        if (resolved.host_value == 1.0f) {
            resolved.setDevicePointer(getTensorRuntimeScalarDevicePtr(binding));
            return resolved;
        }
        resolved.scaleTensorDeviceValueIntoScratch(binding, run_stream);
        return resolved;
    }
    throw std::runtime_error(
        "Dynamic GEMM scale currently requires fp32 runtime scalar, tensor-backed runtime scalar, or single-element tensor bindings.");
}

static ResolvedMatmulScales resolveMatmulRuntimeScales(const Optional<RuntimeInputValue>& alpha_input,
                                                       const Optional<RuntimeInputValue>& beta_input,
                                                       const std::optional<std::string>& alpha_runtime_name,
                                                       const std::optional<std::string>& beta_runtime_name,
                                                       double alpha_base_scale,
                                                       double beta_base_scale,
                                                       const std::unordered_map<std::string, float>& runtime_scalars,
                                                       const Optional<Tensor>& alpha_device_scratch,
                                                       const Optional<Tensor>& beta_device_scratch,
                                                       const Optional<Tensor>& alpha_host_scratch,
                                                       const Optional<Tensor>& beta_host_scratch,
                                                       Stream& run_stream) {
    ResolvedMatmulScales resolved;
    resolved.alpha = resolveMatmulRuntimeScale(
        alpha_input, alpha_runtime_name, alpha_base_scale, runtime_scalars, alpha_device_scratch, alpha_host_scratch, run_stream);
    resolved.beta = resolveMatmulRuntimeScale(
        beta_input, beta_runtime_name, beta_base_scale, runtime_scalars, beta_device_scratch, beta_host_scratch, run_stream);
    resolved.alpha.refreshHostPointer();
    resolved.beta.refreshHostPointer();

    if (resolved.alpha.is_device_pointer || resolved.beta.is_device_pointer) {
        resolved.pointer_mode = CublasScalarPointerMode::Device;
        if (!resolved.alpha.is_device_pointer) {
            resolved.alpha.copyHostValueToDevice(run_stream);
        }
        if (!resolved.beta.is_device_pointer) {
            resolved.beta.copyHostValueToDevice(run_stream);
        }
    }

    return resolved;
}

void StampedMatmul::run() { runOn(stream); }

void StampedMatmul::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedMatmul::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("StampedMatmul currently only supports rank-2 tensors.");
    }

    const auto lhs_dims = lhs.getDimensions();
    const auto rhs_dims = rhs.getDimensions();
    const int32_t a_rows = static_cast<int32_t>(lhs_dims[0]);
    const int32_t a_cols = static_cast<int32_t>(lhs_dims[1]);
    const int32_t b_rows = static_cast<int32_t>(rhs_dims[0]);
    const int32_t b_cols = static_cast<int32_t>(rhs_dims[1]);

    if (compiled_matmul->op == ExprOp::MATMUL) {
        const CublasMatrixMultiply::MatmulDataTypes dataTypes{lhs.getDescriptor().getDataType(),
                                                              rhs.getDescriptor().getDataType(),
                                                              output.getDescriptor().getDataType(),
                                                              output.getDescriptor().getDataType()};
        CublasMatrixMultiply::instance().multiply(lhs,
                                                  rhs,
                                                  output,
                                                  workspace,
                                                  a_rows,
                                                  a_cols,
                                                  b_rows,
                                                  b_cols,
                                                  compiled_matmul->transpose_lhs,
                                                  compiled_matmul->transpose_rhs,
                                                  false,
                                                  false,
                                                  dataTypes,
                                                  run_stream);
        return;
    }

    if (!addend.isPresent()) {
        throw std::runtime_error("Stamped GEMM requires an addend tensor.");
    }
    if (addend.get().getDimensions().size() != 2) {
        throw std::runtime_error("Stamped GEMM currently only supports rank-2 addend tensors.");
    }

    ResolvedMatmulScales resolved_scales = resolveMatmulRuntimeScales(alpha_input,
                                                                      beta_input,
                                                                      alpha_runtime_name,
                                                                      beta_runtime_name,
                                                                      compiled_matmul->alpha,
                                                                      compiled_matmul->beta,
                                                                      runtime_scalars,
                                                                      alpha_device_scratch,
                                                                      beta_device_scratch,
                                                                      alpha_host_scratch,
                                                                      beta_host_scratch,
                                                                      run_stream);

    const CublasMatrixMultiply::MatmulDataTypes dataTypes{lhs.getDescriptor().getDataType(),
                                                          rhs.getDescriptor().getDataType(),
                                                          addend.get().getDescriptor().getDataType(),
                                                          output.getDescriptor().getDataType()};
    CublasMatrixMultiply::instance().gemm(lhs,
                                          rhs,
                                          addend.get(),
                                          output,
                                          workspace,
                                          a_rows,
                                          a_cols,
                                          b_rows,
                                          b_cols,
                                          compiled_matmul->transpose_lhs,
                                          compiled_matmul->transpose_rhs,
                                          compiled_matmul->transpose_aux,
                                          resolved_scales.alpha.ptr,
                                          resolved_scales.beta.ptr,
                                          dataTypes,
                                          run_stream,
                                          resolved_scales.pointer_mode);
}

StampedReduceMinMaxBackward::StampedReduceMinMaxBackward(std::shared_ptr<BuiltReduction> built,
                                                         const Tensor& input,
                                                         const Tensor& grad_output,
                                                         const Tensor& output,
                                                         const Tensor& indices,
                                                         const Tensor& reduction_value_output,
                                                         const Stream& stream,
                                                         Optional<Tensor> workspace)
    : built_reduction(built),
      input(input),
      grad_output(grad_output),
      output(output),
      indices(indices),
      reduction_value_output(reduction_value_output),
      workspace(workspace),
      stream(stream) {
    if (!built_reduction->key.output_indices) {
        throw std::runtime_error("StampedReduceMinMaxBackward requires a BuiltReduction configured for indices.");
    }
    if (built_reduction->workspace_bytes != 0) {
        assert(workspace.isPresent());
        assert(workspace.get().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
}

void StampedReduceMinMaxBackward::run() { runOn(stream); }

void StampedReduceMinMaxBackward::runOn(Stream& run_stream) {
    // std::cerr << "[REDUCE_MINMAX_BW] input dtype=" << TensorDescriptor::getElementTypeName(input.getDataType())
    //           << " built.input_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.input_dtype)
    //           << " built.output_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.output_dtype)
    //           << " built.compute_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.compute_dtype)
    //           << " indices dtype=" << TensorDescriptor::getElementTypeName(indices.getDataType())
    //           << " reduction_value_output dtype=" << TensorDescriptor::getElementTypeName(reduction_value_output.getDataType())
    //           << std::endl;

    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        assert(workspace.isPresent());
        workspace_ptr = workspace.get().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  (void*)indices.getMemPtr(),
                                  built_reduction->indices_bytes,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)reduction_value_output.getMemPtr()));

    output.memsetAsync(run_stream, 0);

    launchReduceMinMaxBackwardScatter(grad_output.getMemPtr(),
                                      static_cast<const uint32_t*>(indices.getMemPtr()),
                                      (void*)output.getMemPtr(),
                                      input.getDimensions(),
                                      built_reduction->key.reduction_axes,
                                      built_reduction->key.squeeze_axes,
                                      grad_output.getDataType(),
                                      output.getDataType(),
                                      run_stream);
}

void StampedExecutionPlan::run() { run({}); }

void StampedExecutionPlan::run(const std::unordered_map<std::string, float>& runtime_scalars) {
    if (steps.empty()) {
        return;
    }

    using StreamEvent = std::decay_t<decltype(std::declval<Stream&>().putEvent())>;

    std::vector<std::optional<StreamEvent>> completion_events(steps.size());

    std::vector<Stream> launch_streams;
    launch_streams.reserve(steps.size());

    std::vector<Stream> helper_streams_used;
    helper_streams_used.reserve(steps.size());

    std::unordered_set<std::string> consumed_runtime_scalar_names;

    auto rememberHelperStream = [&](Stream& helper_stream) {
        if (std::find(helper_streams_used.begin(), helper_streams_used.end(), helper_stream) == helper_streams_used.end()) {
            helper_streams_used.push_back(helper_stream);
        }
    };

    StreamEvent user_stream_ready;
    if (steps.size() > 1)
        user_stream_ready = stream.putEvent();

    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        const bool use_helper_stream = (stage_idx != 0);
        const StampedExecutionStage& stage = steps[stage_idx];

        Stream& launch_stream_ref = use_helper_stream ? Expression::getNextHelperStream(stage.gpu_num) : stream;

        if (use_helper_stream) {
            rememberHelperStream(launch_stream_ref);
            launch_stream_ref.waitEvent(user_stream_ready);
        }

        for (uint32_t dep_stage_idx : stage.dependency_stage_indices) {
            if (dep_stage_idx >= stage_idx) {
                throw std::runtime_error("StampedExecutionPlan::run requires dependency_stage_indices to be topologically ordered.");
            }

            if (!completion_events[dep_stage_idx].has_value()) {
                throw std::runtime_error("StampedExecutionPlan::run missing completion event for dependency stage.");
            }

            if (!(launch_stream_ref == launch_streams[dep_stage_idx])) {
                launch_stream_ref.waitEvent(completion_events[dep_stage_idx].value());
            }
        }

        std::unordered_map<std::string, float> stage_runtime_scalars;

        if (!runtime_scalars.empty()) {
            std::unordered_set<std::string> needed_names;

            if (stage.kind == StampedExecutionStage::Kind::FusedKernel && stage.kernel != nullptr &&
                stage.kernel->requiresRuntimeScalars()) {
                needed_names = stage.kernel->runtimeScalarNames();
            } else if (stage.kind == StampedExecutionStage::Kind::Matmul && stage.matmul != nullptr) {
                if (stage.matmul->alphaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->alphaRuntimeName());
                }
                if (stage.matmul->betaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->betaRuntimeName());
                }
            }

            if (!needed_names.empty()) {
                stage_runtime_scalars.reserve(needed_names.size());

                for (const std::string& name : needed_names) {
                    auto it = runtime_scalars.find(name);
                    if (it == runtime_scalars.end()) {
                        throw std::runtime_error("Missing value for runtime scalar: " + name +
                                                 "  - if it was meant to be constant, use a constant scalar instead.");
                    }
                    stage_runtime_scalars.emplace(name, it->second);
                    consumed_runtime_scalar_names.insert(name);
                }
            }
        }

        if (stage_runtime_scalars.empty())
            stage.runOn(launch_stream_ref);
        else
            stage.runOn(launch_stream_ref, stage_runtime_scalars);

        completion_events[stage_idx] = launch_stream_ref.putEvent();
        launch_streams.push_back(launch_stream_ref);
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_runtime_scalar_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped execution plan: " + name);
        }
    }

    for (Stream& helper_stream : helper_streams_used) {
        if (!(helper_stream == stream)) {
            stream.waitEvent(helper_stream.putEvent());
        }
    }
}

// static unordered_map<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache;
static LruCacheThreadSafe<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache(10'000);

static shared_ptr<BuiltReduction> cacheLookup(const ReductionCacheKey& key) {
    optional<shared_ptr<BuiltReduction>> hit = builtReductionCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static LruCacheThreadSafe<MatmulCacheKey, shared_ptr<BuiltMatmul>> builtMatmulCache(10'000);

static shared_ptr<BuiltMatmul> cacheLookup(const MatmulCacheKey& key) {
    optional<shared_ptr<BuiltMatmul>> hit = builtMatmulCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static cudnnDataType_t toCudnnDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return CUDNN_DATA_FLOAT;

        case TensorDescriptor::DataType::FP16:
            return CUDNN_DATA_HALF;

        case TensorDescriptor::DataType::BF16:
            return CUDNN_DATA_BFLOAT16;

        case TensorDescriptor::DataType::FP8_E4M3:
            return CUDNN_DATA_FP8_E4M3;

        case TensorDescriptor::DataType::FP8_E5M2:
            return CUDNN_DATA_FP8_E5M2;

        default:
            throw std::runtime_error("toCudnnDataType: unsupported TensorDescriptor::DataType value " +
                                     std::to_string(static_cast<int>(dtype)));
    }
}

static cudnnReduceTensorOp_t toCudnnReduceTensorOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
            return CUDNN_REDUCE_TENSOR_ADD;
        case ExprOp::REDUCE_PROD:
            return CUDNN_REDUCE_TENSOR_MUL;
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_ARGMIN:
            return CUDNN_REDUCE_TENSOR_MIN;
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMAX:
            return CUDNN_REDUCE_TENSOR_MAX;
        case ExprOp::REDUCE_AVG:
            return CUDNN_REDUCE_TENSOR_AVG;
        case ExprOp::REDUCE_NORM1:
            return CUDNN_REDUCE_TENSOR_NORM1;
        case ExprOp::REDUCE_NORM2:
            return CUDNN_REDUCE_TENSOR_NORM2;
        default:
            throw std::runtime_error("ExprOp is not a supported cuDNN reduction op.");
    }
}

std::vector<uint64_t> StampedEquation::computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                                  const std::vector<uint64_t>& reduction_axes,
                                                                  const std::vector<uint64_t>& squeeze_axes) {
    std::vector<uint64_t> output_dims = input_dims;

    for (uint64_t axis : reduction_axes) {
        if (axis >= output_dims.size())
            throw std::runtime_error("Reduction axis out of range.");
        output_dims[axis] = 1;
    }

    if (squeeze_axes.empty()) {
        return output_dims;
    }

    std::vector<uint64_t> squeezed;
    squeezed.reserve(output_dims.size());

    if (squeeze_axes.size() == 1 && squeeze_axes[0] == UINT64_MAX) {
        for (uint64_t d : output_dims) {
            if (d != 1)
                squeezed.push_back(d);
        }
    } else {
        uint64_t nextDimToSqueeze = squeeze_axes[0];
        uint64_t nextIndexInSqueezedDims = 1;

        for (uint64_t i = 0; i < output_dims.size(); ++i) {
            if (i == nextDimToSqueeze) {
                if (output_dims[i] != 1) {
                    throw runtime_error("Trying to squeeze axis " + to_string(nextDimToSqueeze) + " but it has size " +
                                        to_string(output_dims[i]) + ", can only squeeze dimensions of size 1.");
                }

                if (nextIndexInSqueezedDims < squeeze_axes.size()) {
                    nextDimToSqueeze = squeeze_axes[nextIndexInSqueezedDims];
                    nextIndexInSqueezedDims += 1;
                } else {
                    nextDimToSqueeze = UINT64_MAX;
                }
            } else {
                squeezed.push_back(output_dims[i]);
            }
        }

        if (nextIndexInSqueezedDims != squeeze_axes.size()) {
            throw runtime_error("Axis " + to_string(nextDimToSqueeze) + " was passed as a dimension to squeeze, but tensor has only " +
                                to_string(output_dims.size()) + " dimensions.");
        }
    }

    if (squeezed.empty())
        squeezed.push_back(1);

    return squeezed;
}

static cudnnTensorDescriptor_t createCudnnTensorDescriptor(std::vector<uint64_t> dims, TensorDescriptor::DataType dtype) {
    while (dims.size() < 4)
        dims.push_back(1);
    if (dims.size() > 8)
        throw std::runtime_error("cuDNN reduction only supports rank <= 8.");

    std::vector<int> cudnn_dims(dims.begin(), dims.end());
    std::vector<int> strides(cudnn_dims.size());
    strides.back() = 1;
    for (int i = static_cast<int>(cudnn_dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * cudnn_dims[i + 1];

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(desc, toCudnnDataType(dtype), static_cast<int>(cudnn_dims.size()), cudnn_dims.data(), strides.data()));
    return desc;
}

static cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(ExprOp op, TensorDescriptor::DataType compute_dtype, bool output_indices) {
    cudnnReduceTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(desc,
                                               toCudnnReduceTensorOp(op),
                                               toCudnnDataType(compute_dtype),
                                               CUDNN_PROPAGATE_NAN,
                                               output_indices ? CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES,
                                               CUDNN_32BIT_INDICES));
    return desc;
}

static size_t getReductionWorkspaceSize(int device_num,
                                        cudnnReduceTensorDescriptor_t reduce_desc,
                                        cudnnTensorDescriptor_t a_desc,
                                        cudnnTensorDescriptor_t c_desc) {
    Stream stream(device_num);
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(stream.getCudnnHandle(), reduce_desc, a_desc, c_desc, &workspace_bytes));
    return workspace_bytes;
}

static size_t getReductionIndicesSize(int device_num,
                                      cudnnReduceTensorDescriptor_t reduce_desc,
                                      cudnnTensorDescriptor_t a_desc,
                                      cudnnTensorDescriptor_t c_desc) {
    Stream stream(device_num);
    size_t indices_bytes = 0;
    CUDNN_CHECK(cudnnGetReductionIndicesSize(stream.getCudnnHandle(), reduce_desc, a_desc, c_desc, &indices_bytes));
    return indices_bytes;
}

static int32_t leadingDimensionForStoredMatrix(const Tensor& matrix) {
    const std::vector<uint64_t> dims = matrix.getDimensions();
    if (dims.size() != 2) {
        throw std::runtime_error("Matmul/gemm workspace planning currently only supports rank-2 tensors.");
    }
    return static_cast<int32_t>(dims[1]);
}

std::shared_ptr<BuiltMatmul> StampedEquation::buildMatmul(const std::shared_ptr<CompiledMatmul>& compiled_matmul,
                                                          const Tensor& lhs,
                                                          const Tensor& rhs,
                                                          const Optional<Tensor>& addend,
                                                          const Tensor& output,
                                                          int device_num) {
    if (!compiled_matmul) {
        throw std::runtime_error("buildMatmul requires non-null compiled payload.");
    }
    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("buildMatmul currently only supports rank-2 tensors.");
    }
    if (compiled_matmul->op == ExprOp::GEMM) {
        if (!addend.isPresent()) {
            throw std::runtime_error("buildMatmul requires an addend tensor for GEMM.");
        }
        if (addend.get().getDimensions().size() != 2) {
            throw std::runtime_error("buildMatmul currently only supports rank-2 GEMM addend tensors.");
        }
        if (compiled_matmul->transpose_aux) {
            throw std::runtime_error("GEMM transpose_aux/transposeC is not supported by CublasMatrixMultiply in this staged path.");
        }
    }

    const std::vector<uint64_t> lhs_dims = lhs.getDimensions();
    const std::vector<uint64_t> rhs_dims = rhs.getDimensions();
    const int32_t a_rows = static_cast<int32_t>(lhs_dims[0]);
    const int32_t a_cols = static_cast<int32_t>(lhs_dims[1]);
    const int32_t b_rows = static_cast<int32_t>(rhs_dims[0]);
    const int32_t b_cols = static_cast<int32_t>(rhs_dims[1]);
    const int32_t ld_a = leadingDimensionForStoredMatrix(lhs);
    const int32_t ld_b = leadingDimensionForStoredMatrix(rhs);
    const int32_t ld_d = leadingDimensionForStoredMatrix(output);
    const int32_t ld_c = addend.isPresent() ? leadingDimensionForStoredMatrix(addend.get()) : ld_d;

    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        lhs.getDescriptor().getDataType(),
        rhs.getDescriptor().getDataType(),
        addend.isPresent() ? addend.get().getDescriptor().getDataType() : output.getDescriptor().getDataType(),
        output.getDescriptor().getDataType()};

    if (dataTypes.A != compiled_matmul->lhs_dtype || dataTypes.B != compiled_matmul->rhs_dtype ||
        dataTypes.C != (compiled_matmul->op == ExprOp::GEMM ? compiled_matmul->aux_dtype : compiled_matmul->output_dtype) ||
        dataTypes.D != compiled_matmul->output_dtype) {
        throw std::runtime_error("buildMatmul tensor dtypes do not match the compiled matmul dtype plan.");
    }

    MatmulCacheKey key(compiled_matmul->op,
                       a_rows,
                       a_cols,
                       b_rows,
                       b_cols,
                       ld_a,
                       ld_b,
                       ld_c,
                       ld_d,
                       compiled_matmul->transpose_lhs,
                       compiled_matmul->transpose_rhs,
                       compiled_matmul->transpose_aux,
                       dataTypes.A,
                       dataTypes.B,
                       dataTypes.C,
                       dataTypes.D,
                       device_num);

    std::shared_ptr<BuiltMatmul> hit = cacheLookup(key);
    if (hit) {
        return hit;
    }

    auto built = std::make_shared<BuiltMatmul>(key);
    bool kernelWillRunOnGpu = false;

    if (compiled_matmul->op == ExprOp::MATMUL) {
        CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(device_num,
                                                                           a_rows,
                                                                           a_cols,
                                                                           b_rows,
                                                                           b_cols,
                                                                           ld_a,
                                                                           ld_b,
                                                                           ld_d,
                                                                           compiled_matmul->transpose_lhs,
                                                                           compiled_matmul->transpose_rhs,
                                                                           dataTypes);
        built->workspace_bytes = CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(device_num,
                                                                                                        a_rows,
                                                                                                        a_cols,
                                                                                                        b_rows,
                                                                                                        b_cols,
                                                                                                        ld_a,
                                                                                                        ld_b,
                                                                                                        ld_d,
                                                                                                        compiled_matmul->transpose_lhs,
                                                                                                        compiled_matmul->transpose_rhs,
                                                                                                        dataTypes,
                                                                                                        kernelWillRunOnGpu);
    } else {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(device_num,
                                                                 a_rows,
                                                                 a_cols,
                                                                 b_rows,
                                                                 b_cols,
                                                                 ld_a,
                                                                 ld_b,
                                                                 ld_c,
                                                                 ld_d,
                                                                 compiled_matmul->transpose_lhs,
                                                                 compiled_matmul->transpose_rhs,
                                                                 compiled_matmul->transpose_aux,
                                                                 dataTypes);
        built->workspace_bytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(device_num,
                                                                                              a_rows,
                                                                                              a_cols,
                                                                                              b_rows,
                                                                                              b_cols,
                                                                                              ld_a,
                                                                                              ld_b,
                                                                                              ld_c,
                                                                                              ld_d,
                                                                                              compiled_matmul->transpose_lhs,
                                                                                              compiled_matmul->transpose_rhs,
                                                                                              compiled_matmul->transpose_aux,
                                                                                              dataTypes,
                                                                                              kernelWillRunOnGpu);
    }

    if (!kernelWillRunOnGpu) {
        throw std::runtime_error("No GPU kernel available for the staged matmul/gemm configuration.");
    }

    builtMatmulCache.put(key, built);
    return built;
}

std::shared_ptr<BuiltConvolution> StampedEquation::buildConvolution(const std::shared_ptr<CompiledConvolution>& compiled_convolution,
                                                                    const Tensor& input,
                                                                    const Tensor& filter,
                                                                    const Tensor& output,
                                                                    const Stream& stream,
                                                                    int device_num) {
    if (!compiled_convolution) {
        throw std::runtime_error("buildConvolution requires non-null compiled payload.");
    }
    if (input.getDimensions().size() != 4 || filter.getDimensions().size() != 4 || output.getDimensions().size() != 4) {
        throw std::runtime_error("buildConvolution currently only supports rank-4 tensors.");
    }

    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> filter_dims = filter.getDimensions();

    auto built = std::make_shared<BuiltConvolution>();

    const std::string gpuType = MachineEvaluator::instance().getGpuType(device_num);
    built->requirement = ConvolutionKernelRequirement(gpuType,
                                                      static_cast<int>(filter_dims[3]),
                                                      static_cast<int>(filter_dims[2]),
                                                      compiled_convolution->stride_w,
                                                      compiled_convolution->stride_h,
                                                      compiled_convolution->pad_w,
                                                      compiled_convolution->pad_h,
                                                      static_cast<int>(input_dims[1]),
                                                      static_cast<int>(filter_dims[0]),
                                                      static_cast<int>(input_dims[0]),
                                                      static_cast<int>(input_dims[3]),
                                                      static_cast<int>(input_dims[2]));

    GpuConvolution::instance().chooseOptimalKernelForward(built->requirement.get(), stream);
    built->workspace_bytes = GpuConvolution::instance().getForwardWorkspaceSizeInBytes(built->requirement.get());
    return built;
}

std::shared_ptr<BuiltConvolution> StampedEquation::buildConvolutionBackward(
    const std::shared_ptr<CompiledConvolutionBackward>& compiled_convolution_backward,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& output,
    const Stream& stream,
    int device_num) {
    if (!compiled_convolution_backward) {
        throw std::runtime_error("buildConvolutionBackward requires non-null compiled payload.");
    }
    if (input.getDimensions().size() != 4 || grad_output.getDimensions().size() != 4 || output.getDimensions().size() != 4) {
        throw std::runtime_error("buildConvolutionBackward currently only supports rank-4 tensors.");
    }

    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> grad_output_dims = grad_output.getDimensions();
    const std::vector<uint64_t> output_dims = output.getDimensions();

    auto built = std::make_shared<BuiltConvolution>();

    const std::string gpuType = MachineEvaluator::instance().getGpuType(device_num);
    if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_DATA) {
        built->requirement = ConvolutionKernelRequirement(gpuType,
                                                          static_cast<int>(input_dims[3]),
                                                          static_cast<int>(input_dims[2]),
                                                          compiled_convolution_backward->stride_w,
                                                          compiled_convolution_backward->stride_h,
                                                          compiled_convolution_backward->pad_w,
                                                          compiled_convolution_backward->pad_h,
                                                          static_cast<int>(output_dims[1]),
                                                          static_cast<int>(grad_output_dims[1]),
                                                          static_cast<int>(output_dims[0]),
                                                          static_cast<int>(output_dims[3]),
                                                          static_cast<int>(output_dims[2]));
    } else if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_FILTER) {
        built->requirement = ConvolutionKernelRequirement(gpuType,
                                                          static_cast<int>(output_dims[3]),
                                                          static_cast<int>(output_dims[2]),
                                                          compiled_convolution_backward->stride_w,
                                                          compiled_convolution_backward->stride_h,
                                                          compiled_convolution_backward->pad_w,
                                                          compiled_convolution_backward->pad_h,
                                                          static_cast<int>(input_dims[1]),
                                                          static_cast<int>(grad_output_dims[1]),
                                                          static_cast<int>(input_dims[0]),
                                                          static_cast<int>(input_dims[3]),
                                                          static_cast<int>(input_dims[2]));
    } else {
        throw std::runtime_error("buildConvolutionBackward received unsupported convolution backward op.");
    }

    GpuConvolution::instance().chooseOptimalKernelBackward(built->requirement.get(), stream);
    built->workspace_bytes = compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_DATA
                                 ? GpuConvolution::instance().getBackwardDataWorkspaceSizeInBytes(built->requirement.get())
                                 : GpuConvolution::instance().getBackwardFilterWorkspaceSizeInBytes(built->requirement.get());
    return built;
}

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                                const Tensor& input,
                                                                int device_num) {
    return buildReduction(compiled_reduction->op,
                          compiled_reduction->reduction_axes,
                          compiled_reduction->squeeze_axes,
                          compiled_reduction->input_dtype,
                          compiled_reduction->output_dtype,
                          compiled_reduction->compute_dtype,
                          /*output_indices=*/false,
                          input,
                          device_num);
}

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(ExprOp op,
                                                                const std::vector<uint64_t>& reduction_axes,
                                                                const std::vector<uint64_t>& squeeze_axes,
                                                                TensorDescriptor::DataType input_dtype,
                                                                TensorDescriptor::DataType output_dtype,
                                                                TensorDescriptor::DataType compute_dtype,
                                                                bool output_indices,
                                                                const Tensor& input,
                                                                int device_num) {
    const std::vector<uint64_t> input_dims = input.getDimensions();

    ReductionCacheKey key(
        op, input_dims, reduction_axes, squeeze_axes, input_dtype, output_dtype, compute_dtype, output_indices, device_num);

    std::shared_ptr<BuiltReduction> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltReduction>(key);

    const std::vector<uint64_t> output_dims = computeReductionOutputDims(input_dims,
                                                                         built->key.reduction_axes,
                                                                         /*squeeze_axes=*/{});
    built->a_desc = createCudnnTensorDescriptor(input_dims, built->key.input_dtype);
    built->reduce_desc = createCudnnReduceDescriptor(built->key.op, built->key.compute_dtype, built->key.output_indices);
    built->c_desc = createCudnnTensorDescriptor(output_dims, built->key.output_dtype);

    built->workspace_bytes = getReductionWorkspaceSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);
    if (built->key.output_indices) {
        built->indices_bytes = getReductionIndicesSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);
    }

    builtReductionCache.put(key, built);
    return built;
}

bool StampedEquation::requiresRuntimeScalars() const {
    if (!compiledEquation) {
        return false;
    }

    for (size_t i = 0; i < compiledEquation->input_kinds.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            return true;
        }
    }
    return false;
}

std::unordered_set<std::string> StampedEquation::runtimeScalarNames() const {
    std::unordered_set<std::string> names;
    if (!compiledEquation) {
        return names;
    }

    for (size_t i = 0; i < compiledEquation->input_names.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            names.insert(compiledEquation->input_names[i]);
        }
    }
    return names;
}

}  // namespace ThorImplementation
