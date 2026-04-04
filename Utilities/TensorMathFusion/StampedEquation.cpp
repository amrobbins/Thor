#include "Utilities/TensorMathFusion/StampedEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/ReduceMinMaxBackwardKernel.h"

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
        if (!runtime_scalars.empty() && stage.kind == StampedExecutionStage::Kind::FusedKernel && stage.kernel != nullptr &&
            stage.kernel->requiresRuntimeScalars()) {
            const std::unordered_set<std::string> needed_names = stage.kernel->runtimeScalarNames();
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
