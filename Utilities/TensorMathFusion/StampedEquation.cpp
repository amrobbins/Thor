#include "Utilities/TensorMathFusion/StampedEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

#include <stdexcept>
#include <vector>

using namespace std;

namespace ThorImplementation {

void StampedEquation::run() {
    if (!compiledEquation) {
        throw std::runtime_error("StampedEquation::run called with null compiled equation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("StampedEquation::run called with no output tensors.");
    }

    if (deviceBroadcastInfo.isPresent()) {
        EquationRunner::run(compiledEquation, inputs, outputs, stream, deviceBroadcastInfo.get());
    } else {
        EquationRunner::run(compiledEquation, inputs, outputs, stream);
    }
}

StampedReduction::StampedReduction(
    std::shared_ptr<BuiltReduction> built, const Tensor& input, const Tensor& output, const Stream& stream, Optional<Tensor> workspace)
    : built_reduction(built), input(input), output(output), workspace(workspace), stream(stream) {
    if (built_reduction->workspace_bytes != 0) {
        assert(workspace.isPresent());
        assert(workspace.get().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
    assert(input.getDataType() == built_reduction->key.inout_dtype);
    assert(output.getDataType() == built_reduction->key.inout_dtype);
}

void StampedReduction::run() {
    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        assert(workspace.isPresent());
        workspace_ptr = workspace.get().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(stream.getCudnnHandle(),
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
                                  output.getMemPtr()));
}

static unordered_map<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache;

static shared_ptr<BuiltReduction> cacheLookup(const ReductionCacheKey& key) {
    auto it = builtReductionCache.find(key);
    if (it == builtReductionCache.end()) {
        return nullptr;
    }
    return it->second;
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
            return CUDNN_REDUCE_TENSOR_MIN;
        case ExprOp::REDUCE_MAX:
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
        // Do not squeeze
        return output_dims;
    }

    std::vector<uint64_t> squeezed;
    squeezed.reserve(output_dims.size());

    if (squeeze_axes.size() == 1 && squeeze_axes[0] == UINT64_MAX) {
        // Squeeze all singletons
        for (uint64_t d : output_dims) {
            if (d != 1)
                squeezed.push_back(d);
        }
    } else {
        // Eliminate the specified dimensions, error if any specified dimension is not a singleton.
        // Precondition: squeeze_dimensions are sorted ascending and uniquified.
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

static cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(ExprOp op, TensorDescriptor::DataType compute_dtype) {
    cudnnReduceTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(desc,
                                               toCudnnReduceTensorOp(op),
                                               toCudnnDataType(compute_dtype),
                                               CUDNN_PROPAGATE_NAN,
                                               CUDNN_REDUCE_TENSOR_NO_INDICES,
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

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                                const Tensor& input,
                                                                int device_num) {
    const std::vector<uint64_t> input_dims = input.getDimensions();

    ReductionCacheKey key(compiled_reduction->op,
                          input_dims,
                          compiled_reduction->reduction_axes,
                          compiled_reduction->squeeze_axes,
                          compiled_reduction->inout_dtype,
                          compiled_reduction->compute_dtype,
                          device_num);

    std::shared_ptr<BuiltReduction> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltReduction>(key);

    const std::vector<uint64_t> output_dims = computeReductionOutputDims(input_dims,
                                                                         built->key.reduction_axes,
                                                                         /*squeeze_axes=*/{});
    built->a_desc = createCudnnTensorDescriptor(input_dims, built->key.inout_dtype);
    built->c_desc = createCudnnTensorDescriptor(output_dims, built->key.inout_dtype);
    built->reduce_desc = createCudnnReduceDescriptor(built->key.op, built->key.compute_dtype);

    built->workspace_bytes = getReductionWorkspaceSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);

    builtReductionCache.emplace(key, built);
    return built;
}

}  // namespace ThorImplementation
