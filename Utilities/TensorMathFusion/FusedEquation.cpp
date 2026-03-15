#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

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
        case ExprOp::REDUCE_AMAX:
            return CUDNN_REDUCE_TENSOR_AMAX;
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

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num, bool use_fast_math) {
    if (device_num < 0) {
        throw std::runtime_error("FusedEquation::compile requires device_num >= 0.");
    }

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(cuda_status));
    }

    if (device_num >= device_count) {
        throw std::runtime_error("FusedEquation::compile device_num is out of range.");
    }

    cudaDeviceProp prop{};
    cuda_status = cudaGetDeviceProperties(&prop, device_num);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cuda_status));
    }

    if (expr.nodes.size() == 1 && isCudnnReduceOp(expr.nodes[0].op)) {
        assert(expr.numInputs() == 1);
        shared_ptr<CompiledReduction> compiledReduction = EquationCompiler::compileReduction(expr, dtype);
        return FusedEquation(compiledReduction);
    }

    EquationSignature sig{};
    sig.num_inputs = expr.numInputs();
    sig.dtype = dtype;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;

    std::shared_ptr<CompiledEquation> flatEquation = EquationCompiler::compile(expr, sig, false);
    std::shared_ptr<CompiledEquation> broadcastEquation = EquationCompiler::compile(expr, sig, true);

    return FusedEquation(std::move(flatEquation), std::move(broadcastEquation));
}

static std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> stripped;
    stripped.reserve(dims.size());

    for (uint64_t dim : dims) {
        if (dim != 1)
            stripped.push_back(dim);
    }

    return stripped;
}

static bool outputDimensionsMatchIgnoringSingletons(const std::vector<uint64_t>& actual, const std::vector<uint64_t>& expected) {
    return stripSingletonDimensions(actual) == stripSingletonDimensions(expected);
}

static void verifyRequestedOutputLayout(const std::vector<uint64_t>& outputDimensions, const std::vector<uint64_t>& expectedDimensions) {
    if (!outputDimensionsMatchIgnoringSingletons(outputDimensions, expectedDimensions)) {
        throw std::runtime_error("Output tensor dimensions are incompatible with the fused equation result.");
    }
}

StampedEquation FusedEquation::stampEquation(std::vector<Tensor>& inputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape) const {
    if (!compiledFlatEquation) {
        throw std::runtime_error("Cannot stamp an empty FusedEquation.");
    }

    if (inputs.empty()) {
        throw std::runtime_error("FusedEquation::stamp requires at least one input tensor.");
    }

    if (inputs.size() != compiledFlatEquation->numInputs()) {
        throw std::runtime_error("Wrong number of inputs passed to FusedEquation::stamp.");
    }

    const Tensor& firstInput = inputs[0];
    if (!firstInput.isInitialized()) {
        throw std::runtime_error("First input tensor is not initialized.");
    }

    if (firstInput.getDescriptor().getDataType() != compiledFlatEquation->dtype) {
        throw std::runtime_error("Input tensor data type does not match compiled equation data type.");
    }

    if (firstInput.getPlacement().getDeviceNum() != compiledFlatEquation->deviceNum) {
        throw std::runtime_error("Input tensor GPU does not match compiled equation device.");
    }

    for (uint64_t i = 1; i < inputs.size(); ++i) {
        if (!inputs[i].isInitialized()) {
            throw std::runtime_error("Input tensor is not initialized.");
        }
        if (inputs[i].getDescriptor().getDataType() != compiledFlatEquation->dtype) {
            throw std::runtime_error("Input tensor data type mismatch.");
        }
        if (inputs[i].getPlacement().getDeviceNum() != compiledFlatEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU mismatch.");
        }
    }

    std::vector<uint64_t> outputDimensions;
    bool requiresBroadcast = resolveLayout(inputs, outputDimensions);

    // Ensure the output shape is valid, when specified explicitly (can only add or remove singleton dimensions)
    if (!requestedOutputShape.empty()) {
        verifyRequestedOutputLayout(requestedOutputShape, outputDimensions);
        // outputDimensions = requestedOutputShape;
    }

    TensorPlacement outputPlacement = firstInput.getPlacement();
    Tensor output(outputPlacement,
                  TensorDescriptor(compiledFlatEquation->dtype, requestedOutputShape.empty() ? outputDimensions : requestedOutputShape));

    if (requiresBroadcast)
        return StampedEquation(
            compiledBroadcastEquation, inputs, output, stream, createDeviceBroadcastInfo(inputs, outputDimensions, stream));
    else
        return StampedEquation(compiledFlatEquation, inputs, output, stream);
}

static uint64_t product(const std::vector<uint64_t>& dims) {
    uint64_t p = 1;
    for (uint64_t d : dims)
        p *= d;
    return p;
}

static std::vector<uint64_t> computePackedOutputStrides(const std::vector<uint64_t>& outputDimensions) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    std::vector<uint64_t> strides(rank, 1);

    if (rank == 0)
        return strides;

    strides[rank - 1] = 1;
    for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * outputDimensions[static_cast<size_t>(i) + 1];
    }

    return strides;
}

static void buildBroadcastInfo(BroadcastInfoBufferView& buffer,
                               const std::vector<Tensor>& inputs,
                               const std::vector<uint64_t>& outputDimensions) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    const uint32_t numInputs = static_cast<uint32_t>(inputs.size());

    buffer.header()->numel = product(outputDimensions);

    const std::vector<uint64_t> outputStrides = computePackedOutputStrides(outputDimensions);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        buffer.outputStrides()[axis] = outputStrides[axis];
    }

    for (uint32_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
        const std::vector<uint64_t>& inputDims = inputs[inputIdx].getDimensions();
        if (inputDims.size() > outputDimensions.size()) {
            throw std::runtime_error("Input rank exceeds output rank in buildBroadcastInfo.");
        }

        const uint32_t rankDiff = rank - static_cast<uint32_t>(inputDims.size());

        uint64_t runningStride = 1;
        std::vector<uint64_t> inputPackedStrides(rank, 0);

        for (int64_t axis = static_cast<int64_t>(rank) - 1; axis >= 0; --axis) {
            if (static_cast<uint32_t>(axis) < rankDiff) {
                inputPackedStrides[static_cast<size_t>(axis)] = 0;
                continue;
            }

            const uint32_t inputAxis = static_cast<uint32_t>(axis) - rankDiff;
            const uint64_t inputDim = inputDims[inputAxis];
            const uint64_t outputDim = outputDimensions[static_cast<size_t>(axis)];

            if (inputDim == outputDim) {
                inputPackedStrides[static_cast<size_t>(axis)] = runningStride;
                runningStride *= inputDim;
            } else if (inputDim == 1) {
                inputPackedStrides[static_cast<size_t>(axis)] = 0;
            } else {
                throw std::runtime_error("Input shape is not broadcast-compatible with output shape.");
            }
        }

        for (uint32_t axis = 0; axis < rank; ++axis) {
            buffer.inputStride(inputIdx, axis) = inputPackedStrides[axis];
        }
    }
}

struct BroadcastInfoHostFunctionArgs : HostFunctionArgsBase {
    Tensor hostBroadcastInfo;
    Tensor deviceBroadcastInfo;

    explicit BroadcastInfoHostFunctionArgs(Tensor hostBroadcastInfo, Tensor deviceBroadcastInfo)
        : hostBroadcastInfo(hostBroadcastInfo), deviceBroadcastInfo(deviceBroadcastInfo) {}
};

Tensor FusedEquation::createDeviceBroadcastInfo(const std::vector<Tensor>& inputs,
                                                const std::vector<uint64_t>& outputDimensions,
                                                Stream stream) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    const uint32_t numInputs = static_cast<uint32_t>(inputs.size());
    const uint64_t numBytes = BroadcastInfoBufferView::bytesRequired(rank, numInputs);

    TensorDescriptor broadcastInfoDescriptor(TensorDescriptor::DataType::UINT8, {numBytes});

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement devicePlacement = inputs[0].getPlacement();

    Tensor hostBroadcastInfo(cpuPlacement, broadcastInfoDescriptor);
    Tensor deviceBroadcastInfo(devicePlacement, broadcastInfoDescriptor);

    BroadcastInfoBufferView buffer(hostBroadcastInfo.getMemPtr(), rank, numInputs);
    buildBroadcastInfo(buffer, inputs, outputDimensions);

    deviceBroadcastInfo.copyFromAsync(hostBroadcastInfo, stream);

    stream.launchCleanUpHostFunctionArgs(std::make_unique<BroadcastInfoHostFunctionArgs>(hostBroadcastInfo, deviceBroadcastInfo));

    return deviceBroadcastInfo;
}

void FusedEquation::runEquation(std::vector<Tensor> inputs, Tensor output, Stream stream) const {
    std::vector<uint64_t> outputDimensions;

    bool requiresBroadcast = resolveLayout(inputs, outputDimensions);
    verifyRequestedOutputLayout(output.getDimensions(), outputDimensions);

    if (requiresBroadcast) {
        Tensor broadcastInfo = createDeviceBroadcastInfo(inputs, outputDimensions, stream);
        EquationRunner::run(compiledBroadcastEquation, inputs, output, stream, broadcastInfo);
    } else {
        EquationRunner::run(compiledFlatEquation, inputs, output, stream);
    }
}

bool FusedEquation::resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty())
        throw std::runtime_error("Tried to create a FusedEquation with 0 tensor inputs. You must have at least one.");

    std::vector<std::vector<uint64_t>> originalInputDimensions;
    originalInputDimensions.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        originalInputDimensions.push_back(input.getDimensions());
    }

    // Find the maximum rank among all inputs.
    uint64_t maxRank = 0;
    for (const Tensor& input : inputs) {
        const std::vector<uint64_t>& dims = input.getDimensions();
        if (dims.empty())
            throw std::runtime_error("Input tensor has 0 dimensions, which is not supported.");
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    // Left-pad each input shape with singleton dimensions so all inputs have the same rank.
    for (Tensor& input : inputs) {
        const std::vector<uint64_t>& oldDims = input.getDimensions();
        if (oldDims.size() == maxRank)
            continue;

        std::vector<uint64_t> paddedDims(maxRank - oldDims.size(), 1);
        paddedDims.insert(paddedDims.end(), oldDims.begin(), oldDims.end());
        input.reshape(paddedDims);
    }

    // Resolve output shape axis by axis, NumPy-style.
    outputDimensions.clear();
    outputDimensions.assign(maxRank, 1);
    bool requiresBroadcast = false;

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const Tensor& input : inputs) {
            const std::vector<uint64_t>& dims = input.getDimensions();
            uint64_t dim = dims[axis];

            if (dim == 1) {
                if (resolvedDim != 1)
                    requiresBroadcast = true;
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
                requiresBroadcast = true;
            } else if (resolvedDim != dim) {
                std::ostringstream err;
                err << "Input tensors are not broadcast-compatible at axis " << axis << ". "
                    << "Encountered dimension " << resolvedDim << " and dimension " << dim << ". "
                    << "Input shapes: ";
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const std::vector<uint64_t>& inDims = originalInputDimensions[i];
                    err << "[";
                    for (size_t j = 0; j < inDims.size(); ++j) {
                        err << inDims[j];
                        if (j + 1 < inDims.size())
                            err << ", ";
                    }
                    err << "]";
                    if (i + 1 < inputs.size())
                        err << ", ";
                }
                throw std::runtime_error(err.str());
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    return requiresBroadcast;
}

std::vector<Tensor> FusedEquation::bindNamedInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const {
    if (!compiledFlatEquation) {
        throw std::runtime_error("Cannot bind inputs for an empty FusedEquation.");
    }

    const std::vector<std::string>& expectedNames = compiledFlatEquation->input_names;

    if (expectedNames.empty()) {
        throw std::runtime_error("This fused equation has no compiled named inputs.");
    }

    std::vector<Tensor> orderedInputs;
    orderedInputs.reserve(expectedNames.size());

    for (const std::string& name : expectedNames) {
        auto it = namedInputs.find(name);
        if (it == namedInputs.end()) {
            throw std::runtime_error("Missing required fused equation input: " + name);
        }
        orderedInputs.push_back(it->second);
    }

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(expectedNames.size());
    for (const std::string& name : expectedNames) {
        expected_input_set.insert(name);
    }
    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }

    return orderedInputs;
}

StampedEquation FusedEquation::stampEquation(const std::unordered_map<std::string, Tensor>& inputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape) const {
    std::vector<Tensor> orderedInputs = bindNamedInputs(inputs);
    return stampEquation(orderedInputs, stream, requestedOutputShape);
}

static unordered_map<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache;
static shared_ptr<BuiltReduction> cacheLookup(const ReductionCacheKey& key) {
    auto it = builtReductionCache.find(key);
    if (it == builtReductionCache.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<BuiltReduction> FusedEquation::buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                              const Tensor& input,
                                                              int device_num) {
    const std::vector<uint64_t> input_dims = input.getDimensions();

    ReductionCacheKey key(compiled_reduction->op,
                          input_dims,
                          compiled_reduction->reduction_axes,
                          compiled_reduction->keepdim,
                          compiled_reduction->inout_dtype,
                          compiled_reduction->compute_dtype,
                          device_num);

    std::shared_ptr<BuiltReduction> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltReduction>(key);

    const std::vector<uint64_t> output_dims = computeReductionOutputDims(input_dims,
                                                                         compiled_reduction->reduction_axes,
                                                                         /*internal_keepdim=*/true);

    built->a_desc = createCudnnTensorDescriptor(input_dims, compiled_reduction->inout_dtype);
    built->c_desc = createCudnnTensorDescriptor(output_dims, compiled_reduction->inout_dtype);
    built->reduce_desc = createCudnnReduceDescriptor(compiled_reduction->op, compiled_reduction->compute_dtype);

    built->workspace_bytes = getReductionWorkspaceSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);

    builtReductionCache.emplace(key, built);
    return built;
}

StampedReduction FusedEquation::stampReduction(Tensor& input, const Stream& stream) const {
    if (!compiledReduction)
        throw std::runtime_error("Tried to stamp reduction on a non-reduction FusedEquation.");

    if (input.getDataType() != compiledReduction->inout_dtype) {
        throw std::runtime_error("Input dtype does not match compiled reduction dtype.");
    }

    std::shared_ptr<BuiltReduction> built = buildReduction(compiledReduction, input, stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(input.getPlacement(), workspaceDescriptor);
    }

    vector<uint64_t> outputDimensions = computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.keepdim);
    TensorDescriptor outputDescriptor(input.getDataType(), outputDimensions);
    Tensor output = Tensor(input.getPlacement(), outputDescriptor);

    return StampedReduction(std::move(built), input, output, stream, workspace);
}

std::vector<uint64_t> FusedEquation::computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                                const std::vector<uint64_t>& reduction_axes,
                                                                bool keepdim) {
    std::vector<uint64_t> output_dims = input_dims;

    for (uint64_t axis : reduction_axes) {
        if (axis >= output_dims.size())
            throw std::runtime_error("Reduction axis out of range.");
        output_dims[axis] = 1;
    }

    if (keepdim)
        return output_dims;

    std::vector<uint64_t> squeezed;
    squeezed.reserve(output_dims.size());
    for (uint64_t d : output_dims) {
        if (d != 1)
            squeezed.push_back(d);
    }
    if (squeezed.empty())
        squeezed.push_back(1);
    return squeezed;
}

cudnnTensorDescriptor_t FusedEquation::createCudnnTensorDescriptor(const std::vector<uint64_t>& dims, TensorDescriptor::DataType dtype) {
    if (dims.empty())
        throw std::runtime_error("cuDNN reduction does not support empty-rank tensor descriptor here.");
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

cudnnReduceTensorDescriptor_t FusedEquation::createCudnnReduceDescriptor(ExprOp op, TensorDescriptor::DataType compute_dtype) {
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

size_t FusedEquation::getReductionWorkspaceSize(int device_num,
                                                cudnnReduceTensorDescriptor_t reduce_desc,
                                                cudnnTensorDescriptor_t a_desc,
                                                cudnnTensorDescriptor_t c_desc) {
    Stream stream(device_num);
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(stream.getCudnnHandle(), reduce_desc, a_desc, c_desc, &workspace_bytes));
    return workspace_bytes;
}

void FusedEquation::runEquation(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    std::vector<Tensor> orderedInputs = bindNamedInputs(inputs);
    runEquation(std::move(orderedInputs), output, stream);
}

}  // namespace ThorImplementation
