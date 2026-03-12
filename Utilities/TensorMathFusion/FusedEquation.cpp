#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace ThorImplementation;

namespace {
cudaDataType_t toCudaDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return CUDA_R_32F;
        default:
            throw std::runtime_error("Unsupported tensor data type for fused equation.");
    }
}
}  // namespace

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num, bool use_fast_math) {
    if (dtype != TensorDescriptor::DataType::FP32) {
        throw std::runtime_error("FusedEquation::compile V1 currently only supports FP32.");
    }

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

    // FIXME add:
    // PhysicalExpressionValidator::validate(expr);

    EquationSignature sig{};
    sig.rank = 1;  // V1 assumes flattened contiguous tensors
    sig.num_inputs = expr.num_inputs;
    sig.dtype = toCudaDataType(dtype);
    sig.contiguous = true;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;

    EquationCompiler compiler;

    std::shared_ptr<CompiledEquation> flatEquation = compiler.compile(expr, sig, false);
    std::shared_ptr<CompiledEquation> broadcastEquation = compiler.compile(expr, sig, true);

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

StampedEquation FusedEquation::stamp(std::vector<Tensor> inputs,
                                     const Stream& stream,
                                     const std::vector<uint64_t>& requestedOutputShape) const {
    if (!compiledFlatEquation) {
        throw std::runtime_error("Cannot stamp an empty FusedEquation.");
    }

    if (inputs.empty()) {
        throw std::runtime_error("FusedEquation::stamp requires at least one input tensor.");
    }

    if (inputs.size() != compiledFlatEquation->num_inputs) {
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

static std::array<unsigned long long, MAX_BROADCAST_DIMS> computeContiguousStrides(const std::vector<uint64_t>& dims) {
    if (dims.empty()) {
        throw std::runtime_error("Cannot compute strides for empty shape.");
    }
    if (dims.size() > MAX_BROADCAST_DIMS) {
        throw std::runtime_error("Shape rank exceeds MAX_BROADCAST_DIMS.");
    }

    std::array<unsigned long long, MAX_BROADCAST_DIMS> strides{};
    strides.fill(0);

    unsigned long long running = 1;
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 1; axis >= 0; --axis) {
        strides[axis] = running;
        running *= static_cast<unsigned long long>(dims[axis]);
    }

    return strides;
}

static unsigned long long computeNumel(const std::vector<uint64_t>& dims) {
    unsigned long long numel = 1;
    for (uint64_t d : dims) {
        numel *= static_cast<unsigned long long>(d);
    }
    return numel;
}

BroadcastInfo FusedEquation::buildBroadcastInfo(const std::vector<Tensor>& inputs, const std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        throw std::runtime_error("buildBroadcastInfo requires at least one input.");
    }
    if (outputDimensions.empty()) {
        throw std::runtime_error("buildBroadcastInfo requires non-empty outputDimensions.");
    }
    if (outputDimensions.size() > MAX_BROADCAST_DIMS) {
        throw std::runtime_error("Output rank exceeds MAX_BROADCAST_DIMS.");
    }
    if (inputs.size() > MAX_FUSED_INPUTS) {
        throw std::runtime_error("Number of inputs exceeds MAX_FUSED_INPUTS.");
    }

    BroadcastInfo info{};
    info.rank = static_cast<unsigned int>(outputDimensions.size());
    info.num_inputs = static_cast<unsigned int>(inputs.size());
    info.numel = computeNumel(outputDimensions);

    for (uint32_t axis = 0; axis < MAX_BROADCAST_DIMS; ++axis) {
        info.output_strides[axis] = 0ULL;
    }

    auto outputStrides = computeContiguousStrides(outputDimensions);
    for (uint32_t axis = 0; axis < info.rank; ++axis) {
        info.output_strides[axis] = outputStrides[axis];
    }

    for (uint32_t inputIdx = 0; inputIdx < inputs.size(); ++inputIdx) {
        const std::vector<uint64_t>& dims = inputs[inputIdx].getDimensions();

        if (dims.size() != info.rank) {
            throw std::runtime_error("Input rank does not match resolved output rank in buildBroadcastInfo.");
        }

        auto inputStrides = computeContiguousStrides(dims);

        for (uint32_t axis = 0; axis < MAX_BROADCAST_DIMS; ++axis) {
            info.inputs[inputIdx].strides[axis] = 0ULL;
        }

        for (uint32_t axis = 0; axis < info.rank; ++axis) {
            const uint64_t inDim = dims[axis];
            const uint64_t outDim = outputDimensions[axis];

            if (inDim == outDim) {
                info.inputs[inputIdx].strides[axis] = inputStrides[axis];
            } else if (inDim == 1) {
                info.inputs[inputIdx].strides[axis] = 0ULL;
            } else {
                std::ostringstream err;
                err << "Input " << inputIdx << " is not broadcast-compatible at axis " << axis << ": input dim = " << inDim
                    << ", output dim = " << outDim;
                throw std::runtime_error(err.str());
            }
        }
    }

    return info;
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
    BroadcastInfo broadcastInfo = buildBroadcastInfo(inputs, outputDimensions);
    TensorDescriptor broadcastInfoDescriptor = TensorDescriptor(TensorDescriptor::DataType::UINT8, {sizeof(broadcastInfo)});

    TensorPlacement cpuPlacement = TensorPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement devicePlacement = inputs[0].getPlacement();
    Tensor hostBroadcastInfo = Tensor(cpuPlacement, broadcastInfoDescriptor);
    Tensor deviceBroadcastInfo(devicePlacement, broadcastInfoDescriptor);

    // Initialize cpu mem and copy to gpu
    memcpy(hostBroadcastInfo.getMemPtr(), &broadcastInfo, sizeof(BroadcastInfo));
    deviceBroadcastInfo.copyFromAsync(hostBroadcastInfo, stream);

    // free tensor memory at the end of the stream
    stream.launchCleanUpHostFunctionArgs(std::make_unique<BroadcastInfoHostFunctionArgs>(hostBroadcastInfo, deviceBroadcastInfo));

    return deviceBroadcastInfo;
}

void FusedEquation::run(std::vector<Tensor> inputs, Tensor output, Stream stream) const {
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
