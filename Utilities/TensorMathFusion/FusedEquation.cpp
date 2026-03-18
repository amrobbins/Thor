#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

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

FusedEquation FusedEquation::compile(const PhysicalOutputs& outputs, TensorDescriptor::DataType dtype, int device_num, bool use_fast_math) {
    if (device_num < 0) {
        throw std::runtime_error("FusedEquation::compile requires device_num >= 0.");
    }

    if (!outputs.expr) {
        throw std::runtime_error("FusedEquation::compile requires non-null PhysicalOutputs.expr.");
    }

    if (outputs.outputs.empty()) {
        throw std::runtime_error("FusedEquation::compile requires at least one named output.");
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

    EquationSignature sig{};
    sig.num_inputs = outputs.expr->numInputs();
    sig.dtype = dtype;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;

    std::shared_ptr<CompiledOutputs> compiled = EquationCompiler::compile(outputs, sig, true);
    return FusedEquation(std::move(compiled), outputs.expr->inputs);
}

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num, bool use_fast_math) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("FusedEquation::compile PhysicalExpression output_node is out of range.");
    }

    PhysicalOutputs outputs;
    outputs.expr = std::make_shared<PhysicalExpression>(expr);
    outputs.outputs.push_back(NamedOutput{
        .name = "output",
        .node_idx = expr.output_node,
    });

    return compile(outputs, dtype, device_num, use_fast_math);
}

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

bool FusedEquation::resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty())
        throw std::runtime_error("Tried to create a FusedEquation with 0 tensor inputs. You must have at least one.");

    std::vector<std::vector<uint64_t>> originalInputDimensions;
    originalInputDimensions.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        originalInputDimensions.push_back(input.getDimensions());
    }

    uint64_t maxRank = 0;
    for (const Tensor& input : inputs) {
        const std::vector<uint64_t>& dims = input.getDimensions();
        if (dims.empty())
            throw std::runtime_error("Input tensor has 0 dimensions, which is not supported.");
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    for (Tensor& input : inputs) {
        const std::vector<uint64_t>& oldDims = input.getDimensions();
        if (oldDims.size() == maxRank)
            continue;

        std::vector<uint64_t> paddedDims(maxRank - oldDims.size(), 1);
        paddedDims.insert(paddedDims.end(), oldDims.begin(), oldDims.end());
        input.reshape(paddedDims);
    }

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

std::unordered_map<uint32_t, Tensor> FusedEquation::bindRootInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const {
    std::unordered_map<uint32_t, Tensor> values;
    values.reserve(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = namedInputs.find(input.name);
        if (it == namedInputs.end()) {
            throw std::runtime_error("Missing required fused equation input: " + input.name);
        }
        values.emplace(input.slot, it->second);
    }

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        expected_input_set.insert(input.name);
    }

    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }

    return values;
}

std::shared_ptr<StampedEquation> FusedEquation::stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                              std::vector<Tensor>& inputs,
                                                              std::vector<Tensor>& outputs,
                                                              const Stream& stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot stamp an empty compiled equation.");
    }

    if (inputs.empty()) {
        throw std::runtime_error("FusedEquation::stampEquation requires at least one input tensor.");
    }

    if (inputs.size() != compiledEquation->numInputs()) {
        throw std::runtime_error("Wrong number of inputs passed to FusedEquation::stampEquation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("FusedEquation::stampEquation requires at least one output tensor.");
    }

    const Tensor& firstInput = inputs[0];
    if (!firstInput.isInitialized()) {
        throw std::runtime_error("First input tensor is not initialized.");
    }

    if (firstInput.getDescriptor().getDataType() != compiledEquation->dtype) {
        throw std::runtime_error("Input tensor data type does not match compiled equation data type.");
    }

    if (firstInput.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Input tensor GPU does not match compiled equation device.");
    }

    for (uint64_t i = 1; i < inputs.size(); ++i) {
        if (!inputs[i].isInitialized()) {
            throw std::runtime_error("Input tensor is not initialized.");
        }
        if (inputs[i].getDescriptor().getDataType() != compiledEquation->dtype) {
            throw std::runtime_error("Input tensor data type mismatch.");
        }
        if (inputs[i].getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU mismatch.");
        }
    }

    for (const Tensor& output : outputs) {
        if (!output.isInitialized()) {
            throw std::runtime_error("Output tensor is not initialized.");
        }
        if (output.getDescriptor().getDataType() != compiledEquation->dtype) {
            throw std::runtime_error("Output tensor data type mismatch.");
        }
        if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Output tensor GPU mismatch.");
        }
    }

    return make_shared<StampedEquation>(compiledEquation, inputs, outputs, stream);
}

std::shared_ptr<StampedEquation> FusedEquation::stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                              std::vector<Tensor>& inputs,
                                                              std::vector<Tensor>& outputs,
                                                              const Stream& stream,
                                                              const Tensor& deviceBroadcastInfo) const {
    return make_shared<StampedEquation>(compiledEquation, inputs, outputs, stream, deviceBroadcastInfo);
}

std::shared_ptr<StampedReduction> FusedEquation::stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                Tensor& input,
                                                                const Stream& stream) const {
    if (!compiledReduction)
        throw std::runtime_error("Tried to stamp reduction on a non-reduction FusedEquation.");

    if (input.getDataType() != compiledReduction->inout_dtype) {
        throw std::runtime_error("Input dtype does not match compiled reduction dtype.");
    }

    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiledReduction, input, stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(input.getPlacement(), workspaceDescriptor);
    }

    vector<uint64_t> outputDimensions =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);
    TensorDescriptor outputDescriptor(input.getDataType(), outputDimensions);
    Tensor output = Tensor(input.getPlacement(), outputDescriptor);

    return make_shared<StampedReduction>(std::move(built), input, output, stream, workspace);
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::stamp requires compiled outputs.");
    }

    std::unordered_map<uint32_t, Tensor> values = bindRootInputs(inputs);
    std::vector<StampedExecutionStage> stampedStages;
    stampedStages.reserve(compiled_outputs->stages.size());

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<Tensor> stageInputs;
        stageInputs.reserve(stage.input_value_ids.size());
        for (uint32_t value_id : stage.input_value_ids) {
            auto it = values.find(value_id);
            if (it == values.end()) {
                throw std::runtime_error("Missing input value for staged execution plan.");
            }
            stageInputs.push_back(it->second);
        }

        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            if (!stage.flat) {
                throw std::runtime_error("Missing compiled fused kernel stage.");
            }

            std::vector<Tensor> layoutInputs = stageInputs;
            std::vector<uint64_t> resolvedOutputDims;
            bool requiresBroadcast = resolveLayout(layoutInputs, resolvedOutputDims);

            std::vector<Tensor> stageOutputs;
            stageOutputs.reserve(stage.outputs.size());

            for (const CompiledStageOutput& produced : stage.outputs) {
                auto requested_it = requestedOutputShapes.find(produced.name);
                const std::vector<uint64_t>* requested_shape =
                    (requested_it != requestedOutputShapes.end()) ? &requested_it->second : nullptr;

                if (requested_shape && !requested_shape->empty()) {
                    verifyRequestedOutputLayout(*requested_shape, resolvedOutputDims);
                }

                TensorPlacement outputPlacement = layoutInputs[0].getPlacement();
                TensorDescriptor outputDescriptor(stage.flat->dtype,
                                                  (requested_shape && !requested_shape->empty()) ? *requested_shape : resolvedOutputDims);
                stageOutputs.emplace_back(outputPlacement, outputDescriptor);
            }

            std::shared_ptr<StampedEquation> stamped;
            if (requiresBroadcast) {
                Tensor broadcastInfo = createDeviceBroadcastInfo(layoutInputs, resolvedOutputDims, stream);
                stamped = stampEquation(stage.broadcast, layoutInputs, stageOutputs, stream, broadcastInfo);
            } else {
                stamped = stampEquation(stage.flat, layoutInputs, stageOutputs, stream);
            }

            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                values[stage.outputs[i].value_id] = stageOutputs[i];
            }

            stampedStages.emplace_back(stamped);
        } else {
            if (!stage.reduction) {
                throw std::runtime_error("Missing compiled reduction stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduction stage expected exactly one input and one output.");
            }

            Tensor& reductionInput = stageInputs[0];
            std::shared_ptr<StampedReduction> stamped = stampReduction(stage.reduction, reductionInput, stream);
            values[stage.outputs[0].value_id] = stamped->getOutputTensor();
            stampedStages.emplace_back(stamped);
        }
    }

    std::unordered_map<std::string, Tensor> finalOutputsByName;
    finalOutputsByName.reserve(compiled_outputs->final_outputs.size());
    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = values.find(final_output.value_id);
        if (it == values.end()) {
            throw std::runtime_error("Missing final output tensor for output: " + final_output.name);
        }
        finalOutputsByName.emplace(final_output.name, it->second);
    }

    return StampedExecutionPlan(std::move(stampedStages), std::move(finalOutputsByName));
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::vector<uint64_t>& requestedOutputShape) const {
    std::unordered_map<std::string, std::vector<uint64_t>> requested;
    if (!requestedOutputShape.empty()) {
        requested["output"] = requestedOutputShape;
    }
    return stamp(inputs, stream, requested);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    // FIXME: Run should support multi-output, it does not yet.

    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::run requires compiled outputs.");
    }

    if (compiled_outputs->stages.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run only supports single-stage fused equations. "
            "Use stamp(...).run() for staged expressions.");
    }

    const CompiledExecutionStage& stage = compiled_outputs->stages[0];

    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error(
            "FusedEquation::run only supports a single fused-kernel stage, not reductions. "
            "Use stamp(...).run() for staged expressions.");
    }

    if (!stage.flat) {
        throw std::runtime_error("FusedEquation::run found a fused stage with no compiled kernel.");
    }

    if (stage.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run only supports single-output fused equations. "
            "Use stamp(...).run() for multi-output expressions.");
    }

    std::unordered_map<uint32_t, Tensor> values = bindRootInputs(inputs);

    std::vector<Tensor> orderedInputs;
    orderedInputs.reserve(stage.input_value_ids.size());
    for (uint32_t value_id : stage.input_value_ids) {
        auto it = values.find(value_id);
        if (it == values.end()) {
            throw std::runtime_error("Missing input value for fused equation run.");
        }
        orderedInputs.push_back(it->second);
    }

    std::vector<Tensor> layoutInputs = orderedInputs;
    std::vector<uint64_t> resolvedOutputDims;
    bool requiresBroadcast = resolveLayout(layoutInputs, resolvedOutputDims);

    verifyRequestedOutputLayout(output.getDimensions(), resolvedOutputDims);

    std::vector<Tensor> outputsVec{output};

    if (requiresBroadcast) {
        Tensor broadcastInfo = createDeviceBroadcastInfo(layoutInputs, resolvedOutputDims, stream);
        EquationRunner::run(stage.broadcast, layoutInputs, outputsVec, stream, broadcastInfo);
    } else {
        EquationRunner::run(stage.flat, layoutInputs, outputsVec, stream);
    }
}

}  // namespace ThorImplementation
