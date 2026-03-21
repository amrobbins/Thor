#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/CudaSourceEmitter.h"
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
                               const std::vector<uint64_t>& outputDimensions,
                               const std::unordered_set<uint32_t>* usedInputSlots) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    const uint32_t numInputs = static_cast<uint32_t>(inputs.size());

    buffer.header()->numel = product(outputDimensions);

    const std::vector<uint64_t> outputStrides = computePackedOutputStrides(outputDimensions);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        buffer.outputStrides()[axis] = outputStrides[axis];
    }

    for (uint32_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
        const bool used = (usedInputSlots == nullptr) || usedInputSlots->contains(inputIdx);

        if (!used) {
            for (uint32_t axis = 0; axis < rank; ++axis) {
                buffer.inputStride(inputIdx, axis) = 0;
            }
            continue;
        }

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

static void buildBroadcastInfo(BroadcastInfoBufferView& buffer,
                               const std::vector<Tensor>& inputs,
                               const std::vector<uint64_t>& outputDimensions) {
    buildBroadcastInfo(buffer, inputs, outputDimensions, nullptr);
}

static void collectReferencedLocalInputSlots(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& slots) {
    if (node_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectReferencedLocalInputSlots saw node index out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];

    if (node.op == ExprOp::INPUT) {
        slots.insert(node.input_slot);
        return;
    }

    if (Expression::isLeafOp(node.op)) {
        return;
    }

    collectReferencedLocalInputSlots(expr, node.lhs, slots);

    if (Expression::isBinaryOp(node.op)) {
        collectReferencedLocalInputSlots(expr, node.rhs, slots);
    }
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<Tensor>& stage_inputs) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput output_idx out of range.");
    }

    std::unordered_set<uint32_t> used_input_slots;
    collectReferencedLocalInputSlots(stage.expr, stage.outputs[output_idx].local_node_idx, used_input_slots);

    if (used_input_slots.empty()) {
        throw std::runtime_error("Broadcast output grouping currently requires each output to depend on at least one tensor input.");
    }

    std::vector<Tensor> subset_inputs;
    subset_inputs.reserve(used_input_slots.size());

    for (uint32_t local_slot = 0; local_slot < stage_inputs.size(); ++local_slot) {
        if (used_input_slots.contains(local_slot)) {
            subset_inputs.push_back(stage_inputs[local_slot]);
        }
    }

    if (subset_inputs.empty()) {
        throw std::runtime_error("Failed to collect subset inputs for stage output.");
    }

    std::vector<std::vector<uint64_t>> original_input_dimensions;
    original_input_dimensions.reserve(subset_inputs.size());
    for (const Tensor& input : subset_inputs) {
        original_input_dimensions.push_back(input.getDimensions());
    }

    uint64_t maxRank = 0;
    for (const Tensor& input : subset_inputs) {
        const std::vector<uint64_t>& dims = input.getDimensions();
        if (dims.empty()) {
            throw std::runtime_error("Input tensor has 0 dimensions, which is not supported.");
        }
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    for (Tensor& input : subset_inputs) {
        const std::vector<uint64_t>& oldDims = input.getDimensions();
        if (oldDims.size() == maxRank) {
            continue;
        }

        std::vector<uint64_t> paddedDims(maxRank - oldDims.size(), 1);
        paddedDims.insert(paddedDims.end(), oldDims.begin(), oldDims.end());
        input.reshape(paddedDims);
    }

    std::vector<uint64_t> outputDimensions(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const Tensor& input : subset_inputs) {
            const std::vector<uint64_t>& dims = input.getDimensions();
            const uint64_t dim = dims[axis];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
            } else if (resolvedDim != dim) {
                std::ostringstream err;
                err << "Stage output inputs are not broadcast-compatible at axis " << axis << ". "
                    << "Encountered dimension " << resolvedDim << " and dimension " << dim << ". "
                    << "Input shapes: ";
                for (size_t i = 0; i < original_input_dimensions.size(); ++i) {
                    const std::vector<uint64_t>& inDims = original_input_dimensions[i];
                    err << "[";
                    for (size_t j = 0; j < inDims.size(); ++j) {
                        err << inDims[j];
                        if (j + 1 < inDims.size()) {
                            err << ", ";
                        }
                    }
                    err << "]";
                    if (i + 1 < original_input_dimensions.size()) {
                        err << ", ";
                    }
                }
                throw std::runtime_error(err.str());
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    return outputDimensions;
}

struct ResolvedBroadcastGroup {
    SpecializedBroadcastGroup specialized;
};

static std::vector<uint64_t> computeInputPackedStridesForBroadcast(const std::vector<uint64_t>& inputDims,
                                                                   const std::vector<uint64_t>& outputDimensions) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    if (inputDims.size() > outputDimensions.size()) {
        throw std::runtime_error("Input rank exceeds output rank in computeInputPackedStridesForBroadcast.");
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

    return inputPackedStrides;
}

static bool canUseNativeVectorLoadForUsedInput(const SpecializedBroadcastGroup& group, size_t used_input_idx) {
    uint64_t contiguous_slice_size = 1;
    uint64_t expected_stride = 1;
    bool found_contiguous_suffix = false;

    for (int64_t axis_i = static_cast<int64_t>(group.active_axes.size()) - 1; axis_i >= 0; --axis_i) {
        const SpecializedBroadcastAxis& axis = group.active_axes[static_cast<size_t>(axis_i)];
        const uint64_t input_stride = axis.input_strides[used_input_idx];

        if (input_stride == 0) {
            break;
        }

        if (input_stride != expected_stride) {
            break;
        }

        found_contiguous_suffix = true;
        contiguous_slice_size *= axis.dim;
        expected_stride *= axis.dim;
    }

    if (!found_contiguous_suffix) {
        return false;
    }

    // Safe when pair boundaries cannot cross an internal odd-sized reset.
    // If the contiguous slice is even, every packed pair start stays aligned.
    // If the contiguous slice is the full group, only the final tail can straddle,
    // and tensor padding makes that safe.
    return (contiguous_slice_size % 2ULL == 0ULL) || (contiguous_slice_size == group.numel);
}

static SpecializedBroadcastGroup buildSpecializedBroadcastGroup(const CompiledExecutionStage& stage,
                                                                const std::vector<Tensor>& stage_inputs,
                                                                const std::vector<uint64_t>& output_dims,
                                                                const std::vector<uint32_t>& output_indices) {
    SpecializedBroadcastGroup group;
    group.output_dims = output_dims;
    group.output_indices = output_indices;
    group.numel = product(output_dims);

    std::unordered_set<uint32_t> used_input_slots_set;
    for (uint32_t out_idx : output_indices) {
        if (out_idx >= stage.outputs.size()) {
            throw std::runtime_error("buildSpecializedBroadcastGroup output index out of range.");
        }
        collectReferencedLocalInputSlots(stage.expr, stage.outputs[out_idx].local_node_idx, used_input_slots_set);
    }

    group.used_input_slots.assign(used_input_slots_set.begin(), used_input_slots_set.end());
    std::sort(group.used_input_slots.begin(), group.used_input_slots.end());

    const uint32_t rank = static_cast<uint32_t>(output_dims.size());
    const std::vector<uint64_t> output_strides = computePackedOutputStrides(output_dims);

    std::vector<std::vector<uint64_t>> per_input_strides;
    per_input_strides.reserve(group.used_input_slots.size());
    for (uint32_t local_slot : group.used_input_slots) {
        if (local_slot >= stage_inputs.size()) {
            throw std::runtime_error("buildSpecializedBroadcastGroup local input slot out of range.");
        }
        per_input_strides.push_back(computeInputPackedStridesForBroadcast(stage_inputs[local_slot].getDimensions(), output_dims));
    }

    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (output_dims[axis] == 1) {
            continue;
        }

        SpecializedBroadcastAxis axis_desc;
        axis_desc.dim = output_dims[axis];
        axis_desc.output_stride = output_strides[axis];
        axis_desc.input_strides.reserve(group.used_input_slots.size());

        bool contributes_to_any_input = false;
        for (const std::vector<uint64_t>& input_strides : per_input_strides) {
            const uint64_t s = input_strides[axis];
            axis_desc.input_strides.push_back(s);
            if (s != 0) {
                contributes_to_any_input = true;
            }
        }

        if (!contributes_to_any_input) {
            continue;
        }

        group.active_axes.push_back(std::move(axis_desc));
    }

    group.used_input_load_kinds.reserve(group.used_input_slots.size());
    for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
        if (canUseNativeVectorLoadForUsedInput(group, used_i)) {
            group.used_input_load_kinds.push_back(SpecializedInputLoadKind::NativeVector);
        } else {
            group.used_input_load_kinds.push_back(SpecializedInputLoadKind::ScalarPack);
        }
    }

    return group;
}

struct BroadcastInfoHostFunctionArgs : HostFunctionArgsBase {
    Tensor hostBroadcastInfo;
    Tensor deviceBroadcastInfo;

    explicit BroadcastInfoHostFunctionArgs(Tensor hostBroadcastInfo, Tensor deviceBroadcastInfo)
        : hostBroadcastInfo(hostBroadcastInfo), deviceBroadcastInfo(deviceBroadcastInfo) {}
};

static Tensor createDeviceBroadcastInfoForUsedInputs(const std::vector<Tensor>& inputs,
                                                     const std::vector<uint64_t>& outputDimensions,
                                                     const std::unordered_set<uint32_t>& usedInputSlots,
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
    buildBroadcastInfo(buffer, inputs, outputDimensions, &usedInputSlots);

    deviceBroadcastInfo.copyFromAsync(hostBroadcastInfo, stream);
    stream.launchCleanUpHostFunctionArgs(std::make_unique<BroadcastInfoHostFunctionArgs>(hostBroadcastInfo, deviceBroadcastInfo));

    return deviceBroadcastInfo;
}

static std::vector<ResolvedBroadcastGroup> buildResolvedBroadcastGroups(const CompiledExecutionStage& stage,
                                                                        const std::vector<Tensor>& stage_inputs) {
    std::map<std::vector<uint64_t>, std::vector<uint32_t>> grouped_output_indices;

    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        std::vector<uint64_t> dims = resolveOutputDimsForStageOutput(stage, i, stage_inputs);
        grouped_output_indices[dims].push_back(i);
    }

    std::vector<ResolvedBroadcastGroup> groups;
    groups.reserve(grouped_output_indices.size());

    for (auto& [dims, output_indices] : grouped_output_indices) {
        ResolvedBroadcastGroup resolved;
        resolved.specialized = buildSpecializedBroadcastGroup(stage, stage_inputs, dims, output_indices);
        groups.push_back(std::move(resolved));
    }

    std::sort(groups.begin(), groups.end(), [](const ResolvedBroadcastGroup& a, const ResolvedBroadcastGroup& b) {
        if (a.specialized.numel != b.specialized.numel) {
            return a.specialized.numel > b.specialized.numel;
        }

        if (a.specialized.output_dims.size() != b.specialized.output_dims.size()) {
            return a.specialized.output_dims.size() > b.specialized.output_dims.size();
        }

        return a.specialized.output_dims < b.specialized.output_dims;
    });

    return groups;
}

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

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const Tensor& input : inputs) {
            const std::vector<uint64_t>& dims = input.getDimensions();
            uint64_t dim = dims[axis];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
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

    bool requiresBroadcast = false;
    for (const Tensor& input : inputs) {
        if (input.getDimensions() != outputDimensions) {
            requiresBroadcast = true;
            break;
        }
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
    std::vector<Tensor> infos{deviceBroadcastInfo};
    return stampEquation(compiledEquation, inputs, outputs, stream, infos);
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

            TensorPlacement outputPlacement = layoutInputs[0].getPlacement();

            if (!requiresBroadcast) {
                for (const CompiledStageOutput& produced : stage.outputs) {
                    auto requested_it = requestedOutputShapes.find(produced.name);
                    const std::vector<uint64_t>* requested_shape =
                        (requested_it != requestedOutputShapes.end()) ? &requested_it->second : nullptr;

                    if (requested_shape && !requested_shape->empty()) {
                        verifyRequestedOutputLayout(*requested_shape, resolvedOutputDims);
                    }

                    TensorDescriptor outputDescriptor(
                        stage.flat->dtype, (requested_shape && !requested_shape->empty()) ? *requested_shape : resolvedOutputDims);
                    stageOutputs.emplace_back(outputPlacement, outputDescriptor);
                }
            } else {
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& produced = stage.outputs[output_idx];

                    std::vector<uint64_t> resolved_output_dims = resolveOutputDimsForStageOutput(stage, output_idx, layoutInputs);

                    auto requested_it = requestedOutputShapes.find(produced.name);
                    const std::vector<uint64_t>* requested_shape =
                        (requested_it != requestedOutputShapes.end()) ? &requested_it->second : nullptr;

                    if (requested_shape && !requested_shape->empty()) {
                        verifyRequestedOutputLayout(*requested_shape, resolved_output_dims);
                    }

                    TensorDescriptor outputDescriptor(
                        stage.flat->dtype, (requested_shape && !requested_shape->empty()) ? *requested_shape : resolved_output_dims);
                    stageOutputs.emplace_back(outputPlacement, outputDescriptor);
                }
            }

            std::shared_ptr<StampedEquation> stamped;
            if (requiresBroadcast) {
                std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, layoutInputs);

                std::vector<SpecializedBroadcastGroup> specialized_groups;
                specialized_groups.reserve(groups.size());
                for (const ResolvedBroadcastGroup& group : groups) {
                    specialized_groups.push_back(group.specialized);
                }

                printf("PATH 1\n");
                std::shared_ptr<CompiledEquation> specialized_broadcast =
                    EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);

                stamped = stampEquation(specialized_broadcast, layoutInputs, stageOutputs, stream);
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

std::shared_ptr<StampedEquation> FusedEquation::stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                              std::vector<Tensor>& inputs,
                                                              std::vector<Tensor>& outputs,
                                                              const Stream& stream,
                                                              const std::vector<Tensor>& deviceBroadcastInfos) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot stamp an empty compiled equation.");
    }

    if (inputs.empty()) {
        throw std::runtime_error("FusedEquation::stampEquation requires at least one input tensor.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("FusedEquation::stampEquation requires at least one output tensor.");
    }

    return std::make_shared<StampedEquation>(compiledEquation, inputs, outputs, stream, deviceBroadcastInfos);
}

void FusedEquation::run(const Tensor& input, Tensor& output, Stream& stream) const {
    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::run requires compiled outputs.");
    }

    for (CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != stage.Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused equations stages, but a stage has type: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }
    }

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages");
    }

    const CompiledExecutionStage& stage = compiled_outputs->stages[0];

    if (stage.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run was only passed a single output, but there are multiple stages"
            "Pass a dict of name -> PhysicalTensor of outputs to run it.");
    }

    if (root_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(root_inputs.size()) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& input_name = root_inputs[0].name;
    const std::string& output_name = stage.outputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{input_name, input}};
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};

    run(input_map, output_map, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::run requires compiled outputs.");
    }

    for (CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != stage.Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused equations stages, but a stage has type: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }
    }

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages");
    }

    const CompiledExecutionStage& stage = compiled_outputs->stages[0];

    if (stage.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run was only passed a single output, but there are multiple stages."
            "Pass a dict of name -> PhysicalTensor of outputs to run it.");
    }

    const std::string& output_name = stage.outputs[0].name;
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};
    run(inputs, output_map, stream);
}

void FusedEquation::run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const {
    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::run requires compiled outputs.");
    }

    for (CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != stage.Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused equations stages, but a stage has type: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }
    }

    if (root_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(root_inputs.size()) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& input_name = root_inputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{input_name, input}};

    run(input_map, outputs, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        std::unordered_map<std::string, Tensor>& outputs,
                        Stream& stream) const {
    if (!compiled_outputs) {
        throw std::runtime_error("FusedEquation::run requires compiled outputs.");
    }

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages.");
    }

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused-kernel stages, but found stage kind: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }
    }

    // Root inputs only. If any later stage depends on a non-root value, this run()
    // overload must reject it and require stamp(...).run().
    const std::unordered_map<uint32_t, Tensor> root_values = bindRootInputs(inputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::run requires at least one bound root input.");
    }

    std::unordered_set<uint32_t> root_value_ids;
    root_value_ids.reserve(root_values.size());
    for (const auto& [value_id, tensor] : root_values) {
        (void)tensor;
        root_value_ids.insert(value_id);
    }

    // Collect all expected final output names across all fused stages.
    std::unordered_set<std::string> expected_output_names;
    std::vector<std::string> expected_output_names_in_order;

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        for (const auto& stage_output : stage.outputs) {
            if (expected_output_names.insert(stage_output.name).second) {
                expected_output_names_in_order.push_back(stage_output.name);
            }
        }
    }

    // Verify caller provided every expected final output.
    for (const std::string& name : expected_output_names_in_order) {
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            throw std::runtime_error("Missing output tensor '" + name + "' for fused equation run.");
        }
    }

    // Verify caller did not provide unexpected outputs.
    for (const auto& [name, tensor] : outputs) {
        (void)tensor;
        if (!expected_output_names.contains(name)) {
            std::string expected_names_str;
            for (size_t i = 0; i < expected_output_names_in_order.size(); ++i) {
                if (i > 0) {
                    expected_names_str += ", ";
                }
                expected_names_str += "'" + expected_output_names_in_order[i] + "'";
            }
            throw std::runtime_error("Unexpected output tensor '" + name +
                                     "' passed to fused equation run. "
                                     "Expected output names: [" +
                                     expected_names_str + "].");
        }
    }

    // Infer GPU number for helper-stream selection.
    const int32_t gpu_num = root_values.begin()->second.getPlacement().getDeviceNum();

    for (const auto& [value_id, tensor] : root_values) {
        (void)value_id;
        if (tensor.getPlacement().getDeviceNum() != gpu_num) {
            throw std::runtime_error("FusedEquation::run requires all root inputs to be on the same GPU.");
        }
    }
    for (const auto& [name, tensor] : outputs) {
        (void)name;
        if (tensor.getPlacement().getDeviceNum() != gpu_num) {
            throw std::runtime_error("FusedEquation::run requires all outputs to be on the same GPU.");
        }
    }

    auto runStageOnStream = [&](const CompiledExecutionStage& stage,
                                const std::vector<Tensor>& orderedInputs,
                                const std::vector<Tensor>& orderedOutputs,
                                Stream& launch_stream) {
        std::vector<Tensor> layoutInputs = orderedInputs;
        std::vector<uint64_t> resolvedOutputDims;
        const bool requiresBroadcast = resolveLayout(layoutInputs, resolvedOutputDims);

        if (!requiresBroadcast) {
            if (!stage.flat) {
                throw std::runtime_error("FusedEquation::run found a flat fused stage with no compiled kernel.");
            }

            for (const Tensor& out : orderedOutputs) {
                verifyRequestedOutputLayout(out.getDimensions(), resolvedOutputDims);
            }

            EquationRunner::run(stage.flat, layoutInputs, orderedOutputs, launch_stream);
            return;
        }

        std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, orderedInputs);
        if (groups.empty()) {
            throw std::runtime_error("FusedEquation::run expected at least one broadcast group.");
        }

        std::vector<SpecializedBroadcastGroup> specialized_groups;
        specialized_groups.reserve(groups.size());

        std::vector<const std::vector<uint64_t>*> expected_output_dims(stage.outputs.size(), nullptr);

        for (const auto& group : groups) {
            specialized_groups.push_back(group.specialized);

            for (uint32_t output_idx : group.specialized.output_indices) {
                if (output_idx >= expected_output_dims.size()) {
                    throw std::runtime_error("Broadcast group output index out of range.");
                }
                expected_output_dims[output_idx] = &group.specialized.output_dims;
            }
        }

        for (size_t i = 0; i < orderedOutputs.size(); ++i) {
            if (!expected_output_dims[i]) {
                throw std::runtime_error("Missing resolved output dims for output '" + stage.outputs[i].name + "'.");
            }

            verifyRequestedOutputLayout(orderedOutputs[i].getDimensions(), *expected_output_dims[i]);
        }

        std::shared_ptr<CompiledEquation> specialized_broadcast =
            EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);

        EquationRunner::run(specialized_broadcast, orderedInputs, orderedOutputs, launch_stream);
    };

    // Track helper streams actually used so the caller's stream can join them at the end.
    std::vector<Stream> helper_streams_used;
    helper_streams_used.reserve(compiled_outputs->stages.size());

    auto rememberHelperStream = [&](Stream& helper_stream) {
        if (std::find(helper_streams_used.begin(), helper_streams_used.end(), helper_stream) == helper_streams_used.end()) {
            helper_streams_used.push_back(helper_stream);
        }
    };

    // Execute each fused stage. This overload still only permits stages whose inputs
    // are all root inputs, so legal stages are independent and can be launched on
    // helper streams without extra inter-stage waits.

    for (uint32_t stage_num = 0; stage_num < compiled_outputs->stages.size(); ++stage_num) {
        bool use_helper_streams = (stage_num != 0);
        const CompiledExecutionStage& stage = compiled_outputs->stages[stage_num];

        std::vector<Tensor> orderedInputs;
        orderedInputs.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            if (!root_value_ids.contains(value_id)) {
                throw std::runtime_error(
                    "FusedEquation::run encountered a stage that depends on a non-root intermediate tensor. "
                    "Use stamp(...).run() for expressions requiring staged intermediates.");
            }

            auto it = root_values.find(value_id);
            if (it == root_values.end()) {
                throw std::runtime_error("Missing input value for fused equation run.");
            }

            orderedInputs.push_back(it->second);
        }

        std::vector<Tensor> orderedOutputs;
        orderedOutputs.reserve(stage.outputs.size());

        for (const auto& stage_output : stage.outputs) {
            auto it = outputs.find(stage_output.name);
            if (it == outputs.end()) {
                throw std::runtime_error("Missing output tensor '" + stage_output.name + "' for fused equation run.");
            }
            orderedOutputs.push_back(it->second);
        }

        if (use_helper_streams) {
            Stream& helper_stream = Expression::getNextHelperStream(gpu_num);
            runStageOnStream(stage, orderedInputs, orderedOutputs, helper_stream);
            rememberHelperStream(helper_stream);
        } else {
            runStageOnStream(stage, orderedInputs, orderedOutputs, stream);
        }
    }

    // Join all helper streams back into the user-provided stream.
    for (Stream& helper_stream : helper_streams_used) {
        stream.waitEvent(helper_stream.putEvent());
    }
}
}  // namespace ThorImplementation
