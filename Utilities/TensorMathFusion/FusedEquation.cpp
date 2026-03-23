#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/AutoDiff.h"
#include "Utilities/TensorMathFusion/CudaSourceEmitter.h"
#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/ExpressionDTypeResolution.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace std;
using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

static RuntimeDTypeKey makeRuntimeDTypeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, Tensor>& root_values) {
    RuntimeDTypeKey key;
    key.root_input_dtypes.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound tensor for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dtypes[input.slot] = it->second.getDataType();
    }

    return key;
}

static RuntimeShapeKey makeRuntimeShapeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, Tensor>& root_values) {
    RuntimeShapeKey key;
    key.dtype_key = makeRuntimeDTypeKey(root_inputs, root_values);
    key.root_input_dims.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound tensor for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dims[input.slot] = it->second.getDimensions();
    }

    return key;
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

static bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        throw std::runtime_error("resolveLayoutFromDims requires at least one input shape.");
    }

    std::vector<std::vector<uint64_t>> originalInputDimensions = inputs;

    uint64_t maxRank = 0;
    for (const std::vector<uint64_t>& dims : inputs) {
        if (dims.empty()) {
            throw std::runtime_error("Input tensor has 0 dimensions, which is not supported.");
        }
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    std::vector<std::vector<uint64_t>> paddedInputs;
    paddedInputs.reserve(inputs.size());

    for (const std::vector<uint64_t>& oldDims : inputs) {
        if (oldDims.size() == maxRank) {
            paddedInputs.push_back(oldDims);
            continue;
        }

        std::vector<uint64_t> paddedDims(maxRank - oldDims.size(), 1);
        paddedDims.insert(paddedDims.end(), oldDims.begin(), oldDims.end());
        paddedInputs.push_back(std::move(paddedDims));
    }

    outputDimensions.clear();
    outputDimensions.assign(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const std::vector<uint64_t>& dims : paddedInputs) {
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
                for (size_t i = 0; i < originalInputDimensions.size(); ++i) {
                    const std::vector<uint64_t>& inDims = originalInputDimensions[i];
                    err << "[";
                    for (size_t j = 0; j < inDims.size(); ++j) {
                        err << inDims[j];
                        if (j + 1 < inDims.size()) {
                            err << ", ";
                        }
                    }
                    err << "]";
                    if (i + 1 < originalInputDimensions.size()) {
                        err << ", ";
                    }
                }
                throw std::runtime_error(err.str());
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    bool requiresBroadcast = false;
    for (const std::vector<uint64_t>& dims : paddedInputs) {
        if (dims != outputDimensions) {
            requiresBroadcast = true;
            break;
        }
    }

    return requiresBroadcast;
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput output_idx out of range.");
    }

    std::unordered_set<uint32_t> used_input_slots;
    collectReferencedLocalInputSlots(stage.expr, stage.outputs[output_idx].local_node_idx, used_input_slots);

    if (used_input_slots.empty()) {
        throw std::runtime_error("Shape inference currently requires each output to depend on at least one tensor input.");
    }

    std::vector<std::vector<uint64_t>> subset_input_dims;
    subset_input_dims.reserve(used_input_slots.size());

    for (uint32_t local_slot = 0; local_slot < stage_input_dims.size(); ++local_slot) {
        if (used_input_slots.contains(local_slot)) {
            subset_input_dims.push_back(stage_input_dims[local_slot]);
        }
    }

    if (subset_input_dims.empty()) {
        throw std::runtime_error("Failed to collect subset input shapes for stage output.");
    }

    std::vector<uint64_t> output_dims;
    resolveLayoutFromDims(subset_input_dims, output_dims);
    return output_dims;
}

std::vector<std::string> FusedEquation::getOutputNames() const {
    std::vector<std::string> output_names;
    output_names.reserve(outputs_template.outputs.size());
    for (const NamedOutput& output : outputs_template.outputs) {
        output_names.push_back(output.name);
    }
    return output_names;
}

std::vector<uint64_t> FusedEquation::getOutputShape(const Tensor& input) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::getOutputShape was called for an equation with multiple final outputs. "
            "Use getOutputShapes(...) instead.");
    }

    if (root_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed a single input, but this equation requires " +
                                 std::to_string(root_inputs.size()) + " inputs. Pass a dict of name -> Tensor to getOutputShapes(...).");
    }

    std::unordered_map<std::string, Tensor> input_map = {
        {root_inputs[0].name, input},
    };

    return getOutputShape(input_map);
}

std::vector<uint64_t> FusedEquation::getOutputShape(const std::unordered_map<std::string, Tensor>& inputs) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::getOutputShape was called for an equation with multiple final outputs. "
            "Use getOutputShapes(...) instead.");
    }

    if (root_inputs.size() != inputs.size()) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed " + to_string(inputs.size()) +
                                 " inputs, but this equation requires " + std::to_string(root_inputs.size()) +
                                 " inputs. Pass a dict of name -> Tensor to getOutputShape(...).");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> output_shapes = getOutputShapes(inputs);
    assert(output_shapes.size() == 1);
    return output_shapes.begin()->second;
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(const Tensor& input) const {
    std::unordered_map<std::string, Tensor> input_map = {
        {root_inputs[0].name, input},
    };

    return getOutputShapes(input_map);
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(
    const std::unordered_map<std::string, Tensor>& inputs) const {
    std::shared_ptr<CompiledOutputs> compiled_outputs = compileForInputs(inputs);

    std::unordered_map<uint32_t, Tensor> root_values = bindRootInputs(inputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::getOutputShapes requires at least one bound root input.");
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> value_dims;
    value_dims.reserve(root_values.size() + compiled_outputs->stages.size());

    for (const auto& [value_id, tensor] : root_values) {
        value_dims.emplace(value_id, tensor.getDimensions());
    }

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<std::vector<uint64_t>> stage_input_dims;
        stage_input_dims.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            auto it = value_dims.find(value_id);
            if (it == value_dims.end()) {
                throw std::runtime_error("Missing input shape for staged output-shape inference.");
            }
            stage_input_dims.push_back(it->second);
        }

        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            std::vector<uint64_t> resolved_output_dims;
            bool requiresBroadcast = resolveLayoutFromDims(stage_input_dims, resolved_output_dims);

            if (!requiresBroadcast) {
                for (const CompiledStageOutput& produced : stage.outputs) {
                    value_dims[produced.value_id] = resolved_output_dims;
                }
            } else {
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    value_dims[stage.outputs[output_idx].value_id] = resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims);
                }
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Reduction) {
            if (!stage.reduction) {
                throw std::runtime_error("Missing compiled reduction stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduction stage expected exactly one input and one output.");
            }

            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.reduction->reduction_axes, stage.reduction->squeeze_axes);
        } else {
            throw std::runtime_error("Unknown execution stage kind in getOutputShapes.");
        }
    }

    std::unordered_map<std::string, std::vector<uint64_t>> final_output_shapes;
    final_output_shapes.reserve(compiled_outputs->final_outputs.size());

    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = value_dims.find(final_output.value_id);
        if (it == value_dims.end()) {
            throw std::runtime_error("Missing final output shape for output: " + final_output.name);
        }
        final_output_shapes.emplace(final_output.name, it->second);
    }

    return final_output_shapes;
}

EquationSignature FusedEquation::buildSignature(uint32_t num_inputs, int device_num, bool use_fast_math) {
    cudaDeviceProp prop{};
    cudaError_t cuda_status = cudaGetDeviceProperties(&prop, device_num);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cuda_status));
    }

    EquationSignature sig{};
    sig.num_inputs = num_inputs;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;
    return sig;
}

PhysicalOutputs FusedEquation::buildShapeSpecializedOutputs(const std::unordered_map<uint32_t, Tensor>& root_values) const {
    if (!backward_config.has_value()) {
        return outputs_template;
    }

    std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims;
    forward_input_dims.reserve(
        backward_config->forward_outputs_template.expr ? backward_config->forward_outputs_template.expr->inputs.size() : 0);

    for (const NamedInput& forward_input : backward_config->forward_outputs_template.expr->inputs) {
        bool found_name = false;
        for (const NamedInput& root_input : root_inputs) {
            if (root_input.name != forward_input.name) {
                continue;
            }
            auto it = root_values.find(root_input.slot);
            if (it == root_values.end()) {
                throw std::runtime_error("Missing bound tensor for backward forward-input shape specialization input: " +
                                         forward_input.name);
            }
            forward_input_dims.emplace(forward_input.name, it->second.getDimensions());
            found_name = true;
            break;
        }
        if (!found_name) {
            throw std::runtime_error("Backward equation root inputs do not contain required forward input: " + forward_input.name);
        }
    }

    return buildBackwardOutputs(
        backward_config->forward_outputs_template, backward_config->wrt_names, backward_config->upstream_input_name, forward_input_dims);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const {
    std::unordered_map<uint32_t, Tensor> root_values = bindRootInputs(namedInputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::compileForInputs requires at least one bound root input.");
    }

    return compileForRootValues(root_values);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForRootValues(const std::unordered_map<uint32_t, Tensor>& root_values) const {
    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::compileForRootValues requires at least one bound root input.");
    }

    const RuntimeDTypeKey dtype_cache_key = makeRuntimeDTypeKey(root_inputs, root_values);

    if (!backward_config.has_value()) {
        std::shared_ptr<CompiledOutputs> cached_compiled_outputs;
        if (compiled_outputs_runtime_cache->tryGet(dtype_cache_key, cached_compiled_outputs)) {
            return cached_compiled_outputs;
        }

        PhysicalOutputs resolved_outputs = outputs_template;
        resolved_outputs.expr = std::make_shared<PhysicalExpression>(*outputs_template.expr);
        resolveOutputsDTypesInPlace(resolved_outputs, dtype_cache_key.root_input_dtypes);

        std::shared_ptr<CompiledOutputs> compiled_outputs = EquationCompiler::compile(resolved_outputs, base_signature, true);
        compiled_outputs_runtime_cache->put(dtype_cache_key, compiled_outputs);
        return compiled_outputs;
    }

    const RuntimeShapeKey shape_cache_key = makeRuntimeShapeKey(root_inputs, root_values);
    std::shared_ptr<CompiledOutputs> cached_compiled_outputs;
    if (compiled_outputs_shape_cache->tryGet(shape_cache_key, cached_compiled_outputs)) {
        return cached_compiled_outputs;
    }

    PhysicalOutputs resolved_outputs = buildShapeSpecializedOutputs(root_values);
    resolved_outputs.expr = std::make_shared<PhysicalExpression>(*resolved_outputs.expr);
    resolveOutputsDTypesInPlace(resolved_outputs, dtype_cache_key.root_input_dtypes);

    std::shared_ptr<CompiledOutputs> compiled_outputs = EquationCompiler::compile(resolved_outputs, base_signature, true);
    compiled_outputs_shape_cache->put(shape_cache_key, compiled_outputs);
    return compiled_outputs;
}

std::shared_ptr<PreparedConvenienceRunPlan> FusedEquation::prepareConvenienceRunPlan(
    const std::unordered_map<uint32_t, Tensor>& root_values) const {
    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::prepareConvenienceRunPlan requires at least one bound root input.");
    }

    const RuntimeShapeKey cache_key = makeRuntimeShapeKey(root_inputs, root_values);

    std::shared_ptr<PreparedConvenienceRunPlan> cached_plan;
    if (convenience_run_plan_cache->tryGet(cache_key, cached_plan)) {
        return cached_plan;
    }

    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    auto plan = std::make_shared<PreparedConvenienceRunPlan>();
    plan->compiled_outputs = compiled_outputs;
    plan->stages.reserve(compiled_outputs->stages.size());

    std::unordered_set<std::string> expected_output_names;

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused-kernel stages, but found stage kind: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }

        std::vector<Tensor> ordered_inputs;
        ordered_inputs.reserve(stage.input_value_ids.size());
        for (uint32_t value_id : stage.input_value_ids) {
            auto it = root_values.find(value_id);
            if (it == root_values.end()) {
                throw std::runtime_error(
                    "FusedEquation::run encountered a stage that depends on a non-root intermediate tensor. "
                    "Use stamp(...).run() for expressions requiring staged intermediates.");
            }
            ordered_inputs.push_back(it->second);
        }

        PreparedConvenienceRunStage prepared_stage;

        std::vector<Tensor> layout_inputs = ordered_inputs;
        std::vector<uint64_t> resolved_output_dims;
        const bool requires_broadcast = resolveLayout(layout_inputs, resolved_output_dims);

        if (!requires_broadcast) {
            if (!stage.flat) {
                throw std::runtime_error("FusedEquation::run found a flat fused stage with no compiled kernel.");
            }

            prepared_stage.compiled_equation = stage.flat;
            prepared_stage.expected_output_dims.assign(stage.outputs.size(), resolved_output_dims);
        } else {
            std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, ordered_inputs);
            if (groups.empty()) {
                throw std::runtime_error("FusedEquation::run expected at least one broadcast group.");
            }

            std::vector<SpecializedBroadcastGroup> specialized_groups;
            specialized_groups.reserve(groups.size());

            prepared_stage.expected_output_dims.resize(stage.outputs.size());

            for (const ResolvedBroadcastGroup& group : groups) {
                specialized_groups.push_back(group.specialized);

                for (uint32_t output_idx : group.specialized.output_indices) {
                    if (output_idx >= prepared_stage.expected_output_dims.size()) {
                        throw std::runtime_error("Broadcast group output index out of range.");
                    }
                    prepared_stage.expected_output_dims[output_idx] = group.specialized.output_dims;
                }
            }

            prepared_stage.compiled_equation =
                EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);
        }

        for (const auto& stage_output : stage.outputs) {
            if (expected_output_names.insert(stage_output.name).second) {
                plan->expected_output_names_in_order.push_back(stage_output.name);
            }
        }

        plan->stages.push_back(std::move(prepared_stage));
    }

    convenience_run_plan_cache->put(cache_key, plan);
    return plan;
}

FusedEquation FusedEquation::compile(const PhysicalOutputs& outputs, int device_num, bool use_fast_math) {
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

    const EquationSignature base_signature = buildSignature(outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(outputs, device_num, use_fast_math, base_signature);
}

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, int device_num, bool use_fast_math) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("FusedEquation::compile PhysicalExpression output_node is out of range.");
    }

    PhysicalOutputs outputs;
    outputs.expr = std::make_shared<PhysicalExpression>(expr);
    outputs.outputs.push_back(NamedOutput{
        .name = "output",
        .node_idx = expr.output_node,
    });

    return compile(outputs, device_num, use_fast_math);
}

FusedEquation FusedEquation::compileBackward(const std::vector<std::string>& wrt_names,
                                             const std::optional<std::string>& upstream_input_name) const {
    PhysicalOutputs backward_outputs = buildBackwardOutputs(outputs_template, wrt_names, upstream_input_name);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = wrt_names,
                             .upstream_input_name = upstream_input_name,
                         });
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
    if (outputs.size() != compiledEquation->numOutputs()) {
        throw std::runtime_error("Wrong number of outputs passed to FusedEquation::stampEquation.");
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

    if (firstInput.getDescriptor().getDataType() != compiledEquation->input_dtypes[0]) {
        throw std::runtime_error("Input tensor data type does not match compiled equation data type.");
    }

    if (firstInput.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Input tensor GPU does not match compiled equation device.");
    }

    for (uint64_t i = 1; i < inputs.size(); ++i) {
        if (!inputs[i].isInitialized()) {
            throw std::runtime_error("Input tensor is not initialized.");
        }
        if (inputs[i].getDescriptor().getDataType() != compiledEquation->input_dtypes[i]) {
            throw std::runtime_error("Input tensor data type mismatch.");
        }
        if (inputs[i].getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU mismatch.");
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& output = outputs[i];
        if (!output.isInitialized()) {
            throw std::runtime_error("Output tensor is not initialized.");
        }
        if (output.getDescriptor().getDataType() != compiledEquation->output_dtypes[i]) {
            throw std::runtime_error("Output tensor data type mismatch.");
        }
        if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Output tensor GPU mismatch.");
        }
    }

    return make_shared<StampedEquation>(compiledEquation, inputs, outputs, stream);
}

std::shared_ptr<StampedReduction> FusedEquation::stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                Tensor& input,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledReduction)
        throw std::runtime_error("Tried to stamp reduction on a non-reduction FusedEquation.");

    if (input.getDataType() != compiledReduction->input_dtype) {
        throw std::runtime_error("Input dtype does not match compiled reduction input dtype.");
    }

    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiledReduction, input, stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(input.getPlacement(), workspaceDescriptor);
    }

    vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);

    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    TensorDescriptor outputDescriptor(compiledReduction->output_dtype, output_dimensions);
    Tensor output = Tensor(input.getPlacement(), outputDescriptor);

    return make_shared<StampedReduction>(std::move(built), input, output, stream, workspace);
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForInputs(inputs);

    std::unordered_map<uint32_t, Tensor> values = bindRootInputs(inputs);

    std::vector<StampedExecutionStage> stampedStages;
    stampedStages.reserve(compiled_outputs->stages.size());

    // Maps each produced value_id to the stamped stage index that materializes it.
    std::unordered_map<uint32_t, uint32_t> producer_stage_by_value_id;
    producer_stage_by_value_id.reserve(compiled_outputs->stages.size() * 2);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<Tensor> stageInputs;
        stageInputs.reserve(stage.input_value_ids.size());

        // Collect dependency stage indices from any inputs produced by prior stages.
        std::vector<uint32_t> dependency_stage_indices;
        dependency_stage_indices.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            auto it = values.find(value_id);
            if (it == values.end()) {
                throw std::runtime_error("Missing input value for staged execution plan.");
            }
            stageInputs.push_back(it->second);

            auto producer_it = producer_stage_by_value_id.find(value_id);
            if (producer_it != producer_stage_by_value_id.end()) {
                dependency_stage_indices.push_back(producer_it->second);
            }
        }

        std::sort(dependency_stage_indices.begin(), dependency_stage_indices.end());
        dependency_stage_indices.erase(std::unique(dependency_stage_indices.begin(), dependency_stage_indices.end()),
                                       dependency_stage_indices.end());

        const uint32_t this_stage_idx = static_cast<uint32_t>(stampedStages.size());

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
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& produced = stage.outputs[output_idx];

                    auto requested_it = requestedOutputShapes.find(produced.name);
                    const std::vector<uint64_t>* requested_shape =
                        (requested_it != requestedOutputShapes.end()) ? &requested_it->second : nullptr;

                    if (requested_shape && !requested_shape->empty()) {
                        verifyRequestedOutputLayout(*requested_shape, resolvedOutputDims);
                    }

                    TensorDescriptor outputDescriptor(
                        stage.flat->output_dtypes[output_idx],
                        (requested_shape && !requested_shape->empty()) ? *requested_shape : resolvedOutputDims);
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
                        stage.flat->output_dtypes[output_idx],
                        (requested_shape && !requested_shape->empty()) ? *requested_shape : resolved_output_dims);
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

                std::shared_ptr<CompiledEquation> specialized_broadcast =
                    EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);

                stamped = stampEquation(specialized_broadcast, layoutInputs, stageOutputs, stream);
            } else {
                stamped = stampEquation(stage.flat, layoutInputs, stageOutputs, stream);
            }

            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                const uint32_t produced_value_id = stage.outputs[i].value_id;
                values[produced_value_id] = stageOutputs[i];
                producer_stage_by_value_id[produced_value_id] = this_stage_idx;
            }

            stampedStages.emplace_back(stamped, std::move(dependency_stage_indices));
        } else {
            if (!stage.reduction) {
                throw std::runtime_error("Missing compiled reduction stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduction stage expected exactly one input and one output.");
            }

            Tensor& reductionInput = stageInputs[0];
            auto requested_it = requestedOutputShapes.find(stage.outputs[0].name);
            std::vector<uint64_t> requested_shape;
            if (requested_it != requestedOutputShapes.end())
                requested_shape = requested_it->second;
            std::shared_ptr<StampedReduction> stamped = stampReduction(stage.reduction, reductionInput, stream, requested_shape);

            const uint32_t produced_value_id = stage.outputs[0].value_id;
            values[produced_value_id] = stamped->getOutputTensor();
            producer_stage_by_value_id[produced_value_id] = this_stage_idx;

            stampedStages.emplace_back(stamped, std::move(dependency_stage_indices));
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

    return StampedExecutionPlan(std::move(stampedStages), std::move(finalOutputsByName), stream);
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

void FusedEquation::run(const Tensor& input, Tensor& output, Stream& stream) const {
    if (root_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(root_inputs.size()) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{root_inputs[0].name, input}};
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};

    run(input_map, output_map, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run was only passed a single output, but this equation has multiple named outputs. "
            "Pass a dict of name -> PhysicalTensor of outputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};
    run(inputs, output_map, stream);
}

void FusedEquation::run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const {
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
    const std::unordered_map<uint32_t, Tensor> root_values = bindRootInputs(inputs);
    const std::shared_ptr<PreparedConvenienceRunPlan> prepared_plan = prepareConvenienceRunPlan(root_values);
    const std::shared_ptr<CompiledOutputs>& compiled_outputs = prepared_plan->compiled_outputs;

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages.");
    }

    // Verify caller provided every expected final output.
    std::unordered_set<std::string> expected_output_names(prepared_plan->expected_output_names_in_order.begin(),
                                                          prepared_plan->expected_output_names_in_order.end());

    for (const std::string& name : prepared_plan->expected_output_names_in_order) {
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
            for (size_t i = 0; i < prepared_plan->expected_output_names_in_order.size(); ++i) {
                if (i > 0) {
                    expected_names_str += ", ";
                }
                expected_names_str += "'" + prepared_plan->expected_output_names_in_order[i] + "'";
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
                                const PreparedConvenienceRunStage& prepared_stage,
                                const std::vector<Tensor>& orderedInputs,
                                const std::vector<Tensor>& orderedOutputs,
                                Stream& launch_stream) {
        if (orderedOutputs.size() != prepared_stage.expected_output_dims.size()) {
            throw std::runtime_error("Prepared convenience run stage output count mismatch.");
        }

        for (size_t i = 0; i < orderedOutputs.size(); ++i) {
            verifyRequestedOutputLayout(orderedOutputs[i].getDimensions(), prepared_stage.expected_output_dims[i]);
        }

        EquationRunner::run(prepared_stage.compiled_equation, orderedInputs, orderedOutputs, launch_stream);
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
        const PreparedConvenienceRunStage& prepared_stage = prepared_plan->stages[stage_num];

        std::vector<Tensor> orderedInputs;
        orderedInputs.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
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
            runStageOnStream(stage, prepared_stage, orderedInputs, orderedOutputs, helper_stream);
            rememberHelperStream(helper_stream);
        } else {
            runStageOnStream(stage, prepared_stage, orderedInputs, orderedOutputs, stream);
        }
    }

    // Join all helper streams back into the user-provided stream.
    for (Stream& helper_stream : helper_streams_used) {
        stream.waitEvent(helper_stream.putEvent());
    }
}
}  // namespace ThorImplementation
