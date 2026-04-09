#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/AutoDiff.h"
#include "Utilities/TensorMathFusion/CudaSourceEmitter.h"
#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/ExpressionDTypeResolution.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

using namespace std;
using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

static bool runtimeInputIsTensor(const RuntimeInputValue& value) { return std::holds_alternative<Tensor>(value); }
static bool runtimeInputIsTensorScalarBinding(const RuntimeInputValue& value) { return std::holds_alternative<TensorScalarBinding>(value); }

static const TensorScalarBinding& runtimeInputTensorScalarBinding(const RuntimeInputValue& value) {
    if (!std::holds_alternative<TensorScalarBinding>(value)) {
        throw std::runtime_error("Expected tensor scalar runtime input.");
    }
    return std::get<TensorScalarBinding>(value);
}

static size_t dataTypeSizeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
            return 2;
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
            return 1;
        case DataType::FP8_E5M2:
            return 1;
        case DataType::UINT8:
            return 1;
        case DataType::UINT16:
            return 2;
        case DataType::UINT32:
            return 4;
        case DataType::INT32:
            return 4;
        default:
            throw std::runtime_error("Unsupported dtype in dataTypeSizeBytes.");
    }
}

static const Tensor& runtimeInputTensor(const RuntimeInputValue& value) {
    if (!std::holds_alternative<Tensor>(value)) {
        throw std::runtime_error("Expected tensor runtime input.");
    }
    return std::get<Tensor>(value);
}

static std::vector<uint64_t> runtimeInputDims(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getDimensions();
    }
    return {};
}

static DataType runtimeInputDType(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getDataType();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return std::get<TensorScalarBinding>(value).sourceDType;
    }
    return DataType::FP32;
}

static const Optional<TensorPlacement> runtimeInputPlacementOrNull(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getPlacement();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return std::get<TensorScalarBinding>(value).buffer.getPlacement();
    }
    return Optional<TensorPlacement>::empty();
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::makeSingleOutputRequestedShapeMap(
    const std::vector<uint64_t>& requestedOutputShape) const {
    std::unordered_map<std::string, std::vector<uint64_t>> requested;

    if (requestedOutputShape.empty()) {
        return requested;
    }

    const auto outputNames = getOutputNames();
    if (outputNames.size() != 1) {
        throw std::runtime_error(
            "Single-output requested-shape stamp overload called on an equation that does not have exactly one output.");
    }

    requested.emplace(outputNames[0], requestedOutputShape);
    return requested;
}

static Tensor adaptReductionInputDTypeIfNeeded(const Tensor& input,
                                               DataType expected_input_dtype,
                                               ExprOp reduction_op,
                                               const Stream& stream) {
    if (input.getDataType() == expected_input_dtype) {
        return input;
    }

    if (toSupportedInputDType(reduction_op, input.getDataType()) != expected_input_dtype) {
        throw std::runtime_error("Input dtype does not match compiled reduction input dtype.");
    }

    TensorDescriptor castDescriptor(expected_input_dtype, input.getDimensions());
    Tensor castInput(input.getPlacement(), castDescriptor);
    castInput.copyFromAsync(input, stream);
    return castInput;
}

static RuntimeDTypeKey makeRuntimeDTypeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    RuntimeDTypeKey key;
    key.root_input_dtypes.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound runtime input for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dtypes[input.slot] = runtimeInputDType(it->second);
    }

    return key;
}

static RuntimeShapeKey makeRuntimeShapeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    RuntimeShapeKey key;
    key.dtype_key = makeRuntimeDTypeKey(root_inputs, root_values);
    key.root_input_dims.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound runtime input for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dims[input.slot] = runtimeInputDims(it->second);
    }

    return key;
}

static bool accumulatesIntoGradOutputs(const std::optional<BackwardEquationConfig>& backward_config) {
    return backward_config.has_value() && backward_config->accumulate_grad_outputs;
}

static std::unordered_set<std::string> backwardAccumulationOutputNames(const std::optional<BackwardEquationConfig>& backward_config) {
    std::unordered_set<std::string> names;
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return names;
    }

    names.reserve(backward_config->wrt_names.size());
    for (const std::string& wrt_name : backward_config->wrt_names) {
        names.insert(wrt_name + "_grad");
    }
    return names;
}

static size_t externalRootInputCount(const std::vector<NamedInput>& root_inputs,
                                     const std::optional<BackwardEquationConfig>& backward_config) {
    const size_t accumulation_count = backwardAccumulationOutputNames(backward_config).size();
    if (accumulation_count > root_inputs.size()) {
        throw std::runtime_error("Invalid backward accumulation input accounting.");
    }
    return root_inputs.size() - accumulation_count;
}

static std::vector<std::string> inferBackwardWrtNamesFromOutputs(const PhysicalOutputs& backward_outputs) {
    std::vector<std::string> wrt_names;
    wrt_names.reserve(backward_outputs.outputs.size());
    for (const NamedOutput& output : backward_outputs.outputs) {
        constexpr const char* suffix = "_grad";
        constexpr size_t suffix_len = 5;
        if (output.name.size() >= suffix_len && output.name.compare(output.name.size() - suffix_len, suffix_len, suffix) == 0) {
            wrt_names.push_back(output.name.substr(0, output.name.size() - suffix_len));
        } else {
            wrt_names.push_back(output.name);
        }
    }
    return wrt_names;
}

static std::string dimsToString(const std::vector<uint64_t>& dims);
static void verifyRequestedOutputLayout(const std::vector<uint64_t>& outputDimensions, const std::vector<uint64_t>& expectedDimensions);

static Optional<DataType> preferredBackwardGradBufferDType(const BackwardEquationConfig& backward_config, const std::string& wrt_name) {
    if (!backward_config.forward_outputs_template.expr) {
        throw std::runtime_error("Backward grad-buffer dtype lookup requires non-null forward expr.");
    }

    uint32_t slot = UINT32_MAX;
    for (const NamedInput& input : backward_config.forward_outputs_template.expr->inputs) {
        if (input.name == wrt_name) {
            slot = input.slot;
            break;
        }
    }
    if (slot == UINT32_MAX) {
        throw std::runtime_error("Unknown backward grad-buffer input name: " + wrt_name);
    }

    for (const ExprNode& node : backward_config.forward_outputs_template.expr->nodes) {
        if ((node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) &&
            node.input_slot == slot) {
            if (node.backward_output_dtype.isPresent()) {
                return node.backward_output_dtype;
            }
            if (node.output_dtype.isPresent()) {
                return node.output_dtype;
            }
            return Optional<DataType>::empty();
        }
    }

    throw std::runtime_error("No INPUT node found for backward grad-buffer input: " + wrt_name);
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

static void validateBackwardAccumulationOutputs(const std::optional<BackwardEquationConfig>& backward_config,
                                                const std::unordered_map<std::string, Tensor>& named_inputs,
                                                const std::unordered_map<std::string, Tensor>& named_outputs,
                                                const std::unordered_map<std::string, std::vector<uint64_t>>& expected_output_shapes) {
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return;
    }

    std::unordered_set<std::string> expected_output_names;
    expected_output_names.reserve(backward_config->wrt_names.size());
    for (const std::string& wrt_name : backward_config->wrt_names) {
        expected_output_names.insert(wrt_name + "_grad");
    }

    if (named_outputs.size() != expected_output_names.size()) {
        throw std::runtime_error("Backward accumulation stamp requires exactly " + std::to_string(expected_output_names.size()) +
                                 " gradient output tensors, but received " + std::to_string(named_outputs.size()) + ".");
    }

    for (const auto& [name, _] : named_outputs) {
        if (!expected_output_names.contains(name)) {
            throw std::runtime_error("Unexpected output tensor supplied for backward accumulation stamp: " + name);
        }
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string grad_output_name = wrt_name + "_grad";

        auto output_it = named_outputs.find(grad_output_name);
        if (output_it == named_outputs.end()) {
            throw std::runtime_error("Missing required gradient accumulator for backward stamp: " + grad_output_name);
        }

        auto input_it = named_inputs.find(wrt_name);
        if (input_it == named_inputs.end()) {
            throw std::runtime_error("Missing forward input required to validate backward accumulator: " + wrt_name);
        }

        const Tensor& accumulator = output_it->second;
        const Tensor& wrt_input = input_it->second;

        std::vector<uint64_t> expected_dims;

        auto expected_it = expected_output_shapes.find(grad_output_name);
        if (expected_it != expected_output_shapes.end() && !expected_it->second.empty()) {
            expected_dims = expected_it->second;
        } else {
            expected_dims = wrt_input.getDimensions();
        }

        if (!outputDimensionsMatchIgnoringSingletons(accumulator.getDimensions(), expected_dims)) {
            throw std::runtime_error("Gradient accumulator tensor dimensions are incompatible for output '" + grad_output_name +
                                     "'. Expected compatible with " + dimsToString(expected_dims) + ", got " +
                                     dimsToString(accumulator.getDimensions()) + ".");
        }

        const Optional<DataType> preferred_dtype = preferredBackwardGradBufferDType(backward_config.value(), wrt_name);
        const DataType expected_dtype = preferred_dtype.isPresent() ? preferred_dtype.get() : wrt_input.getDataType();
        if (accumulator.getDataType() != expected_dtype) {
            throw std::runtime_error("Gradient accumulator tensor dtype mismatch for output '" + grad_output_name + "'.");
        }

        if (accumulator.getPlacement().getMemDevice() != wrt_input.getPlacement().getMemDevice() ||
            accumulator.getPlacement().getDeviceNum() != wrt_input.getPlacement().getDeviceNum()) {
            throw std::runtime_error("Gradient accumulator tensor placement mismatch for output '" + grad_output_name + "'.");
        }
    }
}

static std::unordered_map<std::string, std::vector<uint64_t>> mergeRequestedOutputShapesWithProvidedOutputs(
    const std::unordered_map<std::string, Tensor>& provided_outputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) {
    std::unordered_map<std::string, std::vector<uint64_t>> effective = requested_output_shapes;

    for (const auto& [name, output] : provided_outputs) {
        auto requested_it = effective.find(name);
        if (requested_it != effective.end() && !requested_it->second.empty()) {
            verifyRequestedOutputLayout(output.getDimensions(), requested_it->second);
        }
        effective[name] = output.getDimensions();
    }

    return effective;
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

static uint64_t maxNumel(const std::vector<std::vector<uint64_t>>& dims_by_output) {
    uint64_t max_numel = 0;
    for (const std::vector<uint64_t>& dims : dims_by_output) {
        max_numel = std::max<uint64_t>(max_numel, product(dims));
    }
    return max_numel;
}

static DataType compiledStageOutputDType(const CompiledExecutionStage& stage, size_t output_idx) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("compiledStageOutputDType output index out of range.");
    }

    switch (stage.kind) {
        case CompiledExecutionStage::Kind::FusedKernel:
            if (!stage.flat) {
                throw std::runtime_error("compiledStageOutputDType missing fused stage kernel.");
            }
            return stage.flat->output_dtypes.at(output_idx);
        case CompiledExecutionStage::Kind::Reduction:
            if (!stage.reduction) {
                throw std::runtime_error("compiledStageOutputDType missing reduction stage.");
            }
            return stage.reduction->output_dtype;
        case CompiledExecutionStage::Kind::ArgMinMax:
            if (!stage.arg_minmax) {
                throw std::runtime_error("compiledStageOutputDType missing arg-min/max stage.");
            }
            return stage.arg_minmax->output_dtype;
        case CompiledExecutionStage::Kind::ReduceMinMaxBackward:
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("compiledStageOutputDType missing reduce-min/max-backward stage.");
            }
            return stage.reduce_minmax_backward->output_dtype;
    }

    throw std::runtime_error("compiledStageOutputDType encountered unknown stage kind.");
}

static PhysicalExecutionStage toPhysicalFusedStage(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("toPhysicalFusedStage called on non-fused stage.");
    }

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = stage.expr,
        .input_value_ids = stage.input_value_ids,
        .outputs = stage.outputs,
    };
}

static std::shared_ptr<CompiledEquation> selectFlatCompiledEquation(const CompiledExecutionStage& stage,
                                                                    const EquationSignature& sig,
                                                                    uint64_t max_numel) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("selectFlatCompiledEquation called on non-fused stage.");
    }
    if (max_numel <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        if (!stage.flat) {
            throw std::runtime_error("Missing default flat fused kernel.");
        }
        return stage.flat;
    }

    return EquationCompiler::compileFusedStage(toPhysicalFusedStage(stage), sig, /*use_uint32_index_math=*/false);
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

    if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
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

static bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions);

// static std::vector<uint64_t> applySqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes) {
//     if (squeeze_axes.empty()) {
//         return input_dims;
//     }
//
//     std::vector<uint64_t> normalized = squeeze_axes;
//     std::sort(normalized.begin(), normalized.end());
//     normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
//
//     if (normalized.size() == 1 && normalized[0] == UINT64_MAX) {
//         std::vector<uint64_t> out_dims;
//         out_dims.reserve(input_dims.size());
//         for (uint64_t dim : input_dims) {
//             if (dim != 1) {
//                 out_dims.push_back(dim);
//             }
//         }
//         return out_dims;
//     }
//
//     std::vector<uint64_t> out_dims;
//     out_dims.reserve(input_dims.size());
//     size_t next_axis_i = 0;
//     uint64_t next_axis = normalized.empty() ? UINT64_MAX : normalized[0];
//     for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
//         if (next_axis_i < normalized.size() && axis == next_axis) {
//             if (input_dims[axis] != 1) {
//                 throw std::runtime_error("squeeze axes must refer to singleton dimensions.");
//             }
//             ++next_axis_i;
//             next_axis = next_axis_i < normalized.size() ? normalized[next_axis_i] : UINT64_MAX;
//             continue;
//         }
//         out_dims.push_back(input_dims[axis]);
//     }
//
//     if (next_axis_i != normalized.size()) {
//         throw std::runtime_error("squeeze axes are invalid for the input rank.");
//     }
//
//     return out_dims;
// }

static std::vector<uint64_t> applyNormalizedUnsqueezeDims(const std::vector<uint64_t>& input_dims,
                                                          const std::vector<uint64_t>& unsqueeze_axes) {
    // Scalar literals are rank-0 broadcastable leaves in the IR.
    // Unsqueezing a scalar produces an all-ones shape of the requested rank.
    if (input_dims.empty()) {
        std::vector<uint64_t> normalized = unsqueeze_axes;
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

        for (uint64_t i = 0; i < normalized.size(); ++i) {
            if (normalized[i] != i) {
                throw std::runtime_error("unsqueeze axes are invalid for scalar input.");
            }
        }

        return std::vector<uint64_t>(normalized.size(), 1ULL);
    }

    const std::vector<uint64_t> actual_axes = normalizeUnsqueezeAxesForInputDims(input_dims, unsqueeze_axes);

    std::vector<uint64_t> output_dims = input_dims;
    for (uint64_t axis : actual_axes) {
        output_dims.insert(output_dims.begin() + static_cast<std::ptrdiff_t>(axis), 1ULL);
    }
    return output_dims;
}

static std::vector<uint64_t> applyNormalizedSqueezeDims(const std::vector<uint64_t>& input_dims,
                                                        const std::vector<uint64_t>& squeeze_axes) {
    // Scalar literals are already rank-0. Any squeeze is shape-preserving for stage inference.
    if (input_dims.empty()) {
        return {};
    }

    const std::vector<uint64_t> actual_axes = normalizeSqueezeAxesForInputDims(input_dims, squeeze_axes);

    if (actual_axes.empty()) {
        return input_dims;
    }

    std::vector<uint64_t> output_dims;
    output_dims.reserve(input_dims.size() - actual_axes.size());

    size_t next_remove = 0;
    for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
        if (next_remove < actual_axes.size() && actual_axes[next_remove] == axis) {
            ++next_remove;
            continue;
        }
        output_dims.push_back(input_dims[axis]);
    }

    return output_dims;
}

static std::string dimsToString(const std::vector<uint64_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

// static std::vector<uint64_t> applyUnsqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& unsqueeze_axes) {
//     std::vector<uint64_t> out_dims;
//     out_dims.reserve(input_dims.size() + unsqueeze_axes.size());
//
//     const uint64_t output_rank = static_cast<uint64_t>(input_dims.size() + unsqueeze_axes.size());
//     size_t input_i = 0;
//     size_t axis_i = 0;
//
//     for (uint64_t out_axis = 0; out_axis < output_rank; ++out_axis) {
//         if (axis_i < unsqueeze_axes.size() && unsqueeze_axes[axis_i] == out_axis) {
//             out_dims.push_back(1);
//             ++axis_i;
//         } else {
//             if (input_i >= input_dims.size()) {
//                 throw std::runtime_error("unsqueeze axes are invalid for the input rank.");
//             }
//             out_dims.push_back(input_dims[input_i++]);
//         }
//     }
//
//     if (input_i != input_dims.size() || axis_i != unsqueeze_axes.size()) {
//         throw std::runtime_error("unsqueeze axes are invalid for the input rank.");
//     }
//
//     return out_dims;
// }

static std::vector<std::vector<uint64_t>> inferFusedStageNodeDims(const PhysicalExpression& expr,
                                                                  const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    std::vector<std::vector<uint64_t>> node_dims(expr.nodes.size());

    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        const ExprNode& node = expr.nodes[i];

        // std::cerr << "[FUSION] visiting node"
        //           << " local_node=" << i << " op=" << static_cast<int>(node.op) << " lhs=" << node.lhs << " rhs=" << node.rhs
        //           << " input_slot=" << node.input_slot << std::endl;

        switch (node.op) {
            case ExprOp::INPUT: {
                if (node.input_slot >= stage_input_dims.size()) {
                    throw std::runtime_error("Stage input slot out of range during fused-stage shape inference.");
                }
                node_dims[i] = stage_input_dims[node.input_slot];
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                node_dims[i] = {};
                break;
            case ExprOp::FILL:
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                if (!node_dims[node.rhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
            case ExprOp::NEG:
            case ExprOp::ABS:
            case ExprOp::EXP:
            case ExprOp::EXP2:
            case ExprOp::EXP10:
            case ExprOp::LN:
            case ExprOp::LOG2:
            case ExprOp::LOG10:
            case ExprOp::SQRT:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::UNSQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];

                // std::cerr << "[FUSION] infer node UNSQUEEZE begin"
                //           << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                //           << " unsqueeze_axes=" << dimsToString(node.unsqueeze_axes) << std::endl;

                try {
                    node_dims[i] = applyNormalizedUnsqueezeDims(lhs_dims, node.unsqueeze_axes);
                } catch (const std::exception& e) {
                    std::ostringstream oss;
                    oss << "inferFusedStageNodeDims UNSQUEEZE failed"
                        << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                        << " unsqueeze_axes=" << dimsToString(node.unsqueeze_axes) << " error=" << e.what();
                    throw std::runtime_error(oss.str());
                }

                // std::cerr << "[FUSION] infer node UNSQUEEZE end"
                //           << " local_node=" << i << " out_dims=" << dimsToString(node_dims[i]) << std::endl;
                break;
            }

            case ExprOp::SQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];

                // std::cerr << "[FUSION] infer node SQUEEZE begin"
                //           << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                //           << " squeeze_axes=" << dimsToString(node.squeeze_axes) << std::endl;

                try {
                    node_dims[i] = applyNormalizedSqueezeDims(lhs_dims, node.squeeze_axes);
                } catch (const std::exception& e) {
                    std::ostringstream oss;
                    oss << "inferFusedStageNodeDims SQUEEZE failed"
                        << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                        << " squeeze_axes=" << dimsToString(node.squeeze_axes) << " error=" << e.what();
                    throw std::runtime_error(oss.str());
                }

                // std::cerr << "[FUSION] infer node SQUEEZE end"
                //           << " local_node=" << i << " out_dims=" << dimsToString(node_dims[i]) << std::endl;
                break;
            }
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
                node_dims[i] = StampedEquation::computeReductionOutputDims(node_dims[node.lhs], node.reduction_axes, node.squeeze_axes);
                break;
            default:
                throw std::runtime_error("inferFusedStageNodeDims encountered unknown ExprOp.");
        }
    }

    return node_dims;
}

static std::vector<uint64_t> resolveReductionAxesForInputRank(const std::vector<uint64_t>& reduction_axes, size_t input_rank) {
    if (reduction_axes.empty()) {
        std::vector<uint64_t> axes(input_rank);
        for (size_t i = 0; i < input_rank; ++i) {
            axes[i] = static_cast<uint64_t>(i);
        }
        return axes;
    }
    return reduction_axes;
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput output_idx out of range.");
    }

    switch (stage.kind) {
        case CompiledExecutionStage::Kind::Reduction: {
            if (!stage.reduction) {
                throw std::runtime_error("resolveOutputDimsForStageOutput reduction stage missing payload.");
            }
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput reduction stage expected at least one input shape.");
            }

            const auto reduction_axes = resolveReductionAxesForInputRank(stage.reduction->reduction_axes, stage_input_dims[0].size());

            return StampedEquation::computeReductionOutputDims(stage_input_dims[0], reduction_axes, stage.reduction->squeeze_axes);
        }

        case CompiledExecutionStage::Kind::ArgMinMax: {
            if (!stage.arg_minmax) {
                throw std::runtime_error("resolveOutputDimsForStageOutput argmin/argmax stage missing payload.");
            }
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput argmin/argmax stage expected at least one input shape.");
            }

            const auto reduction_axes = resolveReductionAxesForInputRank(stage.arg_minmax->reduction_axes, stage_input_dims[0].size());

            return StampedEquation::computeReductionOutputDims(stage_input_dims[0], reduction_axes, stage.arg_minmax->squeeze_axes);
        }

        case CompiledExecutionStage::Kind::ReduceMinMaxBackward: {
            if (stage_input_dims.empty()) {
                throw std::runtime_error(
                    "resolveOutputDimsForStageOutput reduce-min/max-backward stage expected at least one input shape.");
            }
            return stage_input_dims[0];
        }

        case CompiledExecutionStage::Kind::FusedKernel:
            break;
    }

    const auto node_dims = inferFusedStageNodeDims(stage.expr, stage_input_dims);
    const uint32_t local_node_idx = stage.outputs[output_idx].local_node_idx;
    if (local_node_idx >= node_dims.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput local_node_idx out of range.");
    }
    return node_dims[local_node_idx];
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<RuntimeInputValue>& stage_inputs) {
    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        stage_input_dims.push_back(runtimeInputDims(input));
    }
    return resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims);
}

struct ResolvedBroadcastGroup {
    SpecializedBroadcastGroup specialized;
};

static bool stageHasShapeOnlyOps(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return false;
    }
    for (const ExprNode& node : stage.expr.nodes) {
        if (node.op == ExprOp::UNSQUEEZE || node.op == ExprOp::SQUEEZE) {
            return true;
        }
    }
    return false;
}
static TensorPlacement pickStageOutputPlacement(const std::vector<RuntimeInputValue>& stage_inputs,
                                                const std::unordered_map<uint32_t, RuntimeInputValue>& available_values) {
    for (const RuntimeInputValue& input : stage_inputs) {
        Optional<TensorPlacement> placement = runtimeInputPlacementOrNull(input);
        if (placement.isPresent()) {
            return placement;
        }
    }
    if (!available_values.empty()) {
        for (const auto& [value_id, value] : available_values) {
            (void)value_id;
            Optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
            if (placement.isPresent()) {
                return placement;
            }
        }
    }
    throw std::runtime_error("Unable to infer output placement for fused stage with no available tensors.");
}

static std::unordered_map<std::string, std::vector<uint64_t>> defaultBackwardRequestedOutputShapes(
    const std::optional<BackwardEquationConfig>& backward_config,
    const std::vector<NamedInput>& root_inputs,
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) {
    std::unordered_map<std::string, std::vector<uint64_t>> effective = requested_output_shapes;

    if (!backward_config.has_value()) {
        return effective;
    }

    std::unordered_map<std::string, uint32_t> root_slot_by_name;
    root_slot_by_name.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        root_slot_by_name.emplace(input.name, input.slot);
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string output_name = wrt_name + "_grad";
        if (effective.contains(output_name)) {
            continue;
        }

        auto slot_it = root_slot_by_name.find(wrt_name);
        if (slot_it == root_slot_by_name.end()) {
            continue;
        }

        auto value_it = root_values.find(slot_it->second);
        if (value_it == root_values.end()) {
            continue;
        }

        effective.emplace(output_name, runtimeInputDims(value_it->second));
    }

    return effective;
}

static bool fusedStageRequiresBroadcastLaunch(const CompiledExecutionStage& stage,
                                              const std::vector<RuntimeInputValue>& stage_inputs,
                                              const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
                                              bool trust_requested_output_shapes,
                                              std::vector<uint64_t>& resolved_output_dims) {
    resolved_output_dims.clear();

    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return false;
    }

    if (stageHasShapeOnlyOps(stage)) {
        if (!stage.outputs.empty()) {
            resolved_output_dims = resolveOutputDimsForStageOutput(stage, 0, stage_inputs);
        }
        return true;
    }

    if (stage_inputs.empty()) {
        if (!stage.outputs.empty()) {
            resolved_output_dims = resolveOutputDimsForStageOutput(stage, 0, stage_inputs);
            auto requested_it = requested_output_shapes.find(stage.outputs[0].name);
            if (requested_it != requested_output_shapes.end() && !requested_it->second.empty()) {
                resolved_output_dims = requested_it->second;
            }
        }
        return false;
    }

    std::vector<Tensor> layout_inputs;
    layout_inputs.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        if (runtimeInputIsTensor(input)) {
            layout_inputs.push_back(runtimeInputTensor(input));
        }
    }
    const bool requires_broadcast = layout_inputs.empty() ? false : FusedEquation::resolveLayout(layout_inputs, resolved_output_dims);

    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
        std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stage_inputs);
        auto requested_it = requested_output_shapes.find(stage.outputs[output_idx].name);
        if (requested_it != requested_output_shapes.end() && !requested_it->second.empty()) {
            if (!trust_requested_output_shapes) {
                verifyRequestedOutputLayout(requested_it->second, output_dims);
            }
            output_dims = requested_it->second;
        }
        if (output_dims != resolved_output_dims) {
            return true;
        }
    }

    return requires_broadcast;
}

static void mergeEffectiveInputDimsMaps(std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& dst,
                                        const std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& src) {
    for (const auto& [slot, dims_set] : src) {
        auto& out = dst[slot];
        out.insert(dims_set.begin(), dims_set.end());
    }
}

static std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>> collectEffectiveInputDimsForNode(
    const PhysicalExpression& expr, const std::vector<std::vector<uint64_t>>& node_dims, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        throw std::runtime_error("collectEffectiveInputDimsForNode node index out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];
    switch (node.op) {
        case ExprOp::INPUT: {
            return {{node.input_slot, {node_dims[node_idx]}}};
        }
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
        case ExprOp::SCALAR_FP:
            return {};
        case ExprOp::FILL:
            return {};
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE: {
            auto result = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            for (auto& [slot, dims_set] : result) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            return result;
        }
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2:
            return collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT: {
            auto lhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            auto rhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.rhs);
            mergeEffectiveInputDimsMaps(lhs_map, rhs_map);
            return lhs_map;
        }
        default:
            throw std::runtime_error("collectEffectiveInputDimsForNode encountered unknown ExprOp.");
    }
}

static bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        throw std::runtime_error("resolveLayoutFromDims requires at least one input shape.");
    }

    uint64_t maxRank = 0;
    for (const auto& dims : inputs) {
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    outputDimensions.assign(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const auto& dims : inputs) {
            const uint64_t pad = maxRank - dims.size();
            const uint64_t dim = (axis < pad) ? 1ULL : dims[axis - pad];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
            } else if (resolvedDim != dim) {
                throw std::runtime_error("resolveLayoutFromDims found non-broadcast-compatible dimensions.");
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    bool requiresBroadcast = false;
    for (const auto& dims : inputs) {
        std::vector<uint64_t> padded(maxRank - dims.size(), 1ULL);
        padded.insert(padded.end(), dims.begin(), dims.end());
        if (padded != outputDimensions) {
            requiresBroadcast = true;
            break;
        }
    }

    return requiresBroadcast;
}

static std::vector<uint64_t> computeInputPackedStridesForBroadcast(const std::vector<uint64_t>& input_dims,
                                                                   const std::vector<uint64_t>& output_dims) {
    if (input_dims.size() > output_dims.size()) {
        throw std::runtime_error("Input rank exceeds broadcast output rank.");
    }

    const size_t rank = output_dims.size();

    std::vector<uint64_t> padded_dims(rank, 1ULL);
    std::copy(input_dims.begin(), input_dims.end(), padded_dims.begin() + (rank - input_dims.size()));

    std::vector<uint64_t> packed_strides(rank, 1ULL);
    if (rank > 0) {
        packed_strides[rank - 1] = 1ULL;
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
            packed_strides[static_cast<size_t>(i)] = packed_strides[static_cast<size_t>(i) + 1] * padded_dims[static_cast<size_t>(i) + 1];
        }
    }

    std::vector<uint64_t> result(rank, 0ULL);
    for (size_t axis = 0; axis < rank; ++axis) {
        const uint64_t in_dim = padded_dims[axis];
        const uint64_t out_dim = output_dims[axis];

        if (in_dim == out_dim) {
            result[axis] = packed_strides[axis];
        } else if (in_dim == 1ULL) {
            result[axis] = 0ULL;
        } else {
            std::ostringstream oss;
            oss << "Input dimensions are not broadcast-compatible with output dimensions. "
                << "axis=" << axis << ", input_dims=" << dimsToString(input_dims) << ", padded_input_dims=" << dimsToString(padded_dims)
                << ", output_dims=" << dimsToString(output_dims) << ", conflicting_in_dim=" << in_dim
                << ", conflicting_out_dim=" << out_dim;
            throw std::runtime_error(oss.str());
        }
    }

    return result;
}

static std::vector<ResolvedBroadcastGroup> buildResolvedBroadcastGroups(const CompiledExecutionStage& stage,
                                                                        const std::vector<RuntimeInputValue>& stage_inputs) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("buildResolvedBroadcastGroups expects a fused-kernel stage.");
    }
    if (stage_inputs.size() != stage.input_value_ids.size()) {
        throw std::runtime_error("buildResolvedBroadcastGroups stage input count mismatch.");
    }

    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        stage_input_dims.push_back(runtimeInputDims(input));
    }
    const auto node_dims = inferFusedStageNodeDims(stage.expr, stage_input_dims);

    std::map<std::vector<uint64_t>, std::vector<uint32_t>> outputs_by_dims;
    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        outputs_by_dims[resolveOutputDimsForStageOutput(stage, out_idx, stage_input_dims)].push_back(out_idx);
    }

    std::vector<ResolvedBroadcastGroup> groups;
    groups.reserve(outputs_by_dims.size());

    for (const auto& [output_dims, output_indices] : outputs_by_dims) {
        std::unordered_set<uint32_t> used_slots_set;
        for (uint32_t out_idx : output_indices) {
            collectReferencedLocalInputSlots(stage.expr, stage.outputs[out_idx].local_node_idx, used_slots_set);
        }

        std::vector<uint32_t> used_input_slots(used_slots_set.begin(), used_slots_set.end());
        std::sort(used_input_slots.begin(), used_input_slots.end());

        SpecializedBroadcastGroup specialized;
        specialized.output_indices = output_indices;
        specialized.output_dims = output_dims;
        specialized.numel = product(output_dims);
        specialized.used_input_slots = used_input_slots;
        specialized.used_input_load_kinds.assign(used_input_slots.size(), SpecializedInputLoadKind::ScalarPack);

        const std::vector<uint64_t> output_strides = computePackedOutputStrides(output_dims);

        const auto effective_dims_by_slot =
            collectEffectiveInputDimsForNode(stage.expr, node_dims, stage.outputs[output_indices.front()].local_node_idx);

        std::vector<std::vector<uint64_t>> input_strides_by_used;
        input_strides_by_used.reserve(used_input_slots.size());
        for (uint32_t slot : used_input_slots) {
            if (slot >= stage_inputs.size()) {
                throw std::runtime_error("Broadcast group input slot out of range.");
            }
            std::vector<uint64_t> effective_dims = runtimeInputDims(stage_inputs[slot]);
            auto dims_it = effective_dims_by_slot.find(slot);
            if (dims_it != effective_dims_by_slot.end()) {
                if (dims_it->second.size() > 1) {
                    std::ostringstream oss;
                    oss << "Broadcast group input slot " << slot << " is used with multiple logical shapes in one fused stage. shapes=";
                    bool first = true;
                    for (const auto& dims : dims_it->second) {
                        if (!first) {
                            oss << ", ";
                        }
                        first = false;
                        oss << dimsToString(dims);
                    }
                    throw std::runtime_error(oss.str());
                }
                if (!dims_it->second.empty()) {
                    effective_dims = *dims_it->second.begin();
                }
            }

            // std::cerr << "[FUSION] broadcast group input slot=" << slot << " raw_dims=" <<
            // dimsToString(stage_inputs[slot].getDimensions())
            //           << " effective_dims=" << dimsToString(effective_dims) << " output_dims=" << dimsToString(output_dims) << std::endl;

            input_strides_by_used.push_back(computeInputPackedStridesForBroadcast(effective_dims, output_dims));
        }

        for (size_t axis = 0; axis < output_dims.size(); ++axis) {
            if (output_dims[axis] == 1ULL) {
                continue;
            }

            SpecializedBroadcastAxis axis_desc;
            axis_desc.dim = output_dims[axis];
            axis_desc.output_stride = output_strides[axis];
            axis_desc.input_strides.resize(used_input_slots.size(), 0ULL);

            bool any_nonzero = false;
            for (size_t used_i = 0; used_i < used_input_slots.size(); ++used_i) {
                axis_desc.input_strides[used_i] = input_strides_by_used[used_i][axis];
                if (axis_desc.input_strides[used_i] != 0ULL) {
                    any_nonzero = true;
                }
            }

            if (any_nonzero) {
                specialized.active_axes.push_back(std::move(axis_desc));
            }
        }

        groups.push_back(ResolvedBroadcastGroup{std::move(specialized)});
    }

    std::sort(groups.begin(), groups.end(), [](const ResolvedBroadcastGroup& a, const ResolvedBroadcastGroup& b) {
        return a.specialized.numel > b.specialized.numel;
    });

    return groups;
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

    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. Pass a dict of name -> Tensor to getOutputShapes(...).");
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

    if (externalRootInputCount(root_inputs, backward_config) != inputs.size()) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed " + to_string(inputs.size()) +
                                 " inputs, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. Pass a dict of name -> Tensor to getOutputShape(...).");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> output_shapes = getOutputShapes(inputs);
    assert(output_shapes.size() == 1);
    return output_shapes.begin()->second;
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(const Tensor& input) const {
    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::getOutputShapes was passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. Pass a dict of name -> Tensor to getOutputShapes(...).");
    }

    std::unordered_map<std::string, Tensor> input_map = {
        {root_inputs[0].name, input},
    };

    return getOutputShapes(input_map);
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(
    const std::unordered_map<std::string, Tensor>& inputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputsForCompilation(inputs);
    std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::getOutputShapes requires at least one bound root input.");
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> value_dims;
    value_dims.reserve(root_values.size() + compiled_outputs->stages.size());

    for (const auto& [value_id, value] : root_values) {
        value_dims.emplace(value_id, runtimeInputDims(value));
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
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                value_dims[stage.outputs[output_idx].value_id] = resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims);
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
        } else if (stage.kind == CompiledExecutionStage::Kind::ArgMinMax) {
            if (!stage.arg_minmax) {
                throw std::runtime_error("Missing compiled arg-min/max stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Arg-min/max stage expected exactly one input and one output.");
            }

            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.arg_minmax->reduction_axes, stage.arg_minmax->squeeze_axes);
        } else if (stage.kind == CompiledExecutionStage::Kind::ReduceMinMaxBackward) {
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("Missing compiled reduce-min/max-backward stage.");
            }

            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduce-min/max-backward stage expected exactly two inputs and one output.");
            }

            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
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

PhysicalOutputs FusedEquation::buildShapeSpecializedOutputs(const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
    if (!backward_config.has_value()) {
        return outputs_template;
    }

    // Resolve the forward template dtypes against the actual runtime forward-input dtypes
    // before rebuilding the backward graph. Otherwise preferredGradValueDType(...) sees
    // unresolved INPUT nodes and returns empty, so terminal grad outputs can stay promoted.
    PhysicalOutputs resolved_forward_outputs = backward_config->forward_outputs_template;
    if (!resolved_forward_outputs.expr) {
        throw std::runtime_error("Backward shape specialization requires non-null forward expr.");
    }
    resolved_forward_outputs.expr = std::make_shared<PhysicalExpression>(*backward_config->forward_outputs_template.expr);

    std::vector<DataType> forward_root_input_dtypes(resolved_forward_outputs.expr->numInputs(), DataType::FP32);
    std::vector<bool> have_forward_root_input_dtype(resolved_forward_outputs.expr->numInputs(), false);

    std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims;
    forward_input_dims.reserve(resolved_forward_outputs.expr->inputs.size());

    for (const NamedInput& forward_input : resolved_forward_outputs.expr->inputs) {
        bool found_name = false;
        for (const NamedInput& root_input : root_inputs) {
            if (root_input.name != forward_input.name) {
                continue;
            }

            auto it = root_values.find(root_input.slot);
            if (it == root_values.end()) {
                throw std::runtime_error("Missing bound runtime input for backward forward-input shape specialization input: " +
                                         forward_input.name);
            }

            forward_input_dims.emplace(forward_input.name, runtimeInputDims(it->second));

            if (forward_input.slot >= forward_root_input_dtypes.size()) {
                throw std::runtime_error("Forward input slot out of range while resolving backward shape-specialized dtypes.");
            }
            forward_root_input_dtypes[forward_input.slot] = runtimeInputDType(it->second);
            have_forward_root_input_dtype[forward_input.slot] = true;

            found_name = true;
            break;
        }

        if (!found_name) {
            throw std::runtime_error("Backward equation root inputs do not contain required forward input: " + forward_input.name);
        }
    }

    for (size_t slot = 0; slot < have_forward_root_input_dtype.size(); ++slot) {
        if (!have_forward_root_input_dtype[slot]) {
            throw std::runtime_error("Missing runtime dtype for forward input slot " + std::to_string(slot) +
                                     " during backward shape specialization.");
        }
    }

    resolveOutputsDTypesInPlace(resolved_forward_outputs, forward_root_input_dtypes);

    if (backward_config->upstream_input_names_by_output.has_value()) {
        return buildBackwardOutputs(resolved_forward_outputs,
                                    backward_config->wrt_names,
                                    backward_config->upstream_input_names_by_output.value(),
                                    forward_input_dims,
                                    backward_config->accumulate_grad_outputs);
    }

    return buildBackwardOutputs(
        resolved_forward_outputs, backward_config->wrt_names, std::nullopt, forward_input_dims, backward_config->accumulate_grad_outputs);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForInputs(const std::unordered_map<std::string, Tensor>& namedInputs,
                                                                 const std::unordered_map<std::string, float>& scalarInputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputsForCompilation(namedInputs, scalarInputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::compileForInputs requires at least one bound root input.");
    }

    return compileForRootValues(root_values);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForRootValues(
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
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
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
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

    const auto effectiveRequestedOutputShapes = defaultBackwardRequestedOutputShapes(backward_config, root_inputs, root_values, {});

    std::unordered_set<std::string> expected_output_names;

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused-kernel stages, but found stage kind: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }

        std::vector<RuntimeInputValue> ordered_inputs;
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

        std::vector<uint64_t> resolved_output_dims;
        const bool requires_broadcast = fusedStageRequiresBroadcastLaunch(
            stage, ordered_inputs, effectiveRequestedOutputShapes, backward_config.has_value(), resolved_output_dims);

        if (!requires_broadcast) {
            if (!stage.flat) {
                throw std::runtime_error("FusedEquation::run found a flat fused stage with no compiled kernel.");
            }

            prepared_stage.expected_output_dims.resize(stage.outputs.size());
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, ordered_inputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    if (!(ordered_inputs.empty() && output_dims.empty())) {
                        verifyRequestedOutputLayout(requested_it->second, output_dims);
                    }
                    output_dims = requested_it->second;
                }
                prepared_stage.expected_output_dims[output_idx] = std::move(output_dims);
            }

            prepared_stage.compiled_equation =
                selectFlatCompiledEquation(stage, compiled_outputs->signature, maxNumel(prepared_stage.expected_output_dims));
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
                    std::vector<uint64_t> output_dims = group.specialized.output_dims;
                    auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                    if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                        if (!(ordered_inputs.empty() && output_dims.empty())) {
                            verifyRequestedOutputLayout(requested_it->second, output_dims);
                        }
                        output_dims = requested_it->second;
                    }
                    prepared_stage.expected_output_dims[output_idx] = std::move(output_dims);
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
                                             const std::optional<std::string>& upstream_input_name,
                                             bool accumulate_grad_outputs) const {
    PhysicalOutputs backward_outputs =
        buildBackwardOutputs(outputs_template, wrt_names, upstream_input_name, std::nullopt, accumulate_grad_outputs);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = inferBackwardWrtNamesFromOutputs(backward_outputs),
                             .upstream_input_names_by_output =
                                 upstream_input_name.has_value() ? std::optional<std::unordered_map<std::string, std::string>>(
                                                                       std::unordered_map<std::string, std::string>{
                                                                           {outputs_template.outputs[0].name, upstream_input_name.value()},
                                                                       })
                                                                 : std::nullopt,
                             .accumulate_grad_outputs = accumulate_grad_outputs,
                         });
}

FusedEquation FusedEquation::compileBackward(const std::vector<std::string>& wrt_names,
                                             const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                             bool accumulate_grad_outputs) const {
    PhysicalOutputs backward_outputs =
        buildBackwardOutputs(outputs_template, wrt_names, upstream_input_names_by_output, std::nullopt, accumulate_grad_outputs);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = inferBackwardWrtNamesFromOutputs(backward_outputs),
                             .upstream_input_names_by_output = upstream_input_names_by_output,
                             .accumulate_grad_outputs = accumulate_grad_outputs,
                         });
}

bool FusedEquation::resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        outputDimensions.clear();
        return false;
    }

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

std::unordered_map<uint32_t, RuntimeInputValue> FusedEquation::bindRootInputs(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalar_inputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, Tensor>* namedOutputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> values;
    values.reserve(root_inputs.size());

    const std::unordered_set<std::string> accumulation_output_names = backwardAccumulationOutputNames(backward_config);

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_scalar_input_set;
    expected_scalar_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_tensor_scalar_input_set;
    expected_tensor_scalar_input_set.reserve(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        const bool bind_from_outputs = namedOutputs != nullptr && accumulation_output_names.contains(input.name);
        if (bind_from_outputs) {
            auto output_it = namedOutputs->find(input.name);
            if (output_it == namedOutputs->end()) {
                throw std::runtime_error("Missing required gradient output tensor for accumulation: " + input.name);
            }
            values.emplace(input.slot, output_it->second);
            continue;
        }

        if (input.kind == NamedInput::Kind::Tensor) {
            auto input_it = namedInputs.find(input.name);
            if (input_it == namedInputs.end()) {
                throw std::runtime_error("Missing required fused equation input: " + input.name);
            }
            values.emplace(input.slot, input_it->second);
            expected_input_set.insert(input.name);
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            auto scalar_it = scalar_inputs.find(input.name);
            const float scalar_value = scalar_it == scalar_inputs.end() ? 0.0f : scalar_it->second;
            values.emplace(input.slot, scalar_value);
            if (scalar_it != scalar_inputs.end()) {
                expected_scalar_input_set.insert(input.name);
            }
        } else {
            auto scalar_it = tensor_scalar_inputs.find(input.name);
            if (scalar_it == tensor_scalar_inputs.end()) {
                throw std::runtime_error("Missing required fused equation tensor runtime scalar input: " + input.name);
            }
            values.emplace(input.slot, scalar_it->second);
            expected_tensor_scalar_input_set.insert(input.name);
        }
    }

    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : scalar_inputs) {
        if (!expected_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }

    return values;
}

std::unordered_map<uint32_t, RuntimeInputValue> FusedEquation::bindRootInputsForCompilation(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalar_inputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return bindRootInputs(namedInputs, scalar_inputs, tensor_scalar_inputs);
    }

    std::unordered_map<uint32_t, RuntimeInputValue> values;
    values.reserve(root_inputs.size());

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_scalar_input_set;
    expected_scalar_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_tensor_scalar_input_set;
    expected_tensor_scalar_input_set.reserve(root_inputs.size());

    std::unordered_map<std::string, uint32_t> root_slot_by_name;
    root_slot_by_name.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        root_slot_by_name.emplace(input.name, input.slot);
    }

    const auto accumulation_output_names = backwardAccumulationOutputNames(backward_config);

    for (const NamedInput& input : root_inputs) {
        if (accumulation_output_names.contains(input.name)) {
            continue;
        }

        if (input.kind == NamedInput::Kind::Tensor) {
            auto it = namedInputs.find(input.name);
            if (it == namedInputs.end()) {
                throw std::runtime_error("Missing required fused equation input: " + input.name);
            }
            values.emplace(input.slot, it->second);
            expected_input_set.insert(input.name);
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            auto it = scalar_inputs.find(input.name);
            const float scalar_value = it == scalar_inputs.end() ? 0.0f : it->second;
            values.emplace(input.slot, scalar_value);
            if (it != scalar_inputs.end()) {
                expected_scalar_input_set.insert(input.name);
            }
        } else {
            auto it = tensor_scalar_inputs.find(input.name);
            if (it == tensor_scalar_inputs.end()) {
                throw std::runtime_error("Missing required fused equation tensor runtime scalar input: " + input.name);
            }
            values.emplace(input.slot, it->second);
            expected_tensor_scalar_input_set.insert(input.name);
        }
    }

    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : scalar_inputs) {
        if (!expected_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string grad_output_name = wrt_name + "_grad";
        auto slot_it = root_slot_by_name.find(grad_output_name);
        if (slot_it == root_slot_by_name.end()) {
            throw std::runtime_error("Missing backward accumulation root input for output: " + grad_output_name);
        }

        auto forward_input_it = namedInputs.find(wrt_name);
        if (forward_input_it == namedInputs.end()) {
            throw std::runtime_error("Missing forward input required to infer backward accumulation output: " + wrt_name);
        }

        std::vector<uint64_t> dims = forward_input_it->second.getDimensions();
        auto requested_it = requestedOutputShapes.find(grad_output_name);
        if (requested_it != requestedOutputShapes.end() && !requested_it->second.empty()) {
            dims = requested_it->second;
        }

        const Optional<DataType> preferred_dtype = preferredBackwardGradBufferDType(backward_config.value(), wrt_name);
        const DataType grad_buffer_dtype = preferred_dtype.isPresent() ? preferred_dtype.get() : forward_input_it->second.getDataType();
        TensorDescriptor descriptor(grad_buffer_dtype, dims);
        values.emplace(slot_it->second, Tensor(forward_input_it->second.getPlacement(), descriptor));
    }

    return values;
}

std::shared_ptr<StampedEquation> FusedEquation::stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                              std::vector<RuntimeInputValue>& inputs,
                                                              std::vector<Tensor>& outputs,
                                                              const Stream& stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot stamp an empty compiled equation.");
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

    for (uint64_t i = 0; i < inputs.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::Tensor) {
            if (!runtimeInputIsTensor(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected tensor input at slot " + std::to_string(i) + ".");
            }
            const Tensor& input = runtimeInputTensor(inputs[i]);
            if (!input.isInitialized()) {
                throw std::runtime_error("Input tensor is not initialized.");
            }
            if (input.getDescriptor().getDataType() != compiledEquation->input_dtypes[i]) {
                throw std::runtime_error("Input tensor data type mismatch.");
            }
            if (input.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
                throw std::runtime_error("Input tensor GPU mismatch.");
            }
        } else if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            if (!std::holds_alternative<float>(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected runtime scalar input at slot " + std::to_string(i) + ".");
            }
            if (compiledEquation->input_dtypes[i] != DataType::FP32) {
                throw std::runtime_error("Runtime scalar inputs currently require FP32 compiled input dtype.");
            }
        } else {
            if (!runtimeInputIsTensorScalarBinding(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected tensor runtime scalar input at slot " + std::to_string(i) +
                                         ".");
            }
            const TensorScalarBinding& binding = runtimeInputTensorScalarBinding(inputs[i]);
            if (!binding.buffer.isInitialized()) {
                throw std::runtime_error("Tensor runtime scalar buffer is not initialized.");
            }
            if (binding.buffer.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::runtime_error("Tensor runtime scalar buffer must be on GPU.");
            }
            if (binding.buffer.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
                throw std::runtime_error("Tensor runtime scalar buffer GPU mismatch.");
            }
            if (binding.sourceDType != compiledEquation->input_dtypes[i]) {
                throw std::runtime_error("Tensor runtime scalar source dtype mismatch.");
            }
            const size_t bytes_needed = binding.byteOffset + dataTypeSizeBytes(binding.sourceDType);
            if (bytes_needed > binding.buffer.getArraySizeInBytes()) {
                throw std::runtime_error("Tensor runtime scalar binding exceeds backing buffer size.");
            }
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
    return stampReduction(compiledReduction, input, Optional<Tensor>::empty(), stream, requested_output_shape);
}

std::shared_ptr<StampedReduction> FusedEquation::stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                Tensor& input,
                                                                const Optional<Tensor>& preallocatedOutput,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledReduction)
        throw std::runtime_error("Tried to stamp reduction on a non-reduction FusedEquation.");

    Tensor adaptedInput = adaptReductionInputDTypeIfNeeded(input, compiledReduction->input_dtype, compiledReduction->op, stream);

    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiledReduction, adaptedInput, stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(adaptedInput.getPlacement(), workspaceDescriptor);
    }

    std::vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(adaptedInput.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);

    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    Tensor output;
    if (preallocatedOutput.isPresent()) {
        output = preallocatedOutput.get();

        if (output.getPlacement() != adaptedInput.getPlacement()) {
            throw std::runtime_error("Preallocated reduction output tensor placement does not match the reduction input placement.");
        }

        if (output.getDescriptor().getDataType() != compiledReduction->output_dtype) {
            throw std::runtime_error("Preallocated reduction output tensor dtype does not match the compiled reduction output dtype.");
        }

        verifyRequestedOutputLayout(output.getDimensions(), resolved_output_dimensions);

        if (!requested_output_shape.empty() && output.getDimensions() != output_dimensions) {
            throw std::runtime_error("Preallocated reduction output tensor shape does not match the requested output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledReduction->output_dtype, output_dimensions);
        output = Tensor(adaptedInput.getPlacement(), outputDescriptor);
    }

    return make_shared<StampedReduction>(std::move(built), adaptedInput, output, stream, workspace);
}

std::shared_ptr<StampedArgMinMax> FusedEquation::stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                Tensor& input,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    return stampArgMinMax(compiledStage, input, Optional<Tensor>::empty(), stream, requested_output_shape);
}

std::shared_ptr<StampedArgMinMax> FusedEquation::stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                Tensor& input,
                                                                const Optional<Tensor>& preallocatedOutput,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledStage) {
        throw std::runtime_error("stampArgMinMax requires non-null compiled stage.");
    }
    Tensor adaptedInput = adaptReductionInputDTypeIfNeeded(input, compiledStage->input_dtype, compiledStage->op, stream);

    const ExprOp reduce_op = compiledStage->op == ExprOp::REDUCE_ARGMIN ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(reduce_op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            DataType::FP32,
                                                                            compiledStage->compute_dtype,
                                                                            /*output_indices=*/true,
                                                                            adaptedInput,
                                                                            stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(adaptedInput.getPlacement(), workspaceDescriptor);
    }

    std::vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(adaptedInput.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);
    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    Tensor output;
    if (preallocatedOutput.isPresent()) {
        output = preallocatedOutput.get();

        if (output.getPlacement() != adaptedInput.getPlacement()) {
            throw std::runtime_error(
                "Preallocated argmin/argmax output tensor placement does not match the argmin/argmax input placement.");
        }

        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error(
                "Preallocated argmin/argmax output tensor dtype does not match the compiled argmin/argmax output dtype.");
        }

        verifyRequestedOutputLayout(output.getDimensions(), resolved_output_dimensions);

        if (!requested_output_shape.empty() && output.getDimensions() != output_dimensions) {
            throw std::runtime_error("Preallocated argmin/argmax output tensor shape does not match the requested output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dimensions);
        output = Tensor(adaptedInput.getPlacement(), outputDescriptor);
    }

    const std::vector<uint64_t> unsqueezed_output_dims =
        StampedEquation::computeReductionOutputDims(adaptedInput.getDimensions(), built->key.reduction_axes, {});
    TensorDescriptor reductionValueDescriptor(built->key.output_dtype, unsqueezed_output_dims);
    Tensor reductionValueOutput(adaptedInput.getPlacement(), reductionValueDescriptor);

    return make_shared<StampedArgMinMax>(std::move(built), adaptedInput, output, reductionValueOutput, stream, workspace);
}

std::shared_ptr<StampedReduceMinMaxBackward> FusedEquation::stampReduceMinMaxBackward(
    const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage, Tensor& input, Tensor& grad_output, const Stream& stream) const {
    return stampReduceMinMaxBackward(compiledStage, input, grad_output, Optional<Tensor>::empty(), stream);
}

std::shared_ptr<StampedReduceMinMaxBackward> FusedEquation::stampReduceMinMaxBackward(
    const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage,
    Tensor& input,
    Tensor& grad_output,
    const Optional<Tensor>& preallocatedOutput,
    const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampReduceMinMaxBackward requires non-null compiled stage.");
    }

    Tensor adaptedInput =
        adaptReductionInputDTypeIfNeeded(input,
                                         compiledStage->input_dtype,
                                         compiledStage->op == ExprOp::REDUCE_MIN_BACKWARD ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX,
                                         stream);

    if (grad_output.getDataType() != compiledStage->grad_output_dtype) {
        throw std::runtime_error("Grad-output dtype does not match compiled reduce-min/max-backward grad-output dtype.");
    }

    const std::vector<uint64_t> expected_grad_dims = StampedEquation::computeReductionOutputDims(
        adaptedInput.getDimensions(), compiledStage->reduction_axes, compiledStage->squeeze_axes);
    if (!outputDimensionsMatchIgnoringSingletons(grad_output.getDimensions(), expected_grad_dims)) {
        throw std::runtime_error("Grad-output tensor dimensions are incompatible with reduce-min/max-backward stage.");
    }

    const ExprOp reduce_op = compiledStage->op == ExprOp::REDUCE_MIN_BACKWARD ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(reduce_op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            DataType::FP32,
                                                                            compiledStage->compute_dtype,
                                                                            /*output_indices=*/true,
                                                                            adaptedInput,
                                                                            stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(adaptedInput.getPlacement(), workspaceDescriptor);
    }

    const std::vector<uint64_t> unsqueezed_output_dims =
        StampedEquation::computeReductionOutputDims(adaptedInput.getDimensions(), built->key.reduction_axes, {});

    TensorDescriptor indicesDescriptor(TensorDescriptor::DataType::UINT32, unsqueezed_output_dims);
    Tensor indices(adaptedInput.getPlacement(), indicesDescriptor);

    TensorDescriptor reductionValueDescriptor(built->key.output_dtype, unsqueezed_output_dims);
    Tensor reductionValueOutput(adaptedInput.getPlacement(), reductionValueDescriptor);

    Tensor output;
    if (preallocatedOutput.isPresent()) {
        output = preallocatedOutput.get();

        if (output.getPlacement() != adaptedInput.getPlacement()) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor placement does not match the input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor dtype does not match the compiled output dtype.");
        }
        if (output.getDimensions() != input.getDimensions()) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor dimensions do not match the input dimensions.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, input.getDimensions());
        output = Tensor(adaptedInput.getPlacement(), outputDescriptor);
    }

    return make_shared<StampedReduceMinMaxBackward>(
        std::move(built), adaptedInput, grad_output, output, indices, reductionValueOutput, stream, workspace);
}

StampedExecutionPlan FusedEquation::stampSingleOutput(const std::unordered_map<std::string, Tensor>& inputs,
                                                      const Stream& stream,
                                                      const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                                      const std::optional<Tensor>& preallocated_output,
                                                      const std::vector<uint64_t>& requestedOutputShape) const {
    std::unordered_map<std::string, Tensor> preallocated_outputs{};

    const auto outputNames = getOutputNames();
    if (outputNames.size() != 1) {
        throw std::runtime_error("Single-output stamp overload called on an equation that does not have exactly one output.");
    }
    if (outputNames.front() != "output") {
        throw std::runtime_error(
            "Single-output stamp overload requires the sole named output to be \"output\" when a preallocated "
            "output tensor is provided.");
    }

    if (preallocated_output.has_value()) {
        preallocated_outputs["output"] = preallocated_output.value();
    }

    return stamp(inputs, stream, tensor_scalar_inputs, preallocated_outputs, makeSingleOutputRequestedShapeMap(requestedOutputShape));
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                          const std::unordered_map<std::string, Tensor>& preallocated_outputs,
                                          const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    if (accumulatesIntoGradOutputs(backward_config) && preallocated_outputs.empty()) {
        throw std::runtime_error(
            "Backward equations compiled with accumulate_grad_outputs=true require caller-provided gradient output tensors when stamping.");
    }

    static const std::unordered_map<std::string, float> empty_scalar_inputs;

    const std::unordered_map<std::string, std::vector<uint64_t>> requestedOutputShapesWithOutputs =
        mergeRequestedOutputShapesWithProvidedOutputs(preallocated_outputs, requestedOutputShapes);

    std::unordered_map<uint32_t, RuntimeInputValue> compile_root_values =
        accumulatesIntoGradOutputs(backward_config)
            ? bindRootInputs(inputs, empty_scalar_inputs, tensor_scalar_inputs, &preallocated_outputs)
            : bindRootInputsForCompilation(inputs, empty_scalar_inputs, tensor_scalar_inputs, requestedOutputShapesWithOutputs);
    const auto effectiveRequestedOutputShapes =
        defaultBackwardRequestedOutputShapes(backward_config, root_inputs, compile_root_values, requestedOutputShapesWithOutputs);

    if (accumulatesIntoGradOutputs(backward_config)) {
        validateBackwardAccumulationOutputs(backward_config, inputs, preallocated_outputs, effectiveRequestedOutputShapes);
    }

    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(compile_root_values);

    std::unordered_map<uint32_t, RuntimeInputValue> values = accumulatesIntoGradOutputs(backward_config)
                                                                 ? compile_root_values
                                                                 : bindRootInputs(inputs, empty_scalar_inputs, tensor_scalar_inputs, {});

    std::unordered_map<std::string, Tensor> preallocated_final_outputs_by_name = preallocated_outputs;

    std::vector<StampedExecutionStage> stampedStages;
    stampedStages.reserve(compiled_outputs->stages.size());

    std::unordered_map<uint32_t, uint32_t> producer_stage_by_value_id;
    producer_stage_by_value_id.reserve(compiled_outputs->stages.size() * 2);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<RuntimeInputValue> stageInputs;
        stageInputs.reserve(stage.input_value_ids.size());

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
                const uint32_t dep_stage_idx = producer_it->second;
                if (std::find(dependency_stage_indices.begin(), dependency_stage_indices.end(), dep_stage_idx) ==
                    dependency_stage_indices.end()) {
                    dependency_stage_indices.push_back(dep_stage_idx);
                }
            }
        }

        switch (stage.kind) {
            case CompiledExecutionStage::Kind::FusedKernel: {
                if (stage.outputs.empty()) {
                    throw std::runtime_error("Fused stage requires at least one output.");
                }

                std::vector<std::vector<uint64_t>> expected_output_dims(stage.outputs.size());
                std::shared_ptr<CompiledEquation> compiledEq;

                std::vector<uint64_t> resolved_output_dims;
                const bool requires_broadcast = fusedStageRequiresBroadcastLaunch(
                    stage, stageInputs, effectiveRequestedOutputShapes, backward_config.has_value(), resolved_output_dims);

                if (!requires_broadcast) {
                    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                        std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stageInputs);
                        auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                        if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                            if (!(stageInputs.empty() && output_dims.empty())) {
                                verifyRequestedOutputLayout(requested_it->second, output_dims);
                            }
                            output_dims = requested_it->second;
                        }
                        expected_output_dims[output_idx] = std::move(output_dims);
                    }
                    compiledEq = selectFlatCompiledEquation(stage, compiled_outputs->signature, maxNumel(expected_output_dims));
                } else {
                    std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, stageInputs);
                    if (groups.empty()) {
                        throw std::runtime_error("Fused stage expected at least one broadcast group.");
                    }

                    std::vector<SpecializedBroadcastGroup> specialized_groups;
                    specialized_groups.reserve(groups.size());
                    for (const ResolvedBroadcastGroup& group : groups) {
                        specialized_groups.push_back(group.specialized);
                        for (uint32_t output_idx : group.specialized.output_indices) {
                            if (output_idx >= expected_output_dims.size()) {
                                throw std::runtime_error("Broadcast group output index out of range.");
                            }
                            std::vector<uint64_t> output_dims = group.specialized.output_dims;
                            auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                            if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                                if (!(stageInputs.empty() && output_dims.empty())) {
                                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                                }
                                output_dims = requested_it->second;
                            }
                            expected_output_dims[output_idx] = std::move(output_dims);
                        }
                    }

                    compiledEq = EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);
                }

                std::vector<Tensor> stageOutputs;
                stageOutputs.reserve(stage.outputs.size());
                const TensorPlacement outputPlacement = pickStageOutputPlacement(stageInputs, values);
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& stageOutput = stage.outputs[output_idx];
                    auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                    if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                        const Tensor& outputTensor = preallocated_it->second;

                        if (!outputTensor.isInitialized()) {
                            throw std::runtime_error("Preallocated fused-stage output tensor is not initialized.");
                        }
                        if (outputTensor.getPlacement() != outputPlacement) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor placement does not match the expected output placement.");
                        }
                        if (outputTensor.getDescriptor().getDataType() != compiledEq->output_dtypes.at(output_idx)) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor dtype does not match the compiled output dtype.");
                        }
                        if (outputTensor.getDimensions() != expected_output_dims[output_idx]) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor dimensions are incompatible with the staged output shape.");
                        }

                        stageOutputs.push_back(outputTensor);
                        values[stageOutput.value_id] = outputTensor;
                    } else {
                        TensorDescriptor outputDescriptor(compiledEq->output_dtypes.at(output_idx), expected_output_dims[output_idx]);
                        Tensor outputTensor(outputPlacement, outputDescriptor);
                        stageOutputs.push_back(outputTensor);
                        values[stageOutput.value_id] = outputTensor;
                    }
                    producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                }

                std::shared_ptr<StampedEquation> stampedKernel = stampEquation(compiledEq, stageInputs, stageOutputs, stream);
                stampedStages.emplace_back(stampedKernel, std::move(dependency_stage_indices));
                break;
            }
            case CompiledExecutionStage::Kind::Reduction: {
                if (!stage.reduction) {
                    throw std::runtime_error("Reduction stage missing compiled reduction payload.");
                }
                if (stageInputs.size() != 1) {
                    throw std::runtime_error("Reduction stage expects exactly one input.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Reduction stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];

                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.reduction->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }

                std::shared_ptr<StampedReduction> stampedReduction =
                    stampReduction(stage.reduction, inputTensor, outputTensor, stream, output_dims);

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedReduction, std::move(dependency_stage_indices));
                break;
            }
            case CompiledExecutionStage::Kind::ArgMinMax: {
                if (!stage.arg_minmax) {
                    throw std::runtime_error("Argmin/argmax stage missing compiled payload.");
                }
                if (stageInputs.size() != 1) {
                    throw std::runtime_error("Argmin/argmax stage expects exactly one input.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Argmin/argmax stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];

                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.arg_minmax->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }

                std::shared_ptr<StampedArgMinMax> stampedArgMinMax =
                    stampArgMinMax(stage.arg_minmax, inputTensor, outputTensor, stream, output_dims);

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedArgMinMax, std::move(dependency_stage_indices));
                break;
            }
            case CompiledExecutionStage::Kind::ReduceMinMaxBackward: {
                if (!stage.reduce_minmax_backward) {
                    throw std::runtime_error("Reduce-min/max-backward stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("Reduce-min/max-backward stage expects exactly two inputs.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor gradOutputTensor = runtimeInputTensor(stageInputs[1]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Reduce-min/max-backward stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.reduce_minmax_backward->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }
                std::shared_ptr<StampedReduceMinMaxBackward> stampedReduceMinMaxBackward =
                    stampReduceMinMaxBackward(stage.reduce_minmax_backward, inputTensor, gradOutputTensor, outputTensor, stream);
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedReduceMinMaxBackward, std::move(dependency_stage_indices));
                break;
            }
        }
    }

    std::unordered_map<std::string, Tensor> finalOutputsByName;
    finalOutputsByName.reserve(compiled_outputs->final_outputs.size());
    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = values.find(final_output.value_id);
        if (it == values.end()) {
            throw std::runtime_error("Missing final output tensor for output: " + final_output.name);
        }
        finalOutputsByName.emplace(final_output.name, runtimeInputTensor(it->second));
    }

    return StampedExecutionPlan(std::move(stampedStages), std::move(finalOutputsByName), stream);
}

void FusedEquation::run(const Tensor& input, Tensor& output, Stream& stream) const {
    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{root_inputs[0].name, input}};
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};

    run(input_map, output_map, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    static const std::unordered_map<std::string, float> empty_scalar_inputs;
    run(inputs, empty_scalar_inputs, output, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        const std::unordered_map<std::string, float>& scalar_inputs,
                        Tensor& output,
                        Stream& stream) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run was only passed a single output, but this equation has multiple named outputs. "
            "Pass a dict of name -> PhysicalTensor of outputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};
    run(inputs, scalar_inputs, output_map, stream);
}

void FusedEquation::run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const {
    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
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
    static const std::unordered_map<std::string, float> empty_scalar_inputs;
    run(inputs, empty_scalar_inputs, outputs, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        const std::unordered_map<std::string, float>& scalar_inputs,
                        std::unordered_map<std::string, Tensor>& outputs,
                        Stream& stream) const {
    const std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputs(inputs, scalar_inputs, {}, &outputs);
    const std::shared_ptr<PreparedConvenienceRunPlan> prepared_plan = prepareConvenienceRunPlan(root_values);
    const std::shared_ptr<CompiledOutputs>& compiled_outputs = prepared_plan->compiled_outputs;

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages.");
    }

    std::unordered_set<std::string> expected_output_names(prepared_plan->expected_output_names_in_order.begin(),
                                                          prepared_plan->expected_output_names_in_order.end());

    for (const std::string& name : prepared_plan->expected_output_names_in_order) {
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            throw std::runtime_error("Missing output tensor '" + name + "' for fused equation run.");
        }
    }

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

    int32_t gpu_num = -1;
    for (const auto& [value_id, value] : root_values) {
        (void)value_id;
        Optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
        if (placement.isPresent()) {
            gpu_num = placement.get().getDeviceNum();
            break;
        }
    }
    if (gpu_num < 0) {
        if (outputs.empty()) {
            throw std::runtime_error("FusedEquation::run requires at least one tensor input or output.");
        }
        gpu_num = outputs.begin()->second.getPlacement().getDeviceNum();
    }

    for (const auto& [value_id, value] : root_values) {
        (void)value_id;
        Optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
        if (placement.isPresent()) {
            if (placement.get().getDeviceNum() != gpu_num) {
                throw std::runtime_error("FusedEquation::run requires all root tensor inputs to be on the same GPU.");
            }
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
                                const std::vector<RuntimeInputValue>& orderedInputs,
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

    std::vector<Stream> helper_streams_used;
    helper_streams_used.reserve(compiled_outputs->stages.size());

    auto rememberHelperStream = [&](Stream& helper_stream) {
        if (std::find(helper_streams_used.begin(), helper_streams_used.end(), helper_stream) == helper_streams_used.end()) {
            helper_streams_used.push_back(helper_stream);
        }
    };

    for (uint32_t stage_num = 0; stage_num < compiled_outputs->stages.size(); ++stage_num) {
        bool use_helper_streams = (stage_num != 0);
        const CompiledExecutionStage& stage = compiled_outputs->stages[stage_num];
        const PreparedConvenienceRunStage& prepared_stage = prepared_plan->stages[stage_num];

        std::vector<RuntimeInputValue> orderedInputs;
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

    for (Stream& helper_stream : helper_streams_used) {
        stream.waitEvent(helper_stream.putEvent());
    }
}

FusedEquation::ParameterFanOverrideMap FusedEquation::getParameterFanOverrides(
    const std::unordered_map<std::string, Tensor>& named_inputs,
    const std::unordered_set<std::string>& parameter_names,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) const {
    ParameterFanOverrideMap result;

    const auto root_values = bindRootInputsForCompilation(named_inputs, {}, tensor_scalar_inputs, requested_output_shapes);
    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            continue;
        }

        for (const ParameterFanOverride& hint : stage.parameter_fan_overrides) {
            if (!parameter_names.contains(hint.input_name)) {
                continue;
            }

            auto [it, inserted] = result.emplace(hint.input_name, hint);
            if (!inserted) {
                it->second.input_name = hint.input_name;
                it->second.fan_in = std::max(it->second.fan_in, hint.fan_in);
                it->second.fan_out = std::max(it->second.fan_out, hint.fan_out);
            }
        }
    }

    return result;
}

}  // namespace ThorImplementation
