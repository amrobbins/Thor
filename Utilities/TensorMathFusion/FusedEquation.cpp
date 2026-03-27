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

static uint64_t maxNumel(const std::vector<std::vector<uint64_t>>& dims_by_output) {
    uint64_t max_numel = 0;
    for (const std::vector<uint64_t>& dims : dims_by_output) {
        max_numel = std::max<uint64_t>(max_numel, product(dims));
    }
    return max_numel;
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

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput output_idx out of range.");
    }

    if (stage.kind == CompiledExecutionStage::Kind::ReduceMinMaxBackward) {
        if (stage_input_dims.empty()) {
            throw std::runtime_error("resolveOutputDimsForStageOutput reduce-min/max-backward stage expected at least one input shape.");
        }
        return stage_input_dims[0];
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
                                                             const std::vector<Tensor>& stage_inputs) {
    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const Tensor& input : stage_inputs) {
        stage_input_dims.push_back(input.getDimensions());
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
static TensorPlacement pickStageOutputPlacement(const std::vector<Tensor>& stage_inputs,
                                                const std::unordered_map<uint32_t, Tensor>& available_values) {
    if (!stage_inputs.empty()) {
        return stage_inputs.front().getPlacement();
    }
    if (!available_values.empty()) {
        return available_values.begin()->second.getPlacement();
    }
    throw std::runtime_error("Unable to infer output placement for fused stage with no available tensors.");
}

static std::unordered_map<std::string, std::vector<uint64_t>> defaultBackwardRequestedOutputShapes(
    const std::optional<BackwardEquationConfig>& backward_config,
    const std::vector<NamedInput>& root_inputs,
    const std::unordered_map<uint32_t, Tensor>& root_values,
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

        effective.emplace(output_name, value_it->second.getDimensions());
    }

    return effective;
}

static bool fusedStageRequiresBroadcastLaunch(const CompiledExecutionStage& stage,
                                              const std::vector<Tensor>& stage_inputs,
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

    std::vector<Tensor> layout_inputs = stage_inputs;
    const bool requires_broadcast = FusedEquation::resolveLayout(layout_inputs, resolved_output_dims);

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
                                                                        const std::vector<Tensor>& stage_inputs) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("buildResolvedBroadcastGroups expects a fused-kernel stage.");
    }
    if (stage_inputs.size() != stage.input_value_ids.size()) {
        throw std::runtime_error("buildResolvedBroadcastGroups stage input count mismatch.");
    }

    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const Tensor& input : stage_inputs) {
        stage_input_dims.push_back(input.getDimensions());
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
            std::vector<uint64_t> effective_dims = stage_inputs[slot].getDimensions();
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

    if (backward_config->upstream_input_names_by_output.has_value()) {
        return buildBackwardOutputs(backward_config->forward_outputs_template,
                                    backward_config->wrt_names,
                                    backward_config->upstream_input_names_by_output.value(),
                                    forward_input_dims);
    }

    return buildBackwardOutputs(backward_config->forward_outputs_template, backward_config->wrt_names, std::nullopt, forward_input_dims);
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

    const auto effectiveRequestedOutputShapes = defaultBackwardRequestedOutputShapes(backward_config, root_inputs, root_values, {});

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
                    if (!backward_config.has_value() && !(ordered_inputs.empty() && output_dims.empty())) {
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
                        if (!backward_config.has_value() && !(ordered_inputs.empty() && output_dims.empty())) {
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
                             .upstream_input_names_by_output =
                                 upstream_input_name.has_value() ? std::optional<std::unordered_map<std::string, std::string>>(
                                                                       std::unordered_map<std::string, std::string>{
                                                                           {outputs_template.outputs[0].name, upstream_input_name.value()},
                                                                       })
                                                                 : std::nullopt,
                         });
}

FusedEquation FusedEquation::compileBackward(const std::vector<std::string>& wrt_names,
                                             const std::unordered_map<std::string, std::string>& upstream_input_names_by_output) const {
    PhysicalOutputs backward_outputs = buildBackwardOutputs(outputs_template, wrt_names, upstream_input_names_by_output);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = wrt_names,
                             .upstream_input_names_by_output = upstream_input_names_by_output,
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

std::shared_ptr<StampedArgMinMax> FusedEquation::stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                Tensor& input,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledStage) {
        throw std::runtime_error("stampArgMinMax requires non-null compiled stage.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error("Input dtype does not match compiled arg-min/max input dtype.");
    }

    const ExprOp reduce_op = compiledStage->op == ExprOp::REDUCE_ARGMIN ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(reduce_op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            compiledStage->input_dtype,
                                                                            compiledStage->compute_dtype,
                                                                            /*output_indices=*/true,
                                                                            input,
                                                                            stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(input.getPlacement(), workspaceDescriptor);
    }

    std::vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);
    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dimensions);
    Tensor output(input.getPlacement(), outputDescriptor);

    const std::vector<uint64_t> unsqueezed_output_dims =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, {});
    TensorDescriptor reductionValueDescriptor(compiledStage->input_dtype, unsqueezed_output_dims);
    Tensor reductionValueOutput(input.getPlacement(), reductionValueDescriptor);

    return make_shared<StampedArgMinMax>(std::move(built), input, output, reductionValueOutput, stream, workspace);
}

std::shared_ptr<StampedReduceMinMaxBackward> FusedEquation::stampReduceMinMaxBackward(
    const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage, Tensor& input, Tensor& grad_output, const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampReduceMinMaxBackward requires non-null compiled stage.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error("Input dtype does not match compiled reduce-min/max-backward input dtype.");
    }
    if (grad_output.getDataType() != compiledStage->output_dtype) {
        throw std::runtime_error("Grad-output dtype does not match compiled reduce-min/max-backward output dtype.");
    }

    const std::vector<uint64_t> expected_grad_dims =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), compiledStage->reduction_axes, compiledStage->squeeze_axes);
    if (!outputDimensionsMatchIgnoringSingletons(grad_output.getDimensions(), expected_grad_dims)) {
        throw std::runtime_error("Grad-output tensor dimensions are incompatible with reduce-min/max-backward stage.");
    }

    const ExprOp reduce_op = compiledStage->op == ExprOp::REDUCE_MIN_BACKWARD ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(reduce_op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            compiledStage->input_dtype,
                                                                            compiledStage->compute_dtype,
                                                                            /*output_indices=*/true,
                                                                            input,
                                                                            stream.getGpuNum());

    Optional<Tensor> workspace = Optional<Tensor>::empty();
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(input.getPlacement(), workspaceDescriptor);
    }

    const std::vector<uint64_t> unsqueezed_output_dims =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, {});

    TensorDescriptor indicesDescriptor(TensorDescriptor::DataType::UINT32, unsqueezed_output_dims);
    Tensor indices(input.getPlacement(), indicesDescriptor);

    TensorDescriptor reductionValueDescriptor(compiledStage->input_dtype, unsqueezed_output_dims);
    Tensor reductionValueOutput(input.getPlacement(), reductionValueDescriptor);

    TensorDescriptor outputDescriptor(compiledStage->output_dtype, input.getDimensions());
    Tensor output(input.getPlacement(), outputDescriptor);

    return make_shared<StampedReduceMinMaxBackward>(
        std::move(built), input, grad_output, output, indices, reductionValueOutput, stream, workspace);
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForInputs(inputs);

    std::unordered_map<uint32_t, Tensor> values = bindRootInputs(inputs);
    const auto effectiveRequestedOutputShapes =
        defaultBackwardRequestedOutputShapes(backward_config, root_inputs, values, requestedOutputShapes);

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

            std::vector<uint64_t> resolvedOutputDims;
            const bool requiresBroadcast = fusedStageRequiresBroadcastLaunch(
                stage, stageInputs, effectiveRequestedOutputShapes, backward_config.has_value(), resolvedOutputDims);

            std::vector<Tensor> stageOutputs;
            stageOutputs.reserve(stage.outputs.size());

            TensorPlacement outputPlacement = pickStageOutputPlacement(stageInputs, values);

            if (!requiresBroadcast) {
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& produced = stage.outputs[output_idx];
                    std::vector<uint64_t> resolved_output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stageInputs);

                    auto requested_it = effectiveRequestedOutputShapes.find(produced.name);
                    const std::vector<uint64_t>* requested_shape =
                        (requested_it != effectiveRequestedOutputShapes.end()) ? &requested_it->second : nullptr;

                    if (requested_shape && !requested_shape->empty()) {
                        if (!backward_config.has_value() && !(stageInputs.empty() && resolved_output_dims.empty())) {
                            verifyRequestedOutputLayout(*requested_shape, resolved_output_dims);
                        }
                    }

                    TensorDescriptor outputDescriptor(
                        stage.flat->output_dtypes[output_idx],
                        (requested_shape && !requested_shape->empty()) ? *requested_shape : resolved_output_dims);
                    stageOutputs.emplace_back(outputPlacement, outputDescriptor);
                }
            } else {
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& produced = stage.outputs[output_idx];

                    std::vector<uint64_t> resolved_output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stageInputs);

                    auto requested_it = effectiveRequestedOutputShapes.find(produced.name);
                    const std::vector<uint64_t>* requested_shape =
                        (requested_it != effectiveRequestedOutputShapes.end()) ? &requested_it->second : nullptr;

                    if (requested_shape && !requested_shape->empty()) {
                        if (!backward_config.has_value() && !(stageInputs.empty() && resolved_output_dims.empty())) {
                            verifyRequestedOutputLayout(*requested_shape, resolved_output_dims);
                        }
                    }

                    TensorDescriptor outputDescriptor(
                        stage.flat->output_dtypes[output_idx],
                        (requested_shape && !requested_shape->empty()) ? *requested_shape : resolved_output_dims);
                    stageOutputs.emplace_back(outputPlacement, outputDescriptor);
                }
            }

            std::shared_ptr<StampedEquation> stamped;
            if (requiresBroadcast) {
                std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, stageInputs);

                std::vector<SpecializedBroadcastGroup> specialized_groups;
                specialized_groups.reserve(groups.size());
                for (const ResolvedBroadcastGroup& group : groups) {
                    specialized_groups.push_back(group.specialized);
                }

                std::shared_ptr<CompiledEquation> specialized_broadcast =
                    EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);

                stamped = stampEquation(specialized_broadcast, stageInputs, stageOutputs, stream);
            } else {
                uint64_t max_output_numel = 0;
                for (const Tensor& output : stageOutputs) {
                    max_output_numel = std::max<uint64_t>(max_output_numel, output.getTotalNumElements());
                }

                stamped = stampEquation(
                    selectFlatCompiledEquation(stage, compiled_outputs->signature, max_output_numel), stageInputs, stageOutputs, stream);
            }

            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                const uint32_t produced_value_id = stage.outputs[i].value_id;
                values[produced_value_id] = stageOutputs[i];
                producer_stage_by_value_id[produced_value_id] = this_stage_idx;
            }

            stampedStages.emplace_back(stamped, std::move(dependency_stage_indices));
        } else if (stage.kind == CompiledExecutionStage::Kind::Reduction) {
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
        } else if (stage.kind == CompiledExecutionStage::Kind::ArgMinMax) {
            if (!stage.arg_minmax) {
                throw std::runtime_error("Missing compiled arg-min/max stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Arg-min/max stage expected exactly one input and one output.");
            }

            Tensor& reductionInput = stageInputs[0];
            auto requested_it = requestedOutputShapes.find(stage.outputs[0].name);
            std::vector<uint64_t> requested_shape;
            if (requested_it != requestedOutputShapes.end())
                requested_shape = requested_it->second;
            std::shared_ptr<StampedArgMinMax> stamped = stampArgMinMax(stage.arg_minmax, reductionInput, stream, requested_shape);

            const uint32_t produced_value_id = stage.outputs[0].value_id;
            values[produced_value_id] = stamped->getOutputTensor();
            producer_stage_by_value_id[produced_value_id] = this_stage_idx;

            stampedStages.emplace_back(stamped, std::move(dependency_stage_indices));
        } else if (stage.kind == CompiledExecutionStage::Kind::ReduceMinMaxBackward) {
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("Missing compiled reduce-min/max-backward stage.");
            }

            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduce-min/max-backward stage expected exactly two inputs and one output.");
            }

            Tensor& reductionInput = stageInputs[0];
            Tensor& gradOutput = stageInputs[1];
            std::shared_ptr<StampedReduceMinMaxBackward> stamped =
                stampReduceMinMaxBackward(stage.reduce_minmax_backward, reductionInput, gradOutput, stream);

            const uint32_t produced_value_id = stage.outputs[0].value_id;
            values[produced_value_id] = stamped->getOutputTensor();
            producer_stage_by_value_id[produced_value_id] = this_stage_idx;

            stampedStages.emplace_back(stamped, std::move(dependency_stage_indices));
        } else {
            throw std::runtime_error("Unknown compiled execution stage kind in FusedEquation::stamp.");
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
