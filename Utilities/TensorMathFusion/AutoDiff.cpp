#include "Utilities/TensorMathFusion/AutoDiff.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
namespace {
uint32_t cloneForwardSubtree(const PhysicalExpression& src,
                             uint32_t src_node_index,
                             PhysicalExpression& dst,
                             std::unordered_map<uint32_t, uint32_t>& old_to_new) {
    auto it = old_to_new.find(src_node_index);
    if (it != old_to_new.end()) {
        return it->second;
    }

    if (src_node_index >= src.nodes.size()) {
        throw std::runtime_error("cloneForwardSubtree source node index out of range.");
    }

    const ExprNode& src_node = src.nodes[src_node_index];
    ExprNode new_node = src_node;

    if (Expression::isUnaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: unary node missing lhs.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new);
        new_node.rhs = UINT32_MAX;
    } else if (Expression::isBinaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX || src_node.rhs == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: binary node missing child.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new);
        new_node.rhs = cloneForwardSubtree(src, src_node.rhs, dst, old_to_new);
    } else if (Expression::isLeafOp(src_node.op)) {
        // Nothing to recurse into.
    } else {
        throw std::runtime_error("Unsupported op while cloning forward subtree for autodiff: " + std::to_string((int)src_node.op));
    }

    const uint32_t new_index = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(new_node));
    old_to_new[src_node_index] = new_index;
    return new_index;
}

class BackwardGraphBuilder {
   public:
    explicit BackwardGraphBuilder(const PhysicalExpression& forward_expr) : forward_expr(forward_expr) {
        grad_expr.inputs = forward_expr.inputs;
    }

    uint32_t input(const std::string& name, Optional<TensorDescriptor::DataType> as_type = Optional<TensorDescriptor::DataType>::empty()) {
        ExprNode node{};
        node.op = ExprOp::INPUT;
        node.input_slot = grad_expr.getOrCreateInputSlot(name);
        if (as_type.isPresent()) {
            node.output_dtype = as_type.get();
        }
        return push(std::move(node));
    }

    uint32_t scalar(double value) {
        ExprNode node{};
        node.op = ExprOp::SCALAR_FP;
        node.scalar_fp = value;
        return push(std::move(node));
    }

    uint32_t unary(ExprOp op, uint32_t lhs) {
        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        return push(std::move(node));
    }

    uint32_t binary(ExprOp op, uint32_t lhs, uint32_t rhs) {
        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.rhs = rhs;
        return push(std::move(node));
    }

    uint32_t reduction(ExprOp op,
                       uint32_t lhs,
                       const std::vector<uint64_t>& reduction_axes,
                       const std::vector<uint64_t>& squeeze_axes,
                       Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) {
        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.reduction_axes = reduction_axes;
        node.squeeze_axes = squeeze_axes;
        node.compute_dtype = compute_dtype;
        return push(std::move(node));
    }

    uint32_t neg(uint32_t value) { return unary(ExprOp::NEG, value); }
    uint32_t add(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::ADD, lhs, rhs); }
    uint32_t sub(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::SUB, lhs, rhs); }
    uint32_t mul(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::MUL, lhs, rhs); }
    uint32_t div(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::DIV, lhs, rhs); }

    uint32_t unsqueeze(uint32_t value, const std::vector<uint64_t>& unsqueeze_axes) {
        ExprNode node{};
        node.op = ExprOp::UNSQUEEZE;
        node.lhs = value;
        node.unsqueeze_axes = unsqueeze_axes;
        return push(std::move(node));
    }

    uint32_t squeeze(uint32_t value, const std::vector<uint64_t>& squeeze_axes) {
        ExprNode node{};
        node.op = ExprOp::SQUEEZE;
        node.lhs = value;
        node.squeeze_axes = squeeze_axes;
        return push(std::move(node));
    }

    uint32_t cloneForward(uint32_t forward_node_index) {
        return cloneForwardSubtree(forward_expr, forward_node_index, grad_expr, forward_to_grad_node_map);
    }

    void addContribution(uint32_t forward_node_index, uint32_t contrib_root) {
        if (forward_node_index >= node_grads.size()) {
            throw std::runtime_error("Autodiff addContribution node index out of range.");
        }

        // std::cerr << "[AUTODIFF] addContribution"
        //           << " forward_node_index=" << forward_node_index << " contrib_root=" << contrib_root << std::endl;

        if (node_grads[forward_node_index].has_value()) {
            node_grads[forward_node_index] = add(node_grads[forward_node_index].value(), contrib_root);
        } else {
            node_grads[forward_node_index] = contrib_root;
        }
    }

    void initializeAdjoints() { node_grads.assign(forward_expr.nodes.size(), std::nullopt); }

    const std::optional<uint32_t>& gradOf(uint32_t forward_node_index) const { return node_grads.at(forward_node_index); }

    PhysicalExpression takeExpression() { return std::move(grad_expr); }

   private:
    uint32_t push(ExprNode node) {
        const uint32_t idx = static_cast<uint32_t>(grad_expr.nodes.size());
        grad_expr.nodes.push_back(std::move(node));
        return idx;
    }

    const PhysicalExpression& forward_expr;
    PhysicalExpression grad_expr;
    std::unordered_map<uint32_t, uint32_t> forward_to_grad_node_map;
    std::vector<std::optional<uint32_t>> node_grads;
};

std::vector<std::string> normalizeWrtNames(const PhysicalExpression& forward_expr, const std::vector<std::string>& wrt_names) {
    if (wrt_names.empty()) {
        std::vector<std::string> all_names;
        all_names.reserve(forward_expr.inputs.size());
        for (const NamedInput& input : forward_expr.inputs) {
            all_names.push_back(input.name);
        }
        return all_names;
    }

    std::vector<std::string> normalized;
    normalized.reserve(wrt_names.size());
    std::unordered_set<std::string> seen;
    for (const std::string& name : wrt_names) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("Duplicate wrt input name passed to compileBackward: " + name);
        }

        bool found = false;
        for (const NamedInput& input : forward_expr.inputs) {
            if (input.name == name) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Requested gradient for unknown input: " + name);
        }

        normalized.push_back(name);
    }

    return normalized;
}

std::optional<std::unordered_map<std::string, std::string>> normalizeUpstreamInputNamesByOutput(
    const PhysicalOutputs& forward_outputs, const std::optional<std::string>& upstream_input_name) {
    if (!upstream_input_name.has_value()) {
        return std::nullopt;
    }

    if (forward_outputs.outputs.size() != 1) {
        throw std::runtime_error("compileBackward single upstream input name overload only supports exactly one forward output.");
    }

    if (upstream_input_name->empty()) {
        throw std::runtime_error("compileBackward explicit upstream input name cannot be empty.");
    }

    const PhysicalExpression& forward_expr = *forward_outputs.expr;
    for (const NamedInput& input : forward_expr.inputs) {
        if (input.name == upstream_input_name.value()) {
            throw std::runtime_error("compileBackward explicit upstream input name collides with an existing forward input: " +
                                     upstream_input_name.value());
        }
    }

    return std::unordered_map<std::string, std::string>{{forward_outputs.outputs[0].name, upstream_input_name.value()}};
}

std::optional<std::unordered_map<std::string, std::string>> normalizeUpstreamInputNamesByOutput(
    const PhysicalOutputs& forward_outputs, const std::unordered_map<std::string, std::string>& upstream_input_names_by_output) {
    if (!forward_outputs.expr) {
        throw std::runtime_error("compileBackward upstream-input validation requires non-null forward expr.");
    }

    const PhysicalExpression& forward_expr = *forward_outputs.expr;
    std::unordered_set<std::string> valid_output_names;
    valid_output_names.reserve(forward_outputs.outputs.size());
    for (const NamedOutput& output : forward_outputs.outputs) {
        valid_output_names.insert(output.name);
    }

    for (const auto& [output_name, upstream_name] : upstream_input_names_by_output) {
        if (!valid_output_names.contains(output_name)) {
            throw std::runtime_error("compileBackward explicit upstream map contains unknown forward output: " + output_name);
        }
        if (upstream_name.empty()) {
            throw std::runtime_error("compileBackward explicit upstream input name cannot be empty for output: " + output_name);
        }
        for (const NamedInput& input : forward_expr.inputs) {
            if (input.name == upstream_name) {
                throw std::runtime_error("compileBackward explicit upstream input name collides with an existing forward input: " +
                                         upstream_name);
            }
        }
    }

    return upstream_input_names_by_output;
}

bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        throw std::runtime_error("resolveLayoutFromDims requires at least one input shape.");
    }

    uint64_t maxRank = 0;
    for (const std::vector<uint64_t>& dims : inputs) {
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    outputDimensions.assign(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const std::vector<uint64_t>& inDims : inputs) {
            const uint64_t rankDiff = maxRank - inDims.size();
            const uint64_t dim = (axis < rankDiff) ? 1 : inDims[axis - rankDiff];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
            } else if (resolvedDim != dim) {
                throw std::runtime_error("Autodiff shape inference encountered non-broadcast-compatible inputs.");
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    bool requiresBroadcast = false;
    for (const std::vector<uint64_t>& inDims : inputs) {
        if (inDims.size() != maxRank) {
            requiresBroadcast = true;
            break;
        }
        for (uint64_t axis = 0; axis < maxRank; ++axis) {
            if (inDims[axis] != outputDimensions[axis]) {
                requiresBroadcast = true;
                break;
            }
        }
        if (requiresBroadcast) {
            break;
        }
    }

    return requiresBroadcast;
}

std::vector<uint64_t> applySqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes) {
    if (squeeze_axes.empty()) {
        return input_dims;
    }

    std::vector<uint64_t> normalized = squeeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

    if (normalized.size() == 1 && normalized[0] == UINT64_MAX) {
        std::vector<uint64_t> out_dims;
        out_dims.reserve(input_dims.size());
        for (uint64_t dim : input_dims) {
            if (dim != 1) {
                out_dims.push_back(dim);
            }
        }
        return out_dims;
    }

    std::vector<uint64_t> out_dims;
    out_dims.reserve(input_dims.size());
    size_t next_axis_i = 0;
    uint64_t next_axis = normalized.empty() ? UINT64_MAX : normalized[0];
    for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
        if (next_axis_i < normalized.size() && axis == next_axis) {
            if (input_dims[axis] != 1) {
                throw std::runtime_error("inferForwardNodeDims squeeze axes must refer to singleton dimensions.");
            }
            ++next_axis_i;
            next_axis = next_axis_i < normalized.size() ? normalized[next_axis_i] : UINT64_MAX;
            continue;
        }
        out_dims.push_back(input_dims[axis]);
    }

    if (next_axis_i != normalized.size()) {
        throw std::runtime_error("inferForwardNodeDims squeeze axes are invalid for the input rank.");
    }

    return out_dims;
}

// std::vector<uint64_t> normalizeSqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>&
// squeeze_axes) {
//     if (squeeze_axes.empty()) {
//         return {};
//     }
//
//     std::vector<uint64_t> normalized = squeeze_axes;
//     std::sort(normalized.begin(), normalized.end());
//     normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
//
//     if (normalized.size() == 1 && normalized[0] == UINT64_MAX) {
//         std::vector<uint64_t> actual_axes;
//         actual_axes.reserve(input_dims.size());
//         for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
//             if (input_dims[axis] == 1) {
//                 actual_axes.push_back(axis);
//             }
//         }
//         return actual_axes;
//     }
//
//     for (uint64_t axis : normalized) {
//         if (axis >= input_dims.size()) {
//             throw std::runtime_error("Autodiff squeeze axes are out of range for the input rank.");
//         }
//         if (input_dims[axis] != 1) {
//             throw std::runtime_error("Autodiff squeeze axes must refer to singleton dimensions.");
//         }
//     }
//
//     return normalized;
// }
//
// std::vector<uint64_t> normalizeUnsqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims,
//                                                          const std::vector<uint64_t>& unsqueeze_axes) {
//     const uint64_t input_rank = input_dims.size();
//
//     if (unsqueeze_axes.empty()) {
//         return {};
//     }
//
//     std::vector<uint64_t> normalized = unsqueeze_axes;
//     std::sort(normalized.begin(), normalized.end());
//     normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
//
//     const uint64_t output_rank = input_rank + normalized.size();
//
//     for (uint64_t axis : normalized) {
//         if (axis == UINT64_MAX) {
//             throw std::runtime_error("Autodiff unsqueeze axes must be explicit.");
//         }
//         if (axis >= output_rank) {
//             throw std::runtime_error("Autodiff unsqueeze axes are out of range for the output rank.");
//         }
//     }
//
//     return normalized;
// }
//
// std::vector<uint64_t> normalizedReductionUnsqueezeAxes(const std::vector<uint64_t>& input_dims,
//                                                        const std::vector<uint64_t>& reduction_axes,
//                                                        const std::vector<uint64_t>& squeeze_axes) {
//     const std::vector<uint64_t> unsqueezed_output_dims = StampedEquation::computeReductionOutputDims(input_dims, reduction_axes, {});
//     return normalizeSqueezeAxesForInputDims(unsqueezed_output_dims, squeeze_axes);
// }

std::vector<std::vector<uint64_t>> inferForwardNodeDims(
    const PhysicalExpression& forward_expr,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    if (!forward_input_dims.has_value()) {
        return {};
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> input_dims_by_slot;
    for (const NamedInput& input : forward_expr.inputs) {
        auto it = forward_input_dims->find(input.name);
        if (it == forward_input_dims->end()) {
            throw std::runtime_error("Missing forward input dimensions for autodiff shape specialization input: " + input.name);
        }
        input_dims_by_slot[input.slot] = it->second;
    }

    std::vector<std::vector<uint64_t>> node_dims(forward_expr.nodes.size());
    for (size_t i = 0; i < forward_expr.nodes.size(); ++i) {
        const ExprNode& node = forward_expr.nodes[i];
        switch (node.op) {
            case ExprOp::INPUT: {
                auto it = input_dims_by_slot.find(node.input_slot);
                if (it == input_dims_by_slot.end()) {
                    throw std::runtime_error("Autodiff shape inference missing INPUT dims for slot " + std::to_string(node.input_slot) +
                                             ".");
                }
                node_dims[i] = it->second;
                break;
            }
            case ExprOp::SCALAR_FP:
                node_dims[i] = {};
                break;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::MIN:
            case ExprOp::MAX: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                }
                if (!node_dims[node.rhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                }
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
                const std::vector<uint64_t>& axes = node.unsqueeze_axes;
                std::vector<uint64_t> out_dims;
                out_dims.reserve(lhs_dims.size() + axes.size());
                const uint64_t output_rank = static_cast<uint64_t>(lhs_dims.size() + axes.size());

                size_t lhs_i = 0;
                size_t axis_i = 0;
                for (uint64_t out_axis = 0; out_axis < output_rank; ++out_axis) {
                    if (axis_i < axes.size() && axes[axis_i] == out_axis) {
                        out_dims.push_back(1);
                        ++axis_i;
                    } else {
                        if (lhs_i >= lhs_dims.size()) {
                            throw std::runtime_error("inferForwardNodeDims unsqueeze axes are out of range.");
                        }
                        out_dims.push_back(lhs_dims[lhs_i++]);
                    }
                }
                if (lhs_i != lhs_dims.size() || axis_i != axes.size()) {
                    throw std::runtime_error("inferForwardNodeDims unsqueeze axes are invalid for the input rank.");
                }
                node_dims[i] = std::move(out_dims);
                break;
            }
            case ExprOp::SQUEEZE:
                node_dims[i] = applySqueezeDims(node_dims[node.lhs], node.squeeze_axes);
                break;
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
                node_dims[i] = StampedEquation::computeReductionOutputDims(node_dims[node.lhs], node.reduction_axes, node.squeeze_axes);
                break;
            default:
                throw std::runtime_error("inferForwardNodeDims encountered unknown ExprOp.");
        }
    }

    return node_dims;
}

uint32_t sumToShape(BackwardGraphBuilder& builder,
                    uint32_t contrib,
                    const std::vector<uint64_t>& contrib_dims,
                    const std::vector<uint64_t>& target_dims) {
    if (contrib_dims == target_dims) {
        return contrib;
    }

    if (contrib_dims.empty() || target_dims.empty()) {
        throw std::runtime_error("Phase-1 autodiff broadcast backward requires tensor-valued shapes.");
    }

    if (contrib_dims.size() < target_dims.size()) {
        throw std::runtime_error("Autodiff cannot sum a contribution to a higher-rank target shape.");
    }

    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;

    int64_t contrib_axis = static_cast<int64_t>(contrib_dims.size()) - 1;
    int64_t target_axis = static_cast<int64_t>(target_dims.size()) - 1;

    while (contrib_axis >= 0 && target_axis >= 0) {
        const uint64_t contrib_dim = contrib_dims[static_cast<size_t>(contrib_axis)];
        const uint64_t target_dim = target_dims[static_cast<size_t>(target_axis)];

        if (contrib_dim == target_dim) {
            --contrib_axis;
            --target_axis;
            continue;
        }

        if (contrib_dim == 1) {
            reduction_axes.push_back(static_cast<uint64_t>(contrib_axis));
            squeeze_axes.push_back(static_cast<uint64_t>(contrib_axis));
            --contrib_axis;
            continue;
        }

        if (target_dim == 1) {
            reduction_axes.push_back(static_cast<uint64_t>(contrib_axis));
            --contrib_axis;
            --target_axis;
            continue;
        }

        throw std::runtime_error("Autodiff broadcast backward found incompatible target shape while summing to input shape.");
    }

    while (contrib_axis >= 0) {
        reduction_axes.push_back(static_cast<uint64_t>(contrib_axis));
        squeeze_axes.push_back(static_cast<uint64_t>(contrib_axis));
        --contrib_axis;
    }

    if (target_axis >= 0) {
        auto formatDims = [](const std::vector<uint64_t>& dims) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << dims[i];
            }
            oss << "]";
            return oss.str();
        };

        auto formatAxes = [](const std::vector<uint64_t>& axes) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < axes.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << axes[i];
            }
            oss << "]";
            return oss.str();
        };

        throw std::runtime_error(
            "Autodiff could not match all target axes while summing to input shape. "
            "contribution shape = " +
            formatDims(contrib_dims) + ", target shape = " + formatDims(target_dims) + ", reduction_axes = " + formatAxes(reduction_axes) +
            ", squeeze_axes = " + formatAxes(squeeze_axes) + ".");
    }

    if (reduction_axes.empty()) {
        return contrib;
    }

    std::sort(reduction_axes.begin(), reduction_axes.end());
    std::sort(squeeze_axes.begin(), squeeze_axes.end());

    bool has_numeric_reduction = false;
    for (uint64_t axis : reduction_axes) {
        if (axis >= contrib_dims.size()) {
            throw std::runtime_error("Autodiff sumToShape produced reduction axis out of range.");
        }
        if (contrib_dims[axis] != 1) {
            has_numeric_reduction = true;
            break;
        }
    }

    if (!has_numeric_reduction) {
        return builder.squeeze(contrib, squeeze_axes);
    }

    return builder.reduction(ExprOp::REDUCE_SUM, contrib, reduction_axes, squeeze_axes);
}

uint64_t reductionElementCount(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& reduction_axes) {
    if (input_dims.empty()) {
        throw std::runtime_error("Phase-1 autodiff reduce_mean backward requires tensor-valued input shapes.");
    }

    uint64_t count = 1;
    if (reduction_axes.empty()) {
        for (uint64_t dim : input_dims) {
            count *= dim;
        }
        return count;
    }

    for (uint64_t axis : reduction_axes) {
        if (axis >= input_dims.size()) {
            throw std::runtime_error("Phase-1 autodiff reduce_mean backward saw reduction axis out of range.");
        }
        count *= input_dims[axis];
    }
    return count;
}

}  // namespace

static std::string dbgDims(const std::vector<uint64_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i)
            oss << ", ";
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

PhysicalOutputs buildBackwardOutputsImpl(const PhysicalOutputs& forward_outputs,
                                         const std::vector<std::string>& wrt_names,
                                         const std::optional<std::unordered_map<std::string, std::string>>& upstream_input_names_by_output,
                                         const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    if (!forward_outputs.expr) {
        throw std::runtime_error("buildBackwardOutputs requires non-null forward_outputs.expr.");
    }

    const PhysicalExpression& forward_expr = *forward_outputs.expr;
    for (const NamedOutput& forward_output : forward_outputs.outputs) {
        if (forward_output.node_idx >= forward_expr.nodes.size()) {
            throw std::runtime_error("Forward output node index is out of range in buildBackwardOutputs.");
        }
    }

    const std::vector<std::string> normalized_wrt = normalizeWrtNames(forward_expr, wrt_names);
    const std::vector<std::vector<uint64_t>> forward_node_dims = inferForwardNodeDims(forward_expr, forward_input_dims);
    const bool has_forward_dims = !forward_node_dims.empty();

    if (forward_outputs.outputs.size() > 1 && !upstream_input_names_by_output.has_value()) {
        throw std::runtime_error(
            "buildBackwardOutputs for multi-output forward equations requires explicit upstream input names for each output.");
    }

    if (upstream_input_names_by_output.has_value()) {
        for (const NamedOutput& forward_output : forward_outputs.outputs) {
            if (!upstream_input_names_by_output->contains(forward_output.name)) {
                throw std::runtime_error("buildBackwardOutputs missing explicit upstream input name for forward output: " +
                                         forward_output.name);
            }
        }
    }

    BackwardGraphBuilder builder(forward_expr);
    builder.initializeAdjoints();

    for (const NamedOutput& forward_output : forward_outputs.outputs) {
        uint32_t output_seed = UINT32_MAX;
        if (upstream_input_names_by_output.has_value()) {
            output_seed = builder.input(upstream_input_names_by_output->at(forward_output.name));
        } else {
            output_seed = builder.scalar(1.0);
        }
        builder.addContribution(forward_output.node_idx, output_seed);
    }

    auto addContributionToChild = [&](uint32_t child_idx, uint32_t contrib, const std::vector<uint64_t>& contrib_dims) {
        uint32_t adjusted_contrib = contrib;
        if (has_forward_dims) {
            const std::vector<uint64_t>& child_dims = forward_node_dims.at(child_idx);
            if (!child_dims.empty()) {
                adjusted_contrib = sumToShape(builder, contrib, contrib_dims, child_dims);
            }
        }
        builder.addContribution(child_idx, adjusted_contrib);
    };

    auto shapeGradLikeNodeOutput =
        [&](uint32_t grad_value, uint32_t forward_node_idx, const std::vector<uint64_t>& forward_node_output_dims) -> uint32_t {
        if (!has_forward_dims || forward_node_output_dims.empty()) {
            return grad_value;
        }

        const uint32_t out = builder.cloneForward(forward_node_idx);
        return builder.add(builder.mul(out, builder.scalar(0.0)), grad_value);
    };

    for (int64_t node_idx = static_cast<int64_t>(forward_expr.nodes.size()) - 1; node_idx >= 0; --node_idx) {
        const auto& grad_opt = builder.gradOf(static_cast<uint32_t>(node_idx));
        if (!grad_opt.has_value()) {
            continue;
        }

        const uint32_t grad = grad_opt.value();
        const ExprNode& node = forward_expr.nodes[static_cast<size_t>(node_idx)];
        const std::vector<uint64_t> node_dims =
            has_forward_dims ? forward_node_dims.at(static_cast<size_t>(node_idx)) : std::vector<uint64_t>{};

        switch (node.op) {
            case ExprOp::INPUT:
            case ExprOp::SCALAR_FP:
                break;

            case ExprOp::ADD:
                addContributionToChild(node.lhs, grad, node_dims);
                addContributionToChild(node.rhs, grad, node_dims);
                break;

            case ExprOp::SUB:
                addContributionToChild(node.lhs, grad, node_dims);
                addContributionToChild(node.rhs, builder.neg(grad), node_dims);
                break;

            case ExprOp::MUL: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t rhs = builder.cloneForward(node.rhs);
                addContributionToChild(node.lhs, builder.mul(grad, rhs), node_dims);
                addContributionToChild(node.rhs, builder.mul(grad, lhs), node_dims);
                break;
            }

            case ExprOp::DIV: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t rhs = builder.cloneForward(node.rhs);
                const uint32_t rhs_sq = builder.mul(rhs, rhs);
                addContributionToChild(node.lhs, builder.div(grad, rhs), node_dims);
                addContributionToChild(node.rhs, builder.neg(builder.div(builder.mul(grad, lhs), rhs_sq)), node_dims);
                break;
            }

            case ExprOp::NEG:
                addContributionToChild(node.lhs, builder.neg(grad), node_dims);
                break;

            case ExprOp::UNSQUEEZE: {
                if (has_forward_dims) {
                    const std::vector<uint64_t>& lhs_dims = forward_node_dims.at(node.lhs);
                    const std::vector<uint64_t> actual_unsqueeze_axes = normalizeUnsqueezeAxesForInputDims(lhs_dims, node.unsqueeze_axes);
                    const uint32_t squeezed_grad = builder.squeeze(grad, actual_unsqueeze_axes);
                    builder.addContribution(node.lhs, squeezed_grad);

                    // std::cerr << "[AUTODIFF] builder.unsqueeze"
                    //           << " input_node=" << node_idx << " actual_unsqueeze_axes=" << dbgDims(actual_unsqueeze_axes)
                    //           << " node.unsqueeze_axes=" << dbgDims(node.unsqueeze_axes) << std::endl;
                } else {
                    const uint32_t squeezed_grad = builder.squeeze(grad, node.unsqueeze_axes);
                    builder.addContribution(node.lhs, squeezed_grad);
                }
                break;
            }

            case ExprOp::SQUEEZE: {
                // std::cerr << "[AUTODIFF] SQUEEZE backward"
                //           << " node=" << node_idx << " lhs=" << node.lhs << " grad_node=" << grad << " node_dims=" << dbgDims(node_dims)
                //           << " raw_squeeze_axes=" << dbgDims(node.squeeze_axes) << std::endl;

                if (has_forward_dims) {
                    const std::vector<uint64_t>& lhs_dims = forward_node_dims.at(node.lhs);
                    const std::vector<uint64_t> actual_squeeze_axes = normalizeSqueezeAxesForInputDims(lhs_dims, node.squeeze_axes);

                    // std::cerr << "[AUTODIFF] SQUEEZE normalized"
                    //           << " lhs_dims=" << dbgDims(lhs_dims) << " actual_squeeze_axes=" << dbgDims(actual_squeeze_axes) <<
                    //           std::endl;

                    const uint32_t unsqueezed_grad = builder.unsqueeze(grad, actual_squeeze_axes);

                    // std::cerr << "[AUTODIFF] SQUEEZE nodes"
                    //           << " incoming_grad_node=" << grad << " unsqueezed_grad_node=" << unsqueezed_grad
                    //           << " expected_incoming_grad_dims=" << dbgDims(node_dims)
                    //           << " expected_unsqueezed_grad_dims=" << dbgDims(lhs_dims) << std::endl;

                    builder.addContribution(node.lhs, unsqueezed_grad);

                    // const uint32_t lhs_grad_after_squeeze = builder.gradOf(node.lhs).value();
                    // std::cerr << "[AUTODIFF] SQUEEZE stored lhs grad"
                    //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_squeeze
                    //           << " expected_lhs_dims=" << dbgDims(lhs_dims) << std::endl;
                } else {
                    const uint32_t unsqueezed_grad = builder.unsqueeze(grad, node.squeeze_axes);

                    // std::cerr << "[AUTODIFF] SQUEEZE nodes (no forward dims)"
                    //           << " incoming_grad_node=" << grad << " unsqueezed_grad_node=" << unsqueezed_grad << std::endl;

                    builder.addContribution(node.lhs, unsqueezed_grad);

                    // const uint32_t lhs_grad_after_squeeze = builder.gradOf(node.lhs).value();
                    // std::cerr << "[AUTODIFF] SQUEEZE stored lhs grad (no forward dims)"
                    //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_squeeze << std::endl;
                }
                break;
            }

            case ExprOp::EXP: {
                const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                addContributionToChild(node.lhs, builder.mul(grad, out), node_dims);
                break;
            }

            case ExprOp::EXP2: {
                const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                const uint32_t scale = builder.scalar(std::log(2.0));
                addContributionToChild(node.lhs, builder.mul(grad, builder.mul(out, scale)), node_dims);
                break;
            }

            case ExprOp::EXP10: {
                const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                const uint32_t scale = builder.scalar(std::log(10.0));
                addContributionToChild(node.lhs, builder.mul(grad, builder.mul(out, scale)), node_dims);
                break;
            }

            case ExprOp::LN: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                addContributionToChild(node.lhs, builder.div(grad, lhs), node_dims);
                break;
            }

            case ExprOp::LOG2: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t denom = builder.mul(lhs, builder.scalar(std::log(2.0)));
                addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                break;
            }

            case ExprOp::LOG10: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t denom = builder.mul(lhs, builder.scalar(std::log(10.0)));
                addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                break;
            }

            case ExprOp::SQRT: {
                const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                const uint32_t denom = builder.mul(builder.scalar(2.0), out);
                addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                break;
            }

            case ExprOp::POW: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t rhs = builder.cloneForward(node.rhs);
                const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                const uint32_t rhs_minus_one = builder.sub(rhs, builder.scalar(1.0));
                const uint32_t lhs_pow_rhs_minus_one = builder.binary(ExprOp::POW, lhs, rhs_minus_one);
                addContributionToChild(node.lhs, builder.mul(grad, builder.mul(rhs, lhs_pow_rhs_minus_one)), node_dims);
                addContributionToChild(node.rhs, builder.mul(grad, builder.mul(out, builder.unary(ExprOp::LN, lhs))), node_dims);
                break;
            }

            case ExprOp::REDUCE_SUM: {
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                if (has_forward_dims && !node.squeeze_axes.empty()) {
                    const std::vector<uint64_t> unsqueeze_axes =
                        normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                    grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                }

                // std::cerr << "[AUTODIFF] REDUCE_SUM backward"
                //           << " node=" << node_idx << " lhs=" << node.lhs << " grad_node=" << grad << " node_dims=" << dbgDims(node_dims)
                //           << " lhs_dims=" << dbgDims(lhs_dims) << " reduction_axes=" << dbgDims(node.reduction_axes)
                //           << " squeeze_axes=" << dbgDims(node.squeeze_axes) << std::endl;

                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t expanded_grad = builder.add(builder.mul(lhs, builder.scalar(0.0)), grad_before_expand);

                // std::cerr << "[AUTODIFF] REDUCE_SUM nodes"
                //           << " grad_before_expand=" << grad_before_expand << " expanded_grad=" << expanded_grad << std::endl;

                addContributionToChild(node.lhs, expanded_grad, lhs_dims);

                // const uint32_t lhs_grad_after_reduce = builder.gradOf(node.lhs).value();
                // std::cerr << "[AUTODIFF] REDUCE_SUM stored lhs grad"
                //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_reduce << " expected_lhs_dims=" <<
                //           dbgDims(lhs_dims)
                //           << std::endl;

                break;
            }

            case ExprOp::REDUCE_AVG: {
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                if (has_forward_dims && !node.squeeze_axes.empty()) {
                    const std::vector<uint64_t> unsqueeze_axes =
                        normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                    grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                }

                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t expanded_grad = builder.add(builder.mul(lhs, builder.scalar(0.0)), grad_before_expand);

                uint32_t scaled_grad = expanded_grad;
                if (has_forward_dims) {
                    const uint64_t count = reductionElementCount(lhs_dims, node.reduction_axes);
                    scaled_grad = builder.div(expanded_grad, builder.scalar(static_cast<double>(count)));
                }

                addContributionToChild(node.lhs, scaled_grad, lhs_dims);
                break;
            }

            case ExprOp::REDUCE_NORM2: {
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));

                if (has_forward_dims && !node.squeeze_axes.empty()) {
                    const std::vector<uint64_t> unsqueeze_axes =
                        normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                    grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                    out = builder.unsqueeze(out, unsqueeze_axes);
                }

                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t expanded_grad = builder.add(builder.mul(lhs, builder.scalar(0.0)), grad_before_expand);
                const uint32_t scaled = builder.div(builder.mul(expanded_grad, lhs), out);
                addContributionToChild(node.lhs, scaled, lhs_dims);
                break;
            }

            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_NORM1:
                throw std::runtime_error("Phase-1 autodiff does not yet support backward for op " + opName(node.op) + ".");

            default:
                throw std::runtime_error("buildBackwardOutputs encountered unknown ExprOp.");
        }
    }

    std::unordered_map<uint32_t, uint32_t> first_input_node_by_slot;
    for (uint32_t i = 0; i < forward_expr.nodes.size(); ++i) {
        const ExprNode& node = forward_expr.nodes[i];
        if (node.op == ExprOp::INPUT && !first_input_node_by_slot.contains(node.input_slot)) {
            first_input_node_by_slot.emplace(node.input_slot, i);
        }
    }

    PhysicalOutputs backward_outputs;
    backward_outputs.expr = std::make_shared<PhysicalExpression>(builder.takeExpression());
    backward_outputs.outputs.reserve(normalized_wrt.size());

    for (const std::string& wrt_name : normalized_wrt) {
        uint32_t slot = UINT32_MAX;
        for (const NamedInput& input : forward_expr.inputs) {
            if (input.name == wrt_name) {
                slot = input.slot;
                break;
            }
        }
        if (slot == UINT32_MAX) {
            throw std::runtime_error("Requested gradient input slot not found for input: " + wrt_name);
        }

        auto first_it = first_input_node_by_slot.find(slot);
        if (first_it == first_input_node_by_slot.end()) {
            throw std::runtime_error("No INPUT node found for requested gradient input: " + wrt_name);
        }

        std::optional<uint32_t> total_grad;
        for (uint32_t i = 0; i < forward_expr.nodes.size(); ++i) {
            const ExprNode& node = forward_expr.nodes[i];
            if (node.op != ExprOp::INPUT || node.input_slot != slot) {
                continue;
            }

            const auto& grad_opt = builder.gradOf(i);
            if (!grad_opt.has_value()) {
                continue;
            }

            if (total_grad.has_value()) {
                ExprNode sum_node{};
                sum_node.op = ExprOp::ADD;
                sum_node.lhs = total_grad.value();
                sum_node.rhs = grad_opt.value();
                total_grad = static_cast<uint32_t>(backward_outputs.expr->nodes.size());
                backward_outputs.expr->nodes.push_back(std::move(sum_node));
            } else {
                total_grad = grad_opt.value();
            }
        }

        if (!total_grad.has_value()) {
            std::unordered_map<uint32_t, uint32_t> zero_clone_memo;
            const uint32_t input_clone = cloneForwardSubtree(forward_expr, first_it->second, *backward_outputs.expr, zero_clone_memo);
            ExprNode zero_node{};
            zero_node.op = ExprOp::SCALAR_FP;
            zero_node.scalar_fp = 0.0;
            const uint32_t zero_idx = static_cast<uint32_t>(backward_outputs.expr->nodes.size());
            backward_outputs.expr->nodes.push_back(std::move(zero_node));

            ExprNode mul_node{};
            mul_node.op = ExprOp::MUL;
            mul_node.lhs = input_clone;
            mul_node.rhs = zero_idx;
            total_grad = static_cast<uint32_t>(backward_outputs.expr->nodes.size());
            backward_outputs.expr->nodes.push_back(std::move(mul_node));
        }

        backward_outputs.outputs.push_back(NamedOutput{
            .name = wrt_name + "_grad",
            .node_idx = total_grad.value(),
        });
    }

    return backward_outputs;
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::optional<std::string>& upstream_input_name,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    return buildBackwardOutputsImpl(
        forward_outputs, wrt_names, normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_name), forward_input_dims);
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_names_by_output),
                                    forward_input_dims);
}

}  // namespace ThorImplementation
