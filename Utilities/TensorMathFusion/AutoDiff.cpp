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

    uint32_t cloneForward(uint32_t forward_node_index) {
        return cloneForwardSubtree(forward_expr, forward_node_index, grad_expr, forward_to_grad_node_map);
    }

    void addContribution(uint32_t forward_node_index, uint32_t contrib_root) {
        if (forward_node_index >= node_grads.size()) {
            throw std::runtime_error("Autodiff addContribution node index out of range.");
        }

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

std::optional<std::string> normalizeUpstreamInputName(const PhysicalExpression& forward_expr,
                                                      const std::optional<std::string>& upstream_input_name) {
    if (!upstream_input_name.has_value()) {
        return std::nullopt;
    }

    if (upstream_input_name->empty()) {
        throw std::runtime_error("compileBackward explicit upstream input name cannot be empty.");
    }

    for (const NamedInput& input : forward_expr.inputs) {
        if (input.name == upstream_input_name.value()) {
            throw std::runtime_error("compileBackward explicit upstream input name collides with an existing forward input: " +
                                     upstream_input_name.value());
        }
    }

    return upstream_input_name;
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

    const uint64_t rank = static_cast<uint64_t>(contrib_dims.size());
    const uint64_t target_rank = static_cast<uint64_t>(target_dims.size());
    const uint64_t rank_diff = rank - target_rank;

    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;

    for (uint64_t axis = 0; axis < rank; ++axis) {
        const uint64_t target_dim = (axis < rank_diff) ? 1 : target_dims[axis - rank_diff];
        const uint64_t contrib_dim = contrib_dims[axis];

        if (target_dim == contrib_dim) {
            continue;
        }

        if (target_dim != 1) {
            throw std::runtime_error("Autodiff broadcast backward found incompatible target shape while summing to input shape.");
        }

        reduction_axes.push_back(axis);
        if (axis < rank_diff) {
            squeeze_axes.push_back(axis);
        }
    }

    if (reduction_axes.empty()) {
        return contrib;
    }

    return builder.reduction(ExprOp::REDUCE_SUM, contrib, reduction_axes, squeeze_axes);
}

}  // namespace

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::optional<std::string>& upstream_input_name,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    if (!forward_outputs.expr) {
        throw std::runtime_error("buildBackwardOutputs requires non-null forward_outputs.expr.");
    }
    if (forward_outputs.outputs.size() != 1) {
        throw std::runtime_error("Phase-1 autodiff currently supports exactly one forward output. Multi-output backward will come later.");
    }

    const PhysicalExpression& forward_expr = *forward_outputs.expr;
    const NamedOutput& forward_output = forward_outputs.outputs[0];
    if (forward_output.node_idx >= forward_expr.nodes.size()) {
        throw std::runtime_error("Forward output node index is out of range in buildBackwardOutputs.");
    }

    const std::vector<std::string> normalized_wrt = normalizeWrtNames(forward_expr, wrt_names);
    const std::optional<std::string> normalized_upstream_input_name = normalizeUpstreamInputName(forward_expr, upstream_input_name);
    const std::vector<std::vector<uint64_t>> forward_node_dims = inferForwardNodeDims(forward_expr, forward_input_dims);
    const bool has_forward_dims = !forward_node_dims.empty();

    BackwardGraphBuilder builder(forward_expr);
    builder.initializeAdjoints();

    const uint32_t output_seed =
        normalized_upstream_input_name.has_value() ? builder.input(normalized_upstream_input_name.value()) : builder.scalar(1.0);
    builder.addContribution(forward_output.node_idx, output_seed);

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
                if (!node.squeeze_axes.empty()) {
                    throw std::runtime_error(
                        "Phase-1 autodiff only supports reduce_sum backward when squeeze=False / no squeeze axes are requested.");
                }

                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t expanded_grad = builder.add(builder.mul(lhs, builder.scalar(0.0)), grad);
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;
                addContributionToChild(node.lhs, expanded_grad, lhs_dims);
                break;
            }

            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
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

}  // namespace ThorImplementation
