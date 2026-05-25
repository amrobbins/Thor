#include "Utilities/Expression/AutoDiff.h"

#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {
namespace {

bool experimentalCudnnAttentionSupportSurfaceProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE");
    return value != nullptr && std::string_view(value) == "1";
}

bool experimentalCudnnRaggedBiasBackwardProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD");
    return (value != nullptr && std::string_view(value) == "1") || experimentalCudnnAttentionSupportSurfaceProbeEnabled();
}

static std::vector<uint64_t> inferTransposeOutputDims(const std::vector<uint64_t>& input_dims);

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
    if (new_node.op == ExprOp::ROPE) {
        // Backward graphs may clone forward RoPE subtrees for saved activations. Keep those clones out-of-place so
        // gradient evaluation cannot destructively mutate recomputed forward values.
        new_node.rope_allow_in_place_materialization = false;
    }

    if (Expression::isUnaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: unary node missing lhs.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new);
        new_node.rhs = UINT32_MAX;
        new_node.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX || src_node.rhs == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: binary node missing child.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new);
        new_node.rhs = cloneForwardSubtree(src, src_node.rhs, dst, old_to_new);
        new_node.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX || src_node.rhs == UINT32_MAX || src_node.aux == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: ternary node missing child.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new);
        new_node.rhs = cloneForwardSubtree(src, src_node.rhs, dst, old_to_new);
        new_node.aux = cloneForwardSubtree(src, src_node.aux, dst, old_to_new);
        if (src_node.alpha_node != UINT32_MAX) {
            new_node.alpha_node = cloneForwardSubtree(src, src_node.alpha_node, dst, old_to_new);
        }
        if (src_node.beta_node != UINT32_MAX) {
            new_node.beta_node = cloneForwardSubtree(src, src_node.beta_node, dst, old_to_new);
        }
        if (src_node.attention_use_padding_mask) {
            if (src_node.attention_seq_len_q_node == UINT32_MAX || src_node.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing padding-mask sequence length node while cloning forward subtree for autodiff.");
            }
            new_node.attention_seq_len_q_node = cloneForwardSubtree(src, src_node.attention_seq_len_q_node, dst, old_to_new);
            new_node.attention_seq_len_kv_node = cloneForwardSubtree(src, src_node.attention_seq_len_kv_node, dst, old_to_new);
        }
        if (src_node.attention_use_ragged_offsets) {
            if (src_node.attention_ragged_offset_q_node == UINT32_MAX || src_node.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing ragged offset node while cloning forward subtree for autodiff.");
            }
            new_node.attention_ragged_offset_q_node = cloneForwardSubtree(src, src_node.attention_ragged_offset_q_node, dst, old_to_new);
            new_node.attention_ragged_offset_kv_node = cloneForwardSubtree(src, src_node.attention_ragged_offset_kv_node, dst, old_to_new);
        }
        if (src_node.attention_use_paged_kv_cache) {
            if (src_node.attention_page_table_k_node == UINT32_MAX || src_node.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing paged KV page-table node while cloning forward subtree for autodiff.");
            }
            new_node.attention_page_table_k_node = cloneForwardSubtree(src, src_node.attention_page_table_k_node, dst, old_to_new);
            new_node.attention_page_table_v_node = cloneForwardSubtree(src, src_node.attention_page_table_v_node, dst, old_to_new);
        }
        if (src_node.attention_dropout_probability > 0.0f) {
            if (src_node.attention_dropout_seed_node == UINT32_MAX || src_node.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing dropout seed/offset node while cloning forward subtree for autodiff.");
            }
            new_node.attention_dropout_seed_node = cloneForwardSubtree(src, src_node.attention_dropout_seed_node, dst, old_to_new);
            new_node.attention_dropout_offset_node = cloneForwardSubtree(src, src_node.attention_dropout_offset_node, dst, old_to_new);
        }
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

std::vector<uint64_t> normalizeAxes(std::vector<uint64_t> axes) {
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

bool axesEqualNormalized(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) { return normalizeAxes(a) == normalizeAxes(b); }

bool dimsAllSingleton(const std::vector<uint64_t>& dims) {
    for (uint64_t dim : dims) {
        if (dim != 1) {
            return false;
        }
    }
    return true;
}

bool resolveBroadcastedDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        outputDimensions.clear();
        return false;
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
                throw std::runtime_error("Autodiff constant-like folding encountered non-broadcast-compatible shapes.");
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

std::vector<uint64_t> applySqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes);

std::optional<DataType> preferredGradValueDType(const ExprNode& forward_node) {
    if (forward_node.backward_output_dtype.has_value()) {
        return forward_node.backward_output_dtype;
    }
    if (forward_node.output_dtype.has_value()) {
        return forward_node.output_dtype;
    }
    return std::nullopt;
}

static bool isAttentionBackwardOp(ExprOp op) {
    return op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K || op == ExprOp::ATTENTION_BACKWARD_V ||
           op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

static bool isStageBoundaryLikeBackwardOutputOp(ExprOp op) {
    switch (op) {
        case ExprOp::MATMUL:
        case ExprOp::GEMM:
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::SOFTMAX:
        case ExprOp::RMSNORM:
        case ExprOp::ATTENTION:
        case ExprOp::ATTENTION_BACKWARD_Q:
        case ExprOp::ATTENTION_BACKWARD_K:
        case ExprOp::ATTENTION_BACKWARD_V:
        case ExprOp::ATTENTION_BACKWARD_BIAS:
        case ExprOp::CONV2D:
        case ExprOp::CONV2D_BACKWARD_DATA:
        case ExprOp::CONV2D_BACKWARD_FILTER:
        case ExprOp::CONV3D:
        case ExprOp::CONV3D_BACKWARD_DATA:
        case ExprOp::CONV3D_BACKWARD_FILTER:
        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD:
            return true;
        default:
            return false;
    }
}

std::vector<bool> computeNodeReachesRequestedInputs(const PhysicalExpression& expr, const std::vector<std::string>& wrt_names) {
    std::unordered_set<uint32_t> wrt_slots;
    wrt_slots.reserve(wrt_names.size());

    for (const std::string& name : wrt_names) {
        bool found = false;
        for (const NamedInput& input : expr.inputs) {
            if (input.name == name) {
                wrt_slots.insert(input.slot);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Requested gradient for unknown input while computing reverse relevance: " + name);
        }
    }

    std::vector<bool> reaches(expr.nodes.size(), false);
    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        const ExprNode& node = expr.nodes[i];
        switch (node.op) {
            case ExprOp::INPUT:
                reaches[i] = wrt_slots.contains(node.input_slot);
                break;
            case ExprOp::SCALAR_FP:
                reaches[i] = false;
                break;
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                reaches[i] = false;
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
            case ExprOp::MAX_GRAD_RIGHT:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs);
                break;
            case ExprOp::NEG:
            case ExprOp::ABS:
            case ExprOp::EXP:
            case ExprOp::EXPM1:
            case ExprOp::EXP2:
            case ExprOp::EXP10:
            case ExprOp::LN:
            case ExprOp::LOG1P:
            case ExprOp::LOG2:
            case ExprOp::LOG10:
            case ExprOp::SQRT:
            case ExprOp::TANH:
            case ExprOp::NORMCDF:
            case ExprOp::ROPE:
            case ExprOp::SOFTMAX:
            case ExprOp::TRANSPOSE:
            case ExprOp::RESHAPE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::STRIDED_VIEW_BACKWARD:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::MATMUL:
            case ExprOp::RMSNORM:
            case ExprOp::CONV2D:
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D:
            case ExprOp::CONV3D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_FILTER:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs);
                break;
            case ExprOp::GEMM:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs) || reaches.at(node.aux);
                break;
            case ExprOp::ATTENTION:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs) || reaches.at(node.aux) ||
                             (node.attention_use_bias && node.alpha_node != UINT32_MAX && reaches.at(node.alpha_node)) ||
                             (node.attention_use_padding_mask && node.attention_seq_len_q_node != UINT32_MAX && reaches.at(node.attention_seq_len_q_node)) ||
                             (node.attention_use_padding_mask && node.attention_seq_len_kv_node != UINT32_MAX && reaches.at(node.attention_seq_len_kv_node)) ||
                             (node.attention_use_ragged_offsets && node.attention_ragged_offset_q_node != UINT32_MAX && reaches.at(node.attention_ragged_offset_q_node)) ||
                             (node.attention_use_ragged_offsets && node.attention_ragged_offset_kv_node != UINT32_MAX && reaches.at(node.attention_ragged_offset_kv_node)) ||
                             (node.attention_use_paged_kv_cache && node.attention_page_table_k_node != UINT32_MAX && reaches.at(node.attention_page_table_k_node)) ||
                             (node.attention_use_paged_kv_cache && node.attention_page_table_v_node != UINT32_MAX && reaches.at(node.attention_page_table_v_node)) ||
                             (node.attention_dropout_probability > 0.0f && node.attention_dropout_seed_node != UINT32_MAX && reaches.at(node.attention_dropout_seed_node)) ||
                             (node.attention_dropout_probability > 0.0f && node.attention_dropout_offset_node != UINT32_MAX && reaches.at(node.attention_dropout_offset_node));
                break;
            case ExprOp::ATTENTION_BACKWARD_Q:
            case ExprOp::ATTENTION_BACKWARD_K:
            case ExprOp::ATTENTION_BACKWARD_V:
            case ExprOp::ATTENTION_BACKWARD_BIAS:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs) || reaches.at(node.aux) || reaches.at(node.alpha_node) ||
                             (node.attention_use_bias && node.beta_node != UINT32_MAX && reaches.at(node.beta_node)) ||
                             (node.attention_use_padding_mask && node.attention_seq_len_q_node != UINT32_MAX && reaches.at(node.attention_seq_len_q_node)) ||
                             (node.attention_use_padding_mask && node.attention_seq_len_kv_node != UINT32_MAX && reaches.at(node.attention_seq_len_kv_node)) ||
                             (node.attention_use_ragged_offsets && node.attention_ragged_offset_q_node != UINT32_MAX && reaches.at(node.attention_ragged_offset_q_node)) ||
                             (node.attention_use_ragged_offsets && node.attention_ragged_offset_kv_node != UINT32_MAX && reaches.at(node.attention_ragged_offset_kv_node)) ||
                             (node.attention_use_paged_kv_cache && node.attention_page_table_k_node != UINT32_MAX && reaches.at(node.attention_page_table_k_node)) ||
                             (node.attention_use_paged_kv_cache && node.attention_page_table_v_node != UINT32_MAX && reaches.at(node.attention_page_table_v_node)) ||
                             (node.attention_dropout_probability > 0.0f && node.attention_dropout_seed_node != UINT32_MAX && reaches.at(node.attention_dropout_seed_node)) ||
                             (node.attention_dropout_probability > 0.0f && node.attention_dropout_offset_node != UINT32_MAX && reaches.at(node.attention_dropout_offset_node));
                break;
            default:
                throw std::runtime_error("Unsupported op while computing reverse relevance: " + std::to_string((int)node.op));
        }
    }

    return reaches;
}

class BackwardGraphBuilder {
   public:
    explicit BackwardGraphBuilder(const PhysicalExpression& forward_expr) : forward_expr(forward_expr) {
        grad_expr.inputs = forward_expr.inputs;
    }

    uint32_t input(const std::string& name, std::optional<DataType> as_type = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::INPUT;
        node.input_slot = grad_expr.getOrCreateInputSlot(name);
        if (as_type.has_value()) {
            node.output_dtype = as_type.value();
        }
        return push(std::move(node));
    }

    uint32_t scalar(double value) {
        ExprNode node{};
        node.op = ExprOp::SCALAR_FP;
        node.scalar_fp = value;
        return push(std::move(node));
    }

    uint32_t fill(double value,
                  const std::vector<uint64_t>& dims,
                  std::optional<DataType> as_type = std::nullopt) {
        if (dims.empty()) {
            return scalar(value);
        }

        ExprNode node{};
        node.op = ExprOp::FILL;
        node.scalar_fp = value;
        node.fill_dims = dims;
        if (as_type.has_value()) {
            node.output_dtype = as_type.value();
        }
        return push(std::move(node));
    }

    uint32_t constantLike(double value,
                          const std::vector<uint64_t>& dims,
                          std::optional<DataType> as_type = std::nullopt) {
        return dims.empty() ? scalar(value) : fill(value, dims, as_type);
    }

    const ExprNode& node(uint32_t node_idx) const {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder node query index out of range.");
        }
        return grad_expr.nodes.at(node_idx);
    }

    std::optional<std::vector<uint64_t>> tryInferKnownGradientDims(uint32_t node_idx) const {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder gradient-dim inference node index out of range.");
        }

        const ExprNode& n = grad_expr.nodes.at(node_idx);
        switch (n.op) {
            case ExprOp::SCALAR_FP:
                return std::vector<uint64_t>{};
            case ExprOp::FILL:
                return n.fill_dims;
            case ExprOp::RESHAPE:
                return n.reshape_dims;
            case ExprOp::STRIDED_VIEW:
                return n.view_dims;
            case ExprOp::STRIDED_VIEW_BACKWARD:
                return n.fill_dims;
            case ExprOp::NEG:
                return tryInferKnownGradientDims(n.lhs);
            case ExprOp::ADD:
            case ExprOp::SUB: {
                const auto lhs_dims = tryInferKnownGradientDims(n.lhs);
                const auto rhs_dims = tryInferKnownGradientDims(n.rhs);
                if (!lhs_dims.has_value() || !rhs_dims.has_value()) {
                    return std::nullopt;
                }
                if (lhs_dims->empty()) {
                    return rhs_dims.value();
                }
                if (rhs_dims->empty()) {
                    return lhs_dims.value();
                }
                if (lhs_dims.value() == rhs_dims.value()) {
                    return lhs_dims.value();
                }

                std::vector<uint64_t> out_dims;
                try {
                    resolveBroadcastedDims({lhs_dims.value(), rhs_dims.value()}, out_dims);
                } catch (const std::runtime_error&) {
                    return std::nullopt;
                }
                return out_dims;
            }
            default:
                return std::nullopt;
        }
    }

    bool tryGetScalarConstant(uint32_t node_idx, double& value) const {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder constant query node index out of range.");
        }
        const ExprNode& node = grad_expr.nodes[node_idx];
        if (node.op != ExprOp::SCALAR_FP) {
            return false;
        }
        value = node.scalar_fp;
        return true;
    }

    bool tryGetConstantLike(uint32_t node_idx, double& value, std::vector<uint64_t>& dims) const {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder constant-like query node index out of range.");
        }
        const ExprNode& node = grad_expr.nodes[node_idx];
        switch (node.op) {
            case ExprOp::SCALAR_FP:
                value = node.scalar_fp;
                dims.clear();
                return true;
            case ExprOp::FILL:
                value = node.scalar_fp;
                dims = node.fill_dims;
                return true;
            case ExprOp::TRANSPOSE: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }
                dims = inferTransposeOutputDims(lhs_dims);
                return true;
            }
            case ExprOp::RESHAPE: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }
                dims = node.reshape_dims;
                return true;
            }
            case ExprOp::STRIDED_VIEW: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }
                dims = node.view_dims;
                return true;
            }
            case ExprOp::STRIDED_VIEW_BACKWARD: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }
                if (value != 0.0) {
                    return false;
                }
                dims = node.fill_dims;
                return true;
            }
            case ExprOp::UNSQUEEZE: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }

                std::vector<uint64_t> actual_axes;
                try {
                    actual_axes = normalizeUnsqueezeAxesForInputDims(lhs_dims, node.unsqueeze_axes);
                } catch (const std::runtime_error&) {
                    // Generic backward graphs can temporarily contain shape ops whose
                    // rank is only well-defined after runtime shape specialization.
                    // In that case this node is not safely foldable as constant-like yet.
                    return false;
                }

                dims.clear();
                dims.reserve(lhs_dims.size() + actual_axes.size());
                const uint64_t output_rank = static_cast<uint64_t>(lhs_dims.size() + actual_axes.size());
                size_t lhs_i = 0;
                size_t axis_i = 0;
                for (uint64_t out_axis = 0; out_axis < output_rank; ++out_axis) {
                    if (axis_i < actual_axes.size() && actual_axes[axis_i] == out_axis) {
                        dims.push_back(1);
                        ++axis_i;
                    } else {
                        if (lhs_i >= lhs_dims.size()) {
                            return false;
                        }
                        dims.push_back(lhs_dims[lhs_i++]);
                    }
                }
                if (lhs_i != lhs_dims.size() || axis_i != actual_axes.size()) {
                    return false;
                }
                return true;
            }
            case ExprOp::SQUEEZE: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }

                try {
                    dims = applySqueezeDims(lhs_dims, normalizeSqueezeAxesForInputDims(lhs_dims, node.squeeze_axes));
                } catch (const std::runtime_error&) {
                    // See UNSQUEEZE case above: leave generic rank-dependent shape ops
                    // untouched until runtime specialization makes them valid.
                    return false;
                }
                return true;
            }
            case ExprOp::NEG:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = -value;
                    return true;
                }
                return false;
            case ExprOp::ABS:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::fabs(value);
                    return true;
                }
                return false;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV: {
                double lhs_value = 0.0;
                double rhs_value = 0.0;
                std::vector<uint64_t> lhs_dims;
                std::vector<uint64_t> rhs_dims;
                if (!tryGetConstantLike(node.lhs, lhs_value, lhs_dims) || !tryGetConstantLike(node.rhs, rhs_value, rhs_dims)) {
                    return false;
                }

                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!lhs_dims.empty()) {
                    non_scalar_inputs.push_back(lhs_dims);
                }
                if (!rhs_dims.empty()) {
                    non_scalar_inputs.push_back(rhs_dims);
                }
                if (non_scalar_inputs.empty()) {
                    dims.clear();
                } else if (non_scalar_inputs.size() == 1) {
                    dims = non_scalar_inputs[0];
                } else {
                    resolveBroadcastedDims(non_scalar_inputs, dims);
                }

                switch (node.op) {
                    case ExprOp::ADD:
                        value = lhs_value + rhs_value;
                        return true;
                    case ExprOp::SUB:
                        value = lhs_value - rhs_value;
                        return true;
                    case ExprOp::MUL:
                        value = lhs_value * rhs_value;
                        return true;
                    case ExprOp::DIV:
                        value = lhs_value / rhs_value;
                        return true;
                    default:
                        return false;
                }
            }
            default:
                return false;
        }
    }

    bool tryGetConstantLikeValue(uint32_t node_idx, double& value) const {
        std::vector<uint64_t> dims;
        return tryGetConstantLike(node_idx, value, dims);
    }

    bool isScalarZero(uint32_t node_idx) const {
        double value = 0.0;
        return tryGetScalarConstant(node_idx, value) && value == 0.0;
    }

    bool isScalarOne(uint32_t node_idx) const {
        double value = 0.0;
        return tryGetScalarConstant(node_idx, value) && value == 1.0;
    }

    bool isConstantLikeZero(uint32_t node_idx) const {
        double value = 0.0;
        return tryGetConstantLikeValue(node_idx, value) && value == 0.0;
    }

    uint32_t unary(ExprOp op, uint32_t lhs) {
        double lhs_value = 0.0;
        std::vector<uint64_t> lhs_dims;
        if (op == ExprOp::NEG) {
            if (tryGetConstantLike(lhs, lhs_value, lhs_dims)) {
                return constantLike(-lhs_value, lhs_dims);
            }

            const ExprNode& lhs_node = grad_expr.nodes.at(lhs);
            if (lhs_node.op == ExprOp::NEG) {
                return lhs_node.lhs;
            }
        } else if (op == ExprOp::ABS) {
            if (tryGetConstantLike(lhs, lhs_value, lhs_dims)) {
                return constantLike(std::fabs(lhs_value), lhs_dims);
            }

            const ExprNode& lhs_node = grad_expr.nodes.at(lhs);
            if (lhs_node.op == ExprOp::ABS) {
                return lhs;
            }
        }

        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        return push(std::move(node));
    }

    uint32_t binary(ExprOp op, uint32_t lhs, uint32_t rhs) {
        double lhs_value = 0.0;
        double rhs_value = 0.0;
        std::vector<uint64_t> lhs_dims;
        std::vector<uint64_t> rhs_dims;
        const bool lhs_const_like = tryGetConstantLike(lhs, lhs_value, lhs_dims);
        const bool rhs_const_like = tryGetConstantLike(rhs, rhs_value, rhs_dims);
        const bool lhs_const = lhs_const_like && lhs_dims.empty();
        const bool rhs_const = rhs_const_like && rhs_dims.empty();

        if (lhs_const_like && rhs_const_like) {
            std::vector<std::vector<uint64_t>> non_scalar_inputs;
            if (!lhs_dims.empty()) {
                non_scalar_inputs.push_back(lhs_dims);
            }
            if (!rhs_dims.empty()) {
                non_scalar_inputs.push_back(rhs_dims);
            }

            std::vector<uint64_t> out_dims;
            if (non_scalar_inputs.empty()) {
                out_dims.clear();
            } else if (non_scalar_inputs.size() == 1) {
                out_dims = non_scalar_inputs[0];
            } else {
                resolveBroadcastedDims(non_scalar_inputs, out_dims);
            }

            switch (op) {
                case ExprOp::ADD:
                    return constantLike(lhs_value + rhs_value, out_dims);
                case ExprOp::SUB:
                    return constantLike(lhs_value - rhs_value, out_dims);
                case ExprOp::MUL:
                    return constantLike(lhs_value * rhs_value, out_dims);
                case ExprOp::DIV:
                    return constantLike(lhs_value / rhs_value, out_dims);
                default:
                    break;
            }
        }

        switch (op) {
            case ExprOp::ADD:
                if (lhs_const && lhs_value == 0.0) {
                    return rhs;
                }
                if (rhs_const && rhs_value == 0.0) {
                    return lhs;
                }
                break;
            case ExprOp::SUB:
                if (rhs_const && rhs_value == 0.0) {
                    return lhs;
                }
                break;
            case ExprOp::MUL:
                if (lhs_const && lhs_value == 1.0) {
                    return rhs;
                }
                if (rhs_const && rhs_value == 1.0) {
                    return lhs;
                }
                break;
            case ExprOp::DIV:
                if (rhs_const && rhs_value == 1.0) {
                    return lhs;
                }
                break;
            default:
                break;
        }

        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.rhs = rhs;
        return push(std::move(node));
    }

    uint32_t matmul(uint32_t lhs,
                    uint32_t rhs,
                    bool transpose_lhs = false,
                    bool transpose_rhs = false,
                    std::optional<DataType> output_dtype = std::nullopt,
                    std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::MATMUL;
        node.lhs = lhs;
        node.rhs = rhs;
        node.transpose_lhs = transpose_lhs;
        node.transpose_rhs = transpose_rhs;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t gemm(uint32_t lhs,
                  uint32_t rhs,
                  uint32_t addend,
                  double alpha,
                  double beta,
                  bool transpose_lhs = false,
                  bool transpose_rhs = false,
                  bool transpose_addend = false,
                  std::optional<DataType> output_dtype = std::nullopt,
                  std::optional<DataType> compute_dtype = std::nullopt,
                  uint32_t alpha_node = UINT32_MAX,
                  uint32_t beta_node = UINT32_MAX) {
        ExprNode node{};
        node.op = ExprOp::GEMM;
        node.lhs = lhs;
        node.rhs = rhs;
        node.aux = addend;
        node.alpha_fp = alpha;
        node.beta_fp = beta;
        node.alpha_node = alpha_node;
        node.beta_node = beta_node;
        node.transpose_lhs = transpose_lhs;
        node.transpose_rhs = transpose_rhs;
        node.transpose_aux = transpose_addend;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }


    uint32_t cloneForwardMatmulPreamble(const ExprNode& forward_node) {
        if (forward_node.op == ExprOp::MATMUL) {
            uint32_t result = matmul(cloneForward(forward_node.lhs),
                                     cloneForward(forward_node.rhs),
                                     forward_node.transpose_lhs,
                                     forward_node.transpose_rhs,
                                     forward_node.output_dtype,
                                     forward_node.compute_dtype);
            ExprNode& result_node = grad_expr.nodes.at(result);
            result_node.alpha_fp = forward_node.alpha_fp;
            result_node.beta_fp = forward_node.beta_fp;
            if (forward_node.alpha_node != UINT32_MAX) {
                result_node.alpha_node = cloneForward(forward_node.alpha_node);
            }
            if (forward_node.beta_node != UINT32_MAX) {
                result_node.beta_node = cloneForward(forward_node.beta_node);
            }
            return result;
        }
        if (forward_node.op == ExprOp::GEMM) {
            return gemm(cloneForward(forward_node.lhs),
                        cloneForward(forward_node.rhs),
                        cloneForward(forward_node.aux),
                        forward_node.alpha_fp,
                        forward_node.beta_fp,
                        forward_node.transpose_lhs,
                        forward_node.transpose_rhs,
                        forward_node.transpose_aux,
                        forward_node.output_dtype,
                        forward_node.compute_dtype,
                        forward_node.alpha_node != UINT32_MAX ? cloneForward(forward_node.alpha_node) : UINT32_MAX,
                        forward_node.beta_node != UINT32_MAX ? cloneForward(forward_node.beta_node) : UINT32_MAX);
        }
        throw std::runtime_error("cloneForwardMatmulPreamble requires a MATMUL or GEMM node.");
    }

    uint32_t duplicateMatmulWithBackwardEpilogue(uint32_t matmul_idx, uint32_t epilogue_aux, MatmulBackwardEpilogue epilogue) {
        if (matmul_idx >= grad_expr.nodes.size()) {
            return UINT32_MAX;
        }
        const ExprNode& source = grad_expr.nodes.at(matmul_idx);
        if (!(source.op == ExprOp::MATMUL || source.op == ExprOp::GEMM) || source.matmul_epilogue != MatmulEpilogue::Default ||
            source.matmul_backward_epilogue != MatmulBackwardEpilogue::Default) {
            return UINT32_MAX;
        }
        ExprNode fused = source;
        if (fused.transpose_lhs) {
            fused.lhs = unary(ExprOp::TRANSPOSE, fused.lhs);
            fused.transpose_lhs = false;
        }
        if (fused.transpose_rhs) {
            fused.rhs = unary(ExprOp::TRANSPOSE, fused.rhs);
            fused.transpose_rhs = false;
        }
        if (fused.transpose_aux) {
            fused.aux = unary(ExprOp::TRANSPOSE, fused.aux);
            fused.transpose_aux = false;
        }
        fused.matmul_backward_epilogue = epilogue;
        fused.matmul_epilogue_aux = epilogue_aux;
        return push(std::move(fused));
    }

    uint32_t applyForwardMatmulEpilogueBackward(const ExprNode& forward_node, uint32_t grad_like_output) {
        if (forward_node.matmul_epilogue == MatmulEpilogue::Default) {
            return grad_like_output;
        }

        const uint32_t preactivation = cloneForwardMatmulPreamble(forward_node);
        if (forward_node.matmul_epilogue == MatmulEpilogue::Relu) {
            const uint32_t fused = duplicateMatmulWithBackwardEpilogue(grad_like_output, preactivation, MatmulBackwardEpilogue::DRelu);
            if (fused != UINT32_MAX) {
                return fused;
            }
            return mul(grad_like_output, binary(ExprOp::MAX_GRAD_LEFT, preactivation, scalar(0.0)));
        }

        if (forward_node.matmul_epilogue == MatmulEpilogue::Gelu) {
            // Only use cuBLASLt DGELU when the forward path was explicitly lowered to cuBLASLt's GELU
            // approximation.  Generic x * normcdf(x) graphs that were not eligible for the forward epilogue
            // continue to use the exact expression derivative through the normal autodiff rules.
            const uint32_t fused = duplicateMatmulWithBackwardEpilogue(grad_like_output, preactivation, MatmulBackwardEpilogue::DGelu);
            if (fused != UINT32_MAX) {
                return fused;
            }
            const uint32_t x2 = mul(preactivation, preactivation);
            const uint32_t x3 = mul(x2, preactivation);
            const uint32_t sqrt_two_over_pi = scalar(0.7978845608028654);
            const uint32_t tanh_arg = mul(sqrt_two_over_pi, add(preactivation, mul(scalar(0.044715), x3)));
            const uint32_t tanh_value = unary(ExprOp::TANH, tanh_arg);
            const uint32_t sech2 = sub(scalar(1.0), mul(tanh_value, tanh_value));
            const uint32_t dt_dx = mul(sqrt_two_over_pi, add(scalar(1.0), mul(scalar(3.0 * 0.044715), x2)));
            const uint32_t term0 = mul(scalar(0.5), add(scalar(1.0), tanh_value));
            const uint32_t term1 = mul(mul(scalar(0.5), preactivation), mul(sech2, dt_dx));
            return mul(grad_like_output, add(term0, term1));
        }

        throw std::runtime_error("Unsupported matmul epilogue in autodiff.");
    }

    uint32_t conv2dBackwardData(uint32_t filter,
                                uint32_t grad_output,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                const std::vector<uint64_t>& target_output_dims = {},
                                std::optional<DataType> output_dtype = std::nullopt,
                                std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV2D_BACKWARD_DATA;
        node.lhs = filter;
        node.rhs = grad_output;
        node.conv_stride_h = stride_h;
        node.conv_stride_w = stride_w;
        node.conv_pad_h = pad_h;
        node.conv_pad_w = pad_w;
        node.fill_dims = target_output_dims;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t conv2dBackwardFilter(uint32_t input,
                                  uint32_t grad_output,
                                  int32_t stride_h,
                                  int32_t stride_w,
                                  int32_t pad_h,
                                  int32_t pad_w,
                                  const std::vector<uint64_t>& target_output_dims = {},
                                  std::optional<DataType> output_dtype = std::nullopt,
                                  std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV2D_BACKWARD_FILTER;
        node.lhs = input;
        node.rhs = grad_output;
        node.conv_stride_h = stride_h;
        node.conv_stride_w = stride_w;
        node.conv_pad_h = pad_h;
        node.conv_pad_w = pad_w;
        node.fill_dims = target_output_dims;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t conv3dBackwardData(uint32_t filter,
                                uint32_t grad_output,
                                int32_t stride_d,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_d,
                                int32_t pad_h,
                                int32_t pad_w,
                                const std::vector<uint64_t>& target_output_dims = {},
                                std::optional<DataType> output_dtype = std::nullopt,
                                std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV3D_BACKWARD_DATA;
        node.lhs = filter;
        node.rhs = grad_output;
        node.conv_stride_d = stride_d;
        node.conv_stride_h = stride_h;
        node.conv_stride_w = stride_w;
        node.conv_pad_d = pad_d;
        node.conv_pad_h = pad_h;
        node.conv_pad_w = pad_w;
        node.fill_dims = target_output_dims;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t conv3dBackwardFilter(uint32_t input,
                                  uint32_t grad_output,
                                  int32_t stride_d,
                                  int32_t stride_h,
                                  int32_t stride_w,
                                  int32_t pad_d,
                                  int32_t pad_h,
                                  int32_t pad_w,
                                  const std::vector<uint64_t>& target_output_dims = {},
                                  std::optional<DataType> output_dtype = std::nullopt,
                                  std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV3D_BACKWARD_FILTER;
        node.lhs = input;
        node.rhs = grad_output;
        node.conv_stride_d = stride_d;
        node.conv_stride_h = stride_h;
        node.conv_stride_w = stride_w;
        node.conv_pad_d = pad_d;
        node.conv_pad_h = pad_h;
        node.conv_pad_w = pad_w;
        node.fill_dims = target_output_dims;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t rotaryPositionEmbedding(uint32_t lhs,
                                    const ExprNode& forward_rope,
                                    bool inverse,
                                    std::optional<DataType> output_dtype = std::nullopt,
                                    std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::ROPE;
        node.lhs = lhs;
        node.rope_sequence_axis = forward_rope.rope_sequence_axis;
        node.rope_head_dim_axis = forward_rope.rope_head_dim_axis;
        node.rope_rotary_dim = forward_rope.rope_rotary_dim;
        node.rope_base = forward_rope.rope_base;
        node.rope_position_offset = forward_rope.rope_position_offset;
        node.rope_interleaved = forward_rope.rope_interleaved;
        node.rope_inverse = inverse;
        node.rope_scaling_kind = forward_rope.rope_scaling_kind;
        node.rope_scaling_factor = forward_rope.rope_scaling_factor;
        node.rope_original_max_position_embeddings = forward_rope.rope_original_max_position_embeddings;
        node.rope_attention_factor = forward_rope.rope_attention_factor;
        node.rope_yarn_beta_fast = forward_rope.rope_yarn_beta_fast;
        node.rope_yarn_beta_slow = forward_rope.rope_yarn_beta_slow;
        node.rope_llama3_low_freq_factor = forward_rope.rope_llama3_low_freq_factor;
        node.rope_llama3_high_freq_factor = forward_rope.rope_llama3_high_freq_factor;
        node.rope_long_rope_short_factors = forward_rope.rope_long_rope_short_factors;
        node.rope_long_rope_long_factors = forward_rope.rope_long_rope_long_factors;
        node.rope_allow_in_place_materialization = false;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        } else if (forward_rope.compute_dtype.has_value()) {
            node.compute_dtype = forward_rope.compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t attentionBackward(ExprOp op,
                               uint32_t q,
                               uint32_t k,
                               uint32_t v,
                               uint32_t dO,
                               uint32_t bias,
                               const ExprNode& forward_attention,
                               std::optional<DataType> output_dtype = std::nullopt,
                               std::optional<DataType> compute_dtype = std::nullopt) {
        if (!isAttentionBackwardOp(op)) {
            throw std::runtime_error("attentionBackward builder called with non-attention-backward op.");
        }
        if (forward_attention.attention_use_ragged_offsets && forward_attention.attention_use_bias &&
            !experimentalCudnnRaggedBiasBackwardProbeEnabled()) {
            throw std::runtime_error(
                "cuDNN primary SDPA backward does not support ragged offsets with additive bias; ragged additive bias is forward-only "
                "until a supported dBias/backward path is implemented. Set THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD=1 "
                "to bypass this guard for cuDNN support-surface probing only.");
        }

        ExprNode node{};
        node.op = op;
        node.lhs = q;
        node.rhs = k;
        node.aux = v;
        node.alpha_node = dO;
        node.beta_node = bias;
        node.attention_q_layout = forward_attention.attention_q_layout;
        node.attention_k_layout = forward_attention.attention_k_layout;
        node.attention_v_layout = forward_attention.attention_v_layout;
        node.attention_o_layout = forward_attention.attention_o_layout;
        node.attention_mask_kind = forward_attention.attention_mask_kind;
        node.attention_diagonal_left_bound = forward_attention.attention_diagonal_left_bound;
        node.attention_diagonal_right_bound = forward_attention.attention_diagonal_right_bound;
        node.attention_has_scale = forward_attention.attention_has_scale;
        node.attention_scale = forward_attention.attention_scale;
        node.attention_use_alibi_mask = forward_attention.attention_use_alibi_mask;
        node.attention_use_bias = forward_attention.attention_use_bias;
        node.attention_use_padding_mask = forward_attention.attention_use_padding_mask;
        node.attention_use_ragged_offsets = forward_attention.attention_use_ragged_offsets;
        node.attention_use_paged_kv_cache = forward_attention.attention_use_paged_kv_cache;
        node.attention_paged_kv_max_sequence_length = forward_attention.attention_paged_kv_max_sequence_length;
        node.attention_dropout_probability = forward_attention.attention_dropout_probability;
        node.attention_seq_len_q_node = forward_attention.attention_seq_len_q_node;
        node.attention_seq_len_kv_node = forward_attention.attention_seq_len_kv_node;
        node.attention_ragged_offset_q_node = forward_attention.attention_ragged_offset_q_node;
        node.attention_ragged_offset_kv_node = forward_attention.attention_ragged_offset_kv_node;
        node.attention_page_table_k_node = forward_attention.attention_page_table_k_node;
        node.attention_page_table_v_node = forward_attention.attention_page_table_v_node;
        node.attention_dropout_seed_node = forward_attention.attention_dropout_seed_node;
        node.attention_dropout_offset_node = forward_attention.attention_dropout_offset_node;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        } else if (forward_attention.compute_dtype.has_value()) {
            node.compute_dtype = forward_attention.compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t reduction(ExprOp op,
                       uint32_t lhs,
                       const std::vector<uint64_t>& reduction_axes,
                       const std::vector<uint64_t>& squeeze_axes,
                       std::optional<DataType> compute_dtype = std::nullopt,
                       std::optional<DataType> output_dtype = std::nullopt) {
        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.reduction_axes = reduction_axes;
        node.squeeze_axes = squeeze_axes;
        node.compute_dtype = compute_dtype;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t softmax(uint32_t lhs, cudnnSoftmaxAlgorithm_t algorithm, cudnnSoftmaxMode_t mode) {
        ExprNode node{};
        node.op = ExprOp::SOFTMAX;
        node.lhs = lhs;
        node.softmax_algorithm = algorithm;
        node.softmax_mode = mode;
        return push(std::move(node));
    }

    uint32_t reduceMinMaxBackward(ExprOp op,
                                  uint32_t lhs,
                                  uint32_t grad,
                                  const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  std::optional<DataType> output_dtype = std::nullopt,
                                  std::optional<DataType> compute_dtype = std::nullopt) {
        if (op != ExprOp::REDUCE_MIN_BACKWARD && op != ExprOp::REDUCE_MAX_BACKWARD) {
            throw std::runtime_error("reduceMinMaxBackward requires REDUCE_MIN_BACKWARD or REDUCE_MAX_BACKWARD.");
        }

        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.rhs = grad;
        node.reduction_axes = reduction_axes;
        node.squeeze_axes = squeeze_axes;
        node.output_dtype = output_dtype;
        node.compute_dtype = compute_dtype;
        return push(std::move(node));
    }

    uint32_t neg(uint32_t value) { return unary(ExprOp::NEG, value); }
    uint32_t exp(uint32_t value) { return unary(ExprOp::EXP, value); }
    uint32_t add(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::ADD, lhs, rhs); }
    uint32_t sub(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::SUB, lhs, rhs); }
    uint32_t mul(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::MUL, lhs, rhs); }
    uint32_t div(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::DIV, lhs, rhs); }

    uint32_t addNoFold(uint32_t lhs,
                       uint32_t rhs,
                       std::optional<DataType> output_dtype = std::nullopt,
                       std::optional<DataType> backward_output_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::ADD;
        node.lhs = lhs;
        node.rhs = rhs;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (backward_output_dtype.has_value()) {
            node.backward_output_dtype = backward_output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t reshape(uint32_t value, const std::vector<uint64_t>& reshape_dims) {
        if (reshape_dims.empty()) {
            throw std::runtime_error("AutoDiff reshape requires non-empty dimensions.");
        }
        const ExprNode& value_node = grad_expr.nodes.at(value);
        if (value_node.op == ExprOp::RESHAPE) {
            // Collapse adjacent reshapes.
            value = value_node.lhs;
        }
        ExprNode node{};
        node.op = ExprOp::RESHAPE;
        node.lhs = value;
        node.reshape_dims = reshape_dims;
        return push(std::move(node));
    }

    uint32_t stridedViewBackward(uint32_t grad_view,
                                 const std::vector<uint64_t>& source_dims,
                                 const std::vector<uint64_t>& view_dims,
                                 const std::vector<uint64_t>& view_strides,
                                 uint64_t view_element_offset,
                                 std::optional<DataType> output_dtype = std::nullopt,
                                 std::optional<DataType> compute_dtype = std::nullopt) {
        if (source_dims.empty()) {
            throw std::runtime_error("AutoDiff strided-view backward requires non-empty source dimensions.");
        }
        if (view_dims.empty() || view_dims.size() != view_strides.size()) {
            throw std::runtime_error("AutoDiff strided-view backward requires view dimensions and strides with the same non-zero rank.");
        }
        // The generated scatter kernel inverts a canonical, non-overlapping row-major-like strided view.
        // Packed-QKV slices satisfy this: [B,S,H,D] strides [S*total,total,D,1].
        uint64_t dense_tail = 1;
        for (int64_t axis = static_cast<int64_t>(view_dims.size()) - 1; axis >= 0; --axis) {
            if (view_dims[axis] == 0 || view_strides[axis] < dense_tail) {
                throw std::runtime_error(
                    "AutoDiff strided-view backward requires canonical non-overlapping row-major-like strides.");
            }
            dense_tail *= view_dims[axis];
        }
        ExprNode node{};
        node.op = ExprOp::STRIDED_VIEW_BACKWARD;
        node.lhs = grad_view;
        node.fill_dims = source_dims;
        node.view_dims = view_dims;
        node.view_strides = view_strides;
        node.view_element_offset = view_element_offset;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t unsqueeze(uint32_t value, const std::vector<uint64_t>& unsqueeze_axes) {
        const std::vector<uint64_t> normalized_axes = normalizeAxes(unsqueeze_axes);
        if (normalized_axes.empty()) {
            return value;
        }

        const ExprNode& value_node = grad_expr.nodes.at(value);
        if (value_node.op == ExprOp::SQUEEZE && axesEqualNormalized(value_node.squeeze_axes, normalized_axes)) {
            return value_node.lhs;
        }

        ExprNode node{};
        node.op = ExprOp::UNSQUEEZE;
        node.lhs = value;
        node.unsqueeze_axes = normalized_axes;
        return push(std::move(node));
    }

    uint32_t squeeze(uint32_t value, const std::vector<uint64_t>& squeeze_axes) {
        const std::vector<uint64_t> normalized_axes = normalizeAxes(squeeze_axes);
        if (normalized_axes.empty()) {
            return value;
        }

        const ExprNode& value_node = grad_expr.nodes.at(value);
        if (value_node.op == ExprOp::UNSQUEEZE && axesEqualNormalized(value_node.unsqueeze_axes, normalized_axes)) {
            return value_node.lhs;
        }

        ExprNode node{};
        node.op = ExprOp::SQUEEZE;
        node.lhs = value;
        node.squeeze_axes = normalized_axes;
        return push(std::move(node));
    }

    uint32_t cloneForward(uint32_t forward_node_index) {
        return cloneForwardSubtree(forward_expr, forward_node_index, grad_expr, forward_to_grad_node_map);
    }

    uint32_t buildScaledByGemmFactor(uint32_t maybe_scale_node, double constant_scale, uint32_t value_node) {
        if (maybe_scale_node != UINT32_MAX) {
            uint32_t scale = cloneForward(maybe_scale_node);
            if (constant_scale != 1.0) {
                scale = mul(scalar(constant_scale), scale);
            }
            return mul(scale, value_node);
        }
        if (constant_scale != 1.0) {
            return mul(scalar(constant_scale), value_node);
        }
        return value_node;
    }

    void addContribution(uint32_t forward_node_index, uint32_t contrib_root) {
        if (forward_node_index >= node_grads.size()) {
            throw std::runtime_error("Autodiff addContribution node index out of range.");
        }

        if (isConstantLikeZero(contrib_root)) {
            return;
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

std::optional<DataType> attentionBackwardBiasOnlyDType(const BackwardGraphBuilder& builder, uint32_t node_idx) {
    const ExprNode& node = builder.node(node_idx);
    if (node.op == ExprOp::ATTENTION_BACKWARD_BIAS) {
        if (!node.output_dtype.has_value()) {
            return std::nullopt;
        }
        return node.output_dtype.value();
    }

    if (node.op == ExprOp::REDUCE_SUM) {
        return attentionBackwardBiasOnlyDType(builder, node.lhs);
    }

    if (node.op != ExprOp::ADD) {
        return std::nullopt;
    }

    const auto lhs_dtype = attentionBackwardBiasOnlyDType(builder, node.lhs);
    if (!lhs_dtype.has_value()) {
        return std::nullopt;
    }
    const auto rhs_dtype = attentionBackwardBiasOnlyDType(builder, node.rhs);
    if (!rhs_dtype.has_value() || rhs_dtype.value() != lhs_dtype.value()) {
        return std::nullopt;
    }
    return lhs_dtype.value();
}

std::vector<std::string> normalizeWrtNames(const PhysicalExpression& forward_expr, const std::vector<std::string>& wrt_names) {
    if (wrt_names.empty()) {
        std::vector<std::string> all_names;
        all_names.reserve(forward_expr.inputs.size());
        for (const NamedInput& input : forward_expr.inputs) {
            if (input.kind == NamedInput::Kind::Tensor) {
                all_names.push_back(input.name);
            }
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
                if (input.kind != NamedInput::Kind::Tensor) {
                    throw std::runtime_error(
                        "compileBackward only supports gradients with respect to tensor inputs. Got runtime scalar input: " + name);
                }
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
    return resolveBroadcastedDims(inputs, outputDimensions);
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

std::vector<uint64_t> inferMatmulOutputDims(const ExprNode& node,
                                            const std::vector<uint64_t>& lhs_dims,
                                            const std::vector<uint64_t>& rhs_dims,
                                            const std::vector<uint64_t>* aux_dims = nullptr) {
    if (lhs_dims.size() != 2 || rhs_dims.size() != 2) {
        throw std::runtime_error("Autodiff shape inference for matmul/gemm currently only supports rank-2 tensors.");
    }

    const uint64_t a_rows = node.transpose_lhs ? lhs_dims[1] : lhs_dims[0];
    const uint64_t a_cols = node.transpose_lhs ? lhs_dims[0] : lhs_dims[1];
    const uint64_t b_rows = node.transpose_rhs ? rhs_dims[1] : rhs_dims[0];
    const uint64_t b_cols = node.transpose_rhs ? rhs_dims[0] : rhs_dims[1];

    if (a_cols != b_rows) {
        throw std::runtime_error("Autodiff shape inference found incompatible matmul/gemm matrix dimensions.");
    }

    std::vector<uint64_t> out_dims{a_rows, b_cols};
    if (aux_dims) {
        if (aux_dims->size() == 1) {
            if (node.transpose_aux || aux_dims->at(0) != out_dims[1]) {
                throw std::runtime_error("Autodiff shape inference found GEMM bias epilogue addend incompatible with output columns.");
            }
        } else if (aux_dims->size() == 2) {
            const std::vector<uint64_t> expected_aux = node.transpose_aux ? std::vector<uint64_t>{out_dims[1], out_dims[0]} : out_dims;
            if (*aux_dims != expected_aux) {
                throw std::runtime_error("Autodiff shape inference found GEMM addend dimensions incompatible with the matmul output.");
            }
        } else {
            throw std::runtime_error("Autodiff shape inference for GEMM currently supports rank-2 addends or rank-1 bias epilogue vectors.");
        }
    }

    return out_dims;
}

struct AttentionTensorLogicalDims {
    uint64_t batch;
    uint64_t heads;
    uint64_t sequence_length;
    uint64_t head_dim;
};

static AttentionTensorLogicalDims logicalAttentionDimsForAutodiff(const std::vector<uint64_t>& dims,
                                                                    AttentionTensorLayout layout,
                                                                    const char* name) {
    if (dims.size() != 4) {
        throw std::runtime_error(std::string("Autodiff attention shape inference requires rank-4 ") + name + " tensor.");
    }
    if (layout == AttentionTensorLayout::BHSD) {
        return {dims.at(0), dims.at(1), dims.at(2), dims.at(3)};
    }
    if (layout == AttentionTensorLayout::BSHD) {
        return {dims.at(0), dims.at(2), dims.at(1), dims.at(3)};
    }
    throw std::runtime_error(std::string("Autodiff attention shape inference does not support the configured layout for ") + name + ".");
}

static std::vector<uint64_t> attentionOutputDimsForAutodiff(const ExprNode& node,
                                                            uint64_t batch,
                                                            uint64_t query_heads,
                                                            uint64_t query_len,
                                                            uint64_t value_dim) {
    if (node.attention_o_layout == AttentionTensorLayout::BHSD) {
        return {batch, query_heads, query_len, value_dim};
    }
    if (node.attention_o_layout == AttentionTensorLayout::BSHD) {
        return {batch, query_len, query_heads, value_dim};
    }
    throw std::runtime_error("Autodiff attention shape inference does not support the configured output layout.");
}

static std::vector<uint64_t> inferAttentionOutputDims(const ExprNode& node,
                                                       const std::vector<uint64_t>& q_dims,
                                                       const std::vector<uint64_t>& k_dims,
                                                       const std::vector<uint64_t>& v_dims) {
    const AttentionTensorLogicalDims q = logicalAttentionDimsForAutodiff(q_dims, node.attention_q_layout, "q");
    const AttentionTensorLogicalDims k = logicalAttentionDimsForAutodiff(k_dims, node.attention_k_layout, "k");
    const AttentionTensorLogicalDims v = logicalAttentionDimsForAutodiff(v_dims, node.attention_v_layout, "v");

    if (q.batch != k.batch || q.batch != v.batch) {
        throw std::runtime_error("Autodiff attention shape inference found mismatched q/k/v batch dimensions.");
    }
    if (k.heads != v.heads) {
        throw std::runtime_error("Autodiff attention shape inference found mismatched k/v head counts.");
    }
    if (k.heads == 0 || q.heads == 0 || q.heads % k.heads != 0) {
        throw std::runtime_error("Autodiff attention query heads must be an integer multiple of key/value heads.");
    }
    if (k.sequence_length != v.sequence_length) {
        throw std::runtime_error("Autodiff attention shape inference found mismatched k/v sequence lengths.");
    }
    if (q.head_dim != k.head_dim) {
        throw std::runtime_error("Autodiff attention q/k head dimensions must match.");
    }
    if (q.sequence_length == 0 || k.sequence_length == 0 || q.head_dim == 0 || v.head_dim == 0) {
        throw std::runtime_error("Autodiff attention q/k/v dimensions must be non-zero.");
    }

    return attentionOutputDimsForAutodiff(node, q.batch, q.heads, q.sequence_length, v.head_dim);
}

static std::vector<uint64_t> inferAttentionDenseBiasDims(const ExprNode& node,
                                                          const std::vector<uint64_t>& q_dims,
                                                          const std::vector<uint64_t>& k_dims) {
    const AttentionTensorLogicalDims q = logicalAttentionDimsForAutodiff(q_dims, node.attention_q_layout, "q");
    const AttentionTensorLogicalDims k = logicalAttentionDimsForAutodiff(k_dims, node.attention_k_layout, "k");
    if (q.batch != k.batch) {
        throw std::runtime_error("Autodiff attention dBias shape inference found mismatched q/k batch dimensions.");
    }
    return {q.batch, q.heads, q.sequence_length, k.sequence_length};
}

static std::vector<uint64_t> inferAttentionBackwardOutputDims(const ExprNode& node,
                                                              ExprOp op,
                                                              const std::vector<uint64_t>& q_dims,
                                                              const std::vector<uint64_t>& k_dims,
                                                              const std::vector<uint64_t>& v_dims,
                                                              const std::vector<uint64_t>& dO_dims) {
    const std::vector<uint64_t> forward_dims = inferAttentionOutputDims(node, q_dims, k_dims, v_dims);
    if (dO_dims != forward_dims) {
        throw std::runtime_error("Autodiff attention-backward dO shape must match attention output shape.");
    }

    switch (op) {
        case ExprOp::ATTENTION_BACKWARD_Q:
            return q_dims;
        case ExprOp::ATTENTION_BACKWARD_K:
            return k_dims;
        case ExprOp::ATTENTION_BACKWARD_V:
            return v_dims;
        case ExprOp::ATTENTION_BACKWARD_BIAS:
            return inferAttentionDenseBiasDims(node, q_dims, k_dims);
        default:
            throw std::runtime_error("Autodiff attention-backward shape inference received a non-attention-backward op.");
    }
}

static bool isConv3DOp(ExprOp op) {
    return op == ExprOp::CONV3D || op == ExprOp::CONV3D_BACKWARD_DATA || op == ExprOp::CONV3D_BACKWARD_FILTER;
}

static std::vector<uint64_t> inferConvolutionOutputDims(const ExprNode& node,
                                                        const std::vector<uint64_t>& input_dims,
                                                        const std::vector<uint64_t>& filter_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (input_dims.size() != rank || filter_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "Autodiff CONV3D shape inference requires rank-5 tensors."
                                       : "Autodiff CONV2D shape inference requires rank-4 tensors.");
    }
    if (input_dims[1] != filter_dims[1]) {
        throw std::runtime_error("Autodiff convolution shape inference found mismatched input/filter channels.");
    }

    std::vector<uint64_t> out_dims{input_dims[0], filter_dims[0]};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                           : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t numer = static_cast<int64_t>(input_dims[dim_idx]) + 2LL * pads[i] - static_cast<int64_t>(filter_dims[dim_idx]);
        if (numer < 0) {
            throw std::runtime_error("Autodiff convolution shape inference produced negative output extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(numer / strides[i] + 1));
    }
    return out_dims;
}

static std::vector<uint64_t> inferConvolutionBackwardDataOutputDims(const ExprNode& node,
                                                                    const std::vector<uint64_t>& filter_dims,
                                                                    const std::vector<uint64_t>& grad_output_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (filter_dims.size() != rank || grad_output_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "Autodiff CONV3D_BACKWARD_DATA shape inference requires rank-5 tensors."
                                       : "Autodiff CONV2D_BACKWARD_DATA shape inference requires rank-4 tensors.");
    }
    const uint64_t k = filter_dims[0];
    const uint64_t c = filter_dims[1];
    const uint64_t grad_k = grad_output_dims[1];
    const uint64_t n = grad_output_dims[0];
    if (k != grad_k) {
        throw std::runtime_error("Autodiff convolution backward-data shape inference found mismatched filter/output channels.");
    }
    if (!node.fill_dims.empty()) {
        if (node.fill_dims.size() != rank) {
            throw std::runtime_error("Autodiff convolution backward-data explicit output shape rank mismatch.");
        }
        if (node.fill_dims[0] != n || node.fill_dims[1] != c) {
            throw std::runtime_error("Autodiff convolution backward-data explicit output shape is incompatible with batch/channels.");
        }
        return node.fill_dims;
    }

    std::vector<uint64_t> out_dims{n, c};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                           : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t extent = static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i] - 2LL * pads[i] +
                               static_cast<int64_t>(filter_dims[dim_idx]);
        if (extent <= 0) {
            throw std::runtime_error("Autodiff convolution backward-data shape inference produced non-positive output extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(extent));
    }
    return out_dims;
}

static std::vector<uint64_t> inferConvolutionBackwardFilterOutputDims(const ExprNode& node,
                                                                      const std::vector<uint64_t>& input_dims,
                                                                      const std::vector<uint64_t>& grad_output_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (input_dims.size() != rank || grad_output_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "Autodiff CONV3D_BACKWARD_FILTER shape inference requires rank-5 tensors."
                                       : "Autodiff CONV2D_BACKWARD_FILTER shape inference requires rank-4 tensors.");
    }
    if (input_dims[0] != grad_output_dims[0]) {
        throw std::runtime_error("Autodiff convolution backward-filter shape inference found mismatched batch sizes.");
    }
    const uint64_t c = input_dims[1];
    const uint64_t k = grad_output_dims[1];
    if (!node.fill_dims.empty()) {
        if (node.fill_dims.size() != rank) {
            throw std::runtime_error("Autodiff convolution backward-filter explicit output shape rank mismatch.");
        }
        if (node.fill_dims[0] != k || node.fill_dims[1] != c) {
            throw std::runtime_error("Autodiff convolution backward-filter explicit output shape is incompatible with channels.");
        }
        return node.fill_dims;
    }

    std::vector<uint64_t> out_dims{k, c};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                           : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t extent = static_cast<int64_t>(input_dims[dim_idx]) + 2LL * pads[i] -
                               static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i];
        if (extent <= 0) {
            throw std::runtime_error("Autodiff convolution backward-filter shape inference produced non-positive filter extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(extent));
    }
    return out_dims;
}

static std::vector<uint64_t> inferTransposeOutputDims(const std::vector<uint64_t>& input_dims) {
    if (input_dims.size() < 2) {
        throw std::runtime_error("Autodiff transpose shape inference requires rank >= 2 tensors.");
    }
    std::vector<uint64_t> out_dims = input_dims;
    std::swap(out_dims[out_dims.size() - 2], out_dims[out_dims.size() - 1]);
    return out_dims;
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
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                node_dims[i] = {};
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
            case ExprOp::ABS:
            case ExprOp::EXP:
            case ExprOp::EXPM1:
            case ExprOp::EXP2:
            case ExprOp::EXP10:
            case ExprOp::LN:
            case ExprOp::LOG1P:
            case ExprOp::LOG2:
            case ExprOp::LOG10:
            case ExprOp::SQRT:
            case ExprOp::TANH:
            case ExprOp::NORMCDF:
            case ExprOp::ROPE:
            case ExprOp::SOFTMAX:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::TRANSPOSE:
                node_dims[i] = inferTransposeOutputDims(node_dims[node.lhs]);
                break;
            case ExprOp::RESHAPE:
                node_dims[i] = node.reshape_dims;
                break;
            case ExprOp::STRIDED_VIEW:
                node_dims[i] = node.view_dims;
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
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
                node_dims[i] = StampedEquation::computeReductionOutputDims(node_dims[node.lhs], node.reduction_axes, node.squeeze_axes);
                break;
            case ExprOp::MATMUL:
                node_dims[i] = inferMatmulOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::GEMM:
                node_dims[i] = inferMatmulOutputDims(node, node_dims[node.lhs], node_dims[node.rhs], &node_dims[node.aux]);
                break;
            case ExprOp::RMSNORM: {
                const std::vector<uint64_t>& input_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& scale_dims = node_dims[node.rhs];
                if (input_dims.size() != 2 || scale_dims.size() != 1 || input_dims[1] != node.rms_norm_normalized_feature_count ||
                    scale_dims[0] != node.rms_norm_normalized_feature_count) {
                    throw std::runtime_error("inferForwardNodeDims RMSNorm expects [outer, hidden] input and [hidden] scale tensors.");
                }
                node_dims[i] = input_dims;
                break;
            }
            case ExprOp::ATTENTION:
                node_dims[i] = inferAttentionOutputDims(node, node_dims[node.lhs], node_dims[node.rhs], node_dims[node.aux]);
                break;
            case ExprOp::ATTENTION_BACKWARD_Q:
            case ExprOp::ATTENTION_BACKWARD_K:
            case ExprOp::ATTENTION_BACKWARD_V:
                node_dims[i] = inferAttentionBackwardOutputDims(node,
                                                                node.op,
                                                                node_dims[node.lhs],
                                                                node_dims[node.rhs],
                                                                node_dims[node.aux],
                                                                node_dims[node.alpha_node]);
                break;
            case ExprOp::ATTENTION_BACKWARD_BIAS:
                if (node.beta_node == UINT32_MAX) {
                    throw std::runtime_error("Autodiff attention-backward bias node is missing the forward bias input.");
                }
                node_dims[i] = inferAttentionBackwardOutputDims(node,
                                                                node.op,
                                                                node_dims[node.lhs],
                                                                node_dims[node.rhs],
                                                                node_dims[node.aux],
                                                                node_dims[node.alpha_node]);
                break;
            case ExprOp::CONV2D:
            case ExprOp::CONV3D:
                node_dims[i] = inferConvolutionOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_DATA:
                node_dims[i] = inferConvolutionBackwardDataOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D_BACKWARD_FILTER:
                node_dims[i] = inferConvolutionBackwardFilterOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
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
                    const std::vector<uint64_t>& target_dims,
                    std::optional<DataType> target_dtype = std::nullopt) {
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

    double contrib_constant = 0.0;
    if (builder.tryGetConstantLikeValue(contrib, contrib_constant)) {
        double reduction_scale = 1.0;
        for (uint64_t axis : reduction_axes) {
            if (axis >= contrib_dims.size()) {
                throw std::runtime_error("Autodiff sumToShape produced reduction axis out of range.");
            }
            reduction_scale *= static_cast<double>(contrib_dims[axis]);
        }
        return builder.fill(contrib_constant * reduction_scale, target_dims, target_dtype);
    }

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

    return builder.reduction(ExprOp::REDUCE_SUM, contrib, reduction_axes, squeeze_axes, std::nullopt, target_dtype);
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


static std::optional<std::vector<uint64_t>> inferPackedDenseSourceDimsForStridedViewBackward(const ExprNode& node) {
    const std::vector<uint64_t>& dims = node.view_dims;
    const std::vector<uint64_t>& strides = node.view_strides;
    if (dims.size() < 2 || dims.size() != strides.size()) {
        return std::nullopt;
    }
    for (uint64_t dim : dims) {
        if (dim == 0) {
            return std::nullopt;
        }
    }
    for (uint64_t stride : strides) {
        if (stride == 0) {
            return std::nullopt;
        }
    }

    // Packed-QKV views are commonly expressed as a higher-rank logical view over a
    // dense 2-D parent [outer, inner], where some prefix axes collapse into
    // `outer`, and the remaining suffix axes address a dense slice within each
    // row of width `inner`.  Example for BSHD Q view into [B*S, QKV]:
    //   dims    = [B, S, H, D]
    //   strides = [S*QKV, QKV, D, 1]
    //   offset  = q_start
    // which infers source dims [B*S, QKV].
    for (size_t collapsed_last_axis = 0; collapsed_last_axis + 1 < dims.size(); ++collapsed_last_axis) {
        bool ok = true;

        uint64_t expected_suffix_stride = 1;
        for (size_t axis = dims.size(); axis-- > collapsed_last_axis + 1;) {
            if (strides[axis] != expected_suffix_stride) {
                ok = false;
                break;
            }
            expected_suffix_stride *= dims[axis];
        }
        if (!ok) {
            continue;
        }

        uint64_t expected_prefix_stride = strides[collapsed_last_axis];
        for (size_t axis = collapsed_last_axis; axis-- > 0;) {
            expected_prefix_stride *= dims[axis + 1];
            if (strides[axis] != expected_prefix_stride) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        const uint64_t inner_width = strides[collapsed_last_axis];
        uint64_t suffix_span = 1;
        for (size_t axis = collapsed_last_axis + 1; axis < dims.size(); ++axis) {
            suffix_span += (dims[axis] - 1) * strides[axis];
        }
        if (node.view_element_offset >= inner_width || node.view_element_offset + suffix_span > inner_width) {
            continue;
        }

        uint64_t outer = 1;
        for (size_t axis = 0; axis <= collapsed_last_axis; ++axis) {
            outer *= dims[axis];
        }
        return std::vector<uint64_t>{outer, inner_width};
    }

    return std::nullopt;
}

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
                                         const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                         bool accumulate_grad_outputs) {
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
    const std::vector<bool> node_reaches_requested_inputs = computeNodeReachesRequestedInputs(forward_expr, normalized_wrt);

    if (forward_outputs.outputs.size() > 1 && !upstream_input_names_by_output.has_value()) {
        throw std::runtime_error(
            "buildBackwardOutputs for multi-output forward equations requires an explicit upstream input name map. The map may be partial "
            "when some outputs have no incoming gradient.");
    }

    if (upstream_input_names_by_output.has_value() && upstream_input_names_by_output->empty()) {
        throw std::runtime_error("buildBackwardOutputs explicit upstream input map must contain at least one forward output.");
    }

    BackwardGraphBuilder builder(forward_expr);
    builder.initializeAdjoints();

    for (const NamedOutput& forward_output : forward_outputs.outputs) {
        uint32_t output_seed = UINT32_MAX;
        if (upstream_input_names_by_output.has_value()) {
            auto upstream_it = upstream_input_names_by_output->find(forward_output.name);
            if (upstream_it == upstream_input_names_by_output->end()) {
                // A partial explicit upstream map means this forward output did not receive
                // an incoming gradient, so it contributes nothing to the requested wrt gradients.
                continue;
            }
            output_seed = builder.input(upstream_it->second);
        } else {
            output_seed = builder.scalar(1.0);
        }
        builder.addContribution(forward_output.node_idx, output_seed);
    }

    auto addContributionToChild = [&](uint32_t child_idx,
                                      uint32_t contrib,
                                      const std::vector<uint64_t>& contrib_dims,
                                      std::optional<DataType> target_grad_dtype = std::nullopt) {
        uint32_t adjusted_contrib = contrib;
        if (has_forward_dims) {
            const std::vector<uint64_t>& child_dims = forward_node_dims.at(child_idx);
            if (!child_dims.empty()) {
                adjusted_contrib = sumToShape(builder,
                                              contrib,
                                              contrib_dims,
                                              child_dims,
                                              target_grad_dtype.has_value()
                                                  ? target_grad_dtype
                                                  : preferredGradValueDType(forward_expr.nodes.at(child_idx)));
            }
        }
        builder.addContribution(child_idx, adjusted_contrib);
    };

    auto broadcastGradToDims = [&](uint32_t grad_value,
                                   const std::vector<uint64_t>& target_dims,
                                   std::optional<DataType> as_type =
                                       std::nullopt) -> uint32_t {
        if (!has_forward_dims || target_dims.empty()) {
            return grad_value;
        }

        double grad_constant = 0.0;
        if (builder.tryGetConstantLikeValue(grad_value, grad_constant)) {
            return builder.fill(grad_constant, target_dims, as_type);
        }

        return builder.add(builder.fill(0.0, target_dims, as_type), grad_value);
    };

    auto shapeGradLikeNodeOutput =
        [&](uint32_t grad_value, uint32_t forward_node_idx, const std::vector<uint64_t>& forward_node_output_dims) -> uint32_t {
        if (!has_forward_dims || forward_node_output_dims.empty()) {
            return grad_value;
        }

        double constant_value = 0.0;
        std::vector<uint64_t> constant_dims;
        if (!builder.tryGetConstantLike(grad_value, constant_value, constant_dims)) {
            // Tensor-valued upstream gradients are already shaped like the forward node output.  Do not
            // wrap them in fill(0)+grad just to force materialization: doing so creates synthetic fused
            // stages around matmul backward inputs and prevents later planner rewrites from recognizing
            // cuBLASLt backward-epilogue and bgrad patterns.
            return grad_value;
        }

        if (constant_dims == forward_node_output_dims) {
            return grad_value;
        }

        return builder.fill(constant_value,
                            forward_node_output_dims,
                            preferredGradValueDType(forward_expr.nodes.at(forward_node_idx)));
    };

    auto shapeAttentionOutputGrad =
        [&](uint32_t grad_value, uint32_t forward_node_idx, const std::vector<uint64_t>& forward_node_output_dims) -> uint32_t {
        if (!has_forward_dims || forward_node_output_dims.empty()) {
            return grad_value;
        }

        double constant_value = 0.0;
        std::vector<uint64_t> constant_dims;
        if (!builder.tryGetConstantLike(grad_value, constant_value, constant_dims)) {
            // Non-constant upstream gradients are already tensor-valued gradients for
            // the attention output. Do not materialize a fill(0)+grad fused kernel
            // just to stamp cuDNN attention backward; that wrapper is a no-op for
            // the common training path and incorrectly hides the single-stage
            // attention-backward plan behind a synthetic fused stage.
            return grad_value;
        }

        if (constant_dims == forward_node_output_dims) {
            return grad_value;
        }

        return builder.fill(constant_value,
                            forward_node_output_dims,
                            preferredGradValueDType(forward_expr.nodes.at(forward_node_idx)));
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
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                break;

            case ExprOp::ADD:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    addContributionToChild(node.lhs, grad, node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    addContributionToChild(node.rhs, grad, node_dims);
                }
                break;

            case ExprOp::SUB:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    addContributionToChild(node.lhs, grad, node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    addContributionToChild(node.rhs, builder.neg(grad), node_dims);
                }
                break;

            case ExprOp::MUL: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    addContributionToChild(node.lhs, builder.mul(grad, rhs), node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.rhs, builder.mul(grad, lhs), node_dims);
                }
                break;
            }

            case ExprOp::DIV: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    addContributionToChild(node.lhs, builder.div(grad, rhs), node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    const uint32_t rhs_sq = builder.mul(rhs, rhs);
                    addContributionToChild(node.rhs, builder.neg(builder.div(builder.mul(grad, lhs), rhs_sq)), node_dims);
                }
                break;
            }

            case ExprOp::NEG:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    addContributionToChild(node.lhs, builder.neg(grad), node_dims);
                }
                break;

            case ExprOp::ABS: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t neg_lhs = builder.neg(lhs);

                    // sign(lhs) with safe 0 handling:
                    //   x > 0  ->  1
                    //   x < 0  -> -1
                    //   x == 0 ->  0
                    const uint32_t sign_lhs = builder.sub(builder.binary(ExprOp::MAX_GRAD_LEFT, lhs, neg_lhs),
                                                          builder.binary(ExprOp::MAX_GRAD_RIGHT, lhs, neg_lhs));

                    const uint32_t scaled = builder.mul(grad, sign_lhs);
                    addContributionToChild(node.lhs, scaled, node_dims);
                }
                break;
            }

            case ExprOp::TRANSPOSE: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    if (has_forward_dims) {
                        addContributionToChild(node.lhs, builder.unary(ExprOp::TRANSPOSE, grad), forward_node_dims.at(node.lhs));
                    } else {
                        builder.addContribution(node.lhs, builder.unary(ExprOp::TRANSPOSE, grad));
                    }
                }
                break;
            }

            case ExprOp::RESHAPE: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    if (has_forward_dims) {
                        builder.addContribution(node.lhs, builder.reshape(grad, forward_node_dims.at(node.lhs)));
                    } else {
                        throw std::runtime_error("AutoDiff reshape backward requires forward shape information.");
                    }
                }
                break;
            }

            case ExprOp::STRIDED_VIEW:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    std::vector<uint64_t> source_dims;
                    if (has_forward_dims) {
                        source_dims = forward_node_dims.at(node.lhs);
                    } else {
                        const auto inferred_source_dims = inferPackedDenseSourceDimsForStridedViewBackward(node);
                        if (!inferred_source_dims.has_value()) {
                            throw std::runtime_error("AutoDiff strided_view backward requires forward shape information.");
                        }
                        source_dims = inferred_source_dims.value();
                    }
                    builder.addContribution(node.lhs,
                                            builder.stridedViewBackward(grad,
                                                                        source_dims,
                                                                        node.view_dims,
                                                                        node.view_strides,
                                                                        node.view_element_offset,
                                                                        preferredGradValueDType(forward_expr.nodes.at(node.lhs)),
                                                                        preferredGradValueDType(forward_expr.nodes.at(node.lhs))));
                }
                break;

            case ExprOp::UNSQUEEZE: {
                if (has_forward_dims) {
                    const std::vector<uint64_t>& lhs_dims = forward_node_dims.at(node.lhs);
                    const std::vector<uint64_t> actual_unsqueeze_axes = normalizeUnsqueezeAxesForInputDims(lhs_dims, node.unsqueeze_axes);
                    const uint32_t squeezed_grad = builder.squeeze(grad, actual_unsqueeze_axes);
                    if (node_reaches_requested_inputs.at(node.lhs)) {
                        builder.addContribution(node.lhs, squeezed_grad);
                    }

                    // std::cerr << "[AUTODIFF] builder.unsqueeze"
                    //           << " input_node=" << node_idx << " actual_unsqueeze_axes=" << dbgDims(actual_unsqueeze_axes)
                    //           << " node.unsqueeze_axes=" << dbgDims(node.unsqueeze_axes) << std::endl;
                } else {
                    const uint32_t squeezed_grad = builder.squeeze(grad, node.unsqueeze_axes);
                    if (node_reaches_requested_inputs.at(node.lhs)) {
                        builder.addContribution(node.lhs, squeezed_grad);
                    }
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

                    if (node_reaches_requested_inputs.at(node.lhs)) {
                        builder.addContribution(node.lhs, unsqueezed_grad);
                    }

                    // const uint32_t lhs_grad_after_squeeze = builder.gradOf(node.lhs).value();
                    // std::cerr << "[AUTODIFF] SQUEEZE stored lhs grad"
                    //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_squeeze
                    //           << " expected_lhs_dims=" << dbgDims(lhs_dims) << std::endl;
                } else {
                    const uint32_t unsqueezed_grad = builder.unsqueeze(grad, node.squeeze_axes);

                    // std::cerr << "[AUTODIFF] SQUEEZE nodes (no forward dims)"
                    //           << " incoming_grad_node=" << grad << " unsqueezed_grad_node=" << unsqueezed_grad << std::endl;

                    if (node_reaches_requested_inputs.at(node.lhs)) {
                        builder.addContribution(node.lhs, unsqueezed_grad);
                    }

                    // const uint32_t lhs_grad_after_squeeze = builder.gradOf(node.lhs).value();
                    // std::cerr << "[AUTODIFF] SQUEEZE stored lhs grad (no forward dims)"
                    //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_squeeze << std::endl;
                }
                break;
            }

            case ExprOp::EXP: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    addContributionToChild(node.lhs, builder.mul(grad, out), node_dims);
                }
                break;
            }

            case ExprOp::EXPM1: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.exp(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::EXP2: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t scale = builder.scalar(std::log(2.0));
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(out, scale)), node_dims);
                }
                break;
            }

            case ExprOp::EXP10: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t scale = builder.scalar(std::log(10.0));
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(out, scale)), node_dims);
                }
                break;
            }

            case ExprOp::LN: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.div(grad, lhs), node_dims);
                }
                break;
            }

            case ExprOp::LOG1P: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.add(builder.scalar(1.0), lhs);
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::LOG2: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.mul(lhs, builder.scalar(std::log(2.0)));
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::LOG10: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.mul(lhs, builder.scalar(std::log(10.0)));
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::SQRT: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t denom = builder.mul(builder.scalar(2.0), out);
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::TANH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t one = builder.scalar(1.0);
                    const uint32_t one_minus_out_squared = builder.sub(one, builder.mul(out, out));
                    addContributionToChild(node.lhs, builder.mul(grad, one_minus_out_squared), node_dims);
                }
                break;
            }

            case ExprOp::NORMCDF: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t neg_half = builder.scalar(-0.5);
                    const uint32_t inv_sqrt_two_pi = builder.scalar(0.3989422804014327);
                    const uint32_t pdf = builder.mul(inv_sqrt_two_pi, builder.exp(builder.mul(neg_half, builder.mul(lhs, lhs))));
                    addContributionToChild(node.lhs, builder.mul(grad, pdf), node_dims);
                }
                break;
            }

            case ExprOp::ROPE: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs_grad = builder.rotaryPositionEmbedding(
                        grad,
                        node,
                        !node.rope_inverse,
                        preferredGradValueDType(forward_expr.nodes.at(node.lhs)),
                        node.compute_dtype);
                    addContributionToChild(node.lhs, lhs_grad, node_dims);
                }
                break;
            }

            case ExprOp::SOFTMAX: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const std::vector<uint64_t>& lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                    if (lhs_dims.size() < 2) {
                        throw std::runtime_error("Autodiff for cuDNN softmax currently expects rank >= 2 tensors.");
                    }

                    std::vector<uint64_t> axes;
                    if (node.softmax_mode == CUDNN_SOFTMAX_MODE_CHANNEL) {
                        axes = {1};
                    } else if (node.softmax_mode == CUDNN_SOFTMAX_MODE_INSTANCE) {
                        for (uint64_t axis = 1; axis < lhs_dims.size(); ++axis) {
                            axes.push_back(axis);
                        }
                    } else {
                        throw std::runtime_error("Autodiff for cuDNN softmax received unsupported mode.");
                    }

                    if (node.softmax_algorithm == CUDNN_SOFTMAX_LOG) {
                        const uint32_t lhs = builder.cloneForward(node.lhs);
                        const uint32_t ordinary_softmax = builder.softmax(lhs, CUDNN_SOFTMAX_ACCURATE, node.softmax_mode);
                        const uint32_t sum_grad = builder.reduction(ExprOp::REDUCE_SUM, grad, axes, {});
                        const uint32_t correction = builder.mul(ordinary_softmax, sum_grad);
                        addContributionToChild(node.lhs, builder.sub(grad, correction), node_dims);
                    } else {
                        const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                        const uint32_t sum_grad_times_out = builder.reduction(ExprOp::REDUCE_SUM, builder.mul(grad, out), axes, {});
                        addContributionToChild(node.lhs, builder.mul(out, builder.sub(grad, sum_grad_times_out)), node_dims);
                    }
                }
                break;
            }

            case ExprOp::RMSNORM: {
                if (!has_forward_dims) {
                    throw std::runtime_error("Autodiff RMSNorm backward requires forward shape information.");
                }
                const std::vector<uint64_t>& x_dims = forward_node_dims.at(node.lhs);
                const std::vector<uint64_t>& scale_dims = forward_node_dims.at(node.rhs);
                if (x_dims.size() != 2 || scale_dims.size() != 1 || x_dims[1] != node.rms_norm_normalized_feature_count ||
                    scale_dims[0] != node.rms_norm_normalized_feature_count) {
                    throw std::runtime_error("Autodiff RMSNorm backward expects [outer, hidden] input and [hidden] scale tensors.");
                }

                uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                const uint32_t x = builder.cloneForward(node.lhs);
                const uint32_t scale = builder.cloneForward(node.rhs);

                const uint32_t x_squared = builder.mul(x, x);
                const uint32_t mean_x_squared =
                    builder.reduction(ExprOp::REDUCE_AVG, x_squared, {1}, {}, node.compute_dtype, node.compute_dtype);
                const uint32_t inv_rms = builder.div(
                    builder.scalar(1.0),
                    builder.unary(ExprOp::SQRT, builder.add(mean_x_squared, builder.scalar(node.rms_norm_epsilon))));

                if (node.rms_norm_fused_activation == CudnnRmsNormFusedActivation::SWISH) {
                    const uint32_t z = builder.mul(builder.mul(x, scale), inv_rms);
                    const uint32_t sigmoid = builder.div(builder.scalar(1.0),
                                                         builder.add(builder.scalar(1.0), builder.exp(builder.neg(z))));
                    const uint32_t one_minus_sigmoid = builder.sub(builder.scalar(1.0), sigmoid);
                    const uint32_t swish_grad = builder.mul(sigmoid, builder.add(builder.scalar(1.0), builder.mul(z, one_minus_sigmoid)));
                    grad_like_output = builder.mul(grad_like_output, swish_grad);
                }

                const uint32_t scaled_grad = builder.mul(grad_like_output, scale);

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t mean_scaled_grad_x =
                        builder.reduction(ExprOp::REDUCE_AVG, builder.mul(scaled_grad, x), {1}, {}, node.compute_dtype, node.compute_dtype);
                    const uint32_t inv_rms_squared = builder.mul(inv_rms, inv_rms);
                    const uint32_t inv_rms_cubed = builder.mul(inv_rms_squared, inv_rms);
                    const uint32_t first = builder.mul(scaled_grad, inv_rms);
                    const uint32_t second = builder.mul(x, builder.mul(inv_rms_cubed, mean_scaled_grad_x));
                    addContributionToChild(node.lhs, builder.sub(first, second), x_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t dscale = builder.reduction(ExprOp::REDUCE_SUM, builder.mul(grad_like_output, builder.mul(x, inv_rms)),
                                                              {0}, {0}, node.compute_dtype, preferredGradValueDType(forward_expr.nodes.at(node.rhs)));
                    addContributionToChild(node.rhs, dscale, scale_dims);
                }
                break;
            }

            case ExprOp::POW: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    const uint32_t rhs_minus_one = builder.sub(rhs, builder.scalar(1.0));
                    const uint32_t lhs_pow_rhs_minus_one = builder.binary(ExprOp::POW, lhs, rhs_minus_one);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(rhs, lhs_pow_rhs_minus_one)), node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    addContributionToChild(node.rhs, builder.mul(grad, builder.mul(out, builder.unary(ExprOp::LN, lhs))), node_dims);
                }
                break;
            }

            case ExprOp::REDUCE_SUM: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                    uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                    if (has_forward_dims && !node.squeeze_axes.empty()) {
                        const std::vector<uint64_t> unsqueeze_axes =
                            normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                        grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                    }

                    // std::cerr << "[AUTODIFF] REDUCE_SUM backward"
                    //           << " node=" << node_idx << " lhs=" << node.lhs << " grad_node=" << grad << " node_dims=" <<
                    //           dbgDims(node_dims)
                    //           << " lhs_dims=" << dbgDims(lhs_dims) << " reduction_axes=" << dbgDims(node.reduction_axes)
                    //           << " squeeze_axes=" << dbgDims(node.squeeze_axes) << std::endl;

                    const uint32_t expanded_grad =
                        broadcastGradToDims(grad_before_expand, lhs_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));

                    // std::cerr << "[AUTODIFF] REDUCE_SUM nodes"
                    //           << " grad_before_expand=" << grad_before_expand << " expanded_grad=" << expanded_grad << std::endl;

                    addContributionToChild(node.lhs, expanded_grad, lhs_dims);

                    // const uint32_t lhs_grad_after_reduce = builder.gradOf(node.lhs).value();
                    // std::cerr << "[AUTODIFF] REDUCE_SUM stored lhs grad"
                    //           << " lhs=" << node.lhs << " lhs_grad_node=" << lhs_grad_after_reduce << " expected_lhs_dims=" <<
                    //           dbgDims(lhs_dims)
                    //           << std::endl;
                }

                break;
            }

            case ExprOp::REDUCE_AVG: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                    uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                    if (has_forward_dims && !node.squeeze_axes.empty()) {
                        const std::vector<uint64_t> unsqueeze_axes =
                            normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                        grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                    }

                    const uint32_t expanded_grad =
                        broadcastGradToDims(grad_before_expand, lhs_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));

                    uint32_t scaled_grad = expanded_grad;
                    if (has_forward_dims) {
                        const uint64_t count = reductionElementCount(lhs_dims, node.reduction_axes);
                        scaled_grad = builder.div(expanded_grad, builder.scalar(static_cast<double>(count)));
                    }

                    addContributionToChild(node.lhs, scaled_grad, lhs_dims);
                }
                break;
            }

            case ExprOp::REDUCE_NORM2: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
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
                    const uint32_t expanded_grad =
                        broadcastGradToDims(grad_before_expand, lhs_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));
                    const uint32_t scaled = builder.div(builder.mul(expanded_grad, lhs), out);
                    addContributionToChild(node.lhs, scaled, lhs_dims);
                }
                break;
            }

            case ExprOp::MIN: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t rhs = builder.cloneForward(node.rhs);
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs_mask = builder.binary(ExprOp::MIN_GRAD_LEFT, lhs, rhs);
                    addContributionToChild(node.lhs, builder.mul(grad, lhs_mask), node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t rhs_mask = builder.binary(ExprOp::MIN_GRAD_RIGHT, lhs, rhs);
                    addContributionToChild(node.rhs, builder.mul(grad, rhs_mask), node_dims);
                }
                break;
            }

            case ExprOp::MAX: {
                const uint32_t lhs = builder.cloneForward(node.lhs);
                const uint32_t rhs = builder.cloneForward(node.rhs);
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs_mask = builder.binary(ExprOp::MAX_GRAD_LEFT, lhs, rhs);
                    addContributionToChild(node.lhs, builder.mul(grad, lhs_mask), node_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t rhs_mask = builder.binary(ExprOp::MAX_GRAD_RIGHT, lhs, rhs);
                    addContributionToChild(node.rhs, builder.mul(grad, rhs_mask), node_dims);
                }
                break;
            }

            case ExprOp::REDUCE_PROD: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
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
                    const uint32_t expanded_grad =
                        broadcastGradToDims(grad_before_expand, lhs_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));

                    // Safe-case assumption: reduced product inputs are nonzero where this backward is used.
                    const uint32_t scaled = builder.div(builder.mul(expanded_grad, out), lhs);
                    addContributionToChild(node.lhs, scaled, lhs_dims);
                }
                break;
            }

            case ExprOp::REDUCE_NORM1: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;

                    uint32_t grad_before_expand = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);

                    if (has_forward_dims && !node.squeeze_axes.empty()) {
                        const std::vector<uint64_t> unsqueeze_axes =
                            normalizedReductionUnsqueezeAxes(lhs_dims, node.reduction_axes, node.squeeze_axes);
                        grad_before_expand = builder.unsqueeze(grad_before_expand, unsqueeze_axes);
                    }

                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t expanded_grad =
                        broadcastGradToDims(grad_before_expand, lhs_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));

                    const uint32_t neg_lhs = builder.neg(lhs);
                    const uint32_t sign_lhs = builder.sub(builder.binary(ExprOp::MAX_GRAD_LEFT, lhs, neg_lhs),
                                                          builder.binary(ExprOp::MAX_GRAD_RIGHT, lhs, neg_lhs));

                    const uint32_t scaled = builder.mul(expanded_grad, sign_lhs);
                    addContributionToChild(node.lhs, scaled, lhs_dims);
                }
                break;
            }

            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                    uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                    const uint32_t routed = builder.reduceMinMaxBackward(
                        node.op == ExprOp::REDUCE_MIN ? ExprOp::REDUCE_MIN_BACKWARD : ExprOp::REDUCE_MAX_BACKWARD,
                        builder.cloneForward(node.lhs),
                        grad_like_output,
                        node.reduction_axes,
                        node.squeeze_axes,
                        preferredGradValueDType(forward_expr.nodes.at(node.lhs)),
                        node.compute_dtype);
                    addContributionToChild(node.lhs, routed, lhs_dims);
                }
                break;
            }

            case ExprOp::MATMUL: {
                uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                grad_like_output = builder.applyForwardMatmulEpilogueBackward(node, grad_like_output);
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> rhs_dims = has_forward_dims ? forward_node_dims.at(node.rhs) : std::vector<uint64_t>{};
                const auto lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                const auto rhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.rhs));

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    uint32_t lhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(grad_like_output, rhs, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        lhs_grad = builder.matmul(grad_like_output, rhs, false, false, lhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(rhs, grad_like_output, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else {
                        lhs_grad = builder.matmul(rhs, grad_like_output, true, true, lhs_grad_dtype, node.compute_dtype);
                    }

                    lhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, lhs_grad);
                    addContributionToChild(node.lhs, lhs_grad, lhs_dims);
                }

                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    uint32_t rhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, grad_like_output, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        rhs_grad = builder.matmul(grad_like_output, lhs, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, grad_like_output, false, false, rhs_grad_dtype, node.compute_dtype);
                    } else {
                        rhs_grad = builder.matmul(grad_like_output, lhs, true, true, rhs_grad_dtype, node.compute_dtype);
                    }

                    rhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, rhs_grad);
                    addContributionToChild(node.rhs, rhs_grad, rhs_dims);
                }
                break;
            }

            case ExprOp::CONV2D:
            case ExprOp::CONV3D: {
                const uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> rhs_dims = has_forward_dims ? forward_node_dims.at(node.rhs) : std::vector<uint64_t>{};
                const auto lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                const auto rhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.rhs));

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t filter = builder.cloneForward(node.rhs);
                    uint32_t lhs_grad = UINT32_MAX;
                    if (node.op == ExprOp::CONV3D) {
                        lhs_grad = builder.conv3dBackwardData(filter,
                                                              grad_like_output,
                                                              node.conv_stride_d,
                                                              node.conv_stride_h,
                                                              node.conv_stride_w,
                                                              node.conv_pad_d,
                                                              node.conv_pad_h,
                                                              node.conv_pad_w,
                                                              lhs_dims,
                                                              lhs_grad_dtype,
                                                              node.compute_dtype);
                    } else {
                        lhs_grad = builder.conv2dBackwardData(filter,
                                                              grad_like_output,
                                                              node.conv_stride_h,
                                                              node.conv_stride_w,
                                                              node.conv_pad_h,
                                                              node.conv_pad_w,
                                                              lhs_dims,
                                                              lhs_grad_dtype,
                                                              node.compute_dtype);
                    }
                    addContributionToChild(node.lhs, lhs_grad, lhs_dims);
                }

                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t input = builder.cloneForward(node.lhs);
                    uint32_t rhs_grad = UINT32_MAX;
                    if (node.op == ExprOp::CONV3D) {
                        rhs_grad = builder.conv3dBackwardFilter(input,
                                                                grad_like_output,
                                                                node.conv_stride_d,
                                                                node.conv_stride_h,
                                                                node.conv_stride_w,
                                                                node.conv_pad_d,
                                                                node.conv_pad_h,
                                                                node.conv_pad_w,
                                                                rhs_dims,
                                                                rhs_grad_dtype,
                                                                node.compute_dtype);
                    } else {
                        rhs_grad = builder.conv2dBackwardFilter(input,
                                                                grad_like_output,
                                                                node.conv_stride_h,
                                                                node.conv_stride_w,
                                                                node.conv_pad_h,
                                                                node.conv_pad_w,
                                                                rhs_dims,
                                                                rhs_grad_dtype,
                                                                node.compute_dtype);
                    }
                    addContributionToChild(node.rhs, rhs_grad, rhs_dims);
                }
                break;
            }

            case ExprOp::GEMM: {
                uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                grad_like_output = builder.applyForwardMatmulEpilogueBackward(node, grad_like_output);
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> rhs_dims = has_forward_dims ? forward_node_dims.at(node.rhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> aux_dims = has_forward_dims ? forward_node_dims.at(node.aux) : std::vector<uint64_t>{};
                const auto lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                const auto rhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.rhs));
                auto aux_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.aux));
                if (!aux_grad_dtype.has_value()) {
                    // A GEMM rank-1/full-rank addend is constrained to the GEMM output dtype by the
                    // cuBLASLt epilogue path.  Plain input nodes do not have their runtime dtype resolved
                    // while the backward graph is being built, so use the forward GEMM's public output
                    // dtype as the requested materialized dtype for dAux.  Without this, an FP16 bias
                    // gradient reduction resolves to the reduction default FP32 output and callers that
                    // request the FP16 public grad need a trailing conversion wrapper.
                    aux_grad_dtype = preferredGradValueDType(node);
                }

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t rhs = builder.cloneForward(node.rhs);
                    uint32_t lhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(grad_like_output, rhs, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        lhs_grad = builder.matmul(grad_like_output, rhs, false, false, lhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(rhs, grad_like_output, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else {
                        lhs_grad = builder.matmul(rhs, grad_like_output, true, true, lhs_grad_dtype, node.compute_dtype);
                    }

                    lhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, lhs_grad);
                    addContributionToChild(node.lhs, lhs_grad, lhs_dims);
                }

                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    uint32_t rhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, grad_like_output, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        rhs_grad = builder.matmul(grad_like_output, lhs, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, grad_like_output, false, false, rhs_grad_dtype, node.compute_dtype);
                    } else {
                        rhs_grad = builder.matmul(grad_like_output, lhs, true, true, rhs_grad_dtype, node.compute_dtype);
                    }

                    rhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, rhs_grad);
                    addContributionToChild(node.rhs, rhs_grad, rhs_dims);
                }

                const std::vector<uint64_t> alpha_dims =
                    (has_forward_dims && node.alpha_node != UINT32_MAX) ? forward_node_dims.at(node.alpha_node) : std::vector<uint64_t>{};
                const std::vector<uint64_t> beta_dims =
                    (has_forward_dims && node.beta_node != UINT32_MAX) ? forward_node_dims.at(node.beta_node) : std::vector<uint64_t>{};

                if (node.alpha_node != UINT32_MAX && node_reaches_requested_inputs.at(node.alpha_node)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t rhs = builder.cloneForward(node.rhs);

                    uint32_t alpha_term = builder.matmul(lhs,
                                                         rhs,
                                                         node.transpose_lhs,
                                                         node.transpose_rhs,
                                                         std::nullopt,
                                                         node.compute_dtype);

                    uint32_t alpha_grad = builder.mul(grad_like_output, alpha_term);
                    addContributionToChild(node.alpha_node, alpha_grad, node_dims);
                }

                if (node.beta_node != UINT32_MAX && node_reaches_requested_inputs.at(node.beta_node)) {
                    if (node.transpose_aux) {
                        throw std::runtime_error(
                            "Thor expressions autodiff does not yet support backward for GEMM beta subexpression with "
                            "transpose_aux/transposeC.");
                    }

                    const uint32_t aux = builder.cloneForward(node.aux);
                    uint32_t beta_grad = builder.mul(grad_like_output, aux);
                    addContributionToChild(node.beta_node, beta_grad, node_dims);
                }

                if (node_reaches_requested_inputs.at(node.aux)) {
                    if (node.transpose_aux) {
                        throw std::runtime_error(
                            "Thor expressions autodiff does not yet support backward for GEMM with transpose_aux/transposeC.");
                    }
                    uint32_t aux_grad = grad_like_output;
                    aux_grad = builder.buildScaledByGemmFactor(node.beta_node, node.beta_fp, aux_grad);

                    if (has_forward_dims && !aux_dims.empty()) {
                        // GEMM supports both full-rank addends and rank-1 bias vectors.  The gradient flowing into
                        // the addend is initially shaped like the GEMM output.  For a bias-vector addend this must be
                        // reduced across the broadcasted batch axis before it is written to the optimizer's [out]
                        // gradient buffer; broadcasting a [out] zero tensor up to [batch, out] would leave the terminal
                        // gradient with the wrong logical shape and fail preallocated-output validation.
                        aux_grad = sumToShape(builder, aux_grad, node_dims, aux_dims, aux_grad_dtype);
                    } else {
                        aux_grad = broadcastGradToDims(aux_grad, aux_dims, aux_grad_dtype);
                    }

                    addContributionToChild(node.aux, aux_grad, aux_dims);
                }
                break;
            }

            case ExprOp::ATTENTION: {
                const uint32_t grad_like_output = shapeAttentionOutputGrad(grad, static_cast<uint32_t>(node_idx), node_dims);
                if (node.attention_use_paged_kv_cache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled() &&
                    (node_reaches_requested_inputs.at(node.lhs) || node_reaches_requested_inputs.at(node.rhs) ||
                     node_reaches_requested_inputs.at(node.aux))) {
                    throw std::runtime_error(
                        "Attention-backward with paged KV cache is not enabled; the paged KV path is inference-only until training semantics are defined.");
                }
                if (node.attention_use_fp8_forward_scaling &&
                    (node_reaches_requested_inputs.at(node.lhs) || node_reaches_requested_inputs.at(node.rhs) ||
                     node_reaches_requested_inputs.at(node.aux) ||
                     (node.attention_use_bias && node_reaches_requested_inputs.at(node.alpha_node)))) {
                    throw std::runtime_error(
                        "FP8 cuDNN attention is forward-only in Thor; cuDNN FP8 SDPA backward is not supported on the validated support surface.");
                }
                const std::vector<uint64_t> q_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> k_dims = has_forward_dims ? forward_node_dims.at(node.rhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> v_dims = has_forward_dims ? forward_node_dims.at(node.aux) : std::vector<uint64_t>{};
                const std::vector<uint64_t> bias_dims =
                    (has_forward_dims && node.attention_use_bias && node.alpha_node != UINT32_MAX)
                        ? forward_node_dims.at(node.alpha_node)
                        : std::vector<uint64_t>{};
                const uint32_t q = builder.cloneForward(node.lhs);
                const uint32_t k = builder.cloneForward(node.rhs);
                const uint32_t v = builder.cloneForward(node.aux);
                uint32_t bias = node.attention_use_bias ? builder.cloneForward(node.alpha_node) : UINT32_MAX;
                if (bias != UINT32_MAX && q_dims.size() == 4 && k_dims.size() == 4 && bias_dims.size() == 4) {
                    const std::vector<uint64_t> dense_score_bias_dims = inferAttentionDenseBiasDims(node, q_dims, k_dims);
                    const bool broadcasts_query_sequence = bias_dims[2] == 1 && dense_score_bias_dims[2] != 1;
                    const bool broadcasts_key_sequence = bias_dims[3] == 1 && dense_score_bias_dims[3] != 1;
                    if ((broadcasts_query_sequence || broadcasts_key_sequence) && bias_dims != dense_score_bias_dims &&
                        !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
                        // cuDNN's native backward surface is not reliable for score bias tensors broadcast across
                        // sequence axes: some shapes are rejected by primary heuristics on SM120 and some accepted
                        // Skv-vector cases produce incorrect dV/dBias.  Keep the public forward surface broad, but
                        // lower production backward through an explicit dense score-bias materialization, then let
                        // the normal broadcast-gradient rule reduce dense dBias back to the original bias shape.
                        const auto bias_backward_dtype = node.compute_dtype.has_value() ? node.compute_dtype : preferredGradValueDType(node);
                        bias = builder.addNoFold(builder.fill(0.0, dense_score_bias_dims, bias_backward_dtype),
                                                 bias,
                                                 bias_backward_dtype,
                                                 bias_backward_dtype);
                    }
                }
                ExprNode attention_for_backward = node;
                if (node.attention_use_padding_mask) {
                    attention_for_backward.attention_seq_len_q_node = builder.cloneForward(node.attention_seq_len_q_node);
                    attention_for_backward.attention_seq_len_kv_node = builder.cloneForward(node.attention_seq_len_kv_node);
                }
                if (node.attention_use_ragged_offsets) {
                    attention_for_backward.attention_ragged_offset_q_node = builder.cloneForward(node.attention_ragged_offset_q_node);
                    attention_for_backward.attention_ragged_offset_kv_node = builder.cloneForward(node.attention_ragged_offset_kv_node);
                }
                if (node.attention_use_paged_kv_cache) {
                    attention_for_backward.attention_page_table_k_node = builder.cloneForward(node.attention_page_table_k_node);
                    attention_for_backward.attention_page_table_v_node = builder.cloneForward(node.attention_page_table_v_node);
                }
                if (node.attention_dropout_probability > 0.0f) {
                    attention_for_backward.attention_dropout_seed_node = builder.cloneForward(node.attention_dropout_seed_node);
                    attention_for_backward.attention_dropout_offset_node = builder.cloneForward(node.attention_dropout_offset_node);
                }

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t dQ = builder.attentionBackward(ExprOp::ATTENTION_BACKWARD_Q,
                                                                  q,
                                                                  k,
                                                                  v,
                                                                  grad_like_output,
                                                                  bias,
                                                                  attention_for_backward,
                                                                  preferredGradValueDType(forward_expr.nodes.at(node.lhs)),
                                                                  node.compute_dtype);
                    addContributionToChild(node.lhs, dQ, q_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t dK = builder.attentionBackward(ExprOp::ATTENTION_BACKWARD_K,
                                                                  q,
                                                                  k,
                                                                  v,
                                                                  grad_like_output,
                                                                  bias,
                                                                  attention_for_backward,
                                                                  preferredGradValueDType(forward_expr.nodes.at(node.rhs)),
                                                                  node.compute_dtype);
                    addContributionToChild(node.rhs, dK, k_dims);
                }
                if (node_reaches_requested_inputs.at(node.aux)) {
                    const uint32_t dV = builder.attentionBackward(ExprOp::ATTENTION_BACKWARD_V,
                                                                  q,
                                                                  k,
                                                                  v,
                                                                  grad_like_output,
                                                                  bias,
                                                                  attention_for_backward,
                                                                  preferredGradValueDType(forward_expr.nodes.at(node.aux)),
                                                                  node.compute_dtype);
                    addContributionToChild(node.aux, dV, v_dims);
                }
                if (node.attention_use_bias && node.alpha_node != UINT32_MAX && node_reaches_requested_inputs.at(node.alpha_node)) {
                    const std::vector<uint64_t> dbias_dense_dims = has_forward_dims
                        ? inferAttentionBackwardOutputDims(node, ExprOp::ATTENTION_BACKWARD_BIAS, q_dims, k_dims, v_dims, node_dims)
                        : std::vector<uint64_t>{};
                    const auto dbias_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                    const uint32_t dBias = builder.attentionBackward(ExprOp::ATTENTION_BACKWARD_BIAS,
                                                                     q,
                                                                     k,
                                                                     v,
                                                                     grad_like_output,
                                                                     bias,
                                                                     attention_for_backward,
                                                                     dbias_dtype,
                                                                     node.compute_dtype);
                    addContributionToChild(node.alpha_node, dBias, dbias_dense_dims, dbias_dtype);
                }
                break;
            }

            case ExprOp::ATTENTION_BACKWARD_Q:
            case ExprOp::ATTENTION_BACKWARD_K:
            case ExprOp::ATTENTION_BACKWARD_V:
            case ExprOp::ATTENTION_BACKWARD_BIAS:
                throw std::runtime_error("Thor expressions autodiff does not support second derivatives for attention backward yet.");

            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
                throw std::runtime_error("Thor expressions autodiff does not support backward for op " + opName(node.op) + ".");

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

    struct PendingBackwardOutput {
        std::string name;
        uint32_t node_idx;
        std::optional<DataType> target_output_dtype;
    };

    std::vector<PendingBackwardOutput> pending_outputs;
    pending_outputs.reserve(normalized_wrt.size());

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

        const ExprNode& forward_input_node = forward_expr.nodes.at(first_it->second);
        std::optional<DataType> grad_dtype = preferredGradValueDType(forward_input_node);
        const std::string grad_output_name = wrt_name + "_grad";

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

            total_grad = total_grad.has_value() ? std::optional<uint32_t>(builder.add(total_grad.value(), grad_opt.value()))
                                                : std::optional<uint32_t>(grad_opt.value());
        }

        if (!total_grad.has_value()) {
            if (accumulate_grad_outputs) {
                total_grad = builder.input(grad_output_name, grad_dtype);
            } else if (has_forward_dims) {
                total_grad = builder.fill(0.0, forward_node_dims.at(first_it->second), grad_dtype);
            } else {
                const uint32_t input_clone = builder.cloneForward(first_it->second);
                total_grad = builder.mul(input_clone, builder.scalar(0.0));
            }
        } else if (accumulate_grad_outputs) {
            total_grad = builder.add(builder.input(grad_output_name, grad_dtype), total_grad.value());
        } else {
            const auto dbias_only_dtype = attentionBackwardBiasOnlyDType(builder, total_grad.value());
            if (dbias_only_dtype.has_value()) {
                // cuDNN SDPA backward produces dBias in the Q/input dtype, not the additive-bias tensor dtype.
                // When the public bias_grad is composed only from attention dBias outputs, keep the terminal dtype
                // there so same-plan duplicated/merged attention expressions do not promote back to the FP32 bias
                // tensor dtype and force an unnecessary down-conversion for callers that want cuDNN's native dBias.
                grad_dtype = dbias_only_dtype.value();
            }
        }

        // Create a dedicated terminal output node so we can force only the final
        // written grad dtype, while leaving internal promoted compute untouched.
        // Stage-boundary gradients that already declare the requested output dtype
        // can be written directly; wrapping them in grad + fill(0) only creates a
        // synthetic fused kernel after the real backend stage.
        uint32_t terminal_grad = total_grad.value();
        bool terminal_already_forces_dtype = false;
        if (grad_dtype.has_value()) {
            const ExprNode& terminal_node = builder.node(terminal_grad);
            terminal_already_forces_dtype = terminal_node.output_dtype.has_value() &&
                                           terminal_node.output_dtype.value() == grad_dtype.value() &&
                                           isStageBoundaryLikeBackwardOutputOp(terminal_node.op);
        }
        if (grad_dtype.has_value() && !terminal_already_forces_dtype) {
            uint32_t zero_like;
            if (has_forward_dims) {
                zero_like = builder.fill(0.0, forward_node_dims.at(first_it->second), grad_dtype);
            } else if (const auto inferred_grad_dims = builder.tryInferKnownGradientDims(total_grad.value()); inferred_grad_dims.has_value()) {
                // Avoid cloning the primal input just to manufacture a zero tensor when the
                // backward graph already carries its terminal shape, as strided-view backward does.
                zero_like = builder.fill(0.0, inferred_grad_dims.value(), grad_dtype);
            } else {
                const uint32_t input_clone = builder.cloneForward(first_it->second);
                zero_like = builder.mul(input_clone, builder.scalar(0.0));
            }

            terminal_grad = builder.addNoFold(total_grad.value(), zero_like, grad_dtype, grad_dtype);
        }

        pending_outputs.push_back(PendingBackwardOutput{
            .name = grad_output_name,
            .node_idx = terminal_grad,
            .target_output_dtype = grad_dtype,
        });
    }

    PhysicalOutputs backward_outputs;
    backward_outputs.expr = std::make_shared<PhysicalExpression>(builder.takeExpression());
    backward_outputs.outputs.reserve(pending_outputs.size());
    for (const PendingBackwardOutput& output : pending_outputs) {
        backward_outputs.outputs.push_back(NamedOutput{
            .name = output.name,
            .node_idx = output.node_idx,
        });
    }

    return backward_outputs;
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::optional<std::string>& upstream_input_name,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                     bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_name),
                                    forward_input_dims,
                                    accumulate_grad_outputs);
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                     bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_names_by_output),
                                    forward_input_dims,
                                    accumulate_grad_outputs);
}

}  // namespace ThorImplementation
