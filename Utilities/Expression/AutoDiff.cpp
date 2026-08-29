#include "Utilities/Expression/AutoDiff.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "Utilities/Expression/BatchedMatmulPlan.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {
namespace {

static constexpr uint64_t EXPRESSION_COPY_DIM = 0;
static constexpr uint64_t EXPRESSION_INFER_DIM = std::numeric_limits<uint64_t>::max();

static uint64_t dynamicDimsNumel(const std::vector<uint64_t>& dims, const std::string& what) {
    uint64_t result = 1;
    for (uint64_t dim : dims) {
        if (dim == EXPRESSION_COPY_DIM || dim == EXPRESSION_INFER_DIM) {
            throw std::runtime_error(what + " contains unresolved dynamic dimensions.");
        }
        if (result > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error(what + " dimensions overflow uint64_t.");
        }
        result *= dim;
    }
    return result;
}

static std::vector<uint64_t> resolveDynamicAliasDims(const std::vector<uint64_t>& source_dims,
                                                    const std::vector<uint64_t>& requested_dims,
                                                    bool must_preserve_numel,
                                                    const std::string& what) {
    if (requested_dims.empty()) {
        throw std::runtime_error(what + " requires non-empty dimensions.");
    }

    std::vector<uint64_t> resolved = requested_dims;
    std::optional<size_t> infer_index;
    uint64_t known_product = 1;
    for (size_t i = 0; i < resolved.size(); ++i) {
        uint64_t dim = resolved[i];
        if (dim == EXPRESSION_COPY_DIM) {
            if (i >= source_dims.size()) {
                throw std::runtime_error(what + " copy-dimension marker is out of range for source rank.");
            }
            dim = source_dims[i];
            if (dim == 0) {
                throw std::runtime_error(what + " resolved copy dimension must be non-zero.");
            }
            resolved[i] = dim;
        } else if (dim == EXPRESSION_INFER_DIM) {
            if (!must_preserve_numel) {
                throw std::runtime_error(what + " does not support infer-dimension markers.");
            }
            if (infer_index.has_value()) {
                throw std::runtime_error(what + " supports at most one infer-dimension marker.");
            }
            infer_index = i;
            continue;
        }

        if (known_product > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error(what + " resolved dimensions overflow uint64_t.");
        }
        known_product *= dim;
    }

    if (must_preserve_numel) {
        const uint64_t source_numel = dynamicDimsNumel(source_dims, what + " source");
        if (infer_index.has_value()) {
            if (known_product == 0 || source_numel % known_product != 0) {
                throw std::runtime_error(what + " cannot infer a dimension that preserves the number of elements.");
            }
            resolved[infer_index.value()] = source_numel / known_product;
        } else if (source_numel != known_product) {
            throw std::runtime_error(what + " must preserve the number of elements.");
        }
    }

    return resolved;
}

double digammaApproxForMl(double x) {
    constexpr double pi = 3.14159265358979323846264338327950288;

    if (std::isnan(x)) {
        return x;
    }
    if (std::isinf(x)) {
        return x > 0.0 ? x : std::numeric_limits<double>::quiet_NaN();
    }

    double result = 0.0;
    if (x <= 0.0) {
        const double floored = std::floor(x);
        if (x == floored) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        result -= pi / std::tan(pi * x);
        x = 1.0 - x;
    }

    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }

    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    const double inv4 = inv2 * inv2;
    const double inv6 = inv4 * inv2;
    const double inv8 = inv4 * inv4;
    const double inv10 = inv8 * inv2;

    result += std::log(x) - 0.5 * inv;
    result -= inv2 * (1.0 / 12.0);
    result += inv4 * (1.0 / 120.0);
    result -= inv6 * (1.0 / 252.0);
    result += inv8 * (1.0 / 240.0);
    result -= inv10 * (1.0 / 132.0);
    return result;
}

bool experimentalCudnnAttentionSupportSurfaceProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE");
    return value != nullptr && std::string_view(value) == "1";
}

bool experimentalCudnnRaggedBiasBackwardProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD");
    return (value != nullptr && std::string_view(value) == "1") || experimentalCudnnAttentionSupportSurfaceProbeEnabled();
}

static std::vector<uint64_t> inferTransposeOutputDims(const std::vector<uint64_t>& input_dims);

static std::vector<uint64_t> resolveReductionAxesForAutodiff(const std::vector<uint64_t>& reduction_axes, size_t input_rank) {
    if (!reduction_axes.empty()) {
        return reduction_axes;
    }

    std::vector<uint64_t> axes(input_rank);
    for (size_t i = 0; i < input_rank; ++i) {
        axes[i] = static_cast<uint64_t>(i);
    }
    return axes;
}

uint32_t cloneForwardSubtree(const PhysicalExpression& src,
                             uint32_t src_node_index,
                             PhysicalExpression& dst,
                             std::unordered_map<uint32_t, uint32_t>& old_to_new,
                             std::unordered_map<uint32_t, uint32_t>& old_cuda_to_new) {
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
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new, old_cuda_to_new);
        new_node.rhs = UINT32_MAX;
        new_node.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX || src_node.rhs == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: binary node missing child.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new, old_cuda_to_new);
        new_node.rhs = cloneForwardSubtree(src, src_node.rhs, dst, old_to_new, old_cuda_to_new);
        new_node.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(src_node.op)) {
        if (src_node.lhs == UINT32_MAX || src_node.rhs == UINT32_MAX || src_node.aux == UINT32_MAX) {
            throw std::runtime_error("Malformed forward expression: ternary node missing child.");
        }
        new_node.lhs = cloneForwardSubtree(src, src_node.lhs, dst, old_to_new, old_cuda_to_new);
        new_node.rhs = cloneForwardSubtree(src, src_node.rhs, dst, old_to_new, old_cuda_to_new);
        new_node.aux = cloneForwardSubtree(src, src_node.aux, dst, old_to_new, old_cuda_to_new);
        if (src_node.alpha_node != UINT32_MAX) {
            new_node.alpha_node = cloneForwardSubtree(src, src_node.alpha_node, dst, old_to_new, old_cuda_to_new);
        }
        if (src_node.beta_node != UINT32_MAX) {
            new_node.beta_node = cloneForwardSubtree(src, src_node.beta_node, dst, old_to_new, old_cuda_to_new);
        }
        if (src_node.attention_use_padding_mask) {
            if (src_node.attention_seq_len_q_node == UINT32_MAX || src_node.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing padding-mask sequence length node while cloning forward subtree for autodiff.");
            }
            new_node.attention_seq_len_q_node = cloneForwardSubtree(src, src_node.attention_seq_len_q_node, dst, old_to_new, old_cuda_to_new);
            new_node.attention_seq_len_kv_node = cloneForwardSubtree(src, src_node.attention_seq_len_kv_node, dst, old_to_new, old_cuda_to_new);
        }
        if (src_node.attention_use_ragged_offsets) {
            if (src_node.attention_ragged_offset_q_node == UINT32_MAX || src_node.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing ragged offset node while cloning forward subtree for autodiff.");
            }
            new_node.attention_ragged_offset_q_node = cloneForwardSubtree(src, src_node.attention_ragged_offset_q_node, dst, old_to_new, old_cuda_to_new);
            new_node.attention_ragged_offset_kv_node = cloneForwardSubtree(src, src_node.attention_ragged_offset_kv_node, dst, old_to_new, old_cuda_to_new);
        }
        if (src_node.attention_use_paged_kv_cache) {
            if (src_node.attention_page_table_k_node == UINT32_MAX || src_node.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing paged KV page-table node while cloning forward subtree for autodiff.");
            }
            new_node.attention_page_table_k_node = cloneForwardSubtree(src, src_node.attention_page_table_k_node, dst, old_to_new, old_cuda_to_new);
            new_node.attention_page_table_v_node = cloneForwardSubtree(src, src_node.attention_page_table_v_node, dst, old_to_new, old_cuda_to_new);
        }
        if (src_node.attention_dropout_probability > 0.0f) {
            if (src_node.attention_dropout_seed_node == UINT32_MAX || src_node.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing dropout seed/offset node while cloning forward subtree for autodiff.");
            }
            new_node.attention_dropout_seed_node = cloneForwardSubtree(src, src_node.attention_dropout_seed_node, dst, old_to_new, old_cuda_to_new);
            new_node.attention_dropout_offset_node = cloneForwardSubtree(src, src_node.attention_dropout_offset_node, dst, old_to_new, old_cuda_to_new);
        }
    } else if (src_node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        if (src_node.cuda_kernel_spec_index >= src.cuda_kernel_expressions.size() ||
            !src.cuda_kernel_expressions[src_node.cuda_kernel_spec_index]) {
            throw std::runtime_error("Malformed forward expression: CudaKernelExpression node references an invalid kernel spec.");
        }
        new_node.cuda_kernel_input_nodes.clear();
        new_node.cuda_kernel_input_nodes.reserve(src_node.cuda_kernel_input_nodes.size());
        for (uint32_t input_node : src_node.cuda_kernel_input_nodes) {
            new_node.cuda_kernel_input_nodes.push_back(
                cloneForwardSubtree(src, input_node, dst, old_to_new, old_cuda_to_new));
        }
        auto spec_it = old_cuda_to_new.find(src_node.cuda_kernel_spec_index);
        if (spec_it == old_cuda_to_new.end()) {
            const uint32_t new_spec_index = static_cast<uint32_t>(dst.cuda_kernel_expressions.size());
            dst.cuda_kernel_expressions.push_back(src.cuda_kernel_expressions[src_node.cuda_kernel_spec_index]);
            old_cuda_to_new.emplace(src_node.cuda_kernel_spec_index, new_spec_index);
            new_node.cuda_kernel_spec_index = new_spec_index;
        } else {
            new_node.cuda_kernel_spec_index = spec_it->second;
        }
    } else if (Expression::isLeafOp(src_node.op)) {
        // Nothing to recurse into.
    } else {
        throw std::runtime_error("Unsupported op while cloning forward subtree for autodiff: " + std::to_string((int)src_node.op));
    }

    if (src_node.op == ExprOp::ROPE && src_node.rope_effective_sequence_length_node != UINT32_MAX) {
        new_node.rope_effective_sequence_length_node =
            cloneForwardSubtree(src, src_node.rope_effective_sequence_length_node, dst, old_to_new, old_cuda_to_new);
    }
    if (src_node.op == ExprOp::ROPE && src_node.rope_position_ids_node != UINT32_MAX) {
        new_node.rope_position_ids_node =
            cloneForwardSubtree(src, src_node.rope_position_ids_node, dst, old_to_new, old_cuda_to_new);
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

std::optional<DataType> matmulLowPrecisionOperandDType(const PhysicalExpression& forward_expr,
                                                        const ExprNode& matmul_node) {
    if (matmul_node.op != ExprOp::MATMUL && matmul_node.op != ExprOp::GEMM) {
        throw std::invalid_argument("matmulLowPrecisionOperandDType requires a MATMUL or GEMM node.");
    }
    if (matmul_node.lhs >= forward_expr.nodes.size() || matmul_node.rhs >= forward_expr.nodes.size()) {
        throw std::runtime_error("Matmul autodiff encountered an invalid matrix operand node index.");
    }

    const std::optional<DataType> lhs_dtype = materializedValueStorageDType(forward_expr, matmul_node.lhs);
    const std::optional<DataType> rhs_dtype = materializedValueStorageDType(forward_expr, matmul_node.rhs);
    if (!lhs_dtype.has_value() || !rhs_dtype.has_value() || lhs_dtype.value() != rhs_dtype.value()) {
        return std::nullopt;
    }
    if (lhs_dtype.value() != DataType::BF16 && lhs_dtype.value() != DataType::FP16) {
        return std::nullopt;
    }
    return lhs_dtype.value();
}

static bool isAttentionBackwardOp(ExprOp op) {
    return op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K || op == ExprOp::ATTENTION_BACKWARD_V ||
           op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

static bool isRmsNormBackwardOp(ExprOp op) {
    return op == ExprOp::RMSNORM_BACKWARD_X || op == ExprOp::RMSNORM_BACKWARD_SCALE;
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
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs);
                break;
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
                // Offsets are structural launch metadata and are never differentiable.
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::RAGGED_CONV1D_CAUSAL:
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA:
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER:
                // Ragged convolution numeric operands are differentiable. Canonical offsets
                // are structural row-boundary metadata and never receive gradients.
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs);
                break;
            case ExprOp::SEGMENTED_BROADCAST:
                // Offsets are structural metadata; only per-segment values differentiate.
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD:
            case ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD:
                // Offsets are structural metadata. Second derivatives are rejected
                // later, but reverse relevance must still ignore the partition.
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs);
                break;
            case ExprOp::TAKE_ALONG_AXIS:
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::NEG:
            case ExprOp::ABS:
            case ExprOp::CEIL:
            case ExprOp::FLOOR:
            case ExprOp::ROUND:
            case ExprOp::TRUNC:
            case ExprOp::SIN:
            case ExprOp::COS:
            case ExprOp::TAN:
            case ExprOp::ASIN:
            case ExprOp::ACOS:
            case ExprOp::ATAN:
            case ExprOp::SINH:
            case ExprOp::COSH:
            case ExprOp::ASINH:
            case ExprOp::ACOSH:
            case ExprOp::ATANH:
            case ExprOp::ERF:
            case ExprOp::ERFC:
            case ExprOp::ERFCX:
            case ExprOp::ERFINV:
            case ExprOp::ERFCINV:
            case ExprOp::TGAMMA:
            case ExprOp::LGAMMA:
            case ExprOp::DIGAMMA:
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
            case ExprOp::LOGICAL_NOT:
            case ExprOp::CAST:
            case ExprOp::ROPE:
            case ExprOp::SOFTMAX:
            case ExprOp::TRANSPOSE:
            case ExprOp::RESHAPE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::STRIDED_VIEW_BACKWARD:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
            case ExprOp::BROADCAST_TO:
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
            case ExprOp::SCAN:
            case ExprOp::SEGMENTED_SCAN:
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX:
            case ExprOp::SEGMENTED_REDUCE_MEAN:
                reaches[i] = reaches.at(node.lhs);
                break;
            case ExprOp::RMSNORM_BACKWARD_X:
            case ExprOp::RMSNORM_BACKWARD_SCALE:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs) || reaches.at(node.aux);
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
            case ExprOp::LAYERNORM:
            case ExprOp::GEMM:
                reaches[i] = reaches.at(node.lhs) || reaches.at(node.rhs) || reaches.at(node.aux);
                break;
            case ExprOp::WHERE:
                // The condition is nondifferentiable control data. Only the selected value branches
                // contribute differentiable reachability.
                reaches[i] = reaches.at(node.rhs) || reaches.at(node.aux);
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
            case ExprOp::CUDA_KERNEL_OUTPUT: {
                bool any = false;
                for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                    if (input_node >= reaches.size()) {
                        throw std::runtime_error("CudaKernelExpression reverse relevance encountered an invalid input node.");
                    }
                    any = any || reaches.at(input_node);
                }
                reaches[i] = any;
                break;
            }
            default:
                throw std::runtime_error("Unsupported op while computing reverse relevance: " + std::to_string((int)node.op));
        }
    }

    return reaches;
}

struct RaggedGradientExtent {
    uint32_t offsetsNode = UINT32_MAX;
    uint32_t offsetsInputSlot = UINT32_MAX;
    uint64_t batchSize = 0;
    uint64_t maxActiveValues = 0;
    uint64_t elementsPerValue = 0;
};

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
            // For synthetic backward inputs, as_type is the actual runtime tensor
            // dtype, not merely a logical expression preference.
            node.input_tensor_dtype = as_type.value();
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

    std::optional<RaggedGradientExtent> tryGetFrontierRaggedExtent(uint32_t root) const {
        if (root >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder ragged-extent query index out of range.");
        }

        std::optional<RaggedGradientExtent> result;
        std::unordered_set<uint32_t> visited;

        auto merge_extent = [&](uint32_t offsets_node_idx, const ExprNode& extent_node) {
            if (offsets_node_idx == UINT32_MAX || offsets_node_idx >= grad_expr.nodes.size()) {
                throw std::runtime_error("BackwardGraphBuilder ragged extent is missing its offsets input.");
            }
            const ExprNode& offsets_node = grad_expr.nodes.at(offsets_node_idx);
            if (offsets_node.op != ExprOp::INPUT) {
                throw std::runtime_error("BackwardGraphBuilder ragged extent requires a direct offsets INPUT node.");
            }
            if (extent_node.ragged_runtime_batch_size == 0 || extent_node.ragged_runtime_max_active_values == 0 ||
                extent_node.ragged_runtime_elements_per_value == 0) {
                throw std::runtime_error("BackwardGraphBuilder encountered incomplete ragged runtime-extent metadata.");
            }

            RaggedGradientExtent candidate;
            candidate.offsetsNode = offsets_node_idx;
            candidate.offsetsInputSlot = offsets_node.input_slot;
            candidate.batchSize = extent_node.ragged_runtime_batch_size;
            candidate.maxActiveValues = extent_node.ragged_runtime_max_active_values;
            candidate.elementsPerValue = extent_node.ragged_runtime_elements_per_value;

            if (result.has_value()) {
                if (result->offsetsInputSlot != candidate.offsetsInputSlot || result->batchSize != candidate.batchSize ||
                    result->maxActiveValues != candidate.maxActiveValues || result->elementsPerValue != candidate.elementsPerValue) {
                    throw std::runtime_error(
                        "BackwardGraphBuilder gradient combines incompatible ragged runtime extents before a shape transform.");
                }
                return;
            }
            result = candidate;
        };

        std::function<void(uint32_t)> visit = [&](uint32_t node_idx) {
            if (node_idx == UINT32_MAX || node_idx >= grad_expr.nodes.size() || !visited.insert(node_idx).second) {
                return;
            }
            const ExprNode& n = grad_expr.nodes.at(node_idx);
            if (n.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
                merge_extent(n.rhs, n);
                return;
            }

            auto visit_parent = [&](uint32_t parent) {
                if (parent != UINT32_MAX) visit(parent);
            };
            visit_parent(n.lhs);
            visit_parent(n.rhs);
            visit_parent(n.aux);
            visit_parent(n.alpha_node);
            visit_parent(n.beta_node);
            visit_parent(n.matmul_epilogue_aux);
            visit_parent(n.rope_effective_sequence_length_node);
            visit_parent(n.rope_position_ids_node);
            visit_parent(n.attention_seq_len_q_node);
            visit_parent(n.attention_seq_len_kv_node);
            visit_parent(n.attention_ragged_offset_q_node);
            visit_parent(n.attention_ragged_offset_kv_node);
            visit_parent(n.attention_page_table_k_node);
            visit_parent(n.attention_page_table_v_node);
            visit_parent(n.attention_dropout_seed_node);
            visit_parent(n.attention_dropout_offset_node);
            if (n.op == ExprOp::CUDA_KERNEL_OUTPUT) {
                for (uint32_t parent : n.cuda_kernel_input_nodes) visit_parent(parent);
            }
        };

        visit(root);
        return result;
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
                if (std::find(n.reshape_dims.begin(), n.reshape_dims.end(), EXPRESSION_COPY_DIM) != n.reshape_dims.end() ||
                    std::find(n.reshape_dims.begin(), n.reshape_dims.end(), EXPRESSION_INFER_DIM) != n.reshape_dims.end()) {
                    return std::nullopt;
                }
                return n.reshape_dims;
            case ExprOp::STRIDED_VIEW:
                if (std::find(n.view_dims.begin(), n.view_dims.end(), EXPRESSION_COPY_DIM) != n.view_dims.end()) {
                    return std::nullopt;
                }
                return n.view_dims;
            case ExprOp::STRIDED_VIEW_BACKWARD:
                return n.fill_dims;
            case ExprOp::BROADCAST_TO:
                return n.broadcast_dims;
            case ExprOp::NEG:
            case ExprOp::CAST:
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
            case ExprOp::SCAN:
            case ExprOp::SEGMENTED_SCAN:
                return tryInferKnownGradientDims(n.lhs);
            case ExprOp::SEGMENTED_BROADCAST:
                if (n.ragged_runtime_max_active_values == 0 || n.ragged_runtime_elements_per_value == 0) {
                    return std::nullopt;
                }
                if (n.ragged_runtime_elements_per_value == 1) {
                    return std::vector<uint64_t>{n.ragged_runtime_max_active_values};
                }
                return std::vector<uint64_t>{n.ragged_runtime_max_active_values, n.ragged_runtime_elements_per_value};
            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX:
            case ExprOp::SEGMENTED_REDUCE_MEAN:
                if (n.ragged_runtime_batch_size == 0 || n.ragged_runtime_elements_per_value == 0) {
                    return std::nullopt;
                }
                if (n.ragged_runtime_elements_per_value == 1) {
                    return std::vector<uint64_t>{n.ragged_runtime_batch_size};
                }
                return std::vector<uint64_t>{n.ragged_runtime_batch_size, n.ragged_runtime_elements_per_value};
            case ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD:
            case ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD:
                if (n.ragged_runtime_max_active_values == 0 || n.ragged_runtime_elements_per_value == 0) {
                    return std::nullopt;
                }
                if (n.ragged_runtime_elements_per_value == 1) {
                    return std::vector<uint64_t>{n.ragged_runtime_max_active_values};
                }
                return std::vector<uint64_t>{n.ragged_runtime_max_active_values, n.ragged_runtime_elements_per_value};
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA:
                if (n.ragged_runtime_max_active_values == 0 || n.ragged_conv1d_input_channels == 0) {
                    return std::nullopt;
                }
                return std::vector<uint64_t>{n.ragged_runtime_max_active_values, n.ragged_conv1d_input_channels};
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER:
                if (n.ragged_conv1d_output_channels == 0 || n.ragged_conv1d_input_channels == 0 ||
                    n.ragged_conv1d_groups == 0 || n.ragged_conv1d_kernel_width == 0) {
                    return std::nullopt;
                }
                return std::vector<uint64_t>{n.ragged_conv1d_output_channels,
                                             n.ragged_conv1d_input_channels / n.ragged_conv1d_groups,
                                             n.ragged_conv1d_kernel_width};
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
                dims = resolveDynamicAliasDims(lhs_dims, node.reshape_dims, true, "AutoDiff constant reshape");
                return true;
            }
            case ExprOp::STRIDED_VIEW: {
                std::vector<uint64_t> lhs_dims;
                if (!tryGetConstantLike(node.lhs, value, lhs_dims)) {
                    return false;
                }
                dims = resolveDynamicAliasDims(lhs_dims, node.view_dims, false, "AutoDiff constant strided_view");
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
            case ExprOp::CEIL:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::ceil(value);
                    return true;
                }
                return false;
            case ExprOp::FLOOR:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::floor(value);
                    return true;
                }
                return false;
            case ExprOp::ROUND:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::round(value);
                    return true;
                }
                return false;
            case ExprOp::TRUNC:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::trunc(value);
                    return true;
                }
                return false;
            case ExprOp::SIN:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::sin(value);
                    return true;
                }
                return false;
            case ExprOp::COS:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::cos(value);
                    return true;
                }
                return false;
            case ExprOp::TAN:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::tan(value);
                    return true;
                }
                return false;
            case ExprOp::ASIN:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::asin(value);
                    return true;
                }
                return false;
            case ExprOp::ACOS:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::acos(value);
                    return true;
                }
                return false;
            case ExprOp::ATAN:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::atan(value);
                    return true;
                }
                return false;
            case ExprOp::SINH:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::sinh(value);
                    return true;
                }
                return false;
            case ExprOp::COSH:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::cosh(value);
                    return true;
                }
                return false;
            case ExprOp::ASINH:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::asinh(value);
                    return true;
                }
                return false;
            case ExprOp::ACOSH:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::acosh(value);
                    return true;
                }
                return false;
            case ExprOp::ATANH:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::atanh(value);
                    return true;
                }
                return false;
            case ExprOp::ERF:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::erf(value);
                    return true;
                }
                return false;
            case ExprOp::ERFC:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::erfc(value);
                    return true;
                }
                return false;
            case ExprOp::ERFCX:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::exp(value * value) * std::erfc(value);
                    return true;
                }
                return false;
            case ExprOp::ERFINV:
            case ExprOp::ERFCINV:
                return false;
            case ExprOp::TGAMMA:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::tgamma(value);
                    return true;
                }
                return false;
            case ExprOp::LGAMMA:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = std::lgamma(value);
                    return true;
                }
                return false;
            case ExprOp::DIGAMMA:
                if (tryGetConstantLike(node.lhs, value, dims)) {
                    value = digammaApproxForMl(value);
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

    uint32_t cast(uint32_t lhs, DataType dtype) {
        const ExprNode& lhs_node = grad_expr.nodes.at(lhs);
        if (lhs_node.output_dtype.has_value() && lhs_node.output_dtype.value() == dtype) {
            return lhs;
        }
        ExprNode node{};
        node.op = ExprOp::CAST;
        node.lhs = lhs;
        node.output_dtype = dtype;
        return push(std::move(node));
    }

    uint32_t materializeMatmulOperandCast(uint32_t lhs, DataType dtype) {
        if (isKnownMaterializedAs(lhs, dtype)) {
            return lhs;
        }

        // Do not use cast() here: a synthetic input can carry a logical
        // output_dtype inherited from the forward node while its bound tensor is
        // actually FP32.  The explicit CAST stage is the storage conversion that
        // makes the subsequent cuBLASLt operand genuinely BF16/FP16.
        ExprNode node{};
        node.op = ExprOp::CAST;
        node.lhs = lhs;
        node.output_dtype = dtype;
        return push(std::move(node));
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
            result_node.matmul_packed_row_binding = forward_node.matmul_packed_row_binding;
            result_node.matmul_packed_row_capacity = forward_node.matmul_packed_row_capacity;
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
            source.matmul_backward_epilogue != MatmulBackwardEpilogue::Default ||
            source.matmul_packed_row_binding != MatmulPackedRowBinding::None) {
            // Packed-row matmuls are lowered through the bucketed backend.  Keep
            // their activation derivative in the ordinary expression tail so
            // it can be fused there rather than converting the matmul to an Lt
            // backward epilogue that the bucketed backend does not implement.
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
                                ConvolutionSpatial2d spatial,
                                uint64_t groups,
                                const std::vector<uint64_t>& target_output_dims = {},
                                std::optional<DataType> output_dtype = std::nullopt,
                                std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV2D_BACKWARD_DATA;
        node.lhs = filter;
        node.rhs = grad_output;
        node.conv_spatial_2d = spatial;
        node.conv_groups = groups;
        node.fill_dims = target_output_dims;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t raggedConv1dCausalBackwardData(uint32_t filter,
                                            uint32_t grad_output,
                                            uint32_t offsets,
                                            const ExprNode& forward_conv,
                                            std::optional<DataType> output_dtype = std::nullopt,
                                            std::optional<DataType> compute_dtype = std::nullopt) {
        if (forward_conv.op != ExprOp::RAGGED_CONV1D_CAUSAL) {
            throw std::runtime_error("raggedConv1dCausalBackwardData requires a forward ragged causal Conv1D node.");
        }
        ExprNode node{};
        node.op = ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA;
        node.lhs = filter;
        node.rhs = grad_output;
        node.aux = offsets;
        node.ragged_conv_spatial_1d = forward_conv.ragged_conv_spatial_1d;
        node.ragged_conv1d_input_channels = forward_conv.ragged_conv1d_input_channels;
        node.ragged_conv1d_output_channels = forward_conv.ragged_conv1d_output_channels;
        node.ragged_conv1d_kernel_width = forward_conv.ragged_conv1d_kernel_width;
        node.ragged_conv1d_groups = forward_conv.ragged_conv1d_groups;
        node.ragged_runtime_batch_size = forward_conv.ragged_runtime_batch_size;
        node.ragged_runtime_max_active_values = forward_conv.ragged_runtime_max_active_values;
        node.ragged_runtime_max_values_per_row = forward_conv.ragged_runtime_max_values_per_row;
        node.ragged_runtime_elements_per_value = forward_conv.ragged_conv1d_input_channels;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        } else if (forward_conv.compute_dtype.has_value()) {
            node.compute_dtype = forward_conv.compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t raggedConv1dCausalBackwardFilter(uint32_t input,
                                              uint32_t grad_output,
                                              uint32_t offsets,
                                              const ExprNode& forward_conv,
                                              std::optional<DataType> output_dtype = std::nullopt,
                                              std::optional<DataType> compute_dtype = std::nullopt) {
        if (forward_conv.op != ExprOp::RAGGED_CONV1D_CAUSAL) {
            throw std::runtime_error("raggedConv1dCausalBackwardFilter requires a forward ragged causal Conv1D node.");
        }
        ExprNode node{};
        node.op = ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER;
        node.lhs = input;
        node.rhs = grad_output;
        node.aux = offsets;
        node.ragged_conv_spatial_1d = forward_conv.ragged_conv_spatial_1d;
        node.ragged_conv1d_input_channels = forward_conv.ragged_conv1d_input_channels;
        node.ragged_conv1d_output_channels = forward_conv.ragged_conv1d_output_channels;
        node.ragged_conv1d_kernel_width = forward_conv.ragged_conv1d_kernel_width;
        node.ragged_conv1d_groups = forward_conv.ragged_conv1d_groups;
        node.ragged_runtime_batch_size = forward_conv.ragged_runtime_batch_size;
        node.ragged_runtime_max_active_values = forward_conv.ragged_runtime_max_active_values;
        node.ragged_runtime_max_values_per_row = forward_conv.ragged_runtime_max_values_per_row;
        node.ragged_runtime_elements_per_value = forward_conv.ragged_conv1d_output_channels;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        } else if (forward_conv.compute_dtype.has_value()) {
            node.compute_dtype = forward_conv.compute_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t conv2dBackwardFilter(uint32_t input,
                                  uint32_t grad_output,
                                  ConvolutionSpatial2d spatial,
                                  uint64_t groups,
                                  const std::vector<uint64_t>& target_output_dims = {},
                                  std::optional<DataType> output_dtype = std::nullopt,
                                  std::optional<DataType> compute_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::CONV2D_BACKWARD_FILTER;
        node.lhs = input;
        node.rhs = grad_output;
        node.conv_spatial_2d = spatial;
        node.conv_groups = groups;
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
                                uint64_t groups,
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
        node.conv_groups = groups;
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
                                  uint64_t groups,
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
        node.conv_groups = groups;
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
                                    uint32_t effective_sequence_length_node = UINT32_MAX,
                                    uint32_t position_ids_node = UINT32_MAX,
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
        node.rope_effective_sequence_length_node = effective_sequence_length_node;
        node.rope_position_ids_node = position_ids_node;
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

    uint32_t rmsNormBackward(ExprOp op,
                             uint32_t x,
                             uint32_t scale,
                             uint32_t dY,
                             const ExprNode& forward_rms_norm,
                             std::optional<DataType> output_dtype = std::nullopt,
                             std::optional<DataType> compute_dtype = std::nullopt) {
        if (!isRmsNormBackwardOp(op)) {
            throw std::runtime_error("rmsNormBackward builder called with non-RMSNorm-backward op.");
        }
        if (forward_rms_norm.rms_norm_fused_activation != CudnnRmsNormFusedActivation::NONE) {
            throw std::runtime_error("cuDNN RMSNorm backward does not support fused activation in training.");
        }

        ExprNode node{};
        node.op = op;
        node.lhs = x;
        node.rhs = scale;
        node.aux = dY;
        node.rms_norm_normalized_feature_count = forward_rms_norm.rms_norm_normalized_feature_count;
        node.rms_norm_epsilon = forward_rms_norm.rms_norm_epsilon;
        node.rms_norm_fused_activation = CudnnRmsNormFusedActivation::NONE;
        node.rms_norm_packed_row_capacity = forward_rms_norm.rms_norm_packed_row_capacity;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        if (compute_dtype.has_value()) {
            node.compute_dtype = compute_dtype.value();
        } else if (forward_rms_norm.compute_dtype.has_value()) {
            node.compute_dtype = forward_rms_norm.compute_dtype.value();
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

    uint32_t segmentedReduce(uint32_t values,
                             uint32_t offsets,
                             ExprOp op,
                             uint64_t batch_size,
                             uint64_t max_active_values,
                             uint64_t elements_per_value,
                             std::optional<DataType> output_dtype = std::nullopt) {
        if (op != ExprOp::SEGMENTED_REDUCE_SUM && op != ExprOp::SEGMENTED_REDUCE_MEAN &&
            op != ExprOp::SEGMENTED_REDUCE_MIN && op != ExprOp::SEGMENTED_REDUCE_MAX) {
            throw std::runtime_error("AutoDiff segmentedReduce requires a segmented reduction op.");
        }
        if (batch_size == 0 || max_active_values == 0 || elements_per_value == 0) {
            throw std::runtime_error("AutoDiff segmentedReduce requires non-zero ragged extent metadata.");
        }

        ExprNode node{};
        node.op = op;
        node.lhs = values;
        node.rhs = offsets;
        node.ragged_runtime_batch_size = batch_size;
        node.ragged_runtime_max_active_values = max_active_values;
        node.ragged_runtime_elements_per_value = elements_per_value;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t segmentedReduceMinMaxBackward(ExprOp op,
                                           uint32_t lhs,
                                           uint32_t grad,
                                           uint32_t offsets,
                                           uint64_t batch_size,
                                           uint64_t max_active_values,
                                           uint64_t elements_per_value,
                                           std::optional<DataType> output_dtype = std::nullopt,
                                           std::optional<DataType> compute_dtype = std::nullopt) {
        if (op != ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD && op != ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD) {
            throw std::runtime_error(
                "segmentedReduceMinMaxBackward requires SEGMENTED_REDUCE_MIN_BACKWARD or SEGMENTED_REDUCE_MAX_BACKWARD.");
        }

        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.rhs = grad;
        node.aux = offsets;
        node.ragged_runtime_batch_size = batch_size;
        node.ragged_runtime_max_active_values = max_active_values;
        node.ragged_runtime_elements_per_value = elements_per_value;
        node.output_dtype = output_dtype;
        node.compute_dtype = compute_dtype;
        return push(std::move(node));
    }

    uint32_t scan(uint32_t lhs,
                  ScanOp op,
                  ScanMode mode,
                  uint64_t axis,
                  bool reverse,
                  std::optional<DataType> output_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::SCAN;
        node.lhs = lhs;
        node.scan_op = op;
        node.scan_mode = mode;
        node.scan_axis = axis;
        node.scan_reverse = reverse;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t segmentedScan(uint32_t lhs,
                           uint32_t offsets,
                           ScanOp op,
                           ScanMode mode,
                           bool reverse,
                           std::optional<DataType> output_dtype = std::nullopt,
                           uint64_t ragged_batch_size = 0,
                           uint64_t ragged_max_active_values = 0,
                           uint64_t ragged_elements_per_value = 1) {
        if ((ragged_batch_size == 0) != (ragged_max_active_values == 0)) {
            throw std::runtime_error("AutoDiff segmented scan ragged metadata requires both batch size and max active values, or neither.");
        }
        ExprNode node{};
        node.op = ExprOp::SEGMENTED_SCAN;
        node.lhs = lhs;
        node.rhs = offsets;
        node.scan_op = op;
        node.scan_mode = mode;
        node.scan_axis = UINT64_MAX;
        node.scan_reverse = reverse;
        node.ragged_runtime_batch_size = ragged_batch_size;
        node.ragged_runtime_max_active_values = ragged_max_active_values;
        node.ragged_runtime_elements_per_value = ragged_elements_per_value;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t raggedValuewiseExtent(uint32_t lhs,
                                   uint32_t offsets,
                                   uint64_t batch_size,
                                   uint64_t max_active_values,
                                   uint64_t elements_per_value = 1,
                                   std::optional<DataType> output_dtype = std::nullopt) {
        ExprNode node{};
        node.op = ExprOp::RAGGED_VALUEWISE_EXTENT;
        node.lhs = lhs;
        node.rhs = offsets;
        node.ragged_runtime_batch_size = batch_size;
        node.ragged_runtime_max_active_values = max_active_values;
        node.ragged_runtime_elements_per_value = elements_per_value;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t segmentedBroadcast(uint32_t per_segment_values,
                                uint32_t offsets,
                                uint64_t batch_size,
                                uint64_t max_active_values,
                                uint64_t elements_per_value,
                                bool normalize_by_segment_length,
                                std::optional<DataType> output_dtype = std::nullopt) {
        if (batch_size == 0 || max_active_values == 0 || elements_per_value == 0) {
            throw std::runtime_error("AutoDiff segmented broadcast requires non-zero ragged extent metadata.");
        }
        ExprNode node{};
        node.op = ExprOp::SEGMENTED_BROADCAST;
        node.lhs = per_segment_values;
        node.rhs = offsets;
        node.ragged_runtime_batch_size = batch_size;
        node.ragged_runtime_max_active_values = max_active_values;
        node.ragged_runtime_elements_per_value = elements_per_value;
        node.segmented_broadcast_normalize_by_length = normalize_by_segment_length;
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }


    uint32_t scanMinMaxBackward(ExprOp op,
                                uint32_t lhs,
                                uint32_t grad,
                                uint32_t offsets,
                                ScanMode mode,
                                uint64_t axis,
                                bool reverse,
                                std::optional<DataType> output_dtype = std::nullopt,
                                uint64_t ragged_batch_size = 0,
                                uint64_t ragged_max_active_values = 0,
                                uint64_t ragged_elements_per_value = 1) {
        if (op != ExprOp::SCAN_MIN_BACKWARD && op != ExprOp::SCAN_MAX_BACKWARD &&
            op != ExprOp::SEGMENTED_SCAN_MIN_BACKWARD && op != ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
            throw std::runtime_error("scanMinMaxBackward requires a scan min/max backward op.");
        }
        if ((ragged_batch_size == 0) != (ragged_max_active_values == 0)) {
            throw std::runtime_error(
                "AutoDiff segmented scan min/max backward ragged metadata requires both batch size and max active values, or neither.");
        }
        ExprNode node{};
        node.op = op;
        node.lhs = lhs;
        node.rhs = grad;
        node.aux = offsets;
        node.scan_mode = mode;
        node.scan_axis = axis;
        node.scan_reverse = reverse;
        if (op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
            node.ragged_runtime_batch_size = ragged_batch_size;
            node.ragged_runtime_max_active_values = ragged_max_active_values;
            node.ragged_runtime_elements_per_value = ragged_elements_per_value;
        }
        if (output_dtype.has_value()) {
            node.output_dtype = output_dtype.value();
        }
        return push(std::move(node));
    }

    uint32_t where(uint32_t condition, uint32_t true_value, uint32_t false_value) {
        ExprNode node{};
        node.op = ExprOp::WHERE;
        node.lhs = condition;
        node.rhs = true_value;
        node.aux = false_value;
        return push(std::move(node));
    }

    uint32_t neg(uint32_t value) { return unary(ExprOp::NEG, value); }
    uint32_t sin(uint32_t value) { return unary(ExprOp::SIN, value); }
    uint32_t cos(uint32_t value) { return unary(ExprOp::COS, value); }
    uint32_t sinh(uint32_t value) { return unary(ExprOp::SINH, value); }
    uint32_t cosh(uint32_t value) { return unary(ExprOp::COSH, value); }
    uint32_t sqrt(uint32_t value) { return unary(ExprOp::SQRT, value); }
    uint32_t exp(uint32_t value) { return unary(ExprOp::EXP, value); }
    uint32_t digamma(uint32_t value) { return unary(ExprOp::DIGAMMA, value); }
    uint32_t add(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::ADD, lhs, rhs); }
    uint32_t sub(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::SUB, lhs, rhs); }
    uint32_t mul(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::MUL, lhs, rhs); }
    uint32_t div(uint32_t lhs, uint32_t rhs) { return binary(ExprOp::DIV, lhs, rhs); }

    uint32_t broadcastTo(uint32_t value, const std::vector<uint64_t>& target_dims) {
        if (target_dims.empty()) {
            throw std::runtime_error("AutoDiff BROADCAST_TO requires non-empty target dimensions.");
        }
        for (uint64_t dim : target_dims) {
            if (dim == 0 || dim == std::numeric_limits<uint64_t>::max()) {
                throw std::runtime_error("AutoDiff BROADCAST_TO requires concrete non-zero target dimensions.");
            }
        }
        ExprNode node{};
        node.op = ExprOp::BROADCAST_TO;
        node.lhs = value;
        node.broadcast_dims = target_dims;
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

        // A dense reduction with keepdims followed immediately by squeezing only
        // reduced axes can express the same shape directly through the reduction's
        // native squeeze_axes contract.  Clone the node instead of mutating it so
        // any other consumer of the keepdims reduction remains unchanged.  This is
        // especially important for low-precision broadcast gradients: CUB can then
        // accumulate in FP32 and store the requested BF16/FP16 result directly,
        // without a trailing shape/materialization kernel.
        if (isValueReductionOp(value_node.op) && value_node.squeeze_axes.empty() && !value_node.reduction_axes.empty()) {
            bool squeezes_only_reduced_axes = true;
            for (uint64_t axis : normalized_axes) {
                if (std::find(value_node.reduction_axes.begin(), value_node.reduction_axes.end(), axis) ==
                    value_node.reduction_axes.end()) {
                    squeezes_only_reduced_axes = false;
                    break;
                }
            }
            if (squeezes_only_reduced_axes) {
                ExprNode folded_reduction = value_node;
                folded_reduction.squeeze_axes = normalized_axes;
                return push(std::move(folded_reduction));
            }
        }

        ExprNode node{};
        node.op = ExprOp::SQUEEZE;
        node.lhs = value;
        node.squeeze_axes = normalized_axes;
        return push(std::move(node));
    }

    std::unordered_map<std::string, uint32_t> cudaKernel(
        const CudaKernelExpression& kernel,
        const std::unordered_map<std::string, uint32_t>& input_nodes_by_name) {
        if (input_nodes_by_name.size() != kernel.inputs().size()) {
            throw std::runtime_error("Autodiff custom CUDA backward kernel input count does not match its declared ABI.");
        }

        std::vector<uint32_t> ordered_inputs;
        ordered_inputs.reserve(kernel.inputs().size());
        for (const auto& input : kernel.inputs()) {
            auto it = input_nodes_by_name.find(input.name);
            if (it == input_nodes_by_name.end()) {
                throw std::runtime_error("Autodiff custom CUDA backward kernel is missing input node '" + input.name + "'.");
            }
            if (it->second >= grad_expr.nodes.size()) {
                throw std::runtime_error("Autodiff custom CUDA backward kernel input node is out of range.");
            }
            const ExprNode& input_node = grad_expr.nodes[it->second];
            if (input.kind == CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar &&
                input_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
                throw std::runtime_error("Autodiff custom CUDA backward tensor-runtime-scalar input is wired to the wrong node kind.");
            }
            if (input.kind == CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar &&
                input_node.op != ExprOp::RUNTIME_SCALAR) {
                throw std::runtime_error("Autodiff custom CUDA backward host-runtime-scalar input is wired to the wrong node kind.");
            }
            if (input.kind == CudaKernelExpression::TensorParamSpec::Kind::Tensor &&
                (input_node.op == ExprOp::TENSOR_RUNTIME_SCALAR || input_node.op == ExprOp::RUNTIME_SCALAR)) {
                throw std::runtime_error("Autodiff custom CUDA backward tensor input is wired to a runtime scalar node.");
            }
            ordered_inputs.push_back(it->second);
        }

        const uint32_t spec_index = static_cast<uint32_t>(grad_expr.cuda_kernel_expressions.size());
        grad_expr.cuda_kernel_expressions.push_back(std::make_shared<CudaKernelExpression>(kernel));

        std::unordered_map<std::string, uint32_t> outputs;
        outputs.reserve(kernel.outputs().size());
        for (uint32_t output_idx = 0; output_idx < kernel.outputs().size(); ++output_idx) {
            ExprNode node;
            node.op = ExprOp::CUDA_KERNEL_OUTPUT;
            node.cuda_kernel_spec_index = spec_index;
            node.cuda_kernel_output_index = output_idx;
            node.cuda_kernel_input_nodes = ordered_inputs;
            node.output_dtype = kernel.outputs()[output_idx].dtype;
            node.compute_dtype = kernel.outputs()[output_idx].dtype;
            node.backward_output_dtype = kernel.outputs()[output_idx].dtype;
            node.backward_compute_dtype = kernel.outputs()[output_idx].dtype;
            const uint32_t node_idx = push(std::move(node));
            outputs.emplace(kernel.outputs()[output_idx].name, node_idx);
        }
        return outputs;
    }

    uint32_t cloneForward(uint32_t forward_node_index) {
        return cloneForwardSubtree(forward_expr, forward_node_index, grad_expr, forward_to_grad_node_map, forward_to_grad_cuda_kernel_map);
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

    void setPackedRowMatmul(uint32_t node_idx, MatmulPackedRowBinding binding, uint64_t capacity) {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("Autodiff packed-row matmul annotation node index out of range.");
        }
        ExprNode& node = grad_expr.nodes.at(node_idx);
        if (node.op != ExprOp::MATMUL) {
            throw std::runtime_error("Autodiff packed-row annotation requires a MATMUL node.");
        }
        node.matmul_packed_row_binding = binding;
        node.matmul_packed_row_capacity = capacity;
    }

    PhysicalExpression takeExpression() { return std::move(grad_expr); }

   private:
    bool isKnownMaterializedAs(uint32_t node_idx, DataType dtype) const {
        if (node_idx >= grad_expr.nodes.size()) {
            throw std::runtime_error("BackwardGraphBuilder materialized-dtype query index out of range.");
        }

        const ExprNode& node = grad_expr.nodes.at(node_idx);
        switch (node.op) {
            case ExprOp::INPUT:
                return node.input_tensor_dtype.has_value() && node.input_tensor_dtype.value() == dtype;
            case ExprOp::RESHAPE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                return isKnownMaterializedAs(node.lhs, dtype);
            case ExprOp::CAST:
                return node.output_dtype.has_value() && node.output_dtype.value() == dtype;
            default:
                // A non-alias producer feeding a matmul is emitted as its own
                // materialized stage by the execution planner.
                return node.output_dtype.has_value() && node.output_dtype.value() == dtype;
        }
    }

    uint32_t push(ExprNode node) {
        const uint32_t idx = static_cast<uint32_t>(grad_expr.nodes.size());
        grad_expr.nodes.push_back(std::move(node));
        return idx;
    }

    const PhysicalExpression& forward_expr;
    PhysicalExpression grad_expr;
    std::unordered_map<uint32_t, uint32_t> forward_to_grad_node_map;
    std::unordered_map<uint32_t, uint32_t> forward_to_grad_cuda_kernel_map;
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

std::optional<std::unordered_map<std::string, uint32_t>> normalizeUpstreamNodeIndicesByOutput(
    const PhysicalOutputs& forward_outputs, const std::unordered_map<std::string, uint32_t>& upstream_node_indices_by_output) {
    if (upstream_node_indices_by_output.empty()) {
        return std::nullopt;
    }
    if (!forward_outputs.expr) {
        throw std::runtime_error("compileBackward upstream-node validation requires non-null forward expr.");
    }

    const PhysicalExpression& forward_expr = *forward_outputs.expr;
    std::unordered_set<std::string> valid_output_names;
    valid_output_names.reserve(forward_outputs.outputs.size());
    for (const NamedOutput& output : forward_outputs.outputs) {
        valid_output_names.insert(output.name);
    }

    for (const auto& [output_name, upstream_node_idx] : upstream_node_indices_by_output) {
        if (!valid_output_names.contains(output_name)) {
            throw std::runtime_error("compileBackward explicit upstream node map contains unknown forward output: " + output_name);
        }
        if (upstream_node_idx >= forward_expr.nodes.size()) {
            throw std::runtime_error("compileBackward explicit upstream node index is out of range for output: " + output_name);
        }
    }

    return upstream_node_indices_by_output;
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
    if (node.op == ExprOp::MATMUL) {
        if (aux_dims != nullptr) {
            throw std::runtime_error("Autodiff MATMUL shape inference does not accept an addend tensor.");
        }
        return planBatchedMatmulShape(lhs_dims, rhs_dims, node.transpose_lhs, node.transpose_rhs).output_dimensions;
    }

    if (node.op != ExprOp::GEMM) {
        throw std::runtime_error("Autodiff matmul shape inference requires a MATMUL or GEMM node.");
    }
    if (lhs_dims.size() != 2 || rhs_dims.size() != 2) {
        throw std::runtime_error("Autodiff shape inference for GEMM currently only supports rank-2 tensors.");
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

std::vector<uint64_t> rawBatchedMatmulOperandGradientDims(const std::vector<uint64_t>& output_dims,
                                                          const std::vector<uint64_t>& operand_dims) {
    if (output_dims.size() < 2 || operand_dims.size() < 2) {
        throw std::runtime_error("Autodiff batched matmul operand-gradient shape inference requires rank >= 2 tensors.");
    }

    std::vector<uint64_t> grad_dims(output_dims.begin(), output_dims.end() - 2);
    grad_dims.push_back(operand_dims[operand_dims.size() - 2]);
    grad_dims.push_back(operand_dims.back());
    return grad_dims;
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
    if (node.attention_use_ragged_offsets) {
        const bool queryRagged = q_dims.size() == 3;
        const bool keyValueRagged = k_dims.size() == 3;
        if ((q_dims.size() != 3 && q_dims.size() != 4) || (k_dims.size() != 3 && k_dims.size() != 4) ||
            (v_dims.size() != 3 && v_dims.size() != 4) || (v_dims.size() == 3) != keyValueRagged) {
            throw std::runtime_error(
                "Autodiff ragged attention requires Q to be packed rank-3 or dense rank-4 and K/V to share the same rank-3/rank-4 domain.");
        }
        if (node.attention_q_layout != AttentionTensorLayout::BSHD || node.attention_k_layout != AttentionTensorLayout::BSHD ||
            node.attention_v_layout != AttentionTensorLayout::BSHD || node.attention_o_layout != AttentionTensorLayout::BSHD) {
            throw std::runtime_error("Autodiff ragged/mixed attention requires BSHD logical layout.");
        }

        const uint64_t queryHeads = queryRagged ? q_dims.at(1) : q_dims.at(2);
        const uint64_t queryHeadDim = q_dims.at(3 - (queryRagged ? 1 : 0));
        const uint64_t keyValueHeads = keyValueRagged ? k_dims.at(1) : k_dims.at(2);
        const uint64_t keyHeadDim = k_dims.at(3 - (keyValueRagged ? 1 : 0));
        const uint64_t valueHeads = keyValueRagged ? v_dims.at(1) : v_dims.at(2);
        const uint64_t valueDim = v_dims.at(3 - (keyValueRagged ? 1 : 0));
        const uint64_t keyTokenExtent = keyValueRagged ? k_dims.at(0) : k_dims.at(1);
        const uint64_t valueTokenExtent = keyValueRagged ? v_dims.at(0) : v_dims.at(1);

        if (queryHeads == 0 || keyValueHeads == 0 || queryHeadDim == 0 || valueDim == 0 || keyTokenExtent == 0 ||
            valueTokenExtent == 0) {
            throw std::runtime_error("Autodiff ragged/mixed attention q/k/v dimensions must be non-zero.");
        }
        if (keyValueHeads != valueHeads || keyTokenExtent != valueTokenExtent) {
            throw std::runtime_error("Autodiff ragged/mixed attention found mismatched K/V dimensions.");
        }
        if (queryHeads % keyValueHeads != 0) {
            throw std::runtime_error("Autodiff attention query heads must be an integer multiple of key/value heads.");
        }
        if (queryHeadDim != keyHeadDim) {
            throw std::runtime_error("Autodiff attention q/k head dimensions must match.");
        }
        if (!queryRagged && !keyValueRagged && q_dims.at(0) != k_dims.at(0)) {
            throw std::runtime_error("Autodiff attention shape inference found mismatched q/k batch dimensions.");
        }
        if (queryRagged) {
            if (q_dims.at(0) == 0) {
                throw std::runtime_error("Autodiff ragged attention query packed capacity must be non-zero.");
            }
            return {q_dims.at(0), queryHeads, valueDim};
        }
        return attentionOutputDimsForAutodiff(node, q_dims.at(0), queryHeads, q_dims.at(1), valueDim);
    }

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
    if (node.attention_use_ragged_offsets) {
        throw std::runtime_error(
            "Autodiff ragged/mixed attention dBias shape inference is unavailable because production ragged score-bias backward is unsupported.");
    }
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
    const uint64_t groups = node.conv_groups;
    if (groups == 0 || input_dims[1] != filter_dims[1] * groups || filter_dims[0] % groups != 0) {
        throw std::runtime_error("Autodiff convolution shape inference found invalid grouped input/filter channels.");
    }

    std::vector<uint64_t> out_dims{input_dims[0], filter_dims[0]};
    const std::vector<int32_t> strides =
        is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
              : std::vector<int32_t>{node.conv_spatial_2d.stride_h, node.conv_spatial_2d.stride_w};
    const std::vector<int32_t> pre_pads =
        is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
              : std::vector<int32_t>{node.conv_spatial_2d.pre_padding_h, node.conv_spatial_2d.pre_padding_w};
    const std::vector<int32_t> post_pads =
        is_3d ? pre_pads : std::vector<int32_t>{node.conv_spatial_2d.post_padding_h, node.conv_spatial_2d.post_padding_w};
    const std::vector<int32_t> dilations =
        is_3d ? std::vector<int32_t>{1, 1, 1}
              : std::vector<int32_t>{node.conv_spatial_2d.dilation_h, node.conv_spatial_2d.dilation_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t effective_filter =
            static_cast<int64_t>(dilations[i]) * (static_cast<int64_t>(filter_dims[dim_idx]) - 1) + 1;
        const int64_t numer = static_cast<int64_t>(input_dims[dim_idx]) + pre_pads[i] + post_pads[i] - effective_filter;
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
    const uint64_t groups = node.conv_groups;
    if (groups == 0 || k % groups != 0)
        throw std::runtime_error("Autodiff convolution backward-data shape inference found invalid groups.");
    const uint64_t c = filter_dims[1] * groups;
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
    const std::vector<int32_t> strides =
        is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
              : std::vector<int32_t>{node.conv_spatial_2d.stride_h, node.conv_spatial_2d.stride_w};
    const std::vector<int32_t> pre_pads =
        is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
              : std::vector<int32_t>{node.conv_spatial_2d.pre_padding_h, node.conv_spatial_2d.pre_padding_w};
    const std::vector<int32_t> post_pads =
        is_3d ? pre_pads : std::vector<int32_t>{node.conv_spatial_2d.post_padding_h, node.conv_spatial_2d.post_padding_w};
    const std::vector<int32_t> dilations =
        is_3d ? std::vector<int32_t>{1, 1, 1}
              : std::vector<int32_t>{node.conv_spatial_2d.dilation_h, node.conv_spatial_2d.dilation_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t effective_filter =
            static_cast<int64_t>(dilations[i]) * (static_cast<int64_t>(filter_dims[dim_idx]) - 1) + 1;
        const int64_t extent =
            static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i] - pre_pads[i] - post_pads[i] + effective_filter;
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
    const uint64_t groups = node.conv_groups;
    const uint64_t c = input_dims[1];
    const uint64_t k = grad_output_dims[1];
    if (groups == 0 || c % groups != 0 || k % groups != 0)
        throw std::runtime_error("Autodiff convolution backward-filter shape inference found invalid groups/channels.");
    const uint64_t filter_c = c / groups;
    if (!node.fill_dims.empty()) {
        if (node.fill_dims.size() != rank) {
            throw std::runtime_error("Autodiff convolution backward-filter explicit output shape rank mismatch.");
        }
        if (node.fill_dims[0] != k || node.fill_dims[1] != filter_c) {
            throw std::runtime_error("Autodiff convolution backward-filter explicit output shape is incompatible with channels.");
        }
        return node.fill_dims;
    }

    std::vector<uint64_t> out_dims{k, filter_c};
    const std::vector<int32_t> strides =
        is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
              : std::vector<int32_t>{node.conv_spatial_2d.stride_h, node.conv_spatial_2d.stride_w};
    const std::vector<int32_t> pre_pads =
        is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
              : std::vector<int32_t>{node.conv_spatial_2d.pre_padding_h, node.conv_spatial_2d.pre_padding_w};
    const std::vector<int32_t> post_pads =
        is_3d ? pre_pads : std::vector<int32_t>{node.conv_spatial_2d.post_padding_h, node.conv_spatial_2d.post_padding_w};
    const std::vector<int32_t> dilations =
        is_3d ? std::vector<int32_t>{1, 1, 1}
              : std::vector<int32_t>{node.conv_spatial_2d.dilation_h, node.conv_spatial_2d.dilation_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t effective_extent = static_cast<int64_t>(input_dims[dim_idx]) + pre_pads[i] + post_pads[i] -
                                         static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i];
        if (effective_extent <= 0) {
            throw std::runtime_error("Autodiff convolution backward-filter shape inference produced non-positive filter extent.");
        }
        if ((effective_extent - 1) % dilations[i] != 0) {
            throw std::runtime_error("Autodiff convolution backward-filter shape inference found geometry incompatible with dilation.");
        }
        const int64_t extent = (effective_extent - 1) / dilations[i] + 1;
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
        // Runtime scalars have no tensor shape to specialize. Callers provide
        // concrete dimensions only for ordinary tensor inputs; requiring a
        // dimension-map entry for a runtime scalar incorrectly rejects valid
        // differentiable graphs such as SDPA dropout seed/offset inputs.
        if (input.kind == NamedInput::Kind::RuntimeScalarFp32 ||
            input.kind == NamedInput::Kind::TensorRuntimeScalar) {
            input_dims_by_slot[input.slot] = {};
            continue;
        }

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
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
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
            case ExprOp::WHERE: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                }
                if (!node_dims[node.rhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                }
                if (!node_dims[node.aux].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.aux]);
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
            case ExprOp::CEIL:
            case ExprOp::FLOOR:
            case ExprOp::ROUND:
            case ExprOp::TRUNC:
            case ExprOp::SIN:
            case ExprOp::COS:
            case ExprOp::TAN:
            case ExprOp::ASIN:
            case ExprOp::ACOS:
            case ExprOp::ATAN:
            case ExprOp::SINH:
            case ExprOp::COSH:
            case ExprOp::ASINH:
            case ExprOp::ACOSH:
            case ExprOp::ATANH:
            case ExprOp::ERF:
            case ExprOp::ERFC:
            case ExprOp::ERFCX:
            case ExprOp::ERFINV:
            case ExprOp::ERFCINV:
            case ExprOp::TGAMMA:
            case ExprOp::LGAMMA:
            case ExprOp::DIGAMMA:
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
            case ExprOp::LOGICAL_NOT:
            case ExprOp::CAST:
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
            case ExprOp::ROPE:
            case ExprOp::SOFTMAX:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::BROADCAST_TO:
                node_dims[i] = inferBroadcastToOutputDims(node_dims[node.lhs], node.broadcast_dims);
                break;
            case ExprOp::TRANSPOSE:
                node_dims[i] = inferTransposeOutputDims(node_dims[node.lhs]);
                break;
            case ExprOp::RESHAPE:
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.reshape_dims, true, "AutoDiff reshape");
                break;
            case ExprOp::STRIDED_VIEW:
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.view_dims, false, "AutoDiff strided_view");
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
            case ExprOp::REDUCE_NORM2: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];
                const std::vector<uint64_t> reduction_axes = resolveReductionAxesForAutodiff(node.reduction_axes, lhs_dims.size());
                node_dims[i] = StampedEquation::computeReductionOutputDims(lhs_dims, reduction_axes, node.squeeze_axes);
                break;
            }
            case ExprOp::SCAN:
            case ExprOp::SEGMENTED_SCAN:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX:
            case ExprOp::SEGMENTED_REDUCE_MEAN: {
                const std::vector<uint64_t>& values_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& offsets_dims = node_dims[node.rhs];
                if (values_dims.empty() || offsets_dims.size() != 1 || offsets_dims[0] == 0) {
                    throw std::runtime_error("inferForwardNodeDims segmented reduce requires values [N,D...] and non-empty rank-1 offsets.");
                }
                if (values_dims[0] == 0 || node.ragged_runtime_elements_per_value == 0 ||
                    dynamicDimsNumel(values_dims, "inferForwardNodeDims segmented reduce values") / values_dims[0] !=
                        node.ragged_runtime_elements_per_value) {
                    throw std::runtime_error("inferForwardNodeDims segmented reduce elements-per-value metadata mismatch.");
                }
                std::vector<uint64_t> output_dims = values_dims;
                output_dims[0] = offsets_dims[0] - 1;
                node_dims[i] = std::move(output_dims);
                break;
            }
            case ExprOp::RAGGED_CONV1D_CAUSAL: {
                const std::vector<uint64_t>& values_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& filter_dims = node_dims[node.rhs];
                const std::vector<uint64_t>& offsets_dims = node_dims[node.aux];
                if (values_dims != std::vector<uint64_t>({node.ragged_runtime_max_active_values,
                                                          node.ragged_conv1d_input_channels})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D packed values shape does not match logical metadata.");
                }
                if (node.ragged_conv1d_groups == 0 || node.ragged_conv1d_input_channels % node.ragged_conv1d_groups != 0 ||
                    node.ragged_conv1d_output_channels % node.ragged_conv1d_groups != 0 ||
                    filter_dims != std::vector<uint64_t>({node.ragged_conv1d_output_channels,
                                                          node.ragged_conv1d_input_channels / node.ragged_conv1d_groups,
                                                          node.ragged_conv1d_kernel_width})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D filter shape must be [K,C/groups,R].");
                }
                if (offsets_dims != std::vector<uint64_t>({node.ragged_runtime_batch_size + 1})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D offsets shape must be [batch+1].");
                }
                node_dims[i] = {node.ragged_runtime_max_active_values, node.ragged_conv1d_output_channels};
                break;
            }
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA: {
                const std::vector<uint64_t>& filter_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& grad_output_dims = node_dims[node.rhs];
                const std::vector<uint64_t>& offsets_dims = node_dims[node.aux];
                if (grad_output_dims != std::vector<uint64_t>({node.ragged_runtime_max_active_values,
                                                               node.ragged_conv1d_output_channels})) {
                    throw std::runtime_error(
                        "inferForwardNodeDims ragged Conv1D dgrad dY shape does not match logical metadata.");
                }
                if (node.ragged_conv1d_groups == 0 || node.ragged_conv1d_input_channels % node.ragged_conv1d_groups != 0 ||
                    node.ragged_conv1d_output_channels % node.ragged_conv1d_groups != 0 ||
                    filter_dims != std::vector<uint64_t>({node.ragged_conv1d_output_channels,
                                                          node.ragged_conv1d_input_channels / node.ragged_conv1d_groups,
                                                          node.ragged_conv1d_kernel_width})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D dgrad filter shape must be [K,C/groups,R].");
                }
                if (offsets_dims != std::vector<uint64_t>({node.ragged_runtime_batch_size + 1})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D dgrad offsets shape must be [batch+1].");
                }
                node_dims[i] = {node.ragged_runtime_max_active_values, node.ragged_conv1d_input_channels};
                break;
            }
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER: {
                const std::vector<uint64_t>& input_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& grad_output_dims = node_dims[node.rhs];
                const std::vector<uint64_t>& offsets_dims = node_dims[node.aux];
                if (input_dims != std::vector<uint64_t>({node.ragged_runtime_max_active_values,
                                                         node.ragged_conv1d_input_channels})) {
                    throw std::runtime_error(
                        "inferForwardNodeDims ragged Conv1D wgrad X shape does not match logical metadata.");
                }
                if (grad_output_dims != std::vector<uint64_t>({node.ragged_runtime_max_active_values,
                                                               node.ragged_conv1d_output_channels})) {
                    throw std::runtime_error(
                        "inferForwardNodeDims ragged Conv1D wgrad dY shape does not match logical metadata.");
                }
                if (node.ragged_conv1d_groups == 0 || node.ragged_conv1d_input_channels % node.ragged_conv1d_groups != 0 ||
                    node.ragged_conv1d_output_channels % node.ragged_conv1d_groups != 0) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D wgrad has invalid grouped channel geometry.");
                }
                if (offsets_dims != std::vector<uint64_t>({node.ragged_runtime_batch_size + 1})) {
                    throw std::runtime_error("inferForwardNodeDims ragged Conv1D wgrad offsets shape must be [batch+1].");
                }
                node_dims[i] = {node.ragged_conv1d_output_channels,
                                node.ragged_conv1d_input_channels / node.ragged_conv1d_groups,
                                node.ragged_conv1d_kernel_width};
                break;
            }
            case ExprOp::SEGMENTED_BROADCAST: {
                if (node.ragged_runtime_max_active_values == 0 || node.ragged_runtime_elements_per_value == 0) {
                    throw std::runtime_error("inferForwardNodeDims segmented broadcast is missing packed capacity metadata.");
                }
                const std::vector<uint64_t>& per_segment_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& offsets_dims = node_dims[node.rhs];
                if (per_segment_dims.empty() || offsets_dims.size() != 1 || offsets_dims[0] != per_segment_dims[0] + 1) {
                    throw std::runtime_error("inferForwardNodeDims segmented broadcast requires values [B,D...] and offsets [B+1].");
                }
                if (per_segment_dims[0] == 0 ||
                    dynamicDimsNumel(per_segment_dims, "inferForwardNodeDims segmented broadcast values") / per_segment_dims[0] !=
                        node.ragged_runtime_elements_per_value) {
                    throw std::runtime_error("inferForwardNodeDims segmented broadcast elements-per-value metadata mismatch.");
                }
                std::vector<uint64_t> output_dims = per_segment_dims;
                output_dims[0] = node.ragged_runtime_max_active_values;
                node_dims[i] = std::move(output_dims);
                break;
            }
            case ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD:
            case ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD:
                node_dims[i] = node_dims[node.lhs];
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
            case ExprOp::LAYERNORM: {
                const std::vector<uint64_t>& input_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& scale_dims = node_dims[node.rhs];
                const std::vector<uint64_t>& bias_dims = node_dims[node.aux];
                if (input_dims.size() != 2 || scale_dims.size() != 1 || bias_dims.size() != 1 ||
                    input_dims[1] != node.layer_norm_normalized_feature_count ||
                    scale_dims[0] != node.layer_norm_normalized_feature_count ||
                    bias_dims[0] != node.layer_norm_normalized_feature_count) {
                    throw std::runtime_error(
                        "inferForwardNodeDims LayerNorm expects [outer, hidden] input and [hidden] scale/bias tensors.");
                }
                node_dims[i] = input_dims;
                break;
            }
            case ExprOp::RMSNORM_BACKWARD_X:
            case ExprOp::RMSNORM_BACKWARD_SCALE: {
                const std::vector<uint64_t>& input_dims = node_dims[node.lhs];
                const std::vector<uint64_t>& scale_dims = node_dims[node.rhs];
                const std::vector<uint64_t>& dy_dims = node_dims[node.aux];
                if (input_dims.size() != 2 || scale_dims.size() != 1 || dy_dims != input_dims ||
                    input_dims[1] != node.rms_norm_normalized_feature_count ||
                    scale_dims[0] != node.rms_norm_normalized_feature_count) {
                    throw std::runtime_error(
                        "inferForwardNodeDims RMSNorm backward expects x/dY [outer, hidden] and scale [hidden].");
                }
                node_dims[i] = node.op == ExprOp::RMSNORM_BACKWARD_X ? input_dims : scale_dims;
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
            case ExprOp::CUDA_KERNEL_OUTPUT: {
                if (node.cuda_kernel_spec_index >= forward_expr.cuda_kernel_expressions.size() ||
                    !forward_expr.cuda_kernel_expressions[node.cuda_kernel_spec_index]) {
                    throw std::runtime_error("inferForwardNodeDims CudaKernelExpression node references an invalid kernel spec.");
                }
                const CudaKernelExpression& kernel = *forward_expr.cuda_kernel_expressions[node.cuda_kernel_spec_index];
                if (node.cuda_kernel_output_index >= kernel.outputs().size() || node.cuda_kernel_input_nodes.size() != kernel.inputs().size()) {
                    throw std::runtime_error("inferForwardNodeDims CudaKernelExpression node has invalid ABI metadata.");
                }
                std::unordered_map<std::string, std::vector<uint64_t>> input_shapes;
                for (size_t input_idx = 0; input_idx < kernel.inputs().size(); ++input_idx) {
                    const uint32_t input_node = node.cuda_kernel_input_nodes[input_idx];
                    if (input_node >= node_dims.size()) {
                        throw std::runtime_error("inferForwardNodeDims CudaKernelExpression input node is out of range.");
                    }
                    input_shapes.emplace(kernel.inputs()[input_idx].name, node_dims[input_node]);
                }
                const auto output_shapes = kernel.inferOutputShapesFromInputShapes(input_shapes);
                node_dims[i] = output_shapes.at(node.cuda_kernel_output_index);
                break;
            }
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

    // A broadcast gradient that still carries a ragged runtime extent must never
    // reduce the packed-capacity axis as an ordinary dense axis: doing so would
    // include storage-only rows beyond offsets[-1].  Consume that axis through a
    // segmented sum first, which produces one dense value per ragged segment, then
    // apply the ordinary dense reduction over the segment axis (and any other
    // broadcasted axes).  This is algebraically the same sum over logical packed
    // values while keeping unused capacity out of parameter/input gradients.
    const std::optional<RaggedGradientExtent> ragged_extent = builder.tryGetFrontierRaggedExtent(contrib);
    if (ragged_extent.has_value() &&
        std::find(reduction_axes.begin(), reduction_axes.end(), 0) != reduction_axes.end()) {
        if (contrib_dims.empty() || contrib_dims.front() != ragged_extent->maxActiveValues) {
            throw std::runtime_error(
                "Autodiff ragged broadcast backward requires packed capacity as contribution dimension zero.");
        }

        uint64_t elements_per_value = 1;
        for (size_t axis = 1; axis < contrib_dims.size(); ++axis) {
            elements_per_value *= contrib_dims[axis];
        }
        if (elements_per_value != ragged_extent->elementsPerValue) {
            throw std::runtime_error(
                "Autodiff ragged broadcast backward contribution width does not match its runtime extent.");
        }

        const uint32_t per_segment = builder.segmentedReduce(contrib,
                                                             ragged_extent->offsetsNode,
                                                             ExprOp::SEGMENTED_REDUCE_SUM,
                                                             ragged_extent->batchSize,
                                                             ragged_extent->maxActiveValues,
                                                             ragged_extent->elementsPerValue);
        return builder.reduction(ExprOp::REDUCE_SUM,
                                 per_segment,
                                 reduction_axes,
                                 squeeze_axes,
                                 std::nullopt,
                                 target_dtype);
    }

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
    if (dims.empty() || dims.size() != strides.size()) {
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

    // Common column views over a dense 2-D parent are expressed as a 1-D view
    // with a row stride and an in-row offset, for example source [N, 4] and
    // view x[:, 1]: dims=[N], strides=[4], offset=1. Without runtime forward
    // shape information, infer the minimal dense 2-D parent that can represent
    // this alias. This keeps ordinary column-slice backward graphs shape-safe
    // while still rejecting ambiguous/non-packed aliases below.
    if (dims.size() == 1) {
        const uint64_t row_width = strides[0];
        if (node.view_element_offset >= row_width) {
            return std::nullopt;
        }
        return std::vector<uint64_t>{dims[0], row_width};
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

static PhysicalOutputs buildFlatBackwardOutputsImpl(const PhysicalOutputs& forward_outputs,
                                         const std::vector<std::string>& wrt_names,
                                         const std::optional<std::unordered_map<std::string, std::string>>& upstream_input_names_by_output,
                                         const std::optional<std::unordered_map<std::string, DataType>>& upstream_input_dtypes_by_output,
                                         const std::optional<std::unordered_map<std::string, uint32_t>>& upstream_node_indices_by_output,
                                         const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                         bool accumulate_grad_outputs,
                                         bool allow_shape_deferred_placeholders = false) {
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
    std::unordered_set<std::string> ragged_metadata_input_names;
    std::unordered_set<uint32_t> visited_metadata_nodes;
    std::function<void(uint32_t)> collect_metadata_inputs = [&](uint32_t node_idx) {
        if (node_idx >= forward_expr.nodes.size()) {
            throw std::runtime_error("Ragged expression operation has an invalid row-partition metadata node during autodiff.");
        }
        if (!visited_metadata_nodes.insert(node_idx).second) {
            return;
        }

        const ExprNode& metadata_node = forward_expr.nodes[node_idx];
        if (metadata_node.op == ExprOp::INPUT) {
            if (metadata_node.input_slot >= forward_expr.inputs.size()) {
                throw std::runtime_error("Ragged row-partition metadata input slot is out of range during autodiff.");
            }
            ragged_metadata_input_names.insert(forward_expr.inputs[metadata_node.input_slot].name);
            return;
        }

        for (uint32_t parent : {metadata_node.lhs, metadata_node.rhs, metadata_node.aux}) {
            if (parent != UINT32_MAX) {
                collect_metadata_inputs(parent);
            }
        }
    };

    for (const ExprNode& node : forward_expr.nodes) {
        const bool uses_row_partition_metadata =
            node.op == ExprOp::RAGGED_VALUEWISE_EXTENT || node.op == ExprOp::SEGMENTED_SCAN ||
            node.op == ExprOp::SEGMENTED_REDUCE_SUM || node.op == ExprOp::SEGMENTED_REDUCE_MIN ||
            node.op == ExprOp::SEGMENTED_REDUCE_MAX || node.op == ExprOp::SEGMENTED_REDUCE_MEAN ||
            node.op == ExprOp::SEGMENTED_BROADCAST;
        if (uses_row_partition_metadata) {
            collect_metadata_inputs(node.rhs);
        }
        if (node.op == ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD) {
            collect_metadata_inputs(node.aux);
        }
    }
    for (const std::string& name : normalized_wrt) {
        if (ragged_metadata_input_names.contains(name)) {
            throw std::runtime_error("Ragged row-partition offsets are metadata and are not differentiable: " + name);
        }
    }

    const std::vector<std::vector<uint64_t>> forward_node_dims = inferForwardNodeDims(forward_expr, forward_input_dims);
    const bool has_forward_dims = !forward_node_dims.empty();
    const std::vector<bool> node_reaches_requested_inputs = computeNodeReachesRequestedInputs(forward_expr, normalized_wrt);

    const bool has_explicit_upstream_seeds = upstream_input_names_by_output.has_value() || upstream_node_indices_by_output.has_value();

    if (forward_outputs.outputs.size() > 1 && !has_explicit_upstream_seeds) {
        throw std::runtime_error(
            "buildBackwardOutputs for multi-output forward equations requires an explicit upstream seed map. The map may be partial "
            "when some outputs have no incoming gradient.");
    }

    if (has_explicit_upstream_seeds &&
        (!upstream_input_names_by_output.has_value() || upstream_input_names_by_output->empty()) &&
        (!upstream_node_indices_by_output.has_value() || upstream_node_indices_by_output->empty())) {
        throw std::runtime_error("buildBackwardOutputs explicit upstream seed map must contain at least one forward output.");
    }

    if (upstream_input_names_by_output.has_value() && upstream_node_indices_by_output.has_value()) {
        for (const auto& [output_name, upstream_name] : upstream_input_names_by_output.value()) {
            (void)upstream_name;
            if (upstream_node_indices_by_output->contains(output_name)) {
                throw std::runtime_error("buildBackwardOutputs received both an upstream input and an upstream node for output: " +
                                         output_name);
            }
        }
    }

    BackwardGraphBuilder builder(forward_expr);
    builder.initializeAdjoints();

    for (const NamedOutput& forward_output : forward_outputs.outputs) {
        uint32_t output_seed = UINT32_MAX;
        if (has_explicit_upstream_seeds) {
            if (upstream_node_indices_by_output.has_value()) {
                auto upstream_node_it = upstream_node_indices_by_output->find(forward_output.name);
                if (upstream_node_it != upstream_node_indices_by_output->end()) {
                    output_seed = builder.cloneForward(upstream_node_it->second);
                }
            }

            if (output_seed == UINT32_MAX && upstream_input_names_by_output.has_value()) {
                auto upstream_it = upstream_input_names_by_output->find(forward_output.name);
                if (upstream_it != upstream_input_names_by_output->end()) {
                    std::optional<DataType> upstream_dtype;
                    if (upstream_input_dtypes_by_output.has_value()) {
                        auto dtype_it = upstream_input_dtypes_by_output->find(forward_output.name);
                        if (dtype_it == upstream_input_dtypes_by_output->end()) {
                            throw std::runtime_error(
                                "buildBackwardOutputs received an upstream gradient input without its runtime dtype for output: " +
                                forward_output.name);
                        }
                        upstream_dtype = dtype_it->second;
                    }
                    output_seed = builder.input(upstream_it->second, upstream_dtype);
                }
            }

            if (output_seed == UINT32_MAX) {
                // A partial explicit upstream map means this forward output did not receive
                // an incoming gradient, so it contributes nothing to the requested wrt gradients.
                continue;
            }
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

    // Dense gradient shape expansion is a mathematical broadcast, so represent
    // it explicitly. Constant-like gradients may still fold directly to FILL;
    // that is genuine constant creation rather than arithmetic used to induce a
    // broadcast. Ragged row-wise expansion remains SEGMENTED_BROADCAST.
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

        return builder.broadcastTo(grad_value, target_dims);
    };

    auto shapeGradLikeNodeOutput =
        [&](uint32_t grad_value, uint32_t forward_node_idx, const std::vector<uint64_t>& forward_node_output_dims) -> uint32_t {
        if (!has_forward_dims || forward_node_output_dims.empty()) {
            return grad_value;
        }

        double constant_value = 0.0;
        std::vector<uint64_t> constant_dims;
        if (!builder.tryGetConstantLike(grad_value, constant_value, constant_dims)) {
            // Tensor-valued upstream gradients usually already have the shape of the forward node output.
            // Only add a metadata reshape when we can prove the gradient's logical shape and it differs.
            // Blindly wrapping unknown tensor gradients is unsafe: the same runtime input can then appear
            // both through that reshape's output domain and through its natural broadcast domain in one
            // fused stage, which gives the broadcast planner incompatible logical shapes for one slot.
            const auto inferred_grad_dims = builder.tryInferKnownGradientDims(grad_value);
            if (!inferred_grad_dims.has_value() || inferred_grad_dims.value() == forward_node_output_dims) {
                return grad_value;
            }
            return builder.reshape(grad_value, forward_node_output_dims);
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
            // the attention output. Do not add a synthetic shape-materialization node
            // just to stamp cuDNN attention backward; that wrapper is a no-op for the
            // common training path and incorrectly hides the single-stage
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

        const uint32_t raw_grad = grad_opt.value();
        const ExprNode& node = forward_expr.nodes[static_cast<size_t>(node_idx)];
        const std::vector<uint64_t> node_dims =
            has_forward_dims ? forward_node_dims.at(static_cast<size_t>(node_idx)) : std::vector<uint64_t>{};
        const bool node_uses_upstream_grad = node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR &&
                                             node.op != ExprOp::TENSOR_RUNTIME_SCALAR && node.op != ExprOp::SCALAR_FP;
        const uint32_t grad = node_uses_upstream_grad ? shapeGradLikeNodeOutput(raw_grad, static_cast<uint32_t>(node_idx), node_dims)
                                                      : raw_grad;

        switch (node.op) {
            case ExprOp::INPUT:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                break;

            case ExprOp::RAGGED_VALUEWISE_EXTENT:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    if (node.ragged_runtime_batch_size == 0 || node.ragged_runtime_max_active_values == 0 ||
                        node.ragged_runtime_elements_per_value == 0) {
                        throw std::runtime_error("Ragged valuewise autodiff encountered incomplete runtime-extent metadata.");
                    }
                    const uint32_t offsets = builder.cloneForward(node.rhs);
                    const uint32_t ragged_grad = builder.raggedValuewiseExtent(
                        grad,
                        offsets,
                        node.ragged_runtime_batch_size,
                        node.ragged_runtime_max_active_values,
                        node.ragged_runtime_elements_per_value,
                        preferredGradValueDType(forward_expr.nodes.at(node.lhs)));
                    addContributionToChild(node.lhs, ragged_grad, node_dims, preferredGradValueDType(forward_expr.nodes.at(node.lhs)));
                }
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

            case ExprOp::WHERE: {
                const uint32_t cond = builder.cloneForward(node.lhs);
                const uint32_t zero = builder.scalar(0.0);
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    const uint32_t true_contrib = builder.where(cond, grad, zero);
                    addContributionToChild(node.rhs, true_contrib, node_dims);
                }
                if (node_reaches_requested_inputs.at(node.aux)) {
                    const uint32_t false_contrib = builder.where(cond, zero, grad);
                    addContributionToChild(node.aux, false_contrib, node_dims);
                }
                break;
            }

            case ExprOp::CEIL:
            case ExprOp::FLOOR:
            case ExprOp::ROUND:
            case ExprOp::TRUNC:
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::LOGICAL_NOT:
                throw std::runtime_error("Thor expressions autodiff does not support backward for op " + opName(node.op) + ".");

            case ExprOp::CAST: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    // Type conversion is a straight-through operation in Thor's layer semantics:
                    // TypeConversion::backProp() converts the incoming gradient back into the
                    // source tensor dtype.  Keep Expression CAST consistent with that behavior so
                    // graph-level type conversion can use the normal Expression/autodiff engine.
                    const ExprNode& lhs_node = forward_expr.nodes.at(node.lhs);
                    const std::optional<DataType> source_dtype = preferredGradValueDType(lhs_node);
                    if (!source_dtype.has_value()) {
                        if (allow_shape_deferred_placeholders) {
                            // FusedEquation::compileBackward builds an initial backward template before
                            // runtime forward-input dtypes are known.  That template is rebuilt from a
                            // dtype-resolved forward graph during stamp-time specialization, at which
                            // point the real cast-back to the source gradient dtype is inserted.  Keep
                            // the deferred template buildable without guessing a dtype here.
                            addContributionToChild(node.lhs, grad, node_dims);
                            break;
                        }
                        throw std::runtime_error(
                            "CAST autodiff requires the source value dtype to be resolved before building backward outputs.");
                    }
                    addContributionToChild(node.lhs, builder.cast(grad, source_dtype.value()), node_dims);
                }
                break;
            }

            case ExprOp::SIN: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.cos(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::COS: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(builder.neg(grad), builder.sin(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::TAN: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t one_plus_out_squared = builder.add(builder.scalar(1.0), builder.mul(out, out));
                    addContributionToChild(node.lhs, builder.mul(grad, one_plus_out_squared), node_dims);
                }
                break;
            }

            case ExprOp::ASIN: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t one_minus_lhs_squared = builder.sub(builder.scalar(1.0), builder.mul(lhs, lhs));
                    addContributionToChild(node.lhs, builder.div(grad, builder.sqrt(one_minus_lhs_squared)), node_dims);
                }
                break;
            }

            case ExprOp::ACOS: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t one_minus_lhs_squared = builder.sub(builder.scalar(1.0), builder.mul(lhs, lhs));
                    addContributionToChild(node.lhs, builder.div(builder.neg(grad), builder.sqrt(one_minus_lhs_squared)), node_dims);
                }
                break;
            }

            case ExprOp::ATAN: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t one_plus_lhs_squared = builder.add(builder.scalar(1.0), builder.mul(lhs, lhs));
                    addContributionToChild(node.lhs, builder.div(grad, one_plus_lhs_squared), node_dims);
                }
                break;
            }

            case ExprOp::SINH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.cosh(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::COSH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.sinh(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::ASINH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.sqrt(builder.add(builder.mul(lhs, lhs), builder.scalar(1.0)));
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::ACOSH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.mul(builder.sqrt(builder.sub(lhs, builder.scalar(1.0))),
                                                       builder.sqrt(builder.add(lhs, builder.scalar(1.0))));
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::ATANH: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t denom = builder.sub(builder.scalar(1.0), builder.mul(lhs, lhs));
                    addContributionToChild(node.lhs, builder.div(grad, denom), node_dims);
                }
                break;
            }

            case ExprOp::ERF: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t lhs_squared = builder.mul(lhs, lhs);
                    const uint32_t scale = builder.scalar(1.1283791670955126);  // 2 / sqrt(pi)
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(scale, builder.exp(builder.neg(lhs_squared)))), node_dims);
                }
                break;
            }

            case ExprOp::ERFC: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t lhs_squared = builder.mul(lhs, lhs);
                    const uint32_t scale = builder.scalar(-1.1283791670955126);  // -2 / sqrt(pi)
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(scale, builder.exp(builder.neg(lhs_squared)))), node_dims);
                }
                break;
            }

            case ExprOp::ERFCX: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t two_x_erfcx = builder.mul(builder.scalar(2.0), builder.mul(lhs, out));
                    const uint32_t two_over_sqrt_pi = builder.scalar(1.1283791670955126);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.sub(two_x_erfcx, two_over_sqrt_pi)), node_dims);
                }
                break;
            }

            case ExprOp::ERFINV: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t scale = builder.scalar(0.8862269254527580);  // sqrt(pi) / 2
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(scale, builder.exp(builder.mul(out, out)))), node_dims);
                }
                break;
            }

            case ExprOp::ERFCINV: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    const uint32_t scale = builder.scalar(-0.8862269254527580);  // -sqrt(pi) / 2
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(scale, builder.exp(builder.mul(out, out)))), node_dims);
                }
                break;
            }

            case ExprOp::TGAMMA: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    const uint32_t out = builder.cloneForward(static_cast<uint32_t>(node_idx));
                    addContributionToChild(node.lhs, builder.mul(grad, builder.mul(out, builder.digamma(lhs))), node_dims);
                }
                break;
            }

            case ExprOp::LGAMMA: {
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    const uint32_t lhs = builder.cloneForward(node.lhs);
                    addContributionToChild(node.lhs, builder.mul(grad, builder.digamma(lhs)), node_dims);
                }
                break;
            }

            case ExprOp::DIGAMMA:
                throw std::runtime_error("Thor expressions autodiff does not support backward for digamma yet; digamma backward requires trigamma.");

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
                    } else if (allow_shape_deferred_placeholders) {
                        // FusedEquation::compileBackward builds an initial template before runtime
                        // forward input shapes are known.  The template is only used to expose
                        // root inputs/output names and is rebuilt with concrete forward dims at
                        // stamp time, so keep the graph buildable without guessing a reshape
                        // target here.  Direct buildBackwardOutputs(...) still rejects this
                        // path unless the caller supplies forward_input_dims.
                        builder.addContribution(node.lhs, grad);
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
                            if (allow_shape_deferred_placeholders) {
                                // See RESHAPE above: this placeholder only keeps the initial
                                // deferred backward template buildable.  The stamped backward
                                // graph is rebuilt with concrete forward input dims.
                                builder.addContribution(node.lhs, grad);
                                break;
                            }
                            throw std::runtime_error("AutoDiff strided_view backward requires forward shape information.");
                        }
                        source_dims = inferred_source_dims.value();
                    }
                    uint32_t strided_view_grad =
                        builder.stridedViewBackward(grad,
                                                    source_dims,
                                                    node.view_dims,
                                                    node.view_strides,
                                                    node.view_element_offset,
                                                    preferredGradValueDType(forward_expr.nodes.at(node.lhs)),
                                                    preferredGradValueDType(forward_expr.nodes.at(node.lhs)));

                    // A trailing ragged slice changes the number of elements represented by
                    // each packed row.  The incoming gradient carries the sliced width, while
                    // STRIDED_VIEW_BACKWARD scatters into the wider source-row domain.  Re-wrap
                    // that scatter in the same row partition using the source width so the fused
                    // kernel launches over active_rows * source_elements_per_value rather than
                    // accidentally reusing the narrower sliced launch extent.
                    const std::optional<RaggedGradientExtent> ragged_extent = builder.tryGetFrontierRaggedExtent(grad);
                    if (ragged_extent.has_value()) {
                        if (source_dims.empty() || node.view_dims.empty() ||
                            source_dims.front() != ragged_extent->maxActiveValues ||
                            node.view_dims.front() != ragged_extent->maxActiveValues) {
                            throw std::runtime_error(
                                "AutoDiff ragged strided_view backward requires the packed-row capacity as dimension zero.");
                        }

                        const std::vector<uint64_t> source_trailing_dims(source_dims.begin() + 1, source_dims.end());
                        const std::vector<uint64_t> view_trailing_dims(node.view_dims.begin() + 1, node.view_dims.end());
                        const uint64_t source_elements_per_value =
                            source_trailing_dims.empty() ? 1 : dynamicDimsNumel(source_trailing_dims, "ragged strided_view source");
                        const uint64_t view_elements_per_value =
                            view_trailing_dims.empty() ? 1 : dynamicDimsNumel(view_trailing_dims, "ragged strided_view view");
                        if (ragged_extent->elementsPerValue != view_elements_per_value) {
                            throw std::runtime_error(
                                "AutoDiff ragged strided_view backward received a gradient extent that does not match the view width.");
                        }
                        strided_view_grad = builder.raggedValuewiseExtent(
                            strided_view_grad,
                            ragged_extent->offsetsNode,
                            ragged_extent->batchSize,
                            ragged_extent->maxActiveValues,
                            source_elements_per_value,
                            preferredGradValueDType(forward_expr.nodes.at(node.lhs)));
                    }

                    builder.addContribution(node.lhs, strided_view_grad);
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

            case ExprOp::CUDA_KERNEL_OUTPUT: {
                if (!node_reaches_requested_inputs.at(static_cast<size_t>(node_idx))) {
                    break;
                }
                if (node.cuda_kernel_spec_index >= forward_expr.cuda_kernel_expressions.size() ||
                    !forward_expr.cuda_kernel_expressions[node.cuda_kernel_spec_index]) {
                    throw std::runtime_error("CudaKernelExpression autodiff node references an invalid kernel spec.");
                }
                const CudaKernelExpression& forward_kernel = *forward_expr.cuda_kernel_expressions[node.cuda_kernel_spec_index];
                if (node.cuda_kernel_output_index >= forward_kernel.outputs().size() ||
                    node.cuda_kernel_input_nodes.size() != forward_kernel.inputs().size()) {
                    throw std::runtime_error("CudaKernelExpression autodiff node has invalid ABI metadata.");
                }
                const std::string& forward_output_name = forward_kernel.outputs()[node.cuda_kernel_output_index].name;
                const CudaKernelExpression::BackwardSpec* backward = forward_kernel.backwardSpecForOutput(forward_output_name);
                if (backward == nullptr || !backward->kernel) {
                    throw std::runtime_error("CudaKernelExpression '" + forward_kernel.name() + "' output '" + forward_output_name +
                                             "' participates in backpropagation but has no explicit backward kernel. Declare one with "
                                             "CudaKernelExpression::Builder::backward(...).");
                }

                std::unordered_map<std::string, size_t> forward_input_index_by_name;
                for (size_t input_idx = 0; input_idx < forward_kernel.inputs().size(); ++input_idx) {
                    forward_input_index_by_name.emplace(forward_kernel.inputs()[input_idx].name, input_idx);
                }

                // A custom backward kernel can explicitly consume a ragged upstream gradient by also
                // receiving the canonical offsets tensor that defines that gradient's active prefix.
                // Keep the compiler's generic stage-boundary guard strict: unwrap the valuewise extent
                // only for a kernel whose declared forward inputs include that exact offsets INPUT, then
                // restore the same extent on backward outputs declared outputLike(dY). This is the
                // consumer-responsibility contract used by the fused ragged dropout post-op: the kernel
                // bounds every read/write by offsets[-1], while downstream gradient expressions retain
                // the partition metadata needed to avoid capacity-tail reads.
                std::optional<RaggedGradientExtent> custom_backward_ragged_extent;
                uint32_t custom_backward_upstream_grad = grad;
                const ExprNode& upstream_grad_node = builder.node(grad);
                if (upstream_grad_node.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
                    if (upstream_grad_node.lhs == UINT32_MAX || upstream_grad_node.rhs == UINT32_MAX ||
                        upstream_grad_node.ragged_runtime_batch_size == 0 ||
                        upstream_grad_node.ragged_runtime_max_active_values == 0 ||
                        upstream_grad_node.ragged_runtime_elements_per_value == 0) {
                        throw std::runtime_error(
                            "CudaKernelExpression backward received malformed ragged upstream-gradient metadata.");
                    }
                    const ExprNode& offsets_node = builder.node(upstream_grad_node.rhs);
                    if (offsets_node.op != ExprOp::INPUT) {
                        throw std::runtime_error(
                            "CudaKernelExpression ragged backward requires canonical direct-input offsets.");
                    }

                    bool kernel_receives_matching_offsets = false;
                    for (const auto& backward_input : backward->kernel->inputs()) {
                        if (backward_input.name == backward->upstream_gradient_input_name) {
                            continue;
                        }
                        auto forward_input_it = forward_input_index_by_name.find(backward_input.name);
                        if (forward_input_it == forward_input_index_by_name.end()) {
                            continue;
                        }
                        const uint32_t forward_input_node = node.cuda_kernel_input_nodes[forward_input_it->second];
                        if (forward_input_node >= forward_expr.nodes.size()) {
                            throw std::runtime_error(
                                "CudaKernelExpression ragged backward offsets candidate is out of range.");
                        }
                        const ExprNode& forward_input = forward_expr.nodes[forward_input_node];
                        if (forward_input.op == ExprOp::INPUT && forward_input.input_slot == offsets_node.input_slot) {
                            kernel_receives_matching_offsets = true;
                            break;
                        }
                    }

                    if (kernel_receives_matching_offsets) {
                        custom_backward_ragged_extent = RaggedGradientExtent{
                            .offsetsNode = upstream_grad_node.rhs,
                            .offsetsInputSlot = offsets_node.input_slot,
                            .batchSize = upstream_grad_node.ragged_runtime_batch_size,
                            .maxActiveValues = upstream_grad_node.ragged_runtime_max_active_values,
                            .elementsPerValue = upstream_grad_node.ragged_runtime_elements_per_value,
                        };
                        custom_backward_upstream_grad = upstream_grad_node.lhs;
                    }
                }

                std::unordered_map<std::string, uint32_t> backward_input_nodes;
                backward_input_nodes.reserve(backward->kernel->inputs().size());
                std::unordered_map<std::string, std::vector<uint64_t>> backward_input_shapes;
                for (const auto& backward_input : backward->kernel->inputs()) {
                    if (backward_input.name == backward->upstream_gradient_input_name) {
                        backward_input_nodes.emplace(backward_input.name, custom_backward_upstream_grad);
                        if (has_forward_dims) {
                            backward_input_shapes.emplace(backward_input.name, node_dims);
                        }
                        continue;
                    }
                    auto forward_input_it = forward_input_index_by_name.find(backward_input.name);
                    if (forward_input_it == forward_input_index_by_name.end()) {
                        throw std::runtime_error("CudaKernelExpression backward kernel contains an input that does not bind to a forward "
                                                 "kernel input: " + backward_input.name);
                    }
                    const uint32_t forward_input_node = node.cuda_kernel_input_nodes[forward_input_it->second];
                    backward_input_nodes.emplace(backward_input.name, builder.cloneForward(forward_input_node));
                    if (has_forward_dims) {
                        backward_input_shapes.emplace(backward_input.name, forward_node_dims.at(forward_input_node));
                    }
                }

                const std::unordered_map<std::string, uint32_t> backward_outputs =
                    builder.cudaKernel(*backward->kernel, backward_input_nodes);
                std::unordered_map<std::string, std::vector<uint64_t>> backward_output_shapes;
                if (has_forward_dims) {
                    const auto inferred = backward->kernel->inferOutputShapesFromInputShapes(backward_input_shapes);
                    for (size_t output_idx = 0; output_idx < backward->kernel->outputs().size(); ++output_idx) {
                        backward_output_shapes.emplace(backward->kernel->outputs()[output_idx].name, inferred[output_idx]);
                    }
                }

                for (const auto& [backward_output_name, forward_input_name] : backward->input_gradients) {
                    auto forward_input_it = forward_input_index_by_name.find(forward_input_name);
                    if (forward_input_it == forward_input_index_by_name.end()) {
                        throw std::runtime_error("CudaKernelExpression backward gradient mapping references unknown forward input: " +
                                                 forward_input_name);
                    }
                    const uint32_t forward_input_node = node.cuda_kernel_input_nodes[forward_input_it->second];
                    if (!node_reaches_requested_inputs.at(forward_input_node)) {
                        continue;
                    }
                    auto backward_output_it = backward_outputs.find(backward_output_name);
                    if (backward_output_it == backward_outputs.end()) {
                        throw std::runtime_error("CudaKernelExpression backward gradient mapping references unknown backward output: " +
                                                 backward_output_name);
                    }
                    if (has_forward_dims) {
                        const auto shape_it = backward_output_shapes.find(backward_output_name);
                        if (shape_it == backward_output_shapes.end() || shape_it->second != forward_node_dims.at(forward_input_node)) {
                            throw std::runtime_error("CudaKernelExpression backward output '" + backward_output_name +
                                                     "' must have exactly the shape of forward input '" + forward_input_name + "'.");
                        }
                    }
                    uint32_t backward_contribution = backward_output_it->second;
                    if (custom_backward_ragged_extent.has_value()) {
                        const auto backward_output_spec_it = std::find_if(
                            backward->kernel->outputs().begin(),
                            backward->kernel->outputs().end(),
                            [&](const CudaKernelExpression::OutputParamSpec& output) {
                                return output.name == backward_output_name;
                            });
                        if (backward_output_spec_it == backward->kernel->outputs().end()) {
                            throw std::runtime_error(
                                "CudaKernelExpression backward output metadata disappeared while propagating ragged extent.");
                        }
                        if (backward_output_spec_it->like_input_name == backward->upstream_gradient_input_name) {
                            const RaggedGradientExtent& extent = custom_backward_ragged_extent.value();
                            backward_contribution = builder.raggedValuewiseExtent(
                                backward_contribution,
                                extent.offsetsNode,
                                extent.batchSize,
                                extent.maxActiveValues,
                                extent.elementsPerValue,
                                preferredGradValueDType(forward_expr.nodes.at(forward_input_node)));
                        }
                    }
                    builder.addContribution(forward_input_node, backward_contribution);
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
                    const uint32_t effective_sequence_length =
                        node.rope_effective_sequence_length_node == UINT32_MAX
                            ? UINT32_MAX
                            : builder.cloneForward(node.rope_effective_sequence_length_node);
                    const uint32_t position_ids =
                        node.rope_position_ids_node == UINT32_MAX
                            ? UINT32_MAX
                            : builder.cloneForward(node.rope_position_ids_node);
                    const uint32_t lhs_grad = builder.rotaryPositionEmbedding(
                        grad,
                        node,
                        !node.rope_inverse,
                        effective_sequence_length,
                        position_ids,
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
                    if (node.rms_norm_fused_activation != CudnnRmsNormFusedActivation::NONE) {
                        throw std::runtime_error(
                            "Training autodiff cannot use a fused RMSNorm activation; keep the activation as a separate expression.");
                    }
                    if (allow_shape_deferred_placeholders) {
                        // FusedEquation::compileBackward first builds a shape-deferred
                        // template before runtime forward-input dimensions are known.
                        // RMSNorm backward needs those dimensions to construct the real
                        // dX/dscale operators, so keep only the requested adjoint routes
                        // alive in this template. buildShapeSpecializedOutputs() rebuilds
                        // the backward graph with concrete forward shapes before it is
                        // compiled or executed.
                        if (node_reaches_requested_inputs.at(node.lhs)) {
                            builder.addContribution(node.lhs, grad);
                        }
                        if (node_reaches_requested_inputs.at(node.rhs)) {
                            builder.addContribution(node.rhs, grad);
                        }
                        break;
                    }
                    throw std::runtime_error("Autodiff RMSNorm backward requires forward shape information.");
                }

                if (node.rms_norm_packed_row_capacity == 0) {
                    if (node.rms_norm_fused_activation != CudnnRmsNormFusedActivation::NONE) {
                        throw std::runtime_error(
                            "Training autodiff cannot use a fused RMSNorm activation; keep the activation as a separate expression.");
                    }
                    const std::vector<uint64_t>& x_dims = forward_node_dims.at(node.lhs);
                    const std::vector<uint64_t>& scale_dims = forward_node_dims.at(node.rhs);
                    if (x_dims.size() != 2 || scale_dims.size() != 1 ||
                        x_dims[1] != node.rms_norm_normalized_feature_count ||
                        scale_dims[0] != node.rms_norm_normalized_feature_count) {
                        throw std::runtime_error(
                            "Autodiff RMSNorm backward expects [outer, hidden] input and [hidden] scale tensors.");
                    }

                    uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                    if (node.output_dtype.has_value()) {
                        grad_like_output = builder.cast(grad_like_output, node.output_dtype.value());
                    }
                    const uint32_t x = builder.cloneForward(node.lhs);
                    const uint32_t scale = builder.cloneForward(node.rhs);
                    const ExprNode& forward_x = forward_expr.nodes.at(node.lhs);
                    const ExprNode& forward_scale = forward_expr.nodes.at(node.rhs);
                    if (!forward_x.output_dtype.has_value() || !forward_scale.output_dtype.has_value()) {
                        throw std::runtime_error("Autodiff RMSNorm backward requires resolved x/scale storage dtypes.");
                    }
                    if (node_reaches_requested_inputs.at(node.lhs)) {
                        uint32_t dx = builder.rmsNormBackward(ExprOp::RMSNORM_BACKWARD_X,
                                                              x,
                                                              scale,
                                                              grad_like_output,
                                                              node,
                                                              forward_x.output_dtype,
                                                              node.compute_dtype);
                        const std::optional<DataType> target_dx_dtype = preferredGradValueDType(forward_x);
                        if (target_dx_dtype.has_value() && target_dx_dtype.value() != forward_x.output_dtype.value()) {
                            dx = builder.cast(dx, target_dx_dtype.value());
                        }
                        addContributionToChild(node.lhs, dx, x_dims);
                    }
                    if (node_reaches_requested_inputs.at(node.rhs)) {
                        uint32_t dscale = builder.rmsNormBackward(ExprOp::RMSNORM_BACKWARD_SCALE,
                                                                  x,
                                                                  scale,
                                                                  grad_like_output,
                                                                  node,
                                                                  forward_scale.output_dtype,
                                                                  node.compute_dtype);
                        const std::optional<DataType> target_dscale_dtype = preferredGradValueDType(forward_scale);
                        if (target_dscale_dtype.has_value() && target_dscale_dtype.value() != forward_scale.output_dtype.value()) {
                            dscale = builder.cast(dscale, target_dscale_dtype.value());
                        }
                        addContributionToChild(node.rhs, dscale, scale_dims);
                    }
                    break;
                }

                if (node.rms_norm_fused_activation != CudnnRmsNormFusedActivation::NONE) {
                    throw std::runtime_error(
                        "Training autodiff cannot use a fused RMSNorm activation; keep the activation as a separate expression.");
                }
                const std::vector<uint64_t>& x_dims = forward_node_dims.at(node.lhs);
                const std::vector<uint64_t>& scale_dims = forward_node_dims.at(node.rhs);
                if (x_dims.size() != 2 || scale_dims.size() != 1 ||
                    x_dims[1] != node.rms_norm_normalized_feature_count ||
                    scale_dims[0] != node.rms_norm_normalized_feature_count) {
                    throw std::runtime_error(
                        "Autodiff packed RMSNorm backward expects [outer, hidden] input and [hidden] scale tensors.");
                }
                const ExprNode& forward_x_extent = forward_expr.nodes.at(node.lhs);
                if (forward_x_extent.op != ExprOp::RAGGED_VALUEWISE_EXTENT ||
                    forward_x_extent.ragged_runtime_max_active_values != node.rms_norm_packed_row_capacity) {
                    throw std::runtime_error(
                        "Autodiff packed RMSNorm backward requires a direct ragged runtime extent matching packed capacity.");
                }

                uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                if (node.output_dtype.has_value()) {
                    grad_like_output = builder.cast(grad_like_output, node.output_dtype.value());
                }
                const uint32_t x = builder.cloneForward(node.lhs);
                const uint32_t scale = builder.cloneForward(node.rhs);
                const ExprNode& forward_scale = forward_expr.nodes.at(node.rhs);
                const ExprNode& forward_values = forward_expr.nodes.at(forward_x_extent.lhs);
                if (!forward_values.output_dtype.has_value() || !forward_scale.output_dtype.has_value()) {
                    throw std::runtime_error("Autodiff packed RMSNorm backward requires resolved x/scale storage dtypes.");
                }
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    uint32_t dx = builder.rmsNormBackward(ExprOp::RMSNORM_BACKWARD_X,
                                                          x,
                                                          scale,
                                                          grad_like_output,
                                                          node,
                                                          forward_values.output_dtype,
                                                          node.compute_dtype);
                    const std::optional<DataType> target_dx_dtype = preferredGradValueDType(forward_values);
                    if (target_dx_dtype.has_value() && target_dx_dtype.value() != forward_values.output_dtype.value()) {
                        dx = builder.cast(dx, target_dx_dtype.value());
                    }
                    addContributionToChild(node.lhs, dx, x_dims);
                }
                if (node_reaches_requested_inputs.at(node.rhs)) {
                    uint32_t dscale = builder.rmsNormBackward(ExprOp::RMSNORM_BACKWARD_SCALE,
                                                              x,
                                                              scale,
                                                              grad_like_output,
                                                              node,
                                                              forward_scale.output_dtype,
                                                              node.compute_dtype);
                    const std::optional<DataType> target_dscale_dtype = preferredGradValueDType(forward_scale);
                    if (target_dscale_dtype.has_value() && target_dscale_dtype.value() != forward_scale.output_dtype.value()) {
                        dscale = builder.cast(dscale, target_dscale_dtype.value());
                    }
                    addContributionToChild(node.rhs, dscale, scale_dims);
                }
                break;
            }

            case ExprOp::LAYERNORM:
                throw std::runtime_error(
                    "LayerNorm expression autodiff is deferred to T9, where retained padded backward reductions and inactive-tail safety are implemented.");

            case ExprOp::RMSNORM_BACKWARD_X:
            case ExprOp::RMSNORM_BACKWARD_SCALE:
                throw std::runtime_error("Thor expressions autodiff does not support second derivatives for RMSNorm backward yet.");

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
                uint32_t matrix_grad_like_output = grad_like_output;
                std::optional<DataType> low_precision_operand_dtype;
                if (node_reaches_requested_inputs.at(node.lhs) || node_reaches_requested_inputs.at(node.rhs)) {
                    low_precision_operand_dtype = matmulLowPrecisionOperandDType(forward_expr, node);
                    if (low_precision_operand_dtype.has_value()) {
                        // cuBLASLt requires the two regular dense matrix operands to have the same
                        // FP16/BF16 dtype.  The upstream gradient can still be an untyped synthetic
                        // CustomLayer input while this graph is built and only resolve to FP32 during
                        // stamping.  Always establish the low-precision matrix-operand boundary here.
                        // The helper elides the conversion only when the producer is known to be
                        // materially in that dtype; this one result is shared by dL/dlhs and dL/drhs.
                        matrix_grad_like_output =
                            builder.materializeMatmulOperandCast(grad_like_output, low_precision_operand_dtype.value());
                    }
                }
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : std::vector<uint64_t>{};
                const std::vector<uint64_t> rhs_dims = has_forward_dims ? forward_node_dims.at(node.rhs) : std::vector<uint64_t>{};
                const auto lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                const auto rhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.rhs));

                auto direct_ragged_marker = [&](uint32_t forward_node_idx) -> const ExprNode* {
                    if (forward_node_idx >= forward_expr.nodes.size()) return nullptr;
                    const ExprNode& candidate = forward_expr.nodes.at(forward_node_idx);
                    if (candidate.op != ExprOp::RAGGED_VALUEWISE_EXTENT) return nullptr;
                    if (candidate.rhs == UINT32_MAX || candidate.ragged_runtime_batch_size == 0 ||
                        candidate.ragged_runtime_max_active_values == 0 || candidate.ragged_runtime_elements_per_value == 0) {
                        throw std::runtime_error("Autodiff found malformed ragged runtime extent on MATMUL operand.");
                    }
                    return &candidate;
                };
                const ExprNode* lhs_ragged_marker = direct_ragged_marker(node.lhs);
                auto wrap_with_marker = [&](uint32_t value, const ExprNode& marker, uint64_t elements_per_value) {
                    return builder.raggedValuewiseExtent(value,
                                                         builder.cloneForward(marker.rhs),
                                                         marker.ragged_runtime_batch_size,
                                                         marker.ragged_runtime_max_active_values,
                                                         elements_per_value);
                };
                auto trailing_elements_per_packed_row = [&](const std::vector<uint64_t>& dims, const ExprNode& marker) {
                    if (node.matmul_packed_row_capacity == 0) {
                        throw std::runtime_error("Packed MATMUL autodiff is missing its packed row capacity.");
                    }
                    if (dims.empty()) {
                        if (allow_shape_deferred_placeholders) {
                            // compileBackward() first builds a shape-deferred template which is
                            // never executed directly. Keep the row-partition dependency alive in
                            // that template; buildShapeSpecializedOutputs() will rebuild this graph
                            // with concrete forward dimensions before compilation/execution.
                            return marker.ragged_runtime_elements_per_value;
                        }
                        throw std::runtime_error("Packed MATMUL autodiff cannot resolve row width without output geometry.");
                    }
                    if (dims[0] != node.matmul_packed_row_capacity) {
                        throw std::runtime_error("Packed MATMUL autodiff output geometry does not match its packed row capacity.");
                    }
                    uint64_t elements = 1;
                    for (size_t axis = 1; axis < dims.size(); ++axis) {
                        if (dims[axis] == 0 || elements > std::numeric_limits<uint64_t>::max() / dims[axis]) {
                            throw std::runtime_error("Packed MATMUL autodiff row width overflows uint64_t.");
                        }
                        elements *= dims[axis];
                    }
                    return elements;
                };

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    uint32_t rhs = builder.cloneForward(node.rhs);
                    if (low_precision_operand_dtype.has_value()) {
                        // A forward operand may be backed by BF16/FP16 storage while deliberately
                        // exposing an FP32 logical value.  Reusing that logical value as a backward
                        // matrix operand requires an explicit down-conversion, independently of the
                        // upstream-gradient conversion above.
                        rhs = builder.cast(rhs, low_precision_operand_dtype.value());
                    }
                    uint32_t row_grad_operand = matrix_grad_like_output;
                    if (node.matmul_packed_row_binding == MatmulPackedRowBinding::RowsA && lhs_ragged_marker != nullptr) {
                        row_grad_operand = wrap_with_marker(
                            row_grad_operand,
                            *lhs_ragged_marker,
                            trailing_elements_per_packed_row(node_dims, *lhs_ragged_marker));
                    }
                    uint32_t lhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(row_grad_operand, rhs, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        lhs_grad = builder.matmul(row_grad_operand, rhs, false, false, lhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(rhs, row_grad_operand, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else {
                        lhs_grad = builder.matmul(rhs, row_grad_operand, true, true, lhs_grad_dtype, node.compute_dtype);
                    }

                    if (node.matmul_packed_row_binding == MatmulPackedRowBinding::RowsA &&
                        node.matmul_packed_row_capacity != 0) {
                        builder.setPackedRowMatmul(lhs_grad, MatmulPackedRowBinding::RowsA, node.matmul_packed_row_capacity);
                    }
                    lhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, lhs_grad);
                    const std::vector<uint64_t> lhs_grad_dims =
                        has_forward_dims ? rawBatchedMatmulOperandGradientDims(node_dims, lhs_dims) : lhs_dims;
                    addContributionToChild(node.lhs, lhs_grad, lhs_grad_dims, lhs_grad_dtype);
                }

                if (node_reaches_requested_inputs.at(node.rhs)) {
                    uint32_t lhs = UINT32_MAX;
                    if (lhs_ragged_marker != nullptr) {
                        lhs = builder.cloneForward(lhs_ragged_marker->lhs);
                        if (low_precision_operand_dtype.has_value()) {
                            lhs = builder.cast(lhs, low_precision_operand_dtype.value());
                        }
                        lhs = wrap_with_marker(lhs, *lhs_ragged_marker, lhs_ragged_marker->ragged_runtime_elements_per_value);
                    } else {
                        lhs = builder.cloneForward(node.lhs);
                        if (low_precision_operand_dtype.has_value()) {
                            lhs = builder.cast(lhs, low_precision_operand_dtype.value());
                        }
                    }
                    uint32_t rhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, matrix_grad_like_output, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        rhs_grad = builder.matmul(matrix_grad_like_output, lhs, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, matrix_grad_like_output, false, false, rhs_grad_dtype, node.compute_dtype);
                    } else {
                        rhs_grad = builder.matmul(matrix_grad_like_output, lhs, true, true, rhs_grad_dtype, node.compute_dtype);
                    }

                    if (node.matmul_packed_row_binding == MatmulPackedRowBinding::RowsA &&
                        node.matmul_packed_row_capacity != 0) {
                        builder.setPackedRowMatmul(rhs_grad, MatmulPackedRowBinding::RowsAAndRowsB, node.matmul_packed_row_capacity);
                    }
                    rhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, rhs_grad);
                    const std::vector<uint64_t> rhs_grad_dims =
                        has_forward_dims ? rawBatchedMatmulOperandGradientDims(node_dims, rhs_dims) : rhs_dims;
                    addContributionToChild(node.rhs, rhs_grad, rhs_grad_dims, rhs_grad_dtype);
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
                                                              node.conv_groups,
                                                              lhs_dims,
                                                              lhs_grad_dtype,
                                                              node.compute_dtype);
                    } else {
                        lhs_grad = builder.conv2dBackwardData(filter,
                                                              grad_like_output,
                                                              node.conv_spatial_2d,
                                                              node.conv_groups,
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
                                                                node.conv_groups,
                                                                rhs_dims,
                                                                rhs_grad_dtype,
                                                                node.compute_dtype);
                    } else {
                        rhs_grad = builder.conv2dBackwardFilter(input,
                                                                grad_like_output,
                                                                node.conv_spatial_2d,
                                                                node.conv_groups,
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
                uint32_t matrix_grad_like_output = grad_like_output;
                std::optional<DataType> low_precision_operand_dtype;
                if (node_reaches_requested_inputs.at(node.lhs) || node_reaches_requested_inputs.at(node.rhs)) {
                    low_precision_operand_dtype = matmulLowPrecisionOperandDType(forward_expr, node);
                    if (low_precision_operand_dtype.has_value()) {
                        // cuBLASLt requires the two regular dense matrix operands to have the same
                        // FP16/BF16 dtype.  The upstream gradient can still be an untyped synthetic
                        // CustomLayer input while this graph is built and only resolve to FP32 during
                        // stamping.  Always establish the low-precision matrix-operand boundary here.
                        // The helper elides the conversion only when the producer is known to be
                        // materially in that dtype; this one result is shared by dL/dlhs and dL/drhs.
                        matrix_grad_like_output =
                            builder.materializeMatmulOperandCast(grad_like_output, low_precision_operand_dtype.value());
                    }
                }
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
                    // gradient reduction resolves to the reduction default FP32 output and M3 must add
                    // an unnecessary compiler-local cast for callers that request the FP16 public grad.
                    aux_grad_dtype = preferredGradValueDType(node);
                }

                if (node_reaches_requested_inputs.at(node.lhs)) {
                    uint32_t rhs = builder.cloneForward(node.rhs);
                    if (low_precision_operand_dtype.has_value()) {
                        rhs = builder.cast(rhs, low_precision_operand_dtype.value());
                    }
                    uint32_t lhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(matrix_grad_like_output, rhs, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        lhs_grad = builder.matmul(matrix_grad_like_output, rhs, false, false, lhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        lhs_grad = builder.matmul(rhs, matrix_grad_like_output, false, true, lhs_grad_dtype, node.compute_dtype);
                    } else {
                        lhs_grad = builder.matmul(rhs, matrix_grad_like_output, true, true, lhs_grad_dtype, node.compute_dtype);
                    }

                    lhs_grad = builder.buildScaledByGemmFactor(node.alpha_node, node.alpha_fp, lhs_grad);
                    addContributionToChild(node.lhs, lhs_grad, lhs_dims);
                }

                if (node_reaches_requested_inputs.at(node.rhs)) {
                    uint32_t lhs = builder.cloneForward(node.lhs);
                    if (low_precision_operand_dtype.has_value()) {
                        lhs = builder.cast(lhs, low_precision_operand_dtype.value());
                    }
                    uint32_t rhs_grad = UINT32_MAX;
                    if (!node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, matrix_grad_like_output, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (!node.transpose_lhs && node.transpose_rhs) {
                        rhs_grad = builder.matmul(matrix_grad_like_output, lhs, true, false, rhs_grad_dtype, node.compute_dtype);
                    } else if (node.transpose_lhs && !node.transpose_rhs) {
                        rhs_grad = builder.matmul(lhs, matrix_grad_like_output, false, false, rhs_grad_dtype, node.compute_dtype);
                    } else {
                        rhs_grad = builder.matmul(matrix_grad_like_output, lhs, true, true, rhs_grad_dtype, node.compute_dtype);
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
                        // Skv-vector cases produce incorrect dV/dBias. Keep the public forward surface broad, but
                        // lower production backward through an explicit dense score-bias broadcast, then let the
                        // normal broadcast-gradient rule reduce dense dBias back to the original bias shape.
                        //
                        // The old fill(0)+bias form also carried an output dtype. Keep dtype conversion orthogonal
                        // to shape expansion: CAST only when needed, then BROADCAST_TO.
                        const auto bias_backward_dtype =
                            node.compute_dtype.has_value() ? node.compute_dtype : preferredGradValueDType(node);
                        if (bias_backward_dtype.has_value()) {
                            bias = builder.cast(bias, bias_backward_dtype.value());
                        }
                        bias = builder.broadcastTo(bias, dense_score_bias_dims);
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

            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MEAN: {
                if (!node_reaches_requested_inputs.at(node.lhs)) {
                    break;
                }

                uint64_t batch_size = node.ragged_runtime_batch_size;
                uint64_t max_active_values = node.ragged_runtime_max_active_values;
                uint64_t elements_per_value = node.ragged_runtime_elements_per_value;
                std::vector<uint64_t> values_dims;
                std::vector<uint64_t> segment_dims;
                if (has_forward_dims) {
                    values_dims = forward_node_dims.at(node.lhs);
                    segment_dims = forward_node_dims.at(static_cast<uint32_t>(node_idx));
                    const std::vector<uint64_t>& offsets_dims = forward_node_dims.at(node.rhs);
                    if (values_dims.empty() || segment_dims.empty() || offsets_dims.size() != 1 || offsets_dims[0] == 0 ||
                        values_dims[0] == 0 || segment_dims[0] != offsets_dims[0] - 1) {
                        throw std::runtime_error(
                            "Segmented ragged reduction backward requires values [N,D...], output [B,D...], and offsets [B+1].");
                    }
                    batch_size = offsets_dims[0] - 1;
                    max_active_values = values_dims[0];
                    const uint64_t inferred_elements =
                        dynamicDimsNumel(values_dims, "segmented reduction backward values") / values_dims[0];
                    if (elements_per_value == 0) {
                        elements_per_value = inferred_elements;
                    } else if (elements_per_value != inferred_elements) {
                        throw std::runtime_error("Segmented ragged reduction backward elements-per-value metadata mismatch.");
                    }
                }

                if (batch_size == 0 || max_active_values == 0 || elements_per_value == 0) {
                    if (allow_shape_deferred_placeholders) {
                        builder.addContribution(node.lhs, grad);
                        break;
                    }
                    throw std::runtime_error(
                        "Segmented ragged reduction backward requires ragged extent metadata or forward input dimensions.");
                }

                const uint32_t offsets = builder.cloneForward(node.rhs);
                const auto grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                uint32_t segment_grad = grad;
                if (has_forward_dims) {
                    segment_grad = shapeGradLikeNodeOutput(
                        grad, static_cast<uint32_t>(node_idx), segment_dims);
                } else {
                    double constant_value = 0.0;
                    if (builder.tryGetConstantLikeValue(grad, constant_value)) {
                        const std::vector<uint64_t> flat_segment_dims =
                            elements_per_value == 1 ? std::vector<uint64_t>{batch_size}
                                                    : std::vector<uint64_t>{batch_size, elements_per_value};
                        segment_grad = builder.fill(constant_value, flat_segment_dims, grad_dtype);
                    }
                }
                if (grad_dtype.has_value()) {
                    segment_grad = builder.cast(segment_grad, grad_dtype.value());
                }
                const uint32_t broadcast = builder.segmentedBroadcast(segment_grad,
                                                                      offsets,
                                                                      batch_size,
                                                                      max_active_values,
                                                                      elements_per_value,
                                                                      node.op == ExprOp::SEGMENTED_REDUCE_MEAN,
                                                                      grad_dtype);
                const uint32_t ragged_grad = builder.raggedValuewiseExtent(
                    broadcast, offsets, batch_size, max_active_values, elements_per_value, grad_dtype);
                const std::vector<uint64_t> child_dims = has_forward_dims
                    ? values_dims
                    : (elements_per_value == 1 ? std::vector<uint64_t>{max_active_values}
                                               : std::vector<uint64_t>{max_active_values, elements_per_value});
                addContributionToChild(node.lhs, ragged_grad, child_dims, grad_dtype);
                break;
            }

            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX: {
                if (!node_reaches_requested_inputs.at(node.lhs)) {
                    break;
                }

                uint64_t batch_size = node.ragged_runtime_batch_size;
                uint64_t max_active_values = node.ragged_runtime_max_active_values;
                uint64_t elements_per_value = node.ragged_runtime_elements_per_value;
                std::vector<uint64_t> values_dims;
                std::vector<uint64_t> segment_dims;
                if (has_forward_dims) {
                    values_dims = forward_node_dims.at(node.lhs);
                    segment_dims = forward_node_dims.at(static_cast<uint32_t>(node_idx));
                    const std::vector<uint64_t>& offsets_dims = forward_node_dims.at(node.rhs);
                    if (values_dims.empty() || segment_dims.empty() || offsets_dims.size() != 1 || offsets_dims[0] == 0 ||
                        values_dims[0] == 0 || segment_dims[0] != offsets_dims[0] - 1) {
                        throw std::runtime_error(
                            "Segmented ragged min/max backward requires values [N,D...], output [B,D...], and offsets [B+1].");
                    }
                    batch_size = offsets_dims[0] - 1;
                    max_active_values = values_dims[0];
                    const uint64_t inferred_elements =
                        dynamicDimsNumel(values_dims, "segmented min/max backward values") / values_dims[0];
                    if (elements_per_value == 0) {
                        elements_per_value = inferred_elements;
                    } else if (elements_per_value != inferred_elements) {
                        throw std::runtime_error("Segmented ragged min/max backward elements-per-value metadata mismatch.");
                    }
                }

                if (batch_size == 0 || max_active_values == 0 || elements_per_value == 0) {
                    if (allow_shape_deferred_placeholders) {
                        builder.addContribution(node.lhs, grad);
                        break;
                    }
                    throw std::runtime_error(
                        "Segmented ragged min/max backward requires ragged extent metadata or forward input dimensions.");
                }

                const uint32_t offsets = builder.cloneForward(node.rhs);
                const auto grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                uint32_t segment_grad = grad;
                if (has_forward_dims) {
                    segment_grad = shapeGradLikeNodeOutput(
                        grad, static_cast<uint32_t>(node_idx), segment_dims);
                } else {
                    double constant_value = 0.0;
                    if (builder.tryGetConstantLikeValue(grad, constant_value)) {
                        const std::vector<uint64_t> flat_segment_dims =
                            elements_per_value == 1 ? std::vector<uint64_t>{batch_size}
                                                    : std::vector<uint64_t>{batch_size, elements_per_value};
                        segment_grad = builder.fill(constant_value, flat_segment_dims, grad_dtype);
                    }
                }
                if (grad_dtype.has_value()) {
                    segment_grad = builder.cast(segment_grad, grad_dtype.value());
                }

                const uint32_t routed = builder.segmentedReduceMinMaxBackward(
                    node.op == ExprOp::SEGMENTED_REDUCE_MIN ? ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD
                                                            : ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD,
                    builder.cloneForward(node.lhs),
                    segment_grad,
                    offsets,
                    batch_size,
                    max_active_values,
                    elements_per_value,
                    grad_dtype,
                    node.compute_dtype);
                const uint32_t ragged_grad = builder.raggedValuewiseExtent(
                    routed, offsets, batch_size, max_active_values, elements_per_value, grad_dtype);
                const std::vector<uint64_t> child_dims = has_forward_dims
                    ? values_dims
                    : (elements_per_value == 1 ? std::vector<uint64_t>{max_active_values}
                                               : std::vector<uint64_t>{max_active_values, elements_per_value});
                addContributionToChild(node.lhs, ragged_grad, child_dims, grad_dtype);
                break;
            }

            case ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD:
            case ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD:
                throw std::runtime_error(
                    "Thor expressions autodiff does not support second derivatives through segmented min/max backward yet.");

            case ExprOp::RAGGED_CONV1D_CAUSAL: {
                const bool need_dgrad = node_reaches_requested_inputs.at(node.lhs);
                const bool need_wgrad = node_reaches_requested_inputs.at(node.rhs);
                if (!need_dgrad && !need_wgrad) {
                    break;
                }
                if (node.ragged_runtime_batch_size == 0 || node.ragged_runtime_max_active_values == 0 ||
                    node.ragged_runtime_max_values_per_row == 0 || node.ragged_conv1d_input_channels == 0 ||
                    node.ragged_conv1d_output_channels == 0 || node.ragged_conv1d_groups == 0) {
                    throw std::runtime_error(
                        "Ragged causal Conv1D backward requires complete retained-representation metadata.");
                }
                const uint32_t grad_like_output =
                    shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                const uint32_t offsets = builder.cloneForward(node.aux);

                if (need_dgrad) {
                    const uint32_t filter = builder.cloneForward(node.rhs);
                    const auto dx_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                    uint32_t dx = builder.raggedConv1dCausalBackwardData(
                        filter, grad_like_output, offsets, node, dx_dtype, node.compute_dtype);
                    dx = builder.raggedValuewiseExtent(dx,
                                                       offsets,
                                                       node.ragged_runtime_batch_size,
                                                       node.ragged_runtime_max_active_values,
                                                       node.ragged_conv1d_input_channels,
                                                       dx_dtype);
                    const std::vector<uint64_t> lhs_dims = has_forward_dims
                        ? forward_node_dims.at(node.lhs)
                        : std::vector<uint64_t>{node.ragged_runtime_max_active_values, node.ragged_conv1d_input_channels};
                    addContributionToChild(node.lhs, dx, lhs_dims, dx_dtype);
                }

                if (need_wgrad) {
                    uint32_t input = builder.cloneForward(node.lhs);
                    input = builder.raggedValuewiseExtent(input,
                                                          offsets,
                                                          node.ragged_runtime_batch_size,
                                                          node.ragged_runtime_max_active_values,
                                                          node.ragged_conv1d_input_channels);
                    const auto dw_dtype = preferredGradValueDType(forward_expr.nodes.at(node.rhs));
                    const uint32_t dw = builder.raggedConv1dCausalBackwardFilter(
                        input, grad_like_output, offsets, node, dw_dtype, node.compute_dtype);
                    const std::vector<uint64_t> filter_dims = has_forward_dims
                        ? forward_node_dims.at(node.rhs)
                        : std::vector<uint64_t>{node.ragged_conv1d_output_channels,
                                                node.ragged_conv1d_input_channels / node.ragged_conv1d_groups,
                                                node.ragged_conv1d_kernel_width};
                    addContributionToChild(node.rhs, dw, filter_dims, dw_dtype);
                }
                break;
            }

            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA:
            case ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER:
                throw std::runtime_error(
                    "Thor expressions autodiff does not support second derivatives through ragged Conv1D backward stages yet.");

            case ExprOp::SEGMENTED_BROADCAST: {
                if (!node_reaches_requested_inputs.at(node.lhs)) {
                    break;
                }
                if (node.ragged_runtime_batch_size == 0 || node.ragged_runtime_max_active_values == 0 ||
                    node.ragged_runtime_elements_per_value == 0) {
                    throw std::runtime_error("Segmented broadcast backward requires complete ragged extent metadata.");
                }

                const uint32_t offsets = builder.cloneForward(node.rhs);
                const auto grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                const uint32_t grad_like_output =
                    has_forward_dims ? shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx),
                                                               forward_node_dims.at(static_cast<uint32_t>(node_idx)))
                                     : grad;
                const ExprOp reduction_op = node.segmented_broadcast_normalize_by_length
                    ? ExprOp::SEGMENTED_REDUCE_MEAN
                    : ExprOp::SEGMENTED_REDUCE_SUM;
                uint32_t per_segment_grad = builder.segmentedReduce(grad_like_output,
                                                                    offsets,
                                                                    reduction_op,
                                                                    node.ragged_runtime_batch_size,
                                                                    node.ragged_runtime_max_active_values,
                                                                    node.ragged_runtime_elements_per_value,
                                                                    grad_dtype);
                if (grad_dtype.has_value()) {
                    per_segment_grad = builder.cast(per_segment_grad, grad_dtype.value());
                }
                const std::vector<uint64_t> lhs_dims = has_forward_dims
                    ? forward_node_dims.at(node.lhs)
                    : (node.ragged_runtime_elements_per_value == 1
                           ? std::vector<uint64_t>{node.ragged_runtime_batch_size}
                           : std::vector<uint64_t>{node.ragged_runtime_batch_size, node.ragged_runtime_elements_per_value});
                addContributionToChild(node.lhs, per_segment_grad, lhs_dims, grad_dtype);
                break;
            }

            case ExprOp::SCAN:
            case ExprOp::SEGMENTED_SCAN: {
                if (!node_reaches_requested_inputs.at(node.lhs)) {
                    break;
                }
                const std::vector<uint64_t> lhs_dims = has_forward_dims ? forward_node_dims.at(node.lhs) : node_dims;
                const uint32_t grad_like_output = shapeGradLikeNodeOutput(grad, static_cast<uint32_t>(node_idx), node_dims);
                const auto lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                uint32_t segmented_offsets = UINT32_MAX;
                if (node.op == ExprOp::SEGMENTED_SCAN) {
                    segmented_offsets = builder.cloneForward(node.rhs);
                }

                uint32_t dx;
                if (node.scan_op == ScanOp::Sum) {
                    if (node.op == ExprOp::SEGMENTED_SCAN) {
                        dx = builder.segmentedScan(grad_like_output,
                                                   segmented_offsets,
                                                   ScanOp::Sum,
                                                   node.scan_mode,
                                                   !node.scan_reverse,
                                                   lhs_grad_dtype,
                                                   node.ragged_runtime_batch_size,
                                                   node.ragged_runtime_max_active_values,
                                                   node.ragged_runtime_elements_per_value);
                    } else {
                        dx = builder.scan(grad_like_output,
                                          ScanOp::Sum,
                                          node.scan_mode,
                                          node.scan_axis,
                                          !node.scan_reverse,
                                          lhs_grad_dtype);
                    }
                } else if (node.scan_op == ScanOp::Min || node.scan_op == ScanOp::Max) {
                    const ExprOp backward_op = node.op == ExprOp::SEGMENTED_SCAN
                        ? (node.scan_op == ScanOp::Min ? ExprOp::SEGMENTED_SCAN_MIN_BACKWARD : ExprOp::SEGMENTED_SCAN_MAX_BACKWARD)
                        : (node.scan_op == ScanOp::Min ? ExprOp::SCAN_MIN_BACKWARD : ExprOp::SCAN_MAX_BACKWARD);
                    const uint32_t grad_for_scatter = builder.cast(grad_like_output, DataType::FP32);
                    dx = builder.scanMinMaxBackward(backward_op,
                                                    builder.cloneForward(node.lhs),
                                                    grad_for_scatter,
                                                    segmented_offsets,
                                                    node.scan_mode,
                                                    node.scan_axis,
                                                    node.scan_reverse,
                                                    DataType::FP32,
                                                    node.op == ExprOp::SEGMENTED_SCAN ? node.ragged_runtime_batch_size : 0,
                                                    node.op == ExprOp::SEGMENTED_SCAN ? node.ragged_runtime_max_active_values : 0,
                                                    node.op == ExprOp::SEGMENTED_SCAN ? node.ragged_runtime_elements_per_value : 1);
                } else {
                    throw std::runtime_error("Thor expressions autodiff currently supports backward only for sum/min/max scan.");
                }

                if (node.op == ExprOp::SEGMENTED_SCAN && node.ragged_runtime_batch_size != 0) {
                    if (node.ragged_runtime_max_active_values == 0 || node.ragged_runtime_elements_per_value == 0) {
                        throw std::runtime_error("Segmented scan autodiff encountered incomplete ragged runtime-extent metadata.");
                    }
                    dx = builder.raggedValuewiseExtent(dx,
                                                       segmented_offsets,
                                                       node.ragged_runtime_batch_size,
                                                       node.ragged_runtime_max_active_values,
                                                       node.ragged_runtime_elements_per_value,
                                                       lhs_grad_dtype);
                }

                addContributionToChild(node.lhs, dx, lhs_dims);
                break;
            }

            case ExprOp::BROADCAST_TO:
                if (node_reaches_requested_inputs.at(node.lhs)) {
                    if (!has_forward_dims) {
                        if (allow_shape_deferred_placeholders) {
                            // FusedEquation::compileBackward builds a shape-deferred template first.
                            // Stamp-time specialization rebuilds it with concrete forward dimensions,
                            // at which point sumToShape inserts the required reduction.
                            addContributionToChild(node.lhs, grad, node_dims);
                            break;
                        }
                        throw std::runtime_error(
                            "BROADCAST_TO autodiff requires resolved forward shapes to reduce the gradient to the input shape.");
                    }
                    const std::vector<uint64_t>& lhs_dims = forward_node_dims.at(node.lhs);
                    const std::optional<DataType> lhs_grad_dtype = preferredGradValueDType(forward_expr.nodes.at(node.lhs));
                    const uint32_t lhs_grad = sumToShape(builder, grad, node_dims, lhs_dims, lhs_grad_dtype);
                    addContributionToChild(node.lhs, lhs_grad, lhs_dims, lhs_grad_dtype);
                }
                break;

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
        bool require_distinct_storage = false;
    };

    std::vector<PendingBackwardOutput> pending_outputs;
    pending_outputs.reserve(normalized_wrt.size());

    // Some expressions legitimately produce the same mathematical gradient value for
    // multiple requested inputs. A residual add is the simplest case: d(lhs + rhs)
    // / d lhs and d rhs are the same upstream tensor. Keep the mathematical graph
    // shared and communicate only the public output-ownership requirement. M5 makes
    // require_distinct_storage a stamped output-materialization concern.
    std::unordered_set<uint32_t> emittedRawGradientNodes;

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
        bool require_distinct_storage = false;
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
                // This is a genuine mathematical zero: the requested forward input
                // has no gradient contribution. Unlike the removed terminal dtype
                // coercion, the zero is the value of dOutput/dInput itself.
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

            const uint32_t raw_grad = total_grad.value();
            if (!emittedRawGradientNodes.insert(raw_grad).second) {
                // The named outputs are mathematically identical, but callers historically
                // receive independent gradient tensors. Preserve that ownership contract
                // without manufacturing WHERE(g, g) solely to obtain a second node id.
                require_distinct_storage = true;
            }
        }

        // Output storage dtype is a physical materialization requirement, not part of
        // the mathematical backward graph.  M3 lowers this contract compiler-locally,
        // so the named output points directly at the mathematical gradient value.
        const uint32_t terminal_grad = total_grad.value();

        pending_outputs.push_back(PendingBackwardOutput{
            .name = grad_output_name,
            .node_idx = terminal_grad,
            .target_output_dtype = grad_dtype,
            .require_distinct_storage = require_distinct_storage,
        });
    }

    PhysicalOutputs backward_outputs;
    backward_outputs.expr = std::make_shared<PhysicalExpression>(builder.takeExpression());
    backward_outputs.outputs.reserve(pending_outputs.size());
    for (const PendingBackwardOutput& output : pending_outputs) {
        NamedOutput named_output{
            .name = output.name,
            .node_idx = output.node_idx,
        };
        named_output.materialization.storage_dtype = output.target_output_dtype;
        named_output.materialization.require_distinct_storage = output.require_distinct_storage;
        backward_outputs.outputs.push_back(std::move(named_output));
    }

    return backward_outputs;
}


static std::optional<DataType> findConditionalInputGradDType(const PhysicalOutputs& outputs, const std::string& input_name) {
    if (!outputs.expr) {
        return std::nullopt;
    }

    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op != ExprOp::INPUT || node.input_slot >= outputs.expr->inputs.size()) {
            continue;
        }
        const NamedInput& input = outputs.expr->inputs[node.input_slot];
        if (input.name == input_name && input.kind == NamedInput::Kind::Tensor) {
            return preferredGradValueDType(node);
        }
    }

    if (!outputs.isConditional() || !outputs.conditional) {
        return std::nullopt;
    }

    for (const PhysicalOutputs* child : {&outputs.conditional->predicate,
                                         &outputs.conditional->then_branch,
                                         &outputs.conditional->else_branch}) {
        const auto dtype = findConditionalInputGradDType(*child, input_name);
        if (dtype.has_value()) {
            return dtype;
        }
    }
    return std::nullopt;
}

static std::optional<std::unordered_map<std::string, std::vector<uint64_t>>> filterForwardInputDimsForOutputs(
    const PhysicalOutputs& outputs,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims) {
    if (!forward_input_dims.has_value()) {
        return std::nullopt;
    }
    if (!outputs.expr) {
        throw std::runtime_error("Conditional autodiff shape filtering requires non-null expression metadata.");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> filtered;
    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.kind != NamedInput::Kind::Tensor) {
            continue;
        }
        auto it = forward_input_dims->find(input.name);
        if (it != forward_input_dims->end()) {
            filtered.emplace(it->first, it->second);
        }
    }
    return filtered;
}

static uint32_t appendBackwardTensorInput(PhysicalExpression& expr,
                                          const std::string& name,
                                          std::optional<DataType> dtype = std::nullopt) {
    uint32_t slot = UINT32_MAX;
    for (const NamedInput& input : expr.inputs) {
        if (input.name == name) {
            if (input.kind != NamedInput::Kind::Tensor) {
                throw std::runtime_error("Conditional backward tensor input name collides with a non-tensor input: " + name);
            }
            slot = input.slot;
            break;
        }
    }
    if (slot == UINT32_MAX) {
        slot = static_cast<uint32_t>(expr.inputs.size());
        expr.inputs.push_back(NamedInput{name, slot, NamedInput::Kind::Tensor});
    }

    ExprNode input_node{};
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = slot;
    if (dtype.has_value()) {
        input_node.output_dtype = dtype.value();
        input_node.backward_output_dtype = dtype.value();
    }
    const uint32_t node_idx = static_cast<uint32_t>(expr.nodes.size());
    expr.nodes.push_back(std::move(input_node));
    return node_idx;
}

static uint32_t appendZeroGradientForMissingConditionalInput(PhysicalExpression& expr,
                                                              const std::string& wrt_name,
                                                              std::optional<DataType> grad_dtype,
                                                              bool accumulate_grad_outputs) {
    const std::string grad_name = wrt_name + "_grad";
    if (accumulate_grad_outputs) {
        return appendBackwardTensorInput(expr, grad_name, grad_dtype);
    }

    const uint32_t input_node = appendBackwardTensorInput(expr, wrt_name, grad_dtype);
    ExprNode zero_scalar{};
    zero_scalar.op = ExprOp::SCALAR_FP;
    zero_scalar.scalar_fp = 0.0;
    const uint32_t zero_scalar_idx = static_cast<uint32_t>(expr.nodes.size());
    expr.nodes.push_back(std::move(zero_scalar));

    ExprNode zero_grad{};
    zero_grad.op = ExprOp::MUL;
    zero_grad.lhs = input_node;
    zero_grad.rhs = zero_scalar_idx;
    if (grad_dtype.has_value()) {
        zero_grad.output_dtype = grad_dtype.value();
        zero_grad.backward_output_dtype = grad_dtype.value();
    }
    const uint32_t zero_grad_idx = static_cast<uint32_t>(expr.nodes.size());
    expr.nodes.push_back(std::move(zero_grad));
    return zero_grad_idx;
}

static void appendConditionalRootInput(PhysicalExpression& root, const NamedInput& child_input) {
    for (const NamedInput& existing : root.inputs) {
        if (existing.name != child_input.name) {
            continue;
        }
        if (existing.kind != child_input.kind) {
            throw std::runtime_error("Conditional backward input kind mismatch for input: " + child_input.name);
        }
        return;
    }
    root.inputs.push_back(NamedInput{child_input.name, static_cast<uint32_t>(root.inputs.size()), child_input.kind});
}

static PhysicalOutputs assembleConditionalBackwardOutputs(const PhysicalOutputs& predicate,
                                                           PhysicalOutputs then_branch,
                                                           PhysicalOutputs else_branch,
                                                           const std::vector<std::string>& desired_wrt) {
    auto output_names = [](const PhysicalOutputs& outputs) {
        std::vector<std::string> names;
        names.reserve(outputs.outputs.size());
        for (const NamedOutput& output : outputs.outputs) {
            names.push_back(output.name);
        }
        return names;
    };

    const std::vector<std::string> then_names = output_names(then_branch);
    const std::vector<std::string> else_names = output_names(else_branch);
    if (then_names != else_names) {
        throw std::runtime_error("Conditional backward branches produced different gradient output contracts.");
    }

    std::vector<std::string> expected_names;
    expected_names.reserve(desired_wrt.size());
    for (const std::string& wrt_name : desired_wrt) {
        expected_names.push_back(wrt_name + "_grad");
    }
    if (then_names != expected_names) {
        throw std::runtime_error("Conditional backward branch gradient outputs do not match the requested wrt order.");
    }

    PhysicalOutputs result;
    result.expr = std::make_shared<PhysicalExpression>();
    const PhysicalOutputs* children[] = {&predicate, &then_branch, &else_branch};
    for (const PhysicalOutputs* child : children) {
        if (!child->expr) {
            throw std::runtime_error("Conditional backward child is missing expression metadata.");
        }
        for (const NamedInput& input : child->expr->inputs) {
            appendConditionalRootInput(*result.expr, input);
        }
    }

    result.outputs.reserve(expected_names.size());
    for (size_t i = 0; i < expected_names.size(); ++i) {
        OutputMaterializationContract& then_materialization = then_branch.outputs[i].materialization;
        OutputMaterializationContract& else_materialization = else_branch.outputs[i].materialization;
        if (then_materialization.storage_dtype != else_materialization.storage_dtype) {
            throw std::runtime_error("Conditional backward branches produced different gradient output storage dtype contracts.");
        }

        // A branch-local duplicate gradient is still a root-level ownership requirement:
        // whichever branch executes, this named output must remain distinct from its
        // sibling output. Promote the ownership bit across both branches so the
        // conditional contract is stable without weakening M1's general requirement
        // that conditional output contracts agree.
        const bool require_distinct_storage = then_materialization.require_distinct_storage ||
                                              else_materialization.require_distinct_storage;
        then_materialization.require_distinct_storage = require_distinct_storage;
        else_materialization.require_distinct_storage = require_distinct_storage;

        result.outputs.push_back(NamedOutput{
            .name = expected_names[i],
            .node_idx = static_cast<uint32_t>(i),
            .materialization = then_materialization,
        });
    }

    result.conditional = std::make_shared<PhysicalConditionalOutputs>();
    result.conditional->predicate = predicate;
    result.conditional->then_branch = std::move(then_branch);
    result.conditional->else_branch = std::move(else_branch);
    return result;
}

static PhysicalOutputs buildConditionalTreeBackwardOutputsImpl(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& desired_wrt,
    const std::optional<std::unordered_map<std::string, std::string>>& upstream_input_names_by_output,
    const std::optional<std::unordered_map<std::string, DataType>>& upstream_input_dtypes_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
    bool accumulate_grad_outputs,
    bool allow_shape_deferred_placeholders,
    const std::unordered_map<std::string, std::optional<DataType>>& grad_dtypes) {
    if (!forward_outputs.expr) {
        throw std::runtime_error("Conditional autodiff requires non-null expression metadata.");
    }

    if (forward_outputs.isConditional()) {
        if (!forward_outputs.conditional) {
            throw std::runtime_error("Conditional autodiff is missing the conditional payload.");
        }
        const PhysicalConditionalOutputs& conditional = *forward_outputs.conditional;
        PhysicalOutputs then_backward = buildConditionalTreeBackwardOutputsImpl(
            conditional.then_branch,
            desired_wrt,
            upstream_input_names_by_output,
            upstream_input_dtypes_by_output,
            filterForwardInputDimsForOutputs(conditional.then_branch, forward_input_dims),
            accumulate_grad_outputs,
            allow_shape_deferred_placeholders,
            grad_dtypes);
        PhysicalOutputs else_backward = buildConditionalTreeBackwardOutputsImpl(
            conditional.else_branch,
            desired_wrt,
            upstream_input_names_by_output,
            upstream_input_dtypes_by_output,
            filterForwardInputDimsForOutputs(conditional.else_branch, forward_input_dims),
            accumulate_grad_outputs,
            allow_shape_deferred_placeholders,
            grad_dtypes);
        return assembleConditionalBackwardOutputs(conditional.predicate,
                                                  std::move(then_backward),
                                                  std::move(else_backward),
                                                  desired_wrt);
    }

    std::unordered_set<std::string> tensor_input_names;
    for (const NamedInput& input : forward_outputs.expr->inputs) {
        if (input.kind == NamedInput::Kind::Tensor) {
            tensor_input_names.insert(input.name);
        }
    }

    std::vector<std::string> active_wrt;
    active_wrt.reserve(desired_wrt.size());
    for (const std::string& wrt_name : desired_wrt) {
        if (tensor_input_names.contains(wrt_name)) {
            active_wrt.push_back(wrt_name);
        }
    }

    PhysicalOutputs backward;
    if (!active_wrt.empty()) {
        backward = buildFlatBackwardOutputsImpl(forward_outputs,
                                               active_wrt,
                                               upstream_input_names_by_output,
                                               upstream_input_dtypes_by_output,
                                               std::nullopt,
                                               forward_input_dims,
                                               accumulate_grad_outputs,
                                               allow_shape_deferred_placeholders);
    } else {
        backward.expr = std::make_shared<PhysicalExpression>();
    }

    std::unordered_map<std::string, NamedOutput> existing_outputs;
    for (const NamedOutput& output : backward.outputs) {
        existing_outputs.emplace(output.name, output);
    }

    std::vector<NamedOutput> ordered_outputs;
    ordered_outputs.reserve(desired_wrt.size());
    for (const std::string& wrt_name : desired_wrt) {
        const std::string grad_name = wrt_name + "_grad";
        auto existing = existing_outputs.find(grad_name);
        if (existing != existing_outputs.end()) {
            ordered_outputs.push_back(existing->second);
            continue;
        }
        auto dtype_it = grad_dtypes.find(wrt_name);
        const std::optional<DataType> grad_dtype = dtype_it == grad_dtypes.end() ? std::nullopt : dtype_it->second;
        const uint32_t zero_node = appendZeroGradientForMissingConditionalInput(
            *backward.expr, wrt_name, grad_dtype, accumulate_grad_outputs);
        NamedOutput zero_output{grad_name, zero_node};
        zero_output.materialization.storage_dtype = grad_dtype;
        ordered_outputs.push_back(std::move(zero_output));
    }
    backward.outputs = std::move(ordered_outputs);
    if (!backward.outputs.empty()) {
        backward.expr->output_node = backward.outputs.front().node_idx;
    }
    return backward;
}

static PhysicalOutputs buildBackwardOutputsImpl(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::optional<std::unordered_map<std::string, std::string>>& upstream_input_names_by_output,
    const std::optional<std::unordered_map<std::string, DataType>>& upstream_input_dtypes_by_output,
    const std::optional<std::unordered_map<std::string, uint32_t>>& upstream_node_indices_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
    bool accumulate_grad_outputs,
    bool allow_shape_deferred_placeholders = false) {
    if (!forward_outputs.expr) {
        throw std::runtime_error("buildBackwardOutputs requires non-null forward_outputs.expr.");
    }
    if (!forward_outputs.isConditional()) {
        return buildFlatBackwardOutputsImpl(forward_outputs,
                                            wrt_names,
                                            upstream_input_names_by_output,
                                            upstream_input_dtypes_by_output,
                                            upstream_node_indices_by_output,
                                            forward_input_dims,
                                            accumulate_grad_outputs,
                                            allow_shape_deferred_placeholders);
    }
    if (upstream_node_indices_by_output.has_value() && !upstream_node_indices_by_output->empty()) {
        throw std::runtime_error(
            "Graph-level conditional autodiff does not support upstream seeds by physical node index; use named upstream inputs.");
    }

    const std::vector<std::string> normalized_wrt = normalizeWrtNames(*forward_outputs.expr, wrt_names);
    std::unordered_map<std::string, std::optional<DataType>> grad_dtypes;
    for (const std::string& wrt_name : normalized_wrt) {
        grad_dtypes.emplace(wrt_name, findConditionalInputGradDType(forward_outputs, wrt_name));
    }

    return buildConditionalTreeBackwardOutputsImpl(forward_outputs,
                                                   normalized_wrt,
                                                   upstream_input_names_by_output,
                                                   upstream_input_dtypes_by_output,
                                                   forward_input_dims,
                                                   accumulate_grad_outputs,
                                                   allow_shape_deferred_placeholders,
                                                   grad_dtypes);
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::optional<std::string>& upstream_input_name,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                     bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_name),
                                    std::nullopt,
                                    std::nullopt,
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
                                    std::nullopt,
                                    std::nullopt,
                                    forward_input_dims,
                                    accumulate_grad_outputs);
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                     const std::unordered_map<std::string, DataType>& upstream_input_dtypes_by_output,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                     bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_names_by_output),
                                    upstream_input_dtypes_by_output,
                                    std::nullopt,
                                    forward_input_dims,
                                    accumulate_grad_outputs);
}

PhysicalOutputs buildBackwardOutputs(const PhysicalOutputs& forward_outputs,
                                     const std::vector<std::string>& wrt_names,
                                     const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                     const std::unordered_map<std::string, uint32_t>& upstream_node_indices_by_output,
                                     const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims,
                                     bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_names_by_output),
                                    std::nullopt,
                                    normalizeUpstreamNodeIndicesByOutput(forward_outputs, upstream_node_indices_by_output),
                                    forward_input_dims,
                                    accumulate_grad_outputs);
}

PhysicalOutputs buildDeferredShapeBackwardOutputsTemplate(const PhysicalOutputs& forward_outputs,
                                                         const std::vector<std::string>& wrt_names,
                                                         const std::optional<std::string>& upstream_input_name,
                                                         bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_name),
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    accumulate_grad_outputs,
                                    true);
}

PhysicalOutputs buildDeferredShapeBackwardOutputsTemplate(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    bool accumulate_grad_outputs) {
    return buildBackwardOutputsImpl(forward_outputs,
                                    wrt_names,
                                    normalizeUpstreamInputNamesByOutput(forward_outputs, upstream_input_names_by_output),
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    accumulate_grad_outputs,
                                    true);
}

}  // namespace ThorImplementation
