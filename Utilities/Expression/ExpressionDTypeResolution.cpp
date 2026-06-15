#include <optional>
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <stdexcept>

namespace ThorImplementation {


bool isSupportedFusionFloatingType(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

bool isFp8Type(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }

static bool isPassthroughInputDType(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
            return true;
        default:
            return isSupportedFusionFloatingType(dtype);
    }
}

static bool isAttentionBackwardOp(ExprOp op) {
    return op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K || op == ExprOp::ATTENTION_BACKWARD_V ||
           op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

static bool isReductionComputeOp(ExprOp op) {
    return isCudnnReduceOp(op) || op == ExprOp::RMSNORM || op == ExprOp::ATTENTION || isAttentionBackwardOp(op) || op == ExprOp::ROPE;
}
static bool isConvolutionOp(ExprOp op) {
    return op == ExprOp::CONV2D || op == ExprOp::CONV3D || op == ExprOp::CONV2D_BACKWARD_DATA ||
           op == ExprOp::CONV2D_BACKWARD_FILTER || op == ExprOp::CONV3D_BACKWARD_DATA || op == ExprOp::CONV3D_BACKWARD_FILTER;
}
static bool isComparisonOp(ExprOp op) {
    return op == ExprOp::EQUAL || op == ExprOp::NOT_EQUAL || op == ExprOp::LESS || op == ExprOp::LESS_EQUAL ||
           op == ExprOp::GREATER || op == ExprOp::GREATER_EQUAL;
}

static bool isLogicalOp(ExprOp op) { return op == ExprOp::LOGICAL_AND || op == ExprOp::LOGICAL_OR || op == ExprOp::LOGICAL_NOT; }

static bool isWhereOp(ExprOp op) { return op == ExprOp::WHERE; }

static bool isScanOp(ExprOp op) { return op == ExprOp::SCAN || op == ExprOp::SEGMENTED_SCAN; }

static bool isCastOp(ExprOp op) { return op == ExprOp::CAST; }

static bool isPassthroughViewOp(ExprOp op) {
    return op == ExprOp::STRIDED_VIEW || op == ExprOp::RESHAPE || op == ExprOp::TRANSPOSE || op == ExprOp::UNSQUEEZE ||
           op == ExprOp::SQUEEZE;
}

static bool isBooleanOutputOp(ExprOp op) { return isComparisonOp(op) || isLogicalOp(op); }

static bool isCudnnSingleInputStageOp(ExprOp op) { return isCudnnReduceOp(op) || isCudnnSoftmaxOp(op); }

static bool isSupportedCudnnReductionOutputDType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

DataType toSupportedComputeDType(ExprOp op, DataType requested_compute_dtype) {
    if (isComparisonOp(op) || isLogicalOp(op) || isWhereOp(op)) {
        if (requested_compute_dtype == DataType::BOOLEAN || requested_compute_dtype == DataType::UINT8) {
            return DataType::BOOLEAN;
        }
        if (isSupportedFusionFloatingType(requested_compute_dtype)) {
            return toSupportedComputeDType(ExprOp::ADD, requested_compute_dtype);
        }
        throw std::runtime_error("Unsupported boolean/comparison/where dtype in toSupportedComputeDType.");
    }

    if (isCastOp(op) || isPassthroughViewOp(op)) {
        if (isPassthroughInputDType(requested_compute_dtype)) {
            return requested_compute_dtype;
        }
        throw std::runtime_error("Unsupported passthrough dtype in toSupportedComputeDType.");
    }

    if (isScanOp(op)) {
        switch (requested_compute_dtype) {
            case DataType::UINT32:
            case DataType::UINT64:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
            case DataType::FP64:
                return requested_compute_dtype;
            default:
                throw std::runtime_error("Unsupported scan dtype in toSupportedComputeDType.");
        }
    }

    if (!isSupportedFusionFloatingType(requested_compute_dtype)) {
        throw std::runtime_error("Unsupported dtype in toSupportedComputeDType.");
    }

    if (isReductionComputeOp(op)) {
        switch (requested_compute_dtype) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
                return DataType::FP32;
            default:
                throw std::runtime_error("Unhandled reduction dtype in toSupportedComputeDType.");
        }
    }

    if (isConvolutionOp(op)) {
        switch (requested_compute_dtype) {
            case DataType::FP16:
                return DataType::FP16;
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::BF16:
            case DataType::FP32:
                return DataType::FP32;
            default:
                throw std::runtime_error("Unhandled convolution dtype in toSupportedComputeDType.");
        }
    }

    switch (requested_compute_dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
            return DataType::FP16;
        case DataType::BF16:
            return DataType::BF16;
        case DataType::FP32:
            return DataType::FP32;
        default:
            throw std::runtime_error("Unhandled pointwise dtype in toSupportedComputeDType.");
    }
}

DataType defaultComputeDType(DataType value_dtype) {
    switch (value_dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return DataType::FP16;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return value_dtype;
        default:
            throw std::runtime_error("Unsupported dtype in defaultComputeDType.");
    }
}

DataType defaultComputeDType(DataType input_dtype, DataType output_dtype) {
    if (!isSupportedFusionFloatingType(input_dtype) || !isSupportedFusionFloatingType(output_dtype)) {
        throw std::runtime_error("Unsupported dtype in defaultComputeDType(input_dtype, output_dtype).");
    }

    return promoteTensorValueDTypes(input_dtype, output_dtype);
}

DataType toSupportedInputDType(ExprOp op, DataType dtype) {
    if (!isSupportedFusionFloatingType(dtype)) {
        throw std::runtime_error("Unsupported dtype in toSupportedInputDType.");
    }

    if (isCudnnSingleInputStageOp(op)) {
        switch (dtype) {
            case DataType::FP16:
            case DataType::FP32:
                return dtype;
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::BF16:
                return DataType::FP16;
            default:
                throw std::runtime_error("Unhandled cuDNN single-input stage dtype conversion, from: " +
                                         TensorDescriptor::getElementTypeName(dtype));
        }
    } else {
        return dtype;
    }
}

DataType promoteTensorValueDTypes(DataType a, DataType b) {
    if (a == b) {
        return a;
    }

    if (!isSupportedFusionFloatingType(a) || !isSupportedFusionFloatingType(b)) {
        throw std::runtime_error("Unsupported dtype in promoteTensorValueDTypes.");
    }

    if ((a == DataType::FP16 && b == DataType::BF16) || (a == DataType::BF16 && b == DataType::FP16)) {
        return DataType::FP32;
    }

    if (a == DataType::FP32 || b == DataType::FP32) {
        return DataType::FP32;
    }

    if (a == DataType::BF16 || b == DataType::BF16) {
        return DataType::BF16;
    }

    if (a == DataType::FP16 || b == DataType::FP16) {
        return DataType::FP16;
    }

    if (isFp8Type(a) && isFp8Type(b)) {
        return DataType::FP16;
    }

    throw std::runtime_error("Unhandled dtype pair in promoteTensorValueDTypes.");
}

DataType promoteTensorValueDTypes(const std::vector<DataType>& dtypes) {
    if (dtypes.empty()) {
        throw std::runtime_error("promoteTensorValueDTypes requires at least one dtype.");
    }

    DataType out = dtypes.front();
    for (size_t i = 1; i < dtypes.size(); ++i) {
        out = promoteTensorValueDTypes(out, dtypes[i]);
    }
    return out;
}

static DataType promoteWhereBranchDTypes(const std::vector<DataType>& dtypes) {
    if (dtypes.empty()) {
        return DataType::FP32;
    }

    bool all_same = true;
    for (DataType dtype : dtypes) {
        if (dtype != dtypes.front()) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        return dtypes.front();
    }

    bool all_floating = true;
    for (DataType dtype : dtypes) {
        if (!isSupportedFusionFloatingType(dtype)) {
            all_floating = false;
            break;
        }
    }
    if (all_floating) {
        return promoteTensorValueDTypes(dtypes);
    }

    throw std::runtime_error("Where branch dtypes must either match exactly or be supported floating dtypes.");
}

static DataType promoteRequiredComputeDType(ExprOp op, DataType current_compute_dtype, DataType propagated_dtype) {
    if (isCastOp(op)) {
        return toSupportedComputeDType(op, current_compute_dtype);
    }

    if (isComparisonOp(op) || isLogicalOp(op) || isWhereOp(op)) {
        if (isSupportedFusionFloatingType(current_compute_dtype) && isSupportedFusionFloatingType(propagated_dtype)) {
            return toSupportedComputeDType(op, promoteTensorValueDTypes(current_compute_dtype, propagated_dtype));
        }
        if (isSupportedFusionFloatingType(propagated_dtype)) {
            return toSupportedComputeDType(op, propagated_dtype);
        }
        return toSupportedComputeDType(op, current_compute_dtype);
    }

    return toSupportedComputeDType(op, promoteTensorValueDTypes(current_compute_dtype, propagated_dtype));
}

static DataType resolveNodeLogicalInputDType(const ExprNode& node,
                                             const std::vector<ExprNode>& nodes,
                                             const std::vector<DataType>& resolved_output_dtypes,
                                             const std::vector<DataType>& root_input_dtypes) {
    if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        if (node.input_slot >= root_input_dtypes.size()) {
            throw std::runtime_error("Input slot out of range in resolveNodeLogicalInputDType.");
        }
        return root_input_dtypes[node.input_slot];
    }

    if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::FILL) {
        return node.output_dtype.has_value() ? node.output_dtype.value() : DataType::FP32;
    }

    if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        if (!node.output_dtype.has_value()) {
            throw std::runtime_error("CudaKernelExpression output node is missing output dtype.");
        }
        return node.output_dtype.value();
    }

    if (node.op == ExprOp::EMBEDDING_LOOKUP) {
        if (node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("EmbeddingLookup weights node has index out of range in resolveNodeLogicalInputDType.");
        }
        return resolved_output_dtypes[node.rhs];
    }

    if (node.op == ExprOp::CAST) {
        if (node.lhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("Cast node has parent index out of range in resolveNodeLogicalInputDType.");
        }
        return resolved_output_dtypes[node.lhs];
    }

    if (node.op == ExprOp::TAKE_ALONG_AXIS) {
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("take_along_axis node has parent index out of range in resolveNodeLogicalInputDType.");
        }
        const DataType indices_dtype = resolved_output_dtypes[node.rhs];
        if (indices_dtype != DataType::UINT32 && indices_dtype != DataType::UINT64) {
            throw std::runtime_error("take_along_axis indices must have UINT32 or UINT64 dtype, received: " +
                                     TensorDescriptor::getElementTypeName(indices_dtype));
        }
        return resolved_output_dtypes[node.lhs];
    }

    if (node.op == ExprOp::SEGMENTED_SCAN) {
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("segmented_scan node has parent index out of range in resolveNodeLogicalInputDType.");
        }
        const DataType offsets_dtype = resolved_output_dtypes[node.rhs];
        if (offsets_dtype != DataType::UINT32 && offsets_dtype != DataType::UINT64) {
            throw std::runtime_error("segmented_scan offsets must have UINT32 or UINT64 dtype, received: " +
                                     TensorDescriptor::getElementTypeName(offsets_dtype));
        }
        return resolved_output_dtypes[node.lhs];
    }

    if (node.op == ExprOp::SCAN_MIN_BACKWARD || node.op == ExprOp::SCAN_MAX_BACKWARD ||
        node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
        const bool segmented = node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD;
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("scan min/max backward node has parent index out of range in resolveNodeLogicalInputDType.");
        }
        if (segmented) {
            if (node.aux >= resolved_output_dtypes.size()) {
                throw std::runtime_error("segmented scan min/max backward offsets node has index out of range in resolveNodeLogicalInputDType.");
            }
            const DataType offsets_dtype = resolved_output_dtypes[node.aux];
            if (offsets_dtype != DataType::UINT32 && offsets_dtype != DataType::UINT64) {
                throw std::runtime_error("segmented scan min/max backward offsets must have UINT32 or UINT64 dtype, received: " +
                                         TensorDescriptor::getElementTypeName(offsets_dtype));
            }
        }

        // Offsets are structural metadata for segmented scan backward; they must
        // not participate in numeric dtype promotion with the input values and
        // upstream gradients.
        return promoteTensorValueDTypes(resolved_output_dtypes[node.lhs], resolved_output_dtypes[node.rhs]);
    }

    std::vector<DataType> tensor_parent_dtypes;
    tensor_parent_dtypes.reserve(4);

    auto add_tensor_parent = [&](uint32_t parent_idx) {
        if (parent_idx >= nodes.size()) {
            throw std::runtime_error("Parent node index out of range in resolveNodeLogicalInputDType.");
        }
        if (nodes[parent_idx].op == ExprOp::SCALAR_FP || nodes[parent_idx].op == ExprOp::RUNTIME_SCALAR ||
            nodes[parent_idx].op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            return;
        }
        tensor_parent_dtypes.push_back(resolved_output_dtypes[parent_idx]);
    };

    if (!Expression::isLeafOp(node.op)) {
        add_tensor_parent(node.lhs);
    }
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        add_tensor_parent(node.rhs);
    }
    if (Expression::isTernaryOp(node.op)) {
        add_tensor_parent(node.aux);
    }
    if (node.op == ExprOp::ATTENTION && node.attention_use_bias) {
        add_tensor_parent(node.alpha_node);
    }
    if (isAttentionBackwardOp(node.op)) {
        add_tensor_parent(node.alpha_node);
        if (node.attention_use_bias) {
            add_tensor_parent(node.beta_node);
        }
    }

    if (isWhereOp(node.op)) {
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size() || node.aux >= resolved_output_dtypes.size()) {
            throw std::runtime_error("Where node has parent index out of range in resolveNodeLogicalInputDType.");
        }
        const DataType condition_dtype = resolved_output_dtypes[node.lhs];
        if (condition_dtype != DataType::BOOLEAN) {
            throw std::runtime_error("Where condition must have BOOLEAN dtype, received: " +
                                     TensorDescriptor::getElementTypeName(condition_dtype));
        }

        std::vector<DataType> branch_dtypes;
        branch_dtypes.reserve(2);
        const ExprNode& true_parent = nodes[node.rhs];
        const ExprNode& false_parent = nodes[node.aux];
        if (true_parent.op != ExprOp::SCALAR_FP && true_parent.op != ExprOp::RUNTIME_SCALAR &&
            true_parent.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            branch_dtypes.push_back(resolved_output_dtypes[node.rhs]);
        }
        if (false_parent.op != ExprOp::SCALAR_FP && false_parent.op != ExprOp::RUNTIME_SCALAR &&
            false_parent.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            branch_dtypes.push_back(resolved_output_dtypes[node.aux]);
        }
        return promoteWhereBranchDTypes(branch_dtypes);
    }

    if (tensor_parent_dtypes.empty()) {
        return DataType::FP32;
    }
    if (isComparisonOp(node.op) || isLogicalOp(node.op)) {
        bool saw_floating = false;
        bool saw_integral = false;
        std::vector<DataType> floating_parent_dtypes;
        for (DataType dtype : tensor_parent_dtypes) {
            if (isSupportedFusionFloatingType(dtype)) {
                saw_floating = true;
                floating_parent_dtypes.push_back(dtype);
            } else if (dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::INT8 || dtype == DataType::UINT16 ||
                       dtype == DataType::INT16 || dtype == DataType::UINT32 || dtype == DataType::INT32 || dtype == DataType::UINT64 ||
                       dtype == DataType::INT64) {
                saw_integral = true;
            } else {
                const char* kind = isComparisonOp(node.op) ? "comparison" : "logical";
                throw std::runtime_error(std::string("Unsupported ") + kind + " expression input dtype: " +
                                         TensorDescriptor::getElementTypeName(dtype));
            }
        }
        if (saw_floating) {
            return promoteTensorValueDTypes(floating_parent_dtypes);
        }
        if (isLogicalOp(node.op) && !saw_integral) {
            return DataType::BOOLEAN;
        }
        if (isLogicalOp(node.op)) {
            return DataType::BOOLEAN;
        }
        return DataType::FP32;
    }
    return promoteTensorValueDTypes(tensor_parent_dtypes);
}

static DataType resolveNodeOutputDType(const ExprNode& node,
                                       const std::vector<ExprNode>& nodes,
                                       const std::vector<DataType>& resolved_output_dtypes,
                                       const std::vector<DataType>& root_input_dtypes) {
    if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        if (node.input_slot >= root_input_dtypes.size()) {
            throw std::runtime_error("Input slot out of range in resolveNodeOutputDType.");
        }

        const DataType default_output = root_input_dtypes[node.input_slot];
        return node.output_dtype.has_value() ? node.output_dtype.value() : default_output;
    }

    if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::FILL) {
        return node.output_dtype.has_value() ? node.output_dtype.value() : DataType::FP32;
    }

    if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        if (!node.output_dtype.has_value()) {
            throw std::runtime_error("CudaKernelExpression output node is missing output dtype.");
        }
        return node.output_dtype.value();
    }

    if (node.op == ExprOp::EMBEDDING_LOOKUP) {
        if (node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("EmbeddingLookup weights node has index out of range in resolveNodeOutputDType.");
        }
        const DataType weights_dtype = resolved_output_dtypes[node.rhs];
        return node.output_dtype.has_value() ? node.output_dtype.value() : weights_dtype;
    }

    if (node.op == ExprOp::CAST) {
        if (!node.output_dtype.has_value()) {
            throw std::runtime_error("Cast node is missing output dtype in resolveNodeOutputDType.");
        }
        return node.output_dtype.value();
    }

    if (node.op == ExprOp::TAKE_ALONG_AXIS) {
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("take_along_axis node has parent index out of range in resolveNodeOutputDType.");
        }
        const DataType indices_dtype = resolved_output_dtypes[node.rhs];
        if (indices_dtype != DataType::UINT32 && indices_dtype != DataType::UINT64) {
            throw std::runtime_error("take_along_axis indices must have UINT32 or UINT64 dtype, received: " +
                                     TensorDescriptor::getElementTypeName(indices_dtype));
        }
        const DataType input_dtype = resolved_output_dtypes[node.lhs];
        return node.output_dtype.has_value() ? node.output_dtype.value() : input_dtype;
    }

    if (node.op == ExprOp::RMSNORM) {
        if (node.lhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("RMSNorm node has input index out of range in resolveNodeOutputDType.");
        }
        const DataType input_dtype = resolved_output_dtypes[node.lhs];
        return node.output_dtype.has_value() ? node.output_dtype.value() : input_dtype;
    }

    if (node.op == ExprOp::SCAN || node.op == ExprOp::SEGMENTED_SCAN) {
        if (node.lhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("Scan node has input index out of range in resolveNodeOutputDType.");
        }
        if (node.op == ExprOp::SEGMENTED_SCAN) {
            if (node.rhs >= resolved_output_dtypes.size()) {
                throw std::runtime_error("segmented_scan offsets node has index out of range in resolveNodeOutputDType.");
            }
            const DataType offsets_dtype = resolved_output_dtypes[node.rhs];
            if (offsets_dtype != DataType::UINT32 && offsets_dtype != DataType::UINT64) {
                throw std::runtime_error("segmented_scan offsets must have UINT32 or UINT64 dtype, received: " +
                                         TensorDescriptor::getElementTypeName(offsets_dtype));
            }
        }
        const DataType input_dtype = resolved_output_dtypes[node.lhs];
        const bool arg_scan = node.scan_op == ScanOp::ArgMin || node.scan_op == ScanOp::ArgMax;
        const DataType default_output_dtype = arg_scan ? DataType::UINT32 : input_dtype;
        if (node.output_dtype.has_value() && node.output_dtype.value() != default_output_dtype) {
            throw std::runtime_error(arg_scan ? "Expression arg scan output dtype must be UINT32."
                                              : "Expression scan currently requires output dtype to match input dtype.");
        }
        return default_output_dtype;
    }


    if (node.op == ExprOp::SCAN_MIN_BACKWARD || node.op == ExprOp::SCAN_MAX_BACKWARD ||
        node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
        const bool segmented = node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD;
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("scan min/max backward node has input index out of range in resolveNodeOutputDType.");
        }
        if (segmented) {
            if (node.aux >= resolved_output_dtypes.size()) {
                throw std::runtime_error("segmented scan min/max backward offsets node has index out of range in resolveNodeOutputDType.");
            }
            const DataType offsets_dtype = resolved_output_dtypes[node.aux];
            if (offsets_dtype != DataType::UINT32 && offsets_dtype != DataType::UINT64) {
                throw std::runtime_error("segmented scan min/max backward offsets must have UINT32 or UINT64 dtype, received: " +
                                         TensorDescriptor::getElementTypeName(offsets_dtype));
            }
        }
        const DataType grad_dtype = resolved_output_dtypes[node.rhs];
        if (node.output_dtype.has_value() && node.output_dtype.value() != grad_dtype) {
            throw std::runtime_error("scan min/max backward output dtype must match grad-output dtype.");
        }
        return grad_dtype;
    }

    if (isCudnnReduceOp(node.op)) {
        if (node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX)
            return DataType::UINT32;

        // cuDNN reductions compute in the promoted reduction compute dtype.  Some low-precision
        // public gradient dtypes still need a final fused conversion stage because cuDNN ReduceTensor
        // does not accept every Thor floating output dtype.  Preserve the single-stage reduction path
        // only for materialized output dtypes that cuDNN supports directly, e.g. FP16 bias gradients.
        if (node.output_dtype.has_value() && isSupportedCudnnReductionOutputDType(node.output_dtype.value())) {
            return node.output_dtype.value();
        }
        return DataType::FP32;
    }

    std::vector<DataType> tensor_parent_dtypes;
    tensor_parent_dtypes.reserve(4);

    auto add_tensor_parent = [&](uint32_t parent_idx) {
        if (parent_idx >= nodes.size()) {
            throw std::runtime_error("Parent node index out of range in resolveNodeOutputDType.");
        }
        if (nodes[parent_idx].op == ExprOp::SCALAR_FP || nodes[parent_idx].op == ExprOp::RUNTIME_SCALAR ||
            nodes[parent_idx].op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            return;
        }
        tensor_parent_dtypes.push_back(resolved_output_dtypes[parent_idx]);
    };

    if (!Expression::isLeafOp(node.op)) {
        add_tensor_parent(node.lhs);
    }
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        add_tensor_parent(node.rhs);
    }
    if (Expression::isTernaryOp(node.op)) {
        add_tensor_parent(node.aux);
    }
    if (node.op == ExprOp::ATTENTION && node.attention_use_bias) {
        add_tensor_parent(node.alpha_node);
    }
    if (isAttentionBackwardOp(node.op)) {
        add_tensor_parent(node.alpha_node);
        if (node.attention_use_bias) {
            add_tensor_parent(node.beta_node);
        }
    }

    if (node.op == ExprOp::ATTENTION_BACKWARD_BIAS) {
        if (node.lhs >= resolved_output_dtypes.size()) {
            throw std::runtime_error("Attention-backward dBias node has q input index out of range in resolveNodeOutputDType.");
        }
        const DataType cudnn_dbias_dtype = resolved_output_dtypes[node.lhs];
        return node.output_dtype.has_value() ? node.output_dtype.value() : cudnn_dbias_dtype;
    }

    if (node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX) {
        return node.output_dtype.has_value() ? node.output_dtype.value() : DataType::UINT32;
    }

    if (isBooleanOutputOp(node.op)) {
        return node.output_dtype.has_value() ? node.output_dtype.value() : DataType::BOOLEAN;
    }

    if (isWhereOp(node.op)) {
        if (node.lhs >= resolved_output_dtypes.size() || node.rhs >= resolved_output_dtypes.size() || node.aux >= resolved_output_dtypes.size()) {
            throw std::runtime_error("Where node has parent index out of range in resolveNodeOutputDType.");
        }
        const DataType condition_dtype = resolved_output_dtypes[node.lhs];
        if (condition_dtype != DataType::BOOLEAN) {
            throw std::runtime_error("Where condition must have BOOLEAN dtype, received: " +
                                     TensorDescriptor::getElementTypeName(condition_dtype));
        }

        std::vector<DataType> branch_dtypes;
        branch_dtypes.reserve(2);
        const ExprNode& true_parent = nodes[node.rhs];
        const ExprNode& false_parent = nodes[node.aux];
        if (true_parent.op != ExprOp::SCALAR_FP && true_parent.op != ExprOp::RUNTIME_SCALAR &&
            true_parent.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            branch_dtypes.push_back(resolved_output_dtypes[node.rhs]);
        }
        if (false_parent.op != ExprOp::SCALAR_FP && false_parent.op != ExprOp::RUNTIME_SCALAR &&
            false_parent.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            branch_dtypes.push_back(resolved_output_dtypes[node.aux]);
        }
        const DataType default_output = promoteWhereBranchDTypes(branch_dtypes);
        return node.output_dtype.has_value() ? node.output_dtype.value() : default_output;
    }

    const DataType default_output = tensor_parent_dtypes.empty() ? DataType::FP32 : promoteTensorValueDTypes(tensor_parent_dtypes);

    return node.output_dtype.has_value() ? node.output_dtype.value() : default_output;
}

static void seedRequiredComputeDType(std::optional<DataType>& slot, DataType dtype) {
    if (!isSupportedFusionFloatingType(dtype)) {
        return;
    }

    if (!slot.has_value()) {
        slot = dtype;
    } else {
        slot = promoteTensorValueDTypes(slot.value(), dtype);
    }
}

static void propagateMaterializedOutputComputeDTypes(PhysicalExpression& expr,
                                                     const std::vector<uint32_t>& materialized_output_nodes,
                                                     const std::vector<uint8_t>& explicit_compute_dtype) {
    std::vector<std::optional<DataType>> required_compute_dtype(expr.nodes.size(), std::nullopt);

    for (uint32_t output_node_idx : materialized_output_nodes) {
        if (output_node_idx >= expr.nodes.size()) {
            throw std::runtime_error("Output node index out of range in propagateMaterializedOutputComputeDTypes.");
        }

        const ExprNode& output_node = expr.nodes[output_node_idx];
        if (output_node.compute_dtype.has_value()) {
            seedRequiredComputeDType(required_compute_dtype[output_node_idx], output_node.compute_dtype.value());
        } else if (output_node.output_dtype.has_value()) {
            seedRequiredComputeDType(required_compute_dtype[output_node_idx], output_node.output_dtype.value());
        }
    }

    for (int64_t node_idx_signed = static_cast<int64_t>(expr.nodes.size()) - 1; node_idx_signed >= 0; --node_idx_signed) {
        const uint32_t node_idx = static_cast<uint32_t>(node_idx_signed);
        if (!required_compute_dtype[node_idx].has_value()) {
            continue;
        }

        ExprNode& node = expr.nodes[node_idx];
        DataType propagated_dtype = required_compute_dtype[node_idx].value();

        if (!Expression::isLeafOp(node.op) && node.compute_dtype.has_value()) {
            if (!explicit_compute_dtype[node_idx]) {
                node.compute_dtype = promoteRequiredComputeDType(node.op, node.compute_dtype.value(), propagated_dtype);
            }
            propagated_dtype = node.compute_dtype.value();
        }

        auto propagate_to_parent = [&](uint32_t parent_idx) {
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Parent node index out of range in propagateMaterializedOutputComputeDTypes.");
            }
            seedRequiredComputeDType(required_compute_dtype[parent_idx], propagated_dtype);
        };

        auto propagate_to_floating_parent = [&](uint32_t parent_idx) {
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Parent node index out of range in propagateMaterializedOutputComputeDTypes.");
            }
            if (expr.nodes[parent_idx].output_dtype.has_value() && isSupportedFusionFloatingType(expr.nodes[parent_idx].output_dtype.value())) {
                seedRequiredComputeDType(required_compute_dtype[parent_idx], propagated_dtype);
            }
        };

        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                propagate_to_parent(input_node);
            }
        } else if (node.op == ExprOp::CAST) {
            // Cast is an explicit dtype-conversion boundary. Do not force the
            // source expression to compute in the destination dtype.
        } else if (node.op == ExprOp::WHERE) {
            // Do not propagate the selected value compute dtype into the boolean condition.
            // Only floating-value branches should be widened by a materialized where output.
            propagate_to_floating_parent(node.rhs);
            propagate_to_floating_parent(node.aux);
        } else if (node.op == ExprOp::SCAN_MIN_BACKWARD || node.op == ExprOp::SCAN_MAX_BACKWARD ||
                   node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
            // Segmented-scan offsets are index metadata. Propagate the requested
            // gradient compute dtype only through the value and upstream-gradient
            // operands, not through the offsets tensor.
            propagate_to_parent(node.lhs);
            propagate_to_parent(node.rhs);
        } else {
            if (!Expression::isLeafOp(node.op)) {
                propagate_to_parent(node.lhs);
            }
            if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
                propagate_to_parent(node.rhs);
            }
            if (Expression::isTernaryOp(node.op)) {
                propagate_to_parent(node.aux);
            }
            if (node.op == ExprOp::ATTENTION && node.attention_use_bias) {
                propagate_to_parent(node.alpha_node);
            }
            if (isAttentionBackwardOp(node.op)) {
                propagate_to_parent(node.alpha_node);
                if (node.attention_use_bias) {
                    propagate_to_parent(node.beta_node);
                }
            }
        }
    }
}

static void resolveExpressionDTypesInPlace(PhysicalExpression& expr,
                                           const std::vector<DataType>& root_input_dtypes,
                                           const std::vector<uint32_t>& materialized_output_nodes) {
    std::vector<DataType> resolved_output_dtypes(expr.nodes.size());
    std::vector<uint8_t> explicit_compute_dtype(expr.nodes.size(), 0);

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        explicit_compute_dtype[node_idx] = expr.nodes[node_idx].compute_dtype.has_value() ? 1 : 0;
    }

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        ExprNode& node = expr.nodes[node_idx];

        if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            if (node.input_slot >= root_input_dtypes.size()) {
                throw std::runtime_error("Input slot out of range in resolveExpressionDTypesInPlace.");
            }

            const DataType actual_input_dtype = root_input_dtypes[node.input_slot];
            if (!isPassthroughInputDType(actual_input_dtype)) {
                throw std::runtime_error("Unsupported input dtype in resolveExpressionDTypesInPlace: " +
                                         TensorDescriptor::getElementTypeName(actual_input_dtype));
            }

            node.input_tensor_dtype = actual_input_dtype;
            const DataType output_dtype = node.output_dtype.has_value() ? node.output_dtype.value() : actual_input_dtype;

            DataType compute_dtype = output_dtype;
            if (isSupportedFusionFloatingType(actual_input_dtype) && isSupportedFusionFloatingType(output_dtype)) {
                const DataType requested_compute_dtype =
                    node.compute_dtype.has_value() ? node.compute_dtype.value() : defaultComputeDType(actual_input_dtype, output_dtype);
                compute_dtype = toSupportedComputeDType(node.op, requested_compute_dtype);
            } else if (node.compute_dtype.has_value()) {
                compute_dtype = node.compute_dtype.value();
            }

            const DataType backward_output_dtype = node.backward_output_dtype.has_value() ? node.backward_output_dtype.value() : output_dtype;
            DataType backward_compute_dtype = compute_dtype;
            if (node.backward_compute_dtype.has_value()) {
                backward_compute_dtype = isSupportedFusionFloatingType(node.backward_compute_dtype.value())
                                             ? toSupportedComputeDType(node.op, node.backward_compute_dtype.value())
                                             : node.backward_compute_dtype.value();
            }

            node.output_dtype = output_dtype;
            node.compute_dtype = compute_dtype;
            node.backward_output_dtype = backward_output_dtype;
            node.backward_compute_dtype = backward_compute_dtype;

            resolved_output_dtypes[node_idx] = output_dtype;
            continue;
        }

        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            if (!node.output_dtype.has_value()) {
                throw std::runtime_error("CudaKernelExpression output node is missing output dtype.");
            }
            const DataType output_dtype = node.output_dtype.value();
            node.compute_dtype = node.compute_dtype.value_or(output_dtype);
            node.backward_output_dtype = node.backward_output_dtype.value_or(output_dtype);
            node.backward_compute_dtype = node.backward_compute_dtype.value_or(node.compute_dtype.value());
            resolved_output_dtypes[node_idx] = output_dtype;
            continue;
        }

        const DataType output_dtype = resolveNodeOutputDType(node, expr.nodes, resolved_output_dtypes, root_input_dtypes);
        const DataType logical_input_dtype = resolveNodeLogicalInputDType(node, expr.nodes, resolved_output_dtypes, root_input_dtypes);

        DataType requested_compute_dtype;
        if (node.op == ExprOp::EMBEDDING_LOOKUP || node.op == ExprOp::CAST || isPassthroughViewOp(node.op)) {
            requested_compute_dtype = output_dtype;
        } else if (isComparisonOp(node.op)) {
            if (node.compute_dtype.has_value()) {
                requested_compute_dtype = node.compute_dtype.value();
            } else if (logical_input_dtype == DataType::BOOLEAN || logical_input_dtype == DataType::UINT8) {
                requested_compute_dtype = DataType::BOOLEAN;
            } else if (isSupportedFusionFloatingType(logical_input_dtype)) {
                requested_compute_dtype = defaultComputeDType(logical_input_dtype);
            } else {
                requested_compute_dtype = DataType::FP32;
            }
        } else if (isLogicalOp(node.op)) {
            requested_compute_dtype = node.compute_dtype.has_value() ? node.compute_dtype.value() : logical_input_dtype;
        } else if (isWhereOp(node.op)) {
            requested_compute_dtype = node.compute_dtype.has_value() ? node.compute_dtype.value() : logical_input_dtype;
        } else if (isScanOp(node.op)) {
            requested_compute_dtype = node.compute_dtype.has_value() ? node.compute_dtype.value() : output_dtype;
        } else {
            requested_compute_dtype =
                node.compute_dtype.has_value() ? node.compute_dtype.value() : defaultComputeDType(logical_input_dtype, output_dtype);
        }
        if (isConvolutionOp(node.op) && !node.compute_dtype.has_value() &&
            (isFp8Type(logical_input_dtype) || isFp8Type(output_dtype))) {
            requested_compute_dtype = DataType::FP32;
        }
        const DataType compute_dtype = toSupportedComputeDType(node.op, requested_compute_dtype);

        const DataType backward_output_dtype = node.backward_output_dtype.has_value() ? node.backward_output_dtype.value() : output_dtype;

        const DataType backward_compute_dtype =
            node.backward_compute_dtype.has_value() ? toSupportedComputeDType(node.op, node.backward_compute_dtype.value()) : compute_dtype;

        node.output_dtype = output_dtype;
        node.compute_dtype = compute_dtype;
        node.backward_output_dtype = backward_output_dtype;
        node.backward_compute_dtype = backward_compute_dtype;

        resolved_output_dtypes[node_idx] = output_dtype;
    }

    propagateMaterializedOutputComputeDTypes(expr, materialized_output_nodes, explicit_compute_dtype);
}

void resolveExpressionDTypesInPlace(PhysicalExpression& expr, const std::vector<DataType>& root_input_dtypes) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("resolveExpressionDTypesInPlace requires a valid output_node.");
    }
    resolveExpressionDTypesInPlace(expr, root_input_dtypes, std::vector<uint32_t>{expr.output_node});
}

void resolveOutputsDTypesInPlace(PhysicalOutputs& outputs, const std::vector<DataType>& root_input_dtypes) {
    if (!outputs.expr) {
        throw std::runtime_error("resolveOutputsDTypesInPlace requires non-null outputs.expr.");
    }

    std::vector<uint32_t> output_nodes;
    output_nodes.reserve(outputs.outputs.size());
    for (const NamedOutput& output : outputs.outputs) {
        output_nodes.push_back(output.node_idx);
    }
    resolveExpressionDTypesInPlace(*outputs.expr, root_input_dtypes, output_nodes);
}

}  // namespace ThorImplementation
