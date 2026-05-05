#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <stdexcept>

namespace ThorImplementation {

using DataType = TensorDescriptor::DataType;

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

static bool isReductionComputeOp(ExprOp op) { return isCudnnReduceOp(op); }
static bool isCudnnSingleInputStageOp(ExprOp op) { return isCudnnReduceOp(op) || isCudnnSoftmaxOp(op); }

DataType toSupportedComputeDType(ExprOp op, DataType requested_compute_dtype) {
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
        return node.output_dtype.isPresent() ? node.output_dtype.get() : DataType::FP32;
    }

    std::vector<DataType> tensor_parent_dtypes;
    tensor_parent_dtypes.reserve(2);

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

    return tensor_parent_dtypes.empty() ? DataType::FP32 : promoteTensorValueDTypes(tensor_parent_dtypes);
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
        return node.output_dtype.isPresent() ? node.output_dtype.get() : default_output;
    }

    if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::FILL) {
        return node.output_dtype.isPresent() ? node.output_dtype.get() : DataType::FP32;
    }

    if (isCudnnReduceOp(node.op)) {
        if (node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX)
            return DataType::UINT32;
        return DataType::FP32;
    }

    std::vector<DataType> tensor_parent_dtypes;
    tensor_parent_dtypes.reserve(2);

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

    if (node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX) {
        return node.output_dtype.isPresent() ? node.output_dtype.get() : DataType::UINT32;
    }

    const DataType default_output = tensor_parent_dtypes.empty() ? DataType::FP32 : promoteTensorValueDTypes(tensor_parent_dtypes);

    return node.output_dtype.isPresent() ? node.output_dtype.get() : default_output;
}

static void seedRequiredComputeDType(Optional<DataType>& slot, DataType dtype) {
    if (!isSupportedFusionFloatingType(dtype)) {
        return;
    }

    if (!slot.isPresent()) {
        slot = dtype;
    } else {
        slot = promoteTensorValueDTypes(slot.get(), dtype);
    }
}

static void propagateMaterializedOutputComputeDTypes(PhysicalExpression& expr,
                                                     const std::vector<uint32_t>& materialized_output_nodes,
                                                     const std::vector<uint8_t>& explicit_compute_dtype) {
    std::vector<Optional<DataType>> required_compute_dtype(expr.nodes.size(), Optional<DataType>::empty());

    for (uint32_t output_node_idx : materialized_output_nodes) {
        if (output_node_idx >= expr.nodes.size()) {
            throw std::runtime_error("Output node index out of range in propagateMaterializedOutputComputeDTypes.");
        }

        const ExprNode& output_node = expr.nodes[output_node_idx];
        if (output_node.compute_dtype.isPresent()) {
            seedRequiredComputeDType(required_compute_dtype[output_node_idx], output_node.compute_dtype.get());
        } else if (output_node.output_dtype.isPresent()) {
            seedRequiredComputeDType(required_compute_dtype[output_node_idx], output_node.output_dtype.get());
        }
    }

    for (int64_t node_idx_signed = static_cast<int64_t>(expr.nodes.size()) - 1; node_idx_signed >= 0; --node_idx_signed) {
        const uint32_t node_idx = static_cast<uint32_t>(node_idx_signed);
        if (!required_compute_dtype[node_idx].isPresent()) {
            continue;
        }

        ExprNode& node = expr.nodes[node_idx];
        DataType propagated_dtype = required_compute_dtype[node_idx].get();

        if (!Expression::isLeafOp(node.op) && node.compute_dtype.isPresent()) {
            if (!explicit_compute_dtype[node_idx]) {
                const DataType promoted_requested_dtype = promoteTensorValueDTypes(node.compute_dtype.get(), propagated_dtype);
                node.compute_dtype = toSupportedComputeDType(node.op, promoted_requested_dtype);
            }
            propagated_dtype = node.compute_dtype.get();
        }

        auto propagate_to_parent = [&](uint32_t parent_idx) {
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Parent node index out of range in propagateMaterializedOutputComputeDTypes.");
            }
            seedRequiredComputeDType(required_compute_dtype[parent_idx], propagated_dtype);
        };

        if (!Expression::isLeafOp(node.op)) {
            propagate_to_parent(node.lhs);
        }
        if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
            propagate_to_parent(node.rhs);
        }
        if (Expression::isTernaryOp(node.op)) {
            propagate_to_parent(node.aux);
        }
    }
}

static void resolveExpressionDTypesInPlace(PhysicalExpression& expr,
                                           const std::vector<DataType>& root_input_dtypes,
                                           const std::vector<uint32_t>& materialized_output_nodes) {
    std::vector<DataType> resolved_output_dtypes(expr.nodes.size());
    std::vector<uint8_t> explicit_compute_dtype(expr.nodes.size(), 0);

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        explicit_compute_dtype[node_idx] = expr.nodes[node_idx].compute_dtype.isPresent() ? 1 : 0;
    }

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        ExprNode& node = expr.nodes[node_idx];

        if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            if (node.input_slot >= root_input_dtypes.size()) {
                throw std::runtime_error("Input slot out of range in resolveExpressionDTypesInPlace.");
            }

            const DataType actual_input_dtype = root_input_dtypes[node.input_slot];
            node.input_tensor_dtype = actual_input_dtype;
            const DataType output_dtype = node.output_dtype.isPresent() ? node.output_dtype.get() : actual_input_dtype;
            const DataType requested_compute_dtype =
                node.compute_dtype.isPresent() ? node.compute_dtype.get() : defaultComputeDType(actual_input_dtype, output_dtype);
            const DataType compute_dtype = toSupportedComputeDType(node.op, requested_compute_dtype);
            const DataType backward_output_dtype = node.backward_output_dtype.isPresent() ? node.backward_output_dtype.get() : output_dtype;
            const DataType backward_compute_dtype = node.backward_compute_dtype.isPresent()
                                                        ? toSupportedComputeDType(node.op, node.backward_compute_dtype.get())
                                                        : compute_dtype;

            node.output_dtype = output_dtype;
            node.compute_dtype = compute_dtype;
            node.backward_output_dtype = backward_output_dtype;
            node.backward_compute_dtype = backward_compute_dtype;

            resolved_output_dtypes[node_idx] = output_dtype;
            continue;
        }

        const DataType output_dtype = resolveNodeOutputDType(node, expr.nodes, resolved_output_dtypes, root_input_dtypes);
        const DataType logical_input_dtype = resolveNodeLogicalInputDType(node, expr.nodes, resolved_output_dtypes, root_input_dtypes);

        const DataType requested_compute_dtype =
            node.compute_dtype.isPresent() ? node.compute_dtype.get() : defaultComputeDType(logical_input_dtype, output_dtype);
        const DataType compute_dtype = toSupportedComputeDType(node.op, requested_compute_dtype);

        const DataType backward_output_dtype = node.backward_output_dtype.isPresent() ? node.backward_output_dtype.get() : output_dtype;

        const DataType backward_compute_dtype =
            node.backward_compute_dtype.isPresent() ? toSupportedComputeDType(node.op, node.backward_compute_dtype.get()) : compute_dtype;

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
