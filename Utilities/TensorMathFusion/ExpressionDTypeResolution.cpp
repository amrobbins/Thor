#include "Utilities/TensorMathFusion/ExpressionDTypeResolution.h"

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

static DataType resolveNodeOutputDType(const ExprNode& node,
                                       const std::vector<ExprNode>& nodes,
                                       const std::vector<DataType>& resolved_output_dtypes,
                                       const std::vector<DataType>& root_input_dtypes) {
    if (node.op == ExprOp::INPUT) {
        if (node.input_slot >= root_input_dtypes.size()) {
            throw std::runtime_error("Input slot out of range in resolveNodeOutputDType.");
        }

        const DataType default_output = root_input_dtypes[node.input_slot];
        return node.output_dtype.isPresent() ? node.output_dtype.get() : default_output;
    }

    if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::FILL) {
        return node.output_dtype.isPresent() ? node.output_dtype.get() : DataType::FP32;
    }

    std::vector<DataType> tensor_parent_dtypes;
    tensor_parent_dtypes.reserve(2);

    auto add_tensor_parent = [&](uint32_t parent_idx) {
        if (parent_idx >= nodes.size()) {
            throw std::runtime_error("Parent node index out of range in resolveNodeOutputDType.");
        }
        if (nodes[parent_idx].op == ExprOp::SCALAR_FP) {
            return;
        }
        tensor_parent_dtypes.push_back(resolved_output_dtypes[parent_idx]);
    };

    if (!Expression::isLeafOp(node.op)) {
        add_tensor_parent(node.lhs);
    }
    if (Expression::isBinaryOp(node.op)) {
        add_tensor_parent(node.rhs);
    }

    if (node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX) {
        return node.output_dtype.isPresent() ? node.output_dtype.get() : DataType::UINT32;
    }

    const DataType default_output = tensor_parent_dtypes.empty() ? DataType::FP32 : promoteTensorValueDTypes(tensor_parent_dtypes);

    return node.output_dtype.isPresent() ? node.output_dtype.get() : default_output;
}

void resolveExpressionDTypesInPlace(PhysicalExpression& expr, const std::vector<DataType>& root_input_dtypes) {
    std::vector<DataType> resolved_output_dtypes(expr.nodes.size());

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        ExprNode& node = expr.nodes[node_idx];

        if (node.op == ExprOp::INPUT) {
            if (node.input_slot >= root_input_dtypes.size()) {
                throw std::runtime_error("Input slot out of range in resolveExpressionDTypesInPlace.");
            }

            const DataType actual_input_dtype = root_input_dtypes[node.input_slot];
            node.input_tensor_dtype = actual_input_dtype;
            const DataType output_dtype = node.output_dtype.isPresent() ? node.output_dtype.get() : actual_input_dtype;
            const DataType compute_dtype = node.compute_dtype.isPresent() ? node.compute_dtype.get() : defaultComputeDType(output_dtype);
            const DataType backward_output_dtype = node.backward_output_dtype.isPresent() ? node.backward_output_dtype.get() : output_dtype;
            const DataType backward_compute_dtype =
                node.backward_compute_dtype.isPresent() ? node.backward_compute_dtype.get() : compute_dtype;

            node.output_dtype = output_dtype;
            node.compute_dtype = compute_dtype;
            node.backward_output_dtype = backward_output_dtype;
            node.backward_compute_dtype = backward_compute_dtype;

            resolved_output_dtypes[node_idx] = output_dtype;
            continue;
        }

        const DataType output_dtype = resolveNodeOutputDType(node, expr.nodes, resolved_output_dtypes, root_input_dtypes);

        const bool is_arg_reduction = node.op == ExprOp::REDUCE_ARGMIN || node.op == ExprOp::REDUCE_ARGMAX;
        const DataType compute_dtype = node.compute_dtype.isPresent()
                                           ? node.compute_dtype.get()
                                           : (is_arg_reduction ? DataType::FP32 : defaultComputeDType(output_dtype));

        const DataType backward_output_dtype = node.backward_output_dtype.isPresent() ? node.backward_output_dtype.get() : output_dtype;

        const DataType backward_compute_dtype = node.backward_compute_dtype.isPresent() ? node.backward_compute_dtype.get() : compute_dtype;

        node.output_dtype = output_dtype;
        node.compute_dtype = compute_dtype;
        node.backward_output_dtype = backward_output_dtype;
        node.backward_compute_dtype = backward_compute_dtype;

        resolved_output_dtypes[node_idx] = output_dtype;
    }
}

void resolveOutputsDTypesInPlace(PhysicalOutputs& outputs, const std::vector<DataType>& root_input_dtypes) {
    if (!outputs.expr) {
        throw std::runtime_error("resolveOutputsDTypesInPlace requires non-null outputs.expr.");
    }

    resolveExpressionDTypesInPlace(*outputs.expr, root_input_dtypes);
}

}  // namespace ThorImplementation
