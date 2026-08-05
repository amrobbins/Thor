#include "Utilities/TensorOperations/Einsum/Einsum.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr const char* kGenericOutputName = "einsum_output";
constexpr const char* kPreReductionOutputName = "einsum_pre_reduction";

[[nodiscard]] bool isSupportedStorageDType(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

[[nodiscard]] uint64_t checkedProduct(const std::vector<uint64_t>& dimensions, const char* role) {
    uint64_t product = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument(std::string("Einsum ") + role + " dimensions must be non-zero.");
        }
        if (product > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::invalid_argument(std::string("Einsum ") + role + " element count overflows uint64_t.");
        }
        product *= dimension;
    }
    return product;
}

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* role) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::invalid_argument(std::string("Einsum ") + role + " size overflows uint64_t.");
    }
    return lhs * rhs;
}

[[nodiscard]] std::vector<uint64_t> physicalOutputDimensions(const EinsumPlan& plan) {
    if (plan.equation.output_dimensions.empty()) {
        return {1};
    }
    return plan.equation.output_dimensions;
}

void requireSupportedInputs(const std::vector<Tensor>& inputs, const Stream& stream) {
    if (inputs.empty()) {
        throw std::invalid_argument("Einsum requires at least one input tensor.");
    }
    if (!stream.isInitialized()) {
        throw std::invalid_argument("Einsum requires an initialized GPU stream.");
    }

    if (!inputs.front().isInitialized()) {
        throw std::invalid_argument("Einsum input tensor 0 is uninitialized.");
    }
    const TensorPlacement expected_placement = inputs.front().getPlacement();
    const DataType expected_dtype = inputs.front().getDataType();
    if (expected_placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Einsum Patch 3 execution currently requires GPU input tensors.");
    }
    if (expected_placement.getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Einsum input tensors and stream must use the same GPU.");
    }
    if (!isSupportedStorageDType(expected_dtype)) {
        throw std::invalid_argument("Einsum Patch 3 execution supports FP16, BF16, and FP32 storage dtypes.");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& input = inputs[i];
        if (!input.isInitialized()) {
            throw std::invalid_argument("Einsum input tensor " + std::to_string(i) + " is uninitialized.");
        }
        if (input.getPlacement() != expected_placement) {
            throw std::invalid_argument("Einsum input tensors must all use the same GPU placement.");
        }
        if (input.getDataType() != expected_dtype) {
            throw std::invalid_argument("Einsum Patch 3 execution requires all input tensors to have the same storage dtype.");
        }
        if (!input.isDenseContiguous()) {
            throw std::invalid_argument("Einsum Patch 3 execution currently requires dense contiguous input tensors.");
        }
    }
}

[[nodiscard]] bool tensorStorageOverlaps(const Tensor& lhs, const Tensor& rhs) {
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs.getMemPtr<void>());
    const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs.getMemPtr<void>());
    const uintptr_t lhs_end = lhs_begin + lhs.getArraySizeInBytes();
    const uintptr_t rhs_end = rhs_begin + rhs.getArraySizeInBytes();
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

void requireOutput(const Tensor& output,
                   const std::vector<Tensor>& inputs,
                   const std::vector<uint64_t>& expected_dimensions,
                   DataType expected_dtype) {
    if (!output.isInitialized()) {
        throw std::invalid_argument("Einsum preallocated output tensor is uninitialized.");
    }
    if (output.getPlacement() != inputs.front().getPlacement()) {
        throw std::invalid_argument("Einsum preallocated output must use the input GPU placement.");
    }
    if (output.getDataType() != expected_dtype) {
        throw std::invalid_argument("Einsum preallocated output dtype must match the input dtype.");
    }
    if (!output.isDenseContiguous()) {
        throw std::invalid_argument("Einsum preallocated output must be dense contiguous.");
    }
    if (output.getDimensions() != expected_dimensions) {
        throw std::invalid_argument("Einsum preallocated output dimensions do not match the resolved equation.");
    }
    for (const Tensor& input : inputs) {
        if (tensorStorageOverlaps(input, output)) {
            throw std::invalid_argument("Einsum input and output storage must not overlap.");
        }
    }
}

[[nodiscard]] std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> strides(dimensions.size(), 1);
    if (dimensions.empty()) {
        return strides;
    }
    for (size_t axis = dimensions.size() - 1; axis > 0; --axis) {
        strides[axis - 1] = checkedMultiply(strides[axis], dimensions[axis], "dense stride");
    }
    return strides;
}

[[nodiscard]] std::vector<uint64_t> diagonalizedStrides(const ResolvedEinsumOperand& operand,
                                                         const std::vector<uint64_t>& input_dimensions,
                                                         const EinsumOperandPlan& operand_plan) {
    const std::vector<uint64_t> source_strides = denseStrides(input_dimensions);
    std::vector<uint64_t> result;
    result.reserve(operand_plan.diagonalized_labels.size());

    for (int32_t label : operand_plan.diagonalized_labels) {
        uint64_t combined_stride = 0;
        bool found = false;
        for (size_t source_axis = 0; source_axis < operand.axis_labels.size(); ++source_axis) {
            if (operand.axis_labels[source_axis] != label) {
                continue;
            }
            if (combined_stride > std::numeric_limits<uint64_t>::max() - source_strides[source_axis]) {
                throw std::invalid_argument("Einsum diagonal stride overflows uint64_t.");
            }
            combined_stride += source_strides[source_axis];
            found = true;
        }
        if (!found || combined_stride == 0) {
            throw std::runtime_error("Einsum planner produced an invalid diagonalized label mapping.");
        }
        result.push_back(combined_stride);
    }
    return result;
}

[[nodiscard]] Expression alignedOperandExpression(size_t operand_index,
                                                  const Tensor& input,
                                                  const EinsumPlan& plan) {
    Expression expression = Expression::input("einsum_input_" + std::to_string(operand_index));
    const EinsumOperandPlan& operand_plan = plan.operands.at(operand_index);
    const ResolvedEinsumOperand& resolved_operand = plan.equation.inputs.at(operand_index);

    if (operand_plan.requiresDiagonalExtraction() || operand_plan.requiresPermutation()) {
        const std::vector<uint64_t> logical_strides =
            diagonalizedStrides(resolved_operand, input.getDimensions(), operand_plan);
        std::vector<uint64_t> view_dimensions;
        std::vector<uint64_t> view_strides;
        view_dimensions.reserve(operand_plan.permutation.size());
        view_strides.reserve(operand_plan.permutation.size());
        for (uint32_t source_axis : operand_plan.permutation) {
            if (source_axis >= operand_plan.diagonalized_dimensions.size() || source_axis >= logical_strides.size()) {
                throw std::runtime_error("Einsum planner produced an out-of-range operand permutation.");
            }
            view_dimensions.push_back(operand_plan.diagonalized_dimensions[source_axis]);
            view_strides.push_back(logical_strides[source_axis]);
        }
        if (view_dimensions.empty()) {
            throw std::runtime_error("Einsum execution cannot construct an empty-rank strided view.");
        }
        expression = expression.stridedView(view_dimensions, view_strides);
    }

    if (!operand_plan.inserted_axes.empty()) {
        std::vector<uint64_t> inserted_axes;
        inserted_axes.reserve(operand_plan.inserted_axes.size());
        for (uint32_t axis : operand_plan.inserted_axes) {
            inserted_axes.push_back(axis);
        }
        expression = expression.unsqueeze(inserted_axes);
    }

    return expression;
}

[[nodiscard]] std::unordered_map<std::string, Tensor> expressionInputs(const std::vector<Tensor>& inputs) {
    std::unordered_map<std::string, Tensor> named_inputs;
    named_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        named_inputs.emplace("einsum_input_" + std::to_string(i), inputs[i]);
    }
    return named_inputs;
}

struct GenericStampedStages {
    std::shared_ptr<StampedExecutionPlan> preparation;
    std::shared_ptr<StampedCubReduction> reduction;
};

[[nodiscard]] GenericStampedStages stampGeneric(const EinsumPlan& plan,
                                                 const std::vector<Tensor>& inputs,
                                                 const Tensor& output,
                                                 const Stream& stream) {
    std::vector<Expression> aligned_operands;
    aligned_operands.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        aligned_operands.push_back(alignedOperandExpression(i, inputs[i], plan));
    }

    Expression combined = aligned_operands.front();
    if (aligned_operands.size() > 1) {
        combined = combined.cast(DataType::FP32);
        for (size_t i = 1; i < aligned_operands.size(); ++i) {
            combined = combined * aligned_operands[i].cast(DataType::FP32);
        }
    }

    const auto named_inputs = expressionInputs(inputs);
    const DataType output_dtype = output.getDataType();

    if (plan.reduction_axes.empty()) {
        if (aligned_operands.size() > 1 && output_dtype != DataType::FP32) {
            combined = combined.cast(output_dtype);
        }
        FusedEquation equation = FusedEquation::compile(Expression::outputs({{kGenericOutputName, combined}}).physicalOutputs(),
                                                       output.getPlacement().getDeviceNum());
        StampedExecutionPlan stamped = equation.stamp(named_inputs, stream, {}, {{kGenericOutputName, output}});
        return GenericStampedStages{
            .preparation = std::make_shared<StampedExecutionPlan>(std::move(stamped)),
            .reduction = nullptr,
        };
    }

    // A pure identity-layout unary reduction can feed the dense input straight
    // to CubReduction.  Otherwise materialize all diagonal/permutation/
    // broadcast/product work into one dense pre-reduction tensor first.
    const bool direct_unary_reduction = inputs.size() == 1 && !plan.operands.front().requiresDiagonalExtraction()
                                        && !plan.operands.front().requiresPermutation()
                                        && plan.operands.front().inserted_axes.empty()
                                        && plan.operands.front().broadcast_axes.empty()
                                        && inputs.front().getDimensions() == plan.iteration_dimensions;

    Tensor reduction_input;
    std::shared_ptr<StampedExecutionPlan> preparation;
    if (direct_unary_reduction) {
        reduction_input = inputs.front();
    } else {
        const DataType intermediate_dtype = inputs.size() > 1 ? DataType::FP32 : inputs.front().getDataType();
        reduction_input = Tensor(inputs.front().getPlacement(), TensorDescriptor(intermediate_dtype, plan.iteration_dimensions));
        FusedEquation equation =
            FusedEquation::compile(Expression::outputs({{kPreReductionOutputName, combined}}).physicalOutputs(),
                                   output.getPlacement().getDeviceNum());
        StampedExecutionPlan stamped =
            equation.stamp(named_inputs, stream, {}, {{kPreReductionOutputName, reduction_input}});
        preparation = std::make_shared<StampedExecutionPlan>(std::move(stamped));
    }

    // Centralization invariant: every standalone einsum reduction reaches this
    // exact utility.  Do not replace this with an einsum-local reduction.
    auto reduction = CubReduction(CubReductionOp::Sum, plan.reduction_axes, output_dtype).stamp(reduction_input, output, stream);
    return GenericStampedStages{
        .preparation = std::move(preparation),
        .reduction = std::move(reduction),
    };
}

[[nodiscard]] bool matrixDimensionsFitBackend(const EinsumMatrixMultiplyPlan& matrix_plan) {
    constexpr uint64_t kMax = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return matrix_plan.m <= kMax && matrix_plan.n <= kMax && matrix_plan.k <= kMax && matrix_plan.batch_count <= kMax;
}

[[nodiscard]] int64_t checkedBatchStride(uint64_t elements, const char* role) {
    if (elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument(std::string("Einsum ") + role + " batch stride exceeds the cuBLASLt int64 limit.");
    }
    return static_cast<int64_t>(elements);
}

struct StridedBatchedMatrixMultiplyStamp {
    std::shared_ptr<CublasKernel> kernel;
    std::optional<Tensor> workspace;
};

[[nodiscard]] std::vector<uint64_t> matrixStoredDimensions(uint64_t rows, uint64_t columns) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Einsum matrix dimensions must be non-zero.");
    }
    return {rows, columns};
}

[[nodiscard]] std::vector<std::shared_ptr<StampedMatmul>> stampDirectMatrixMultiply(
    const EinsumMatrixMultiplyPlan& matrix_plan,
    const std::vector<Tensor>& inputs,
    const Tensor& output,
    const Stream& stream) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Einsum direct matrix plan requires exactly two inputs.");
    }
    if (!matrix_plan.direct) {
        throw std::runtime_error("Einsum attempted to stamp a non-direct matrix plan as direct GEMM.");
    }
    if (matrix_plan.batch_count != 1) {
        throw std::runtime_error("Einsum single-GEMM stamper received a batched matrix plan.");
    }
    if (!matrixDimensionsFitBackend(matrix_plan)) {
        throw std::invalid_argument("Einsum direct GEMM dimensions exceed the current int32 cuBLAS backend limit.");
    }

    const uint64_t lhs_rows = matrix_plan.lhs.transpose ? matrix_plan.k : matrix_plan.m;
    const uint64_t lhs_cols = matrix_plan.lhs.transpose ? matrix_plan.m : matrix_plan.k;
    const uint64_t rhs_rows = matrix_plan.rhs.transpose ? matrix_plan.n : matrix_plan.k;
    const uint64_t rhs_cols = matrix_plan.rhs.transpose ? matrix_plan.k : matrix_plan.n;
    const uint64_t output_rows = matrix_plan.m;
    const uint64_t output_cols = matrix_plan.n;

    const uint64_t lhs_elements = checkedMultiply(lhs_rows, lhs_cols, "left matrix");
    const uint64_t rhs_elements = checkedMultiply(rhs_rows, rhs_cols, "right matrix");
    const uint64_t output_elements = checkedMultiply(output_rows, output_cols, "output matrix");
    if (lhs_elements != inputs[0].getTotalNumElements() || rhs_elements != inputs[1].getTotalNumElements()
        || output_elements != output.getTotalNumElements()) {
        throw std::runtime_error("Einsum direct matrix plan element counts do not match physical tensors.");
    }

    const DataType dtype = inputs.front().getDataType();
    auto compiled = std::make_shared<CompiledMatmul>(ExprOp::MATMUL,
                                                     matrix_plan.lhs.transpose,
                                                     matrix_plan.rhs.transpose,
                                                     false,
                                                     1.0,
                                                     0.0,
                                                     UINT32_MAX,
                                                     UINT32_MAX,
                                                     dtype,
                                                     dtype,
                                                     dtype,
                                                     dtype,
                                                     dtype,
                                                     DataType::FP32);

    Tensor lhs = inputs[0];
    Tensor rhs = inputs[1];
    Tensor dst = output;
    lhs.reshape(matrixStoredDimensions(lhs_rows, lhs_cols));
    rhs.reshape(matrixStoredDimensions(rhs_rows, rhs_cols));
    dst.reshape(matrixStoredDimensions(output_rows, output_cols));

    std::shared_ptr<BuiltMatmul> built =
        StampedEquation::buildMatmul(compiled, lhs, rhs, std::nullopt, dst, output.getPlacement().getDeviceNum());

    std::optional<Tensor> workspace;
    if (built->workspace_bytes != 0) {
        workspace = Tensor(output.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(built->workspace_bytes)}));
    }

    std::vector<std::shared_ptr<StampedMatmul>> batches;
    batches.push_back(std::make_shared<StampedMatmul>(compiled,
                                                       built,
                                                       lhs,
                                                       rhs,
                                                       std::nullopt,
                                                       dst,
                                                       stream,
                                                       workspace,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::nullopt));
    return batches;
}

[[nodiscard]] StridedBatchedMatrixMultiplyStamp stampDirectStridedBatchedMatrixMultiply(
    const EinsumMatrixMultiplyPlan& matrix_plan,
    const std::vector<Tensor>& inputs,
    const Tensor& output) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Einsum direct strided-batched matrix plan requires exactly two inputs.");
    }
    if (!matrix_plan.direct || matrix_plan.batch_count <= 1) {
        throw std::runtime_error("Einsum strided-batched stamper requires a direct matrix plan with batch_count > 1.");
    }
    if (!matrixDimensionsFitBackend(matrix_plan)) {
        throw std::invalid_argument("Einsum direct batched GEMM dimensions exceed the current cuBLASLt int32 backend limit.");
    }

    const uint64_t lhs_rows = matrix_plan.lhs.transpose ? matrix_plan.k : matrix_plan.m;
    const uint64_t lhs_cols = matrix_plan.lhs.transpose ? matrix_plan.m : matrix_plan.k;
    const uint64_t rhs_rows = matrix_plan.rhs.transpose ? matrix_plan.n : matrix_plan.k;
    const uint64_t rhs_cols = matrix_plan.rhs.transpose ? matrix_plan.k : matrix_plan.n;
    const uint64_t output_rows = matrix_plan.m;
    const uint64_t output_cols = matrix_plan.n;

    const uint64_t lhs_batch_elements = checkedMultiply(lhs_rows, lhs_cols, "left matrix batch");
    const uint64_t rhs_batch_elements = checkedMultiply(rhs_rows, rhs_cols, "right matrix batch");
    const uint64_t output_batch_elements = checkedMultiply(output_rows, output_cols, "output matrix batch");
    if (checkedMultiply(matrix_plan.batch_count, lhs_batch_elements, "left batched matrix") != inputs[0].getTotalNumElements()
        || checkedMultiply(matrix_plan.batch_count, rhs_batch_elements, "right batched matrix") != inputs[1].getTotalNumElements()
        || checkedMultiply(matrix_plan.batch_count, output_batch_elements, "batched output matrix") != output.getTotalNumElements()) {
        throw std::runtime_error("Einsum direct batched matrix plan element counts do not match physical tensors.");
    }

    const int32_t batch_count = static_cast<int32_t>(matrix_plan.batch_count);
    const int32_t lhs_rows_i32 = static_cast<int32_t>(lhs_rows);
    const int32_t lhs_cols_i32 = static_cast<int32_t>(lhs_cols);
    const int32_t rhs_rows_i32 = static_cast<int32_t>(rhs_rows);
    const int32_t rhs_cols_i32 = static_cast<int32_t>(rhs_cols);
    const int32_t output_cols_i32 = static_cast<int32_t>(output_cols);
    const CublasStridedBatchConfig batch_config =
        CublasStridedBatchConfig::strided(batch_count,
                                          checkedBatchStride(lhs_batch_elements, "left"),
                                          checkedBatchStride(rhs_batch_elements, "right"),
                                          checkedBatchStride(output_batch_elements, "addend"),
                                          checkedBatchStride(output_batch_elements, "output"));

    const DataType dtype = inputs.front().getDataType();
    const CublasMatrixMultiply::MatmulDataTypes data_types(dtype, dtype, dtype, dtype, DataType::FP32);
    const int gpu_num = output.getPlacement().getDeviceNum();
    CublasMatrixMultiply& matrix_multiply = CublasMatrixMultiply::instance();
    matrix_multiply.chooseOptimalStridedBatchedGemmKernel(gpu_num,
                                                           lhs_rows_i32,
                                                           lhs_cols_i32,
                                                           rhs_rows_i32,
                                                           rhs_cols_i32,
                                                           lhs_cols_i32,
                                                           rhs_cols_i32,
                                                           output_cols_i32,
                                                           output_cols_i32,
                                                           matrix_plan.lhs.transpose,
                                                           matrix_plan.rhs.transpose,
                                                           false,
                                                           data_types,
                                                           batch_config);

    auto kernel = std::make_shared<CublasKernel>(matrix_multiply.getCachedGemmKernel(gpu_num,
                                                                                     lhs_rows_i32,
                                                                                     lhs_cols_i32,
                                                                                     rhs_rows_i32,
                                                                                     rhs_cols_i32,
                                                                                     lhs_cols_i32,
                                                                                     rhs_cols_i32,
                                                                                     output_cols_i32,
                                                                                     output_cols_i32,
                                                                                     matrix_plan.lhs.transpose,
                                                                                     matrix_plan.rhs.transpose,
                                                                                     false,
                                                                                     data_types,
                                                                                     true,
                                                                                     batch_config));

    std::optional<Tensor> workspace;
    const uint64_t workspace_bytes = kernel->getWorkspaceSizeInBytes(gpu_num);
    if (workspace_bytes != 0) {
        workspace = Tensor(output.getPlacement(), TensorDescriptor(DataType::UINT8, {workspace_bytes}));
    }

    return StridedBatchedMatrixMultiplyStamp{.kernel = std::move(kernel), .workspace = std::move(workspace)};
}

[[nodiscard]] bool canUseDirectMatrixPath(const EinsumPlan& plan) {
    return plan.matrix_multiply.has_value() && plan.matrix_multiply->direct && matrixDimensionsFitBackend(*plan.matrix_multiply);
}

[[nodiscard]] std::shared_ptr<StampedEinsum> stampImpl(const std::string& equation,
                                                       const std::vector<Tensor>& inputs,
                                                       const std::optional<Tensor>& preallocated_output,
                                                       const Stream& stream) {
    requireSupportedInputs(inputs, stream);

    std::vector<std::vector<uint64_t>> input_dimensions;
    input_dimensions.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_dimensions.push_back(input.getDimensions());
    }
    EinsumPlan plan = EinsumPlanner::parseAndPlan(equation, input_dimensions);
    const std::vector<uint64_t> output_dimensions = physicalOutputDimensions(plan);
    (void)checkedProduct(output_dimensions, "output");

    Tensor output;
    if (preallocated_output.has_value()) {
        output = preallocated_output.value();
        requireOutput(output, inputs, output_dimensions, inputs.front().getDataType());
    } else {
        output = Tensor(inputs.front().getPlacement(), TensorDescriptor(inputs.front().getDataType(), output_dimensions));
    }

    if (canUseDirectMatrixPath(plan)) {
        const EinsumExecutionPath path = plan.kind == EinsumPlanKind::BATCHED_GEMM ? EinsumExecutionPath::BATCHED_GEMM
                                                                                  : EinsumExecutionPath::GEMM;
        std::vector<std::shared_ptr<StampedMatmul>> matrix_batches;
        std::shared_ptr<CublasKernel> batched_matrix_kernel;
        std::optional<Tensor> matrix_workspace;
        if (path == EinsumExecutionPath::BATCHED_GEMM && plan.matrix_multiply->batch_count > 1) {
            StridedBatchedMatrixMultiplyStamp batched =
                stampDirectStridedBatchedMatrixMultiply(*plan.matrix_multiply, inputs, output);
            batched_matrix_kernel = std::move(batched.kernel);
            matrix_workspace = std::move(batched.workspace);
        } else {
            matrix_batches = stampDirectMatrixMultiply(*plan.matrix_multiply, inputs, output, stream);
        }
        return std::shared_ptr<StampedEinsum>(new StampedEinsum(std::move(plan),
                                                               inputs,
                                                               output,
                                                               stream,
                                                               path,
                                                               nullptr,
                                                               nullptr,
                                                               std::move(matrix_batches),
                                                               std::move(batched_matrix_kernel),
                                                               std::move(matrix_workspace)));
    }

    GenericStampedStages generic = stampGeneric(plan, inputs, output, stream);
    return std::shared_ptr<StampedEinsum>(new StampedEinsum(std::move(plan),
                                                           inputs,
                                                           output,
                                                           stream,
                                                           EinsumExecutionPath::GENERIC,
                                                           std::move(generic.preparation),
                                                           std::move(generic.reduction),
                                                           {},
                                                           nullptr,
                                                           std::nullopt));
}

}  // namespace

Einsum::Einsum(std::string equation) : equation(std::move(equation)) {
    if (this->equation.empty()) {
        throw std::invalid_argument("Einsum equation must not be empty.");
    }
}

std::shared_ptr<StampedEinsum> Einsum::stamp(const std::vector<Tensor>& inputs, const Stream& stream) const {
    return stampImpl(equation, inputs, std::nullopt, stream);
}

std::shared_ptr<StampedEinsum> Einsum::stamp(const std::vector<Tensor>& inputs,
                                             const Tensor& preallocated_output,
                                             const Stream& stream) const {
    return stampImpl(equation, inputs, preallocated_output, stream);
}

StampedEinsum::StampedEinsum(EinsumPlan plan,
                             std::vector<Tensor> inputs,
                             Tensor output,
                             const Stream& stream,
                             EinsumExecutionPath execution_path,
                             std::shared_ptr<StampedExecutionPlan> generic_preparation,
                             std::shared_ptr<StampedCubReduction> cub_reduction,
                             std::vector<std::shared_ptr<StampedMatmul>> matrix_batches,
                             std::shared_ptr<CublasKernel> batched_matrix_kernel,
                             std::optional<Tensor> matrix_workspace)
    : plan(std::move(plan)),
      inputs(std::move(inputs)),
      output(std::move(output)),
      stream(stream),
      execution_path(execution_path),
      generic_preparation(std::move(generic_preparation)),
      cub_reduction(std::move(cub_reduction)),
      matrix_batches(std::move(matrix_batches)),
      batched_matrix_kernel(std::move(batched_matrix_kernel)),
      matrix_workspace(std::move(matrix_workspace)) {}

void StampedEinsum::run() { runOn(stream); }

void StampedEinsum::runOn(Stream& run_stream) const {
    if (!run_stream.isInitialized()) {
        throw std::invalid_argument("StampedEinsum::runOn requires an initialized stream.");
    }
    if (run_stream.getGpuNum() != output.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("StampedEinsum::runOn stream must use the stamped output GPU.");
    }

    if (execution_path == EinsumExecutionPath::BATCHED_GEMM && batched_matrix_kernel) {
        const float alpha_one = 1.0f;
        const float beta_zero = 0.0f;
        CHECK_CUBLAS(batched_matrix_kernel->launchUncheckedPrevalidated(inputs.at(0),
                                                                        inputs.at(1),
                                                                        output,
                                                                        output,
                                                                        matrix_workspace,
                                                                        &alpha_one,
                                                                        &beta_zero,
                                                                        run_stream,
                                                                        CublasScalarPointerMode::Host));
        return;
    }

    if (execution_path == EinsumExecutionPath::GEMM || execution_path == EinsumExecutionPath::BATCHED_GEMM) {
        if (matrix_batches.size() != 1) {
            throw std::runtime_error("StampedEinsum non-strided matrix execution path requires exactly one stamped GEMM.");
        }
        matrix_batches.front()->runOn(run_stream);
        return;
    }

    if (generic_preparation) {
        generic_preparation->runOn(run_stream);
    }
    if (cub_reduction) {
        cub_reduction->runOn(run_stream);
    }
}

}  // namespace ThorImplementation
