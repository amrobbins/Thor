#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for GEMM lowering dtype tests.";                                  \
        }                                                                                                               \
    } while (false)

void expectPromotedLowPrecisionAddendStaysOutsideFp32Gemm(DataType storage_dtype) {
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor lhs(placement, TensorDescriptor(storage_dtype, {2, 3}));
    Tensor rhs(placement, TensorDescriptor(storage_dtype, {3, 4}));
    Tensor addend(placement, TensorDescriptor(storage_dtype, {2, 4}));
    Stream stream(0);

    const Expression lhs_expression = Expression::input("lhs", storage_dtype, storage_dtype);
    const Expression rhs_expression = Expression::input("rhs", storage_dtype, storage_dtype);

    // Generic fused kernels may read low-precision storage and expose it as an FP32
    // logical value. cuBLASLt cannot use that tensor as C for a regular FP32-output
    // GEMM, so matmul + addend must remain two backend stages.
    const Expression promoted_addend = Expression::input("addend", DataType::FP32, DataType::FP32);
    const Expression output =
        Expression::matmul(lhs_expression, rhs_expression, false, false, DataType::FP32, DataType::FP32) + promoted_addend;

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"output", output}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> inputs{{"lhs", lhs}, {"rhs", rhs}, {"addend", addend}};

    const std::shared_ptr<CompiledOutputs> compiled = equation.compileForInputs(inputs);
    size_t matmul_stage_count = 0;
    for (const CompiledExecutionStage& stage : compiled->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::Matmul) {
            continue;
        }
        ++matmul_stage_count;
        ASSERT_NE(stage.matmul, nullptr);
        EXPECT_EQ(stage.matmul->op, ExprOp::MATMUL)
            << "A promoted low-precision addend must not be absorbed as GEMM C.";
    }
    EXPECT_EQ(matmul_stage_count, 1u);

    EXPECT_NO_THROW({
        StampedExecutionPlan plan = equation.stamp(inputs, stream);
        (void)plan;
    });
}

void expectPromotedLowPrecisionOptimizerAddendStaysOutsideLowPrecisionGemm(DataType storage_dtype) {
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor lhs(placement, TensorDescriptor(storage_dtype, {2, 3}));
    Tensor rhs(placement, TensorDescriptor(storage_dtype, {3, 4}));
    Tensor weights(placement, TensorDescriptor(storage_dtype, {2, 4}));
    Stream stream(0);

    const Expression lhs_expression = Expression::input("lhs", storage_dtype, storage_dtype);
    const Expression rhs_expression = Expression::input("rhs", storage_dtype, storage_dtype);

    // Dense optimizer fusion intentionally exposes low-precision parameter storage
    // as an FP32 logical value so the update arithmetic is performed in FP32 before
    // the final low-precision store.  Even though C and D would both be physically
    // low precision, that promoted input is an explicit conversion boundary and must
    // not be swallowed by GEMM lowering.
    const Expression promoted_weights = Expression::input("weights", DataType::FP32, DataType::FP32);
    const Expression gradient =
        Expression::matmul(lhs_expression, rhs_expression, false, false, DataType::FP32, storage_dtype);
    const Expression output =
        (promoted_weights - Expression::constantScalar(0.125) * gradient).withOutputDType(storage_dtype);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"output", output}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> inputs{{"lhs", lhs}, {"rhs", rhs}, {"weights", weights}};

    const std::shared_ptr<CompiledOutputs> compiled = equation.compileForInputs(inputs);
    size_t matmul_stage_count = 0;
    for (const CompiledExecutionStage& stage : compiled->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::Matmul) {
            continue;
        }
        ++matmul_stage_count;
        ASSERT_NE(stage.matmul, nullptr);
        EXPECT_EQ(stage.matmul->op, ExprOp::MATMUL)
            << "A promoted low-precision optimizer addend must remain outside GEMM C.";
    }
    EXPECT_EQ(matmul_stage_count, 1u);

    EXPECT_NO_THROW({
        StampedExecutionPlan plan = equation.stamp(inputs, stream);
        (void)plan;
    });
}

}  // namespace

TEST(GemmLoweringDType, PromotedBf16AddendStaysOutsideFp32Gemm) {
    REQUIRE_CUDA_DEVICE();
    expectPromotedLowPrecisionAddendStaysOutsideFp32Gemm(DataType::BF16);
}

TEST(GemmLoweringDType, PromotedFp16AddendStaysOutsideFp32Gemm) {
    REQUIRE_CUDA_DEVICE();
    expectPromotedLowPrecisionAddendStaysOutsideFp32Gemm(DataType::FP16);
}

TEST(GemmLoweringDType, PromotedBf16OptimizerAddendStaysOutsideLowPrecisionGemm) {
    REQUIRE_CUDA_DEVICE();
    expectPromotedLowPrecisionOptimizerAddendStaysOutsideLowPrecisionGemm(DataType::BF16);
}

TEST(GemmLoweringDType, PromotedFp16OptimizerAddendStaysOutsideLowPrecisionGemm) {
    REQUIRE_CUDA_DEVICE();
    expectPromotedLowPrecisionOptimizerAddendStaysOutsideLowPrecisionGemm(DataType::FP16);
}

TEST(GemmLoweringDType, PackedRowMatmulKeepsBiasAndUnsupportedEpilogueInExpressionFusionTail) {
    REQUIRE_CUDA_DEVICE();
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor lhs(placement, TensorDescriptor(DataType::FP32, {66, 3}));
    Tensor rhs(placement, TensorDescriptor(DataType::FP32, {3, 4}));
    Tensor bias(placement, TensorDescriptor(DataType::FP32, {4}));

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression b = Expression::input("b", DataType::FP32, DataType::FP32);
    const Expression projected = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32, 66);
    const Expression output = (projected + b).swish();

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"output", output}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> inputs{{"x", lhs}, {"w", rhs}, {"b", bias}};
    const std::shared_ptr<CompiledOutputs> compiled = equation.compileForInputs(inputs);

    ASSERT_EQ(compiled->stages.size(), 2u);
    ASSERT_EQ(compiled->stages[0].kind, CompiledExecutionStage::Kind::Matmul);
    ASSERT_NE(compiled->stages[0].matmul, nullptr);
    EXPECT_EQ(compiled->stages[0].matmul->op, ExprOp::MATMUL);
    EXPECT_EQ(compiled->stages[0].matmul->packed_row_binding, MatmulPackedRowBinding::RowsA);
    EXPECT_EQ(compiled->stages[0].matmul->packed_row_capacity, 66u);
    EXPECT_EQ(compiled->stages[1].kind, CompiledExecutionStage::Kind::FusedKernel)
        << "bias + swish should stay in one normal Expression fusion tail after the bucketed matmul stage";
}
