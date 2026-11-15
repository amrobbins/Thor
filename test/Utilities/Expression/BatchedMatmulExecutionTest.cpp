#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for Expression batched-matmul execution tests.";                  \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const std::vector<uint64_t>& dims) {
    uint64_t count = 1;
    for (uint64_t dim : dims) {
        count *= dim;
    }
    return count;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    if (numel(dims) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpu(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(gpu.getDataType(), gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(numel(cpu.getDimensions()));
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

void expectValues(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.0e-5f) << "index " << i;
    }
}

FusedEquation compileMatmul(bool transpose_lhs = false, bool transpose_rhs = false) {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(lhs, rhs, transpose_lhs, transpose_rhs, DataType::FP32, DataType::FP32);
    return FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
}

FusedEquation compileMatmulPlusOne() {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP32);
    const Expression z = y + Expression::constantScalar(1.0);
    return FusedEquation::compile(Expression::outputs({{"z", z}}).physicalOutputs(), 0);
}

FusedEquation compileGroupedMatmulCudaKernelPlusOne() {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression matmul = Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP32);

    auto identity = CudaKernelExpression::builder("grouped_matmul_cuda_identity")
                        .source(R"cuda(
extern "C" __global__
void grouped_matmul_cuda_identity_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}
)cuda")
                        .entry("grouped_matmul_cuda_identity_kernel")
                        .input("x", DataType::FP32)
                        .outputLike("y", DataType::FP32, "x")
                        .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                        .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                        .build();

    Outputs cuda_outputs = identity.apply({{"x", matmul}});
    if (cuda_outputs.outputNames().size() != 1) {
        throw std::runtime_error("Expected one CudaKernel output.");
    }
    const Expression cuda_y = cuda_outputs.outputExpression(0);
    const Expression z = cuda_y + Expression::constantScalar(1.0);
    return FusedEquation::compile(Expression::outputs({{"z", z}}).physicalOutputs(), 0);
}

}  // namespace

TEST(ExpressionBatchedMatmulExecution, DenseRegularBatchUsesOneStridedBatchedMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream streamA(0);
    Stream streamB(0);

    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, streamA);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1}, streamA);

    FusedEquation equation = compileMatmul();
    auto& cublas = CublasMatrixMultiply::instance();
    cublas.clearOptimalKernelSelectionCacheForTests();
    ASSERT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), 0u);

    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, streamA);
    const size_t selectionCountAfterFirstStamp = cublas.cachedOptimalKernelSelectionCountForTests();
    EXPECT_GT(selectionCountAfterFirstStamp, 0u);
    ASSERT_EQ(plan.output("y").getDimensions(), (std::vector<uint64_t>{2, 2, 2}));
    EXPECT_EQ(plan.flopCount(), 48u);
    const std::vector<StampedMatmulStageDiagnostic> diagnostics = plan.matmulStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 1u);
    EXPECT_EQ(diagnostics[0].stage_index, 0u);
    EXPECT_EQ(diagnostics[0].lane_index, 0u);
    EXPECT_EQ(diagnostics[0].dependency_count, 0u);
    EXPECT_EQ(diagnostics[0].kernel.m, 2);
    EXPECT_EQ(diagnostics[0].kernel.n, 2);
    EXPECT_EQ(diagnostics[0].kernel.k, 3);
    EXPECT_EQ(diagnostics[0].kernel.batch_count, 2);
    EXPECT_EQ(diagnostics[0].kernel.flop_count, 48u);
    EXPECT_TRUE(diagnostics[0].kernel.has_measured_kernel);
    EXPECT_GT(diagnostics[0].kernel.waves_count, 0.0f);
    EXPECT_GT(diagnostics[0].kernel.picker_runtime_ms, 0.0);
    EXPECT_GE(diagnostics[0].kernel.algorithm_id, 0);
    EXPECT_NE(diagnostics[0].kernel.execution_state_id, 0u);

    // C9: an equivalent independently stamped expression may reuse the immutable
    // measured selection, but it must materialize its own descriptor-bearing kernel.
    StampedExecutionPlan second_plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, streamB);
    const std::vector<StampedMatmulStageDiagnostic> second_diagnostics = second_plan.matmulStageDiagnostics();
    ASSERT_EQ(second_diagnostics.size(), 1u);
    EXPECT_NE(second_diagnostics[0].kernel.execution_state_id, 0u);
    EXPECT_NE(second_diagnostics[0].kernel.execution_state_id, diagnostics[0].kernel.execution_state_id);
    EXPECT_EQ(second_diagnostics[0].kernel.algorithm_id, diagnostics[0].kernel.algorithm_id);
    EXPECT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), selectionCountAfterFirstStamp);
    if (diagnostics[0].kernel.workspace_bytes > 0) {
        EXPECT_NE(diagnostics[0].kernel.workspace_state_id, 0u);
        EXPECT_NE(second_diagnostics[0].kernel.workspace_state_id, 0u);
        EXPECT_NE(second_diagnostics[0].kernel.workspace_state_id, diagnostics[0].kernel.workspace_state_id);
    }

    const uint64_t materializationsAfterStamping = CublasKernel::materializationCountForTests();
    cublas.clearOptimalKernelSelectionCacheForTests();
    ASSERT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), 0u);

    for (int repetition = 0; repetition < 8; ++repetition) {
        plan.run();
        second_plan.run();
    }
    streamA.synchronize();
    streamB.synchronize();

    EXPECT_EQ(CublasKernel::materializationCountForTests(), materializationsAfterStamping)
        << "Ordinary cuBLASLt runtime must not materialize descriptor-bearing kernels.";
    EXPECT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), 0u)
        << "Ordinary cuBLASLt runtime must not consult/repopulate the global selection cache.";

    expectValues(copyToCpu(plan.output("y"), streamA), {4, 5, 10, 11, 22, 16, 31, 22});
    expectValues(copyToCpu(second_plan.output("y"), streamB), {4, 5, 10, 11, 22, 16, 31, 22});
}

TEST(ExpressionBatchedMatmulExecution, WholeOperandBatchBroadcastUsesZeroStride) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 0, 0, 1, 1, 1}, stream);

    FusedEquation equation = compileMatmul();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    plan.run();

    expectValues(copyToCpu(plan.output("y"), stream), {4, 5, 10, 11, 16, 17, 22, 23});
}

TEST(ExpressionBatchedMatmulExecution, LogicalTransposeAppliesOnlyToFinalMatrixAxes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1}, stream);

    FusedEquation equation = compileMatmul(true, false);
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    plan.run();

    expectValues(copyToCpu(plan.output("y"), stream), {6, 8, 8, 10, 23, 18, 26, 20});
}

TEST(ExpressionBatchedMatmulExecution, PhysicallyTransposedMatrixViewUsesBackendTransposeWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // lhs is the transpose of each stored [3, 2] matrix, taken inside the expression (a span of the stored tensor).
    Tensor lhs_storage = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1}, stream);

    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32).stridedView({2, 2, 3}, {6, 1, 2}, 0);
    const Expression y = Expression::matmul(
        lhs, Expression::input("rhs", DataType::FP32, DataType::FP32), false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs_storage}, {"rhs", rhs}}, stream);
    EXPECT_EQ(plan.stageKindNames(), (std::vector<std::string>{"Matmul"}));
    plan.run();

    expectValues(copyToCpu(plan.output("y"), stream), {6, 8, 8, 10, 23, 18, 26, 20});
}

TEST(ExpressionBatchedMatmulExecution, IrregularCrossBroadcastLowersToIndependentStridedBatchedGroupsWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs =
        makeGpuTensor({2, 1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);

    FusedEquation equation = compileMatmul();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);

    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier"}));
    EXPECT_EQ(plan.stageFlopCounts(), (std::vector<uint64_t>{72, 72, 0}));
    EXPECT_EQ(plan.flopCount(), 144u);

    plan.run();
    expectValues(copyToCpu(plan.output("y"), stream),
                 {22, 28, 49, 64,
                  58, 64, 139, 154,
                  94, 100, 229, 244,
                  76, 100, 103, 136,
                  220, 244, 301, 334,
                  364, 388, 499, 532});
}

TEST(ExpressionBatchedMatmulExecution, GroupedMatmulReadsAStridedOperandThroughSpansWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // lhs [2, 1, 2, 3] x rhs [1, 3, 3, 2] cross-broadcasts into two strided-batched groups. rhs is a transposed view
    // (each 3x2 matrix stored as 2x3), as an einsum operand permutation produces: each group's rhs is a span of that
    // strided tensor's storage, read with backend transpose.
    const std::vector<float> lhs_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> rhs_logical(18);
    for (size_t i = 0; i < rhs_logical.size(); ++i) rhs_logical[i] = static_cast<float>(i + 1);
    std::vector<float> rhs_stored(18);
    for (uint64_t c = 0; c < 3; ++c)
        for (uint64_t k = 0; k < 3; ++k)
            for (uint64_t n = 0; n < 2; ++n) rhs_stored[c * 6 + n * 3 + k] = rhs_logical[(c * 3 + k) * 2 + n];

    Tensor lhs = makeGpuTensor({2, 1, 2, 3}, lhs_values, stream);
    Tensor rhs_storage = makeGpuTensor({1, 3, 2, 3}, rhs_stored, stream);

    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32).stridedView({1, 3, 3, 2}, {18, 6, 1, 3}, 0);
    const Expression y = Expression::matmul(
        Expression::input("lhs", DataType::FP32, DataType::FP32), rhs, false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs_storage}}, stream);
    EXPECT_EQ(plan.stageKindNames(), (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier"}));
    plan.run();

    std::vector<float> expected(24, 0.0f);
    for (uint64_t b = 0; b < 2; ++b)
        for (uint64_t c = 0; c < 3; ++c)
            for (uint64_t m = 0; m < 2; ++m)
                for (uint64_t n = 0; n < 2; ++n)
                    for (uint64_t k = 0; k < 3; ++k)
                        expected[((b * 3 + c) * 2 + m) * 2 + n] += lhs_values[(b * 2 + m) * 3 + k] * rhs_logical[(c * 3 + k) * 2 + n];
    expectValues(copyToCpu(plan.output("y"), stream), expected);
}

TEST(ExpressionBatchedMatmulExecution, ExpressionStridedViewOperandIsReadInPlaceByThePlainMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // lhs is an expression strided view (the transpose of a stored [3, 2] matrix): the matmul stage receives it as a
    // span of x and reads it with backend transpose, without a view or a copy.
    Tensor x = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 0, 0, 1, 1, 1}, stream);

    const Expression lhs_view = Expression::input("x", DataType::FP32, DataType::FP32).stridedView({2, 3}, {1, 2}, 0);
    const Expression rhs_input = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(lhs_view, rhs_input, false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"rhs", rhs}}, stream);
    EXPECT_EQ(plan.stageKindNames(), (std::vector<std::string>{"Matmul"}));
    plan.run();

    // lhs = [[1, 3, 5], [2, 4, 6]].
    expectValues(copyToCpu(plan.output("y"), stream), {6, 8, 8, 10});
}

TEST(ExpressionBatchedMatmulExecution, StridedOperandViewingARankOneSourceIsReadInPlaceByThePlainMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // lhs is a [2, 3] strided view of a [6] vector: the operand's rank is 2 although its source's is 1.
    Tensor x = makeGpuTensor({6}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 0, 0, 1, 1, 1}, stream);

    const Expression lhs_view = Expression::input("x", DataType::FP32, DataType::FP32).stridedView({2, 3}, {1, 2}, 0);
    const Expression rhs_input = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(lhs_view, rhs_input, false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"rhs", rhs}}, stream);
    EXPECT_EQ(plan.stageKindNames(), (std::vector<std::string>{"Matmul"}));
    plan.run();

    // lhs = [[1, 3, 5], [2, 4, 6]].
    expectValues(copyToCpu(plan.output("y"), stream), {6, 8, 8, 10});
}

TEST(ExpressionBatchedMatmulExecution, GluHalvesOfAMatmulOutputAreReadInPlaceByTheFusedStage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // y = a @ b is [2, 4]; value = y[:, 0:2] and gate = y[:, 2:4] are strided views of the matmul's output, read by
    // the fused multiply through their row stride 4.
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({3, 4}, {1, 0, 2, 1, 0, 1, 1, 2, 1, 1, 0, 1}, stream);

    const Expression ea = Expression::input("a", DataType::FP32, DataType::FP32);
    const Expression eb = Expression::input("b", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(ea, eb, false, false, DataType::FP32, DataType::FP32);
    const Expression value = y.stridedView({2, 2}, {4, 1}, 0);
    const Expression gate = y.stridedView({2, 2}, {4, 1}, 2);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"z", value * gate}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"a", a}, {"b", b}}, stream);
    plan.run();

    // y = [[4, 5, 4, 8], [10, 11, 13, 20]]; z = value * gate.
    expectValues(copyToCpu(plan.output("z"), stream), {16, 40, 130, 220});
}

TEST(ExpressionBatchedMatmulExecution, GroupedMatmulDependencyBarrierFeedsDownstreamExpressionStage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs =
        makeGpuTensor({2, 1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);

    FusedEquation equation = compileMatmulPlusOne();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier", "FusedKernel"}));

    plan.run();
    expectValues(copyToCpu(plan.output("z"), stream),
                 {23, 29, 50, 65,
                  59, 65, 140, 155,
                  95, 101, 230, 245,
                  77, 101, 104, 137,
                  221, 245, 302, 335,
                  365, 389, 500, 533});
}

TEST(ExpressionBatchedMatmulExecution, CudaKernelProducerUsesPhysicalStageIndexAfterGroupedMatmulExpansion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs =
        makeGpuTensor({2, 1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);

    FusedEquation equation = compileGroupedMatmulCudaKernelPlusOne();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier", "CudaKernel", "FusedKernel"}));

    const std::vector<std::vector<uint32_t>> dependencies = plan.stageDependencyIndices();
    ASSERT_EQ(dependencies.size(), 5u);
    EXPECT_EQ(dependencies[2], (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(dependencies[3], (std::vector<uint32_t>{2}));
    EXPECT_EQ(dependencies[4], (std::vector<uint32_t>{3}));

    plan.run();
    expectValues(copyToCpu(plan.output("z"), stream),
                 {23, 29, 50, 65,
                  59, 65, 140, 155,
                  95, 101, 230, 245,
                  77, 101, 104, 137,
                  221, 245, 302, 335,
                  365, 389, 500, 533});
}

TEST(ExpressionBatchedMatmulExecution, GroupedMatmulWritesDirectlyIntoPreallocatedOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs =
        makeGpuTensor({2, 1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 2, 2}));

    FusedEquation equation = compileMatmul();
    StampedExecutionPlan plan =
        equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream, {}, {{"y", output}});
    plan.run();

    EXPECT_EQ(plan.output("y").getMemPtr(), output.getMemPtr());
    expectValues(copyToCpu(output, stream),
                 {22, 28, 49, 64,
                  58, 64, 139, 154,
                  94, 100, 229, 244,
                  76, 100, 103, 136,
                  220, 244, 301, 334,
                  364, 388, 499, 532});
}
