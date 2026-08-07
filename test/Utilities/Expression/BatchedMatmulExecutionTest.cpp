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

}  // namespace

TEST(ExpressionBatchedMatmulExecution, DenseRegularBatchUsesOneStridedBatchedMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1}, stream);

    FusedEquation equation = compileMatmul();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
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
    plan.run();

    expectValues(copyToCpu(plan.output("y"), stream), {4, 5, 10, 11, 22, 16, 31, 22});
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

    Tensor lhs_storage = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor lhs = lhs_storage.aliasView({2, 2, 3}, {6, 1, 2});
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1}, stream);

    FusedEquation equation = compileMatmul();
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
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
