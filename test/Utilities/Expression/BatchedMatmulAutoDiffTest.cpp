#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                         \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for Expression batched-matmul autodiff tests.";                 \
        }                                                                                                              \
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

void expectValues(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

Outputs batchedMatmulOutputs(bool transpose_lhs = false, bool transpose_rhs = false) {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression y = Expression::matmul(lhs, rhs, transpose_lhs, transpose_rhs, DataType::FP32, DataType::FP32);
    return Expression::outputs({{"y", y}});
}

struct BackwardRunResult {
    std::vector<float> lhs_grad;
    std::vector<float> rhs_grad;
    std::vector<std::string> stage_kinds;
};

BackwardRunResult runBackward(const Outputs& forward_outputs,
                              const Tensor& lhs,
                              const Tensor& rhs,
                              const Tensor& dy,
                              Stream& stream) {
    FusedEquation forward = FusedEquation::compile(forward_outputs.physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"lhs", "rhs"}, "dy");
    StampedExecutionPlan plan = backward.stamp({{"lhs", lhs}, {"rhs", rhs}, {"dy", dy}}, stream);
    const std::vector<std::string> stage_kinds = plan.stageKindNames();
    plan.run();

    return {
        .lhs_grad = copyToCpu(plan.output("lhs_grad"), stream),
        .rhs_grad = copyToCpu(plan.output("rhs_grad"), stream),
        .stage_kinds = stage_kinds,
    };
}

}  // namespace

TEST(ExpressionBatchedMatmulAutoDiff, ShapeSpecializationUsesMatmulsAndCentralReductionsForCrossBroadcast) {
    PhysicalOutputs forward = batchedMatmulOutputs().physicalOutputs();
    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"lhs", {2, 1, 2, 3}},
        {"rhs", {1, 3, 3, 2}},
    };

    PhysicalOutputs backward = buildBackwardOutputs(forward, {"lhs", "rhs"}, std::optional<std::string>{"dy"}, forward_input_dims);
    ASSERT_NE(backward.expr, nullptr);

    size_t matmul_count = 0;
    size_t reduce_sum_count = 0;
    for (const ExprNode& node : backward.expr->nodes) {
        matmul_count += node.op == ExprOp::MATMUL ? 1u : 0u;
        reduce_sum_count += node.op == ExprOp::REDUCE_SUM ? 1u : 0u;
    }

    EXPECT_EQ(matmul_count, 2u);
    EXPECT_EQ(reduce_sum_count, 2u);
}

TEST(ExpressionBatchedMatmulAutoDiff, DenseRegularBatchProducesExpectedOperandGradientsWithoutBroadcastReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({2, 3, 2},
                               {1, 0, 0, 1, 1, 1,
                                2, 1, 1, 0, 0, 1},
                               stream);
    Tensor dy = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    const BackwardRunResult result = runBackward(batchedMatmulOutputs(), lhs, rhs, dy, stream);

    expectValues(result.lhs_grad,
                 {1, 2, 3, 3, 4, 7,
                  16, 5, 6, 22, 7, 8});
    expectValues(result.rhs_grad,
                 {13, 18, 17, 24, 21, 30,
                  105, 122, 117, 136, 129, 150});
    EXPECT_EQ(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "Reduction"), 0);
}

TEST(ExpressionBatchedMatmulAutoDiff, WholeOperandBatchBroadcastSumsGradientBackToUnbatchedOperand) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 0, 0, 1, 1, 1}, stream);
    Tensor dy = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    const BackwardRunResult result = runBackward(batchedMatmulOutputs(), lhs, rhs, dy, stream);

    expectValues(result.lhs_grad,
                 {1, 2, 3, 3, 4, 7,
                  5, 6, 11, 7, 8, 15});
    expectValues(result.rhs_grad,
                 {118, 140,
                  134, 160,
                  150, 180});
    EXPECT_GE(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "Reduction"), 1);
}

TEST(ExpressionBatchedMatmulAutoDiff, IrregularCrossBroadcastUsesGroupedMatmulsAndSumsBothOperandGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 1, 2, 3},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18},
                               stream);
    Tensor dy = makeGpuTensor({2, 3, 2, 2},
                              {1, 2, 3, 4,
                               5, 6, 7, 8,
                               9, 10, 11, 12,
                               13, 14, 15, 16,
                               17, 18, 19, 20,
                               21, 22, 23, 24},
                              stream);

    const BackwardRunResult result = runBackward(batchedMatmulOutputs(), lhs, rhs, dy, stream);

    // dL/dlhs[b,0,m,k] = sum_{c,n} dY[b,c,m,n] * rhs[0,c,k,n]
    expectValues(result.lhs_grad,
                 {345, 411, 477,
                  435, 525, 615,
                  885, 1095, 1305,
                  975, 1209, 1443});

    // dL/drhs[0,c,k,n] = sum_{b,m} lhs[b,0,m,k] * dY[b,c,m,n]
    expectValues(result.rhs_grad,
                 {254, 276, 286, 312, 318, 348,
                  342, 364, 390, 416, 438, 468,
                  430, 452, 494, 520, 558, 588});

    EXPECT_GE(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "Matmul"), 4);
    EXPECT_GE(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "DependencyBarrier"), 2);
    EXPECT_GE(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "Reduction"), 2);
}

TEST(ExpressionBatchedMatmulAutoDiff, BatchedTransposeFlagsPreservePhysicalOperandGradientShapesAndBroadcastCollapse) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor lhs = makeGpuTensor({2, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor dy = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    const BackwardRunResult result = runBackward(batchedMatmulOutputs(true, true), lhs, rhs, dy, stream);

    expectValues(result.lhs_grad,
                 {9, 19, 12, 26, 15, 33,
                  29, 39, 40, 54, 51, 69});
    expectValues(result.rhs_grad,
                 {98, 130, 162,
                  116, 156, 196});
    EXPECT_GE(std::count(result.stage_kinds.begin(), result.stage_kinds.end(), "Reduction"), 1);
}
