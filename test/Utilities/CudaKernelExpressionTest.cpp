#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"
#include <algorithm>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t d : tensor.getDimensions())
        n *= d;
    return n;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    if (numel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

std::vector<float> copyToCpuValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(numel(cpu));
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.0e-5f) << "index " << i;
    }
}

}  // namespace

TEST(CudaKernelExpression, SingleOutputRawPointerKernelRunsAsCustomStage) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("scale")
                  .source(R"cuda(
extern "C" __global__
void scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("scale_kernel")
                  .input("x", DataType::FP32)
                  .output("y", DataType::FP32, {CudaKernelExpression::DimExpr::dim("x", 0), CudaKernelExpression::DimExpr::dim("x", 1)})
                  .scalar("alpha", DataType::FP32, 2.5f)
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    auto plan = op.asDynamicExpression().stamp({{"x", x}}, {}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor y = plan.output("y");
    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), {2.5f, -5.0f, 7.5f, 11.25f, -12.5f, 15.0f});
}

TEST(CudaKernelExpression, MultiOutputKernelReturnsNamedOutputs) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}, stream);

    auto op = CudaKernelExpression::builder("split_math")
                  .source(R"cuda(
extern "C" __global__
void split_math_kernel(const float* x, float* twice, float* plus_one, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        twice[i] = 2.0f * x[i];
        plus_one[i] = x[i] + 1.0f;
    }
}
)cuda")
                  .entry("split_math_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("twice", DataType::FP32, "x")
                  .outputLike("plus_one", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("twice"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 64;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("twice") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    auto plan = op.stamp({{"x", x}}, {}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor twice = plan.output("twice");
    Tensor plusOne = plan.output("plus_one");
    EXPECT_EQ(twice.getDimensions(), (std::vector<uint64_t>{2, 2, 2}));
    EXPECT_EQ(plusOne.getDimensions(), (std::vector<uint64_t>{2, 2, 2}));

    expectNear(copyToCpuValues(twice, stream), {2.0f, 4.0f, 6.0f, 8.0f, -2.0f, -4.0f, -6.0f, -8.0f});
    expectNear(copyToCpuValues(plusOne, stream), {2.0f, 3.0f, 4.0f, 5.0f, 0.0f, -1.0f, -2.0f, -3.0f});
}

TEST(CudaKernelExpression, RejectsInputDTypeMismatchBeforeLaunch) {
    Stream stream(0);
    Tensor x = makeGpuTensor({4}, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    auto op = CudaKernelExpression::builder("reject_dtype")
                  .source(R"cuda(
extern "C" __global__
void reject_dtype_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("reject_dtype_kernel")
                  .input("x", DataType::FP16)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      return CudaKernelLaunchConfig{dim3(static_cast<uint32_t>(ctx.numel("y")), 1, 1), dim3(1, 1, 1), 0};
                  })
                  .build();

    EXPECT_THROW((void)op.stamp({{"x", x}}, {}, stream), std::runtime_error);
}
TEST(CudaKernelExpression, TensorRuntimeScalarInputPassesDevicePointerThroughStagedPath) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);
    Tensor alphaBuffer = makeGpuTensor({2}, {123.0f, -1.5f}, stream);

    auto op = CudaKernelExpression::builder("runtime_scalar_scale")
                  .source(R"cuda(
extern "C" __global__
void runtime_scalar_scale_kernel(const float* x, const float* alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (*alpha) * x[i];
    }
}
)cuda")
                  .entry("runtime_scalar_scale_kernel")
                  .input("x", DataType::FP32)
                  .tensorRuntimeScalarInput("alpha", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({
        {"x", Expression::input("x")},
        {"alpha", Expression::tensorRuntimeScalar("alpha", DataType::FP32, DataType::FP32)},
    });
    FusedEquation eq = FusedEquation::compile(outputs.physicalOutputs(), 0);

    TensorScalarBinding alphaBinding{alphaBuffer, sizeof(float), DataType::FP32};
    auto plan = eq.stamp({{"x", x}}, stream, {{"alpha", alphaBinding}});
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor y = plan.output("y");
    expectNear(copyToCpuValues(y, stream), {-1.5f, 3.0f, -4.5f, -6.75f, 7.5f, -9.0f});
}

TEST(CudaKernelExpression, HostRuntimeScalarInputPassesByValueThroughStagedPath) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("host_runtime_scalar_scale")
                  .source(R"cuda(
extern "C" __global__
void host_runtime_scalar_scale_kernel(const float* x, float alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("host_runtime_scalar_scale_kernel")
                  .input("x", DataType::FP32)
                  .hostRuntimeScalarInput("alpha", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({
        {"x", Expression::input("x")},
        {"alpha", Expression::runtimeScalar("alpha", DataType::FP32, DataType::FP32)},
    });
    FusedEquation eq = FusedEquation::compile(outputs.physicalOutputs(), 0);

    auto plan = eq.stamp({{"x", x}}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});

    EXPECT_THROW(plan.run(), std::runtime_error);
    plan.run({{"alpha", -2.0f}});

    Tensor y = plan.output("y");
    expectNear(copyToCpuValues(y, stream), {-2.0f, 4.0f, -6.0f, -9.0f, 10.0f, -12.0f});
}

TEST(CudaKernelExpression, ConditionalHostRuntimeScalarInputUpdatesCapturedKernelNode) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);
    Tensor predicateValue = makeGpuTensor({1}, {1.0f}, stream);

    auto op = CudaKernelExpression::builder("conditional_host_runtime_scalar_scale")
                  .source(R"cuda(
extern "C" __global__
void conditional_host_runtime_scalar_scale_kernel(const float* x, float alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("conditional_host_runtime_scalar_scale_kernel")
                  .input("x", DataType::FP32)
                  .hostRuntimeScalarInput("alpha", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Expression xExpr = Expression::input("x");
    Outputs cudaBranch = op.apply({
        {"x", xExpr},
        {"alpha", Expression::runtimeScalar("alpha", DataType::FP32, DataType::FP32)},
    });
    Expression predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs conditional = Outputs::conditional(
        predicate,
        cudaBranch,
        Expression::outputs({{"y", xExpr - Expression::constantScalar(1.0)}}));

    FusedEquation eq = FusedEquation::compile(conditional.physicalOutputs(), 0);
    auto plan = eq.stamp({{"x", x}, {"predicate_value", predicateValue}}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"Conditional"});
    EXPECT_EQ(plan.runtimeScalarNames(), (std::unordered_set<std::string>{"alpha"}));

    plan.run({{"alpha", 2.0f}});
    expectNear(copyToCpuValues(plan.output("y"), stream), {2.0f, -4.0f, 6.0f, 9.0f, -10.0f, 12.0f});
    plan.run({{"alpha", -3.0f}});
    expectNear(copyToCpuValues(plan.output("y"), stream), {-3.0f, 6.0f, -9.0f, -13.5f, 15.0f, -18.0f});
}

TEST(CudaKernelExpression, HostRuntimeScalarInputRejectsNonFp32DType) {
    EXPECT_THROW((void)CudaKernelExpression::builder("host_runtime_scalar_dtype_reject")
                     .source(R"cuda(
extern "C" __global__
void host_runtime_scalar_dtype_reject_kernel(const float* x, float alpha, float* y, int64_t n) {}
)cuda")
                     .entry("host_runtime_scalar_dtype_reject_kernel")
                     .input("x", DataType::FP32)
                     .hostRuntimeScalarInput("alpha", DataType::INT64),
                 std::invalid_argument);
}



TEST(CudaKernelExpression, ExplicitBackwardKernelMatchesAnalyticVectorJacobianProduct) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);
    Tensor dy = makeGpuTensor({2, 3}, {0.5f, 1.0f, -2.0f, 3.0f, -0.25f, 4.0f}, stream);

    auto backward = CudaKernelExpression::builder("square_backward")
                        .source(R"cuda(
extern "C" __global__
void square_backward_kernel(const float* x, const float* dy, float* dx, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = 2.0f * x[i] * dy[i];
    }
}
)cuda")
                        .entry("square_backward_kernel")
                        .input("x", DataType::FP32)
                        .input("dy", DataType::FP32)
                        .outputLike("dx", DataType::FP32, "x")
                        .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("dx"))
                        .launchGrid1D(CudaKernelExpression::DimExpr::numel("dx"), 128)
                        .build();

    auto square = CudaKernelExpression::builder("square")
                      .source(R"cuda(
extern "C" __global__
void square_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * x[i];
    }
}
)cuda")
                      .entry("square_kernel")
                      .input("x", DataType::FP32)
                      .outputLike("y", DataType::FP32, "x")
                      .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                      .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                      .backward("y", std::move(backward), "dy", {{"dx", "x"}})
                      .build();

    Outputs outputs = square.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    FusedEquation forward = FusedEquation::compile(outputs.physicalOutputs(), 0);
    FusedEquation backward_equation = forward.compileBackward({"x"}, "dy");
    auto plan = backward_equation.stamp({{"x", x}, {"dy", dy}}, stream);
    const std::vector<std::string> stage_kinds = plan.stageKindNames();
    EXPECT_NE(std::find(stage_kinds.begin(), stage_kinds.end(), "CudaKernel"), stage_kinds.end());
    plan.run();

    expectNear(copyToCpuValues(plan.output("x_grad"), stream),
               {1.0f, -4.0f, -12.0f, 27.0f, 2.5f, 48.0f});
}


TEST(CudaKernelExpression, ConditionalCompileBackwardExecutesExplicitCudaVjpOnSelectedBranch) {
    Stream stream(0);

    auto backward = CudaKernelExpression::builder("conditional_square_backward")
                        .source(R"cuda(
extern "C" __global__
void conditional_square_backward_kernel(const float* x, const float* dy, float* dx, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = 2.0f * x[i] * dy[i];
    }
}
)cuda")
                        .entry("conditional_square_backward_kernel")
                        .input("x", DataType::FP32)
                        .input("dy", DataType::FP32)
                        .outputLike("dx", DataType::FP32, "x")
                        .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("dx"))
                        .launchGrid1D(CudaKernelExpression::DimExpr::numel("dx"), 128)
                        .build();

    auto square = CudaKernelExpression::builder("conditional_square")
                      .source(R"cuda(
extern "C" __global__
void conditional_square_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * x[i];
    }
}
)cuda")
                      .entry("conditional_square_kernel")
                      .input("x", DataType::FP32)
                      .outputLike("y", DataType::FP32, "x")
                      .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                      .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                      .backward("y", std::move(backward), "dy", {{"dx", "x"}})
                      .build();

    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto predicate_value = Expression::input("predicate_value", DataType::FP32, DataType::FP32);
    Outputs cuda_branch = square.apply({{"x", x}});
    Outputs ordinary_branch =
        Expression::outputs({{"y", x * Expression::constantScalar(3.0)}});
    Outputs conditional = Outputs::conditional(
        predicate_value.greaterThan(Expression::constantScalar(0.0)),
        cuda_branch,
        ordinary_branch);

    FusedEquation backward_equation =
        FusedEquation::compile(conditional.physicalOutputs(), 0).compileBackward({"x"}, "dy");

    Tensor x_tensor = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);
    Tensor dy = makeGpuTensor({2, 3}, {0.5f, 1.0f, -2.0f, 3.0f, -0.25f, 4.0f}, stream);
    Tensor positive = makeGpuTensor({1}, {1.0f}, stream);
    Tensor negative = makeGpuTensor({1}, {-1.0f}, stream);

    StampedExecutionPlan cuda_plan = backward_equation.stamp(
        {{"x", x_tensor}, {"predicate_value", positive}, {"dy", dy}}, stream);
    cuda_plan.run();
    expectNear(copyToCpuValues(cuda_plan.output("x_grad"), stream),
               {1.0f, -4.0f, -12.0f, 27.0f, 2.5f, 48.0f});

    StampedExecutionPlan ordinary_plan = backward_equation.stamp(
        {{"x", x_tensor}, {"predicate_value", negative}, {"dy", dy}}, stream);
    ordinary_plan.run();
    expectNear(copyToCpuValues(ordinary_plan.output("x_grad"), stream),
               {1.5f, 3.0f, -6.0f, 9.0f, -0.75f, 12.0f});
}

TEST(CudaKernelExpression, MissingExplicitBackwardIsRejectedOnlyWhenCudaOutputParticipatesInBackpropagation) {
    auto forward_only = CudaKernelExpression::builder("forward_only_scale")
                            .source(R"cuda(
extern "C" __global__
void forward_only_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
)cuda")
                            .entry("forward_only_scale_kernel")
                            .input("x", DataType::FP32)
                            .outputLike("y", DataType::FP32, "x")
                            .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                            .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                            .build();

    Outputs outputs = forward_only.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    FusedEquation forward = FusedEquation::compile(outputs.physicalOutputs(), 0);
    EXPECT_THROW((void)forward.compileBackward({"x"}, "dy"), std::runtime_error);
}

TEST(CudaKernelExpression, ExplicitBackwardCudaSourceIsEncryptedSignedAndTamperDetected) {
    auto backward = CudaKernelExpression::builder("signed_backward")
                        .source(R"cuda(
extern "C" __global__
void signed_backward_kernel(const float* x, const float* dy, float* dx, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = 3.0f * dy[i];
}
)cuda")
                        .entry("signed_backward_kernel")
                        .input("x", DataType::FP32)
                        .input("dy", DataType::FP32)
                        .outputLike("dx", DataType::FP32, "x")
                        .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("dx"))
                        .launchGrid1D(CudaKernelExpression::DimExpr::numel("dx"), 128)
                        .build();
    auto forward = CudaKernelExpression::builder("signed_forward")
                       .source(R"cuda(
extern "C" __global__
void signed_forward_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 3.0f * x[i];
}
)cuda")
                       .entry("signed_forward_kernel")
                       .input("x", DataType::FP32)
                       .outputLike("y", DataType::FP32, "x")
                       .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                       .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                       .backward("y", std::move(backward), "dy", {{"dx", "x"}})
                       .build();

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        forward.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    nlohmann::json protected_json = definition.architectureJsonWithCudaKernelManifestSignature();

    ASSERT_TRUE(protected_json.contains("cuda_kernels"));
    ASSERT_EQ(protected_json.at("cuda_kernels").size(), 1u);
    const auto& forward_json = protected_json.at("cuda_kernels").at(0);
    ASSERT_TRUE(forward_json.contains("encrypted_source"));
    ASSERT_FALSE(forward_json.contains("source"));
    ASSERT_TRUE(forward_json.contains("backward"));
    ASSERT_EQ(forward_json.at("backward").size(), 1u);
    const auto& backward_json = forward_json.at("backward").at(0).at("kernel");
    EXPECT_TRUE(backward_json.contains("encrypted_source"));
    EXPECT_FALSE(backward_json.contains("source"));

    const auto in_memory_sources = definition.cudaKernelSourceInfo();
    ASSERT_EQ(in_memory_sources.size(), 2u);
    EXPECT_EQ(in_memory_sources[0].name, "signed_forward");
    EXPECT_EQ(in_memory_sources[1].name, "signed_backward");

    const auto protected_sources = collectCudaKernelSourceInfo(protected_json);
    ASSERT_EQ(protected_sources.size(), 2u);
    EXPECT_TRUE(protected_sources[0].source_encrypted);
    EXPECT_TRUE(protected_sources[1].source_encrypted);
    EXPECT_FALSE(protected_sources[0].signature.empty());
    EXPECT_EQ(protected_sources[1].signature, protected_sources[0].signature);

    const auto keys = collectCudaKernelOutOfBandKeys(protected_json);
    ASSERT_EQ(keys.size(), 1u);
    ExpressionDefinition loaded = ExpressionDefinition::deserialize(
        protected_json, true, keys.at(0).signing_public_key, keys.at(0).source_decryption_key);

    Stream stream(0);
    Tensor x = makeGpuTensor({2, 2}, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    Tensor dy = makeGpuTensor({2, 2}, {0.5f, 1.0f, -2.0f, 4.0f}, stream);
    FusedEquation loaded_forward = FusedEquation::compile(loaded.outputs, 0);
    FusedEquation loaded_backward = loaded_forward.compileBackward({"x"}, "dy");
    auto loaded_plan = loaded_backward.stamp({{"x", x}, {"dy", dy}}, stream);
    loaded_plan.run();
    expectNear(copyToCpuValues(loaded_plan.output("x_grad"), stream), {1.5f, 3.0f, -6.0f, 12.0f});

    nlohmann::json tampered = protected_json;
    std::string ciphertext = tampered.at("cuda_kernels").at(0).at("backward").at(0).at("kernel").at("encrypted_source").get<std::string>();
    ASSERT_FALSE(ciphertext.empty());
    ciphertext.back() = ciphertext.back() == '0' ? '1' : '0';
    tampered["cuda_kernels"][0]["backward"][0]["kernel"]["encrypted_source"] = ciphertext;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(
                     tampered, false, keys.at(0).signing_public_key, keys.at(0).source_decryption_key),
                 std::runtime_error);
}

TEST(CudaKernelExpression, ConditionalExpressionDefinitionProtectsAndRoundTripsCudaKernelBranches) {
    auto op = CudaKernelExpression::builder("conditional_identity")
                  .source(R"cuda(
extern "C" __global__
void conditional_identity_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}
)cuda")
                  .entry("conditional_identity_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    Outputs cuda_branch = op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs ordinary_branch = Expression::outputs({{"y", Expression::input("x")}});
    Outputs conditional = Outputs::conditional(predicate, cuda_branch, ordinary_branch);

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(conditional);
    EXPECT_TRUE(definition.hasCudaKernelExpressions());
    std::vector<CudaKernelSourceInspection> sourceInfo = definition.cudaKernelSourceInfo();
    ASSERT_EQ(sourceInfo.size(), 1u);
    EXPECT_EQ(sourceInfo.front().name, "conditional_identity");

    nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();
    EXPECT_FALSE(payload.contains("cuda_kernel_manifest_signature"));
    const nlohmann::json& thenPayload = payload.at("conditional").at("then_branch");
    ASSERT_TRUE(thenPayload.contains("cuda_kernels"));
    ASSERT_TRUE(thenPayload.contains("cuda_kernel_manifest_signature"));
    ASSERT_EQ(thenPayload.at("cuda_kernels").size(), 1u);
    EXPECT_FALSE(thenPayload.at("cuda_kernels").at(0).contains("source"));
    EXPECT_TRUE(thenPayload.at("cuda_kernels").at(0).contains("encrypted_source"));

    EXPECT_EQ(definition.cudaKernelSigningPublicKeys().size(), 1u);
    EXPECT_EQ(definition.cudaKernelOutOfBandKeys().size(), 1u);

    std::vector<CudaKernelOutOfBandKeys> keys = collectCudaKernelOutOfBandKeys(payload);
    ASSERT_EQ(keys.size(), 1u);
    ASSERT_FALSE(keys.front().signing_public_key.empty());
    ASSERT_FALSE(keys.front().source_decryption_key.empty());

    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload), std::runtime_error);

    ExpressionDefinition loaded = ExpressionDefinition::deserialize(
        payload, false, keys.front().signing_public_key, keys.front().source_decryption_key);
    EXPECT_TRUE(loaded.outputs.isConditional());
    EXPECT_TRUE(loaded.hasCudaKernelExpressions());
    std::vector<CudaKernelSourceInspection> loadedSourceInfo = loaded.cudaKernelSourceInfo();
    ASSERT_EQ(loadedSourceInfo.size(), 1u);
    EXPECT_EQ(loadedSourceInfo.front().name, "conditional_identity");
    EXPECT_FALSE(loadedSourceInfo.front().loaded_source_compilation_allowed);

    nlohmann::json tamperedSource = payload;
    std::string encryptedSource = tamperedSource["conditional"]["then_branch"]["cuda_kernels"][0]["encrypted_source"].get<std::string>();
    ASSERT_FALSE(encryptedSource.empty());
    encryptedSource.back() = encryptedSource.back() == '0' ? '1' : '0';
    tamperedSource["conditional"]["then_branch"]["cuda_kernels"][0]["encrypted_source"] = encryptedSource;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(
                     tamperedSource, true, keys.front().signing_public_key, keys.front().source_decryption_key),
                 std::runtime_error);

    nlohmann::json tamperedLaunch = payload;
    tamperedLaunch["conditional"]["then_branch"]["cuda_kernels"][0]["launch"]["block_size"] = 256;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(
                     tamperedLaunch, true, keys.front().signing_public_key, keys.front().source_decryption_key),
                 std::runtime_error);

    nlohmann::json unsignedNestedKernel = payload;
    unsignedNestedKernel["conditional"]["then_branch"].erase("cuda_kernel_manifest_signature");
    EXPECT_THROW((void)ExpressionDefinition::deserialize(
                     unsignedNestedKernel, true, keys.front().signing_public_key, keys.front().source_decryption_key),
                 std::runtime_error);
}

TEST(CudaKernelExpression, ConditionalExpressionDefinitionProtectsCudaKernelsInElseBothAndNestedBranches) {
    auto op = CudaKernelExpression::builder("conditional_branch_identity")
                  .source(R"cuda(
extern "C" __global__
void conditional_branch_identity_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("conditional_branch_identity_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    auto cudaBranch = [&]() { return op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}); };
    auto ordinaryBranch = [&]() { return Expression::outputs({{"y", Expression::input("x")}}); };
    auto predicate = [](const char* name) { return Expression::input(name).greaterThan(Expression::constantScalar(0.0)); };

    const std::vector<Outputs> cases = {
        Outputs::conditional(predicate("else_predicate"), ordinaryBranch(), cudaBranch()),
        Outputs::conditional(predicate("both_predicate"), cudaBranch(), cudaBranch()),
        Outputs::conditional(predicate("outer_predicate"),
                             Outputs::conditional(predicate("inner_predicate"), ordinaryBranch(), cudaBranch()),
                             ordinaryBranch()),
    };

    for (const Outputs& outputs : cases) {
        ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
        EXPECT_TRUE(definition.hasCudaKernelExpressions());
        nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();
        std::vector<CudaKernelSourceInspection> protectedSources = collectCudaKernelSourceInfo(payload);
        ASSERT_FALSE(protectedSources.empty());
        for (const CudaKernelSourceInspection& source : protectedSources) {
            EXPECT_TRUE(source.source_encrypted);
            EXPECT_FALSE(source.signature.empty());
        }

        std::vector<CudaKernelOutOfBandKeys> keys = collectCudaKernelOutOfBandKeys(payload);
        ASSERT_EQ(keys.size(), 1u);
        ExpressionDefinition loaded = ExpressionDefinition::deserialize(
            payload, false, keys.front().signing_public_key, keys.front().source_decryption_key);
        EXPECT_TRUE(loaded.hasCudaKernelExpressions());
        EXPECT_EQ(loaded.cudaKernelSourceInfo().size(), protectedSources.size());
    }
}


TEST(CudaKernelExpression, SerializedCudaSourceIsInspectableAndRequiresUnsafeOptInToRunAfterLoad) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("serializable_scale")
                  .source(R"cuda(
extern "C" __global__
void serializable_scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("serializable_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("alpha", DataType::FP32, 3.0f)
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    CudaKernelSourceInspection directInfo;
    {
        const auto info = op.sourceInfo();
        directInfo.name = info.name;
        directInfo.entrypoint = info.entrypoint;
        directInfo.source = info.source;
        directInfo.compiled_source = info.compiled_source;
        directInfo.compiled_source_hash = info.source_hash;
        directInfo.loaded_source_compilation_allowed = info.loaded_source_compilation_allowed;
    }
    EXPECT_EQ(directInfo.name, "serializable_scale");
    EXPECT_NE(directInfo.source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_NE(directInfo.compiled_source.find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"), std::string::npos);

    Outputs outputs = op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
    nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();

    ASSERT_TRUE(payload.contains("cuda_kernels"));
    ASSERT_TRUE(payload.contains("cuda_kernel_manifest_signature"));
    ASSERT_EQ(payload.at("cuda_kernels").size(), 1u);
    EXPECT_EQ(payload.at("cuda_kernels").at(0).at("name").get<std::string>(), "serializable_scale");
    EXPECT_FALSE(payload.at("cuda_kernels").at(0).contains("source"));
    // Entrypoint names remain plaintext ABI metadata; the CUDA source body must not.
    EXPECT_EQ(payload.dump().find("y[i] = alpha * x[i]"), std::string::npos);
    EXPECT_TRUE(payload.at("cuda_kernels").at(0).contains("encrypted_source"));
    EXPECT_TRUE(payload.at("cuda_kernels").at(0).contains("source_encryption"));
    EXPECT_EQ(payload.at("cuda_kernels").at(0).at("source_encryption").at("algorithm").get<std::string>(), "aes-256-gcm");
    const nlohmann::json& signatureJson = payload.at("cuda_kernel_manifest_signature");
    EXPECT_EQ(signatureJson.at("algorithm").get<std::string>(), "ed25519");
    EXPECT_FALSE(signatureJson.contains("public_key"));
    EXPECT_FALSE(signatureJson.at("public_key_fingerprint").get<std::string>().empty());
    std::vector<CudaKernelOutOfBandKeys> outOfBandKeys = collectCudaKernelOutOfBandKeys(payload);
    ASSERT_EQ(outOfBandKeys.size(), 1u);
    const std::string trustedPublicKey = outOfBandKeys.front().signing_public_key;
    const std::string trustedSourceDecryptionKey = outOfBandKeys.front().source_decryption_key;
    EXPECT_FALSE(trustedPublicKey.empty());
    EXPECT_FALSE(trustedSourceDecryptionKey.empty());
    EXPECT_NE(signatureJson.at("public_key_fingerprint").get<std::string>(), trustedPublicKey);
    EXPECT_NE(payload.at("cuda_kernels").at(0).at("source_encryption").at("source_decryption_key_fingerprint").get<std::string>(),
              trustedSourceDecryptionKey);

    std::vector<CudaKernelSourceInspection> serializedInfo = collectCudaKernelSourceInfo(payload);
    ASSERT_EQ(serializedInfo.size(), 1u);
    EXPECT_EQ(serializedInfo.front().name, "serializable_scale");
    EXPECT_TRUE(serializedInfo.front().source_encrypted);
    EXPECT_TRUE(serializedInfo.front().source.empty());
    EXPECT_TRUE(serializedInfo.front().compiled_source.empty());
    EXPECT_EQ(serializedInfo.front().source_encryption_algorithm, "aes-256-gcm");
    EXPECT_FALSE(serializedInfo.front().loaded_source_compilation_allowed);
    EXPECT_EQ(serializedInfo.front().signing_public_key_fingerprint, signatureJson.at("public_key_fingerprint").get<std::string>());

    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload), std::runtime_error);

    ExpressionDefinition loadedDefault = ExpressionDefinition::deserialize(payload, false, trustedPublicKey, trustedSourceDecryptionKey);
    std::vector<CudaKernelSourceInspection> firstClassSourceInfo = loadedDefault.cudaKernelSourceInfo();
    ASSERT_EQ(firstClassSourceInfo.size(), 1u);
    EXPECT_EQ(firstClassSourceInfo.front().name, "serializable_scale");
    EXPECT_NE(firstClassSourceInfo.front().source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_EQ(loadedDefault.cudaKernelSources(), std::vector<std::string>{firstClassSourceInfo.front().source});

    nlohmann::json sourceInfo = loadedDefault.cudaKernelSourceInfoJson();
    ASSERT_EQ(sourceInfo.size(), 1u);
    EXPECT_FALSE(sourceInfo.at(0).at("loaded_source_compilation_allowed").get<bool>());
    EXPECT_NE(sourceInfo.at(0).at("compiled_source").get<std::string>().find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"),
              std::string::npos);
    EXPECT_NE(sourceInfo.at(0).at("compiled_source").get<std::string>().find("serializable_scale_kernel"), std::string::npos);
    EXPECT_TRUE(sourceInfo.at(0).contains("signing_public_key_fingerprint"));
    EXPECT_FALSE(sourceInfo.at(0).contains("signing_public_key"));

    EXPECT_THROW((void)DynamicExpression::fromExpressionDefinition(loadedDefault).stamp({{"x", x}}, {}, stream), std::runtime_error);

    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload, true), std::runtime_error);
    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload, false, trustedPublicKey), std::runtime_error);

    nlohmann::json missingSignature = payload;
    missingSignature.erase("cuda_kernel_manifest_signature");
    EXPECT_THROW((void)ExpressionDefinition::deserialize(missingSignature, true, trustedPublicKey, trustedSourceDecryptionKey), std::runtime_error);

    nlohmann::json encryptedWithExtraPlaintextSource = payload;
    encryptedWithExtraPlaintextSource["cuda_kernels"][0]["source"] = directInfo.source;
    try {
        (void)ExpressionDefinition::deserialize(encryptedWithExtraPlaintextSource, true, trustedPublicKey, trustedSourceDecryptionKey);
        FAIL() << "Expected encrypted serialized kernel carrying plaintext source to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("plaintext CUDA source"), std::string::npos);
    }

    nlohmann::json plaintextOnlySource = payload;
    plaintextOnlySource["cuda_kernels"][0]["source"] = directInfo.source;
    plaintextOnlySource["cuda_kernels"][0].erase("encrypted_source");
    plaintextOnlySource["cuda_kernels"][0].erase("source_encryption");
    try {
        (void)ExpressionDefinition::deserialize(plaintextOnlySource, true, trustedPublicKey, trustedSourceDecryptionKey);
        FAIL() << "Expected serialized plaintext CUDA source to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("plaintext CUDA source"), std::string::npos);
    }

    nlohmann::json publicKeyInFingerprint = payload;
    publicKeyInFingerprint["cuda_kernel_manifest_signature"]["public_key_fingerprint"] = trustedPublicKey;
    try {
        (void)ExpressionDefinition::deserialize(publicKeyInFingerprint, true, trustedPublicKey, trustedSourceDecryptionKey);
        FAIL() << "Expected manifest public_key_fingerprint containing public key material to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("public_key_fingerprint contains public key material"), std::string::npos);
    }

    auto wrongKeyOp = CudaKernelExpression::builder("wrong_key_source")
                          .source(R"cuda(
extern "C" __global__
void wrong_key_source_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                          .entry("wrong_key_source_kernel")
                          .input("x", DataType::FP32)
                          .outputLike("y", DataType::FP32, "x")
                          .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                          .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                          .build();
    ExpressionDefinition wrongKeyDefinition =
        ExpressionDefinition::fromOutputs(wrongKeyOp.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    nlohmann::json wrongKeyPayload = wrongKeyDefinition.architectureJsonWithCudaKernelManifestSignature();
    EXPECT_FALSE(wrongKeyPayload.at("cuda_kernel_manifest_signature").contains("public_key"));
    std::vector<CudaKernelOutOfBandKeys> wrongOutOfBandKeys = collectCudaKernelOutOfBandKeys(wrongKeyPayload);
    ASSERT_EQ(wrongOutOfBandKeys.size(), 1u);
    const std::string wrongTrustedPublicKey = wrongOutOfBandKeys.front().signing_public_key;
    ASSERT_FALSE(wrongTrustedPublicKey.empty());
    ASSERT_NE(wrongTrustedPublicKey, trustedPublicKey);
    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload, true, wrongTrustedPublicKey, trustedSourceDecryptionKey),
                 std::runtime_error);

    nlohmann::json tampered = payload;
    tampered["cuda_kernels"][0]["encrypted_source"] = tampered["cuda_kernels"][0]["encrypted_source"].get<std::string>() + "00";
    EXPECT_THROW((void)ExpressionDefinition::deserialize(tampered, true, trustedPublicKey, trustedSourceDecryptionKey),
                 std::runtime_error);

    nlohmann::json tamperedLaunch = payload;
    tamperedLaunch["cuda_kernels"][0]["launch"]["block"] = 256;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(tamperedLaunch, true, trustedPublicKey, trustedSourceDecryptionKey),
                 std::runtime_error);

    ExpressionDefinition loadedAllowed = ExpressionDefinition::deserialize(payload, true, trustedPublicKey, trustedSourceDecryptionKey);
    nlohmann::json allowedSourceInfo = loadedAllowed.cudaKernelSourceInfoJson();
    EXPECT_TRUE(allowedSourceInfo.at(0).at("loaded_source_compilation_allowed").get<bool>());

    auto plan = DynamicExpression::fromExpressionDefinition(loadedAllowed).stamp({{"x", x}}, {}, stream);
    plan.run();
    expectNear(copyToCpuValues(plan.output("y"), stream), {3.0f, -6.0f, 9.0f, 13.5f, -15.0f, 18.0f});
}


TEST(CudaKernelExpression, ArchitectureJsonDoesNotMintCudaManifestSignature) {
    auto op = CudaKernelExpression::builder("unsigned_inspection_scale")
                  .source(R"cuda(
extern "C" __global__
void unsigned_inspection_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("unsigned_inspection_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));

    nlohmann::json unsignedPayload = definition.architectureJson();
    ASSERT_TRUE(unsignedPayload.contains("cuda_kernels"));
    EXPECT_FALSE(unsignedPayload.contains("cuda_kernel_manifest_signature"));
    std::vector<CudaKernelSourceInspection> unsignedInfo = collectCudaKernelSourceInfo(unsignedPayload);
    ASSERT_EQ(unsignedInfo.size(), 1u);
    EXPECT_NE(unsignedInfo.front().compiled_source.find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"), std::string::npos);
    EXPECT_NE(unsignedInfo.front().compiled_source.find("unsigned_inspection_scale_kernel"), std::string::npos);

    nlohmann::json signedPayload = definition.architectureJsonWithCudaKernelManifestSignature();
    ASSERT_TRUE(signedPayload.contains("cuda_kernel_manifest_signature"));
    ASSERT_EQ(definition.cudaKernelSigningPublicKeys().size(), 1u);

    nlohmann::json loadedUnsignedPayload = signedPayload;
    loadedUnsignedPayload.erase("cuda_kernel_manifest_signature");
    EXPECT_THROW((void)ExpressionDefinition::deserialize(loadedUnsignedPayload), std::runtime_error);

    nlohmann::json plaintextSavedModelPayload = unsignedPayload;
    EXPECT_TRUE(plaintextSavedModelPayload.at("cuda_kernels").at(0).contains("source"));
    try {
        (void)ExpressionDefinition::deserialize(plaintextSavedModelPayload, true, "ed25519:" + std::string(64, '0'), "aes256-gcm:" + std::string(64, '0'));
        FAIL() << "Expected plaintext saved-model CUDA source to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("plaintext CUDA source"), std::string::npos);
    }
}

TEST(CudaKernelExpression, RecursiveModelSigningUsesOnePublicKeyForAllCudaExpressions) {
    auto opA = CudaKernelExpression::builder("model_signing_scale_a")
                   .source(R"cuda(
extern "C" __global__
void model_signing_scale_a_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                   .entry("model_signing_scale_a_kernel")
                   .input("x", DataType::FP32)
                   .outputLike("y", DataType::FP32, "x")
                   .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                   .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                   .build();
    auto opB = CudaKernelExpression::builder("model_signing_scale_b")
                   .source(R"cuda(
extern "C" __global__
void model_signing_scale_b_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
)cuda")
                   .entry("model_signing_scale_b_kernel")
                   .input("x", DataType::FP32)
                   .outputLike("y", DataType::FP32, "x")
                   .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                   .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                   .build();

    ExpressionDefinition definitionA = ExpressionDefinition::fromOutputs(
        opA.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    ExpressionDefinition definitionB = ExpressionDefinition::fromOutputs(
        opB.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));

    nlohmann::json modelJson{{"layers", nlohmann::json::array()}};
    modelJson["layers"].push_back(nlohmann::json{{"expression", definitionA.architectureJson()}});
    modelJson["layers"].push_back(nlohmann::json{{"expression", definitionB.architectureJson()}});
    nlohmann::json unsignedModelJson = modelJson;

    std::vector<CudaKernelOutOfBandKeys> outOfBandKeys = cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    ASSERT_EQ(outOfBandKeys.size(), 1u);
    const std::string& trustedPublicKey = outOfBandKeys.front().signing_public_key;
    ASSERT_FALSE(trustedPublicKey.empty());
    ASSERT_FALSE(outOfBandKeys.front().source_decryption_key.empty());

    const nlohmann::json& expressionA = modelJson.at("layers").at(0).at("expression");
    const nlohmann::json& expressionB = modelJson.at("layers").at(1).at("expression");
    ASSERT_TRUE(expressionA.contains("cuda_kernel_manifest_signature"));
    ASSERT_TRUE(expressionB.contains("cuda_kernel_manifest_signature"));
    EXPECT_EQ(expressionA.at("cuda_kernel_manifest_signature").at("public_key_fingerprint").get<std::string>(),
              expressionB.at("cuda_kernel_manifest_signature").at("public_key_fingerprint").get<std::string>());

    CudaKernelSignatureVerificationResult verificationA = cudaKernelVerifyManifestSignature(expressionA, trustedPublicKey);
    CudaKernelSignatureVerificationResult verificationB = cudaKernelVerifyManifestSignature(expressionB, trustedPublicKey);
    EXPECT_TRUE(verificationA.verified) << verificationA.message;
    EXPECT_TRUE(verificationB.verified) << verificationB.message;

    nlohmann::json modelJsonAgain = unsignedModelJson;
    std::vector<CudaKernelOutOfBandKeys> outOfBandKeysAgain = cudaKernelGenerateAndAttachManifestSignatures(modelJsonAgain);
    ASSERT_EQ(outOfBandKeysAgain.size(), outOfBandKeys.size());
    EXPECT_EQ(outOfBandKeysAgain.front().signing_public_key, outOfBandKeys.front().signing_public_key);
    EXPECT_EQ(outOfBandKeysAgain.front().source_decryption_key, outOfBandKeys.front().source_decryption_key);
    EXPECT_EQ(modelJsonAgain.dump(), modelJson.dump());
}

TEST(CudaKernelExpression, MalformedCudaKernelGraphNodesAreRejectedDuringDeserializeValidation) {
    auto op = CudaKernelExpression::builder("validation_scale")
                  .source(R"cuda(
extern "C" __global__
void validation_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("validation_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();

    auto findCudaOutputNodeIndex = [](const nlohmann::json& expression_json) -> size_t {
        const auto& nodes = expression_json.at("nodes");
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes.at(i).at("op").get<std::string>() == "cuda_kernel_output") {
                return i;
            }
        }
        throw std::runtime_error("test payload did not contain a CUDA kernel output node");
    };

    const size_t cudaNodeIndex = findCudaOutputNodeIndex(payload);

    nlohmann::json badOutputIndex = payload;
    badOutputIndex["nodes"][cudaNodeIndex]["cuda_kernel_output_index"] = 999;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badOutputIndex), std::runtime_error);

    nlohmann::json badInputCount = payload;
    badInputCount["nodes"][cudaNodeIndex]["cuda_kernel_input_nodes"] = nlohmann::json::array();
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badInputCount), std::runtime_error);

    nlohmann::json badInputKind = payload;
    const uint32_t kernelInputNode = payload["nodes"][cudaNodeIndex]["cuda_kernel_input_nodes"].at(0).get<uint32_t>();
    badInputKind["nodes"][kernelInputNode]["op"] = "runtime_scalar";
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badInputKind), std::runtime_error);

    nlohmann::json badOutputDType = payload;
    badOutputDType["nodes"][cudaNodeIndex]["output_dtype"] = DataType::FP64;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badOutputDType), std::runtime_error);
}

TEST(CudaKernelExpression, NonSerializableLaunchCallbackIsRejectedWhenSavingExpressionDefinition) {
    auto op = CudaKernelExpression::builder("callback_launch_not_serializable")
                  .source(R"cuda(
extern "C" __global__
void callback_launch_not_serializable_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("callback_launch_not_serializable_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
    EXPECT_THROW((void)definition.architectureJson(), std::runtime_error);
}
