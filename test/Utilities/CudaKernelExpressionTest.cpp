#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using DataType = TensorDescriptor::DataType;

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
