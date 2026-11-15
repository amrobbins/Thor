#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

bool haveCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const std::vector<uint64_t>& dims) {
    uint64_t count = 1;
    for (uint64_t dim : dims) {
        count *= dim;
    }
    return count;
}

template <typename T>
T fromFloat(float value);
template <>
half fromFloat<half>(float value) {
    return __float2half(value);
}
template <>
__nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
float toFloat(half value) { return __half2float(value); }
float toFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

// Small integers, so every product and sum is exact in FP16 and BF16.
std::vector<float> integerValues(uint64_t count, uint64_t salt) {
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int64_t>((i * 5 + salt) % 7) - 3);
    }
    return values;
}

template <typename T>
Tensor makeGpuTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtype, dims));
    T* data = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = fromFloat<T>(values[i]);
    }
    Tensor gpu(gpuPlacement, TensorDescriptor(dtype, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<float> copyToHost(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(gpu.getDataType(), gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(numel(gpu.getDimensions()));
    const T* data = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = toFloat(data[i]);
    }
    return values;
}

// lhs [2, 1, M, K] x rhs [1, 3, K, N] broadcasts to [2, 3, M, N], which is not one strided batch:
// it lowers to one strided-batched matmul per leading index, over spans of lhs and the output that
// start index * M * K and index * 3 * M * N elements in. With M = 3, K = 5, N = 7 and 2-byte elements, the second
// group's lhs span starts 30 bytes in and its output span 126 bytes in, so cuBLASLt must be told the
// operands are only 2-byte aligned.
template <typename T>
void runOddShapeBroadcastBatchedMatmul(DataType dtype) {
    constexpr uint64_t groups = 2;
    constexpr uint64_t batch = 3;
    constexpr uint64_t m = 3;
    constexpr uint64_t k = 5;
    constexpr uint64_t n = 7;
    Stream stream(0);

    const std::vector<float> lhsValues = integerValues(groups * m * k, 1);
    const std::vector<float> rhsValues = integerValues(batch * k * n, 4);
    Tensor lhs = makeGpuTensor<T>(dtype, {groups, 1, m, k}, lhsValues, stream);
    Tensor rhs = makeGpuTensor<T>(dtype, {1, batch, k, n}, rhsValues, stream);

    const Expression lhsInput = Expression::input("lhs", dtype, dtype);
    const Expression rhsInput = Expression::input("rhs", dtype, dtype);
    const Expression y = Expression::matmul(lhsInput, rhsInput, false, false, DataType::FP32, dtype);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    ASSERT_EQ(plan.stageKindNames(), (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier"}));
    ASSERT_EQ(plan.output("y").getDimensions(), (std::vector<uint64_t>{groups, batch, m, n}));

    plan.run();
    const std::vector<float> actual = copyToHost<T>(plan.output("y"), stream);

    ASSERT_EQ(actual.size(), groups * batch * m * n);
    for (uint64_t g = 0; g < groups; ++g) {
        for (uint64_t b = 0; b < batch; ++b) {
            for (uint64_t row = 0; row < m; ++row) {
                for (uint64_t col = 0; col < n; ++col) {
                    float expected = 0.0f;
                    for (uint64_t inner = 0; inner < k; ++inner) {
                        expected += lhsValues[(g * m + row) * k + inner] * rhsValues[(b * k + inner) * n + col];
                    }
                    EXPECT_EQ(actual[((g * batch + b) * m + row) * n + col], expected)
                        << "group " << g << " batch " << b << " row " << row << " col " << col;
                }
            }
        }
    }
}

}  // namespace

TEST(MatmulOperandAlignmentExecution, OddShapeBroadcastBatchedMatmulFp16MatchesHostReference) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";
    runOddShapeBroadcastBatchedMatmul<half>(DataType::FP16);
}

TEST(MatmulOperandAlignmentExecution, OddShapeBroadcastBatchedMatmulBf16MatchesHostReference) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";
    runOddShapeBroadcastBatchedMatmul<__nv_bfloat16>(DataType::BF16);
}
