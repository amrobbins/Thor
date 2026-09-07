#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for FP8/BF16 Expression execution tests.";  \
        }                                                                                         \
    } while (false)

TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeGpuTensorWithDType(const std::vector<float>& values, DataType dtype, Stream& stream) {
    const std::vector<uint64_t> dims{static_cast<uint64_t>(values.size())};
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, dims));
    auto* cpu_values = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_values[i] = values[i];
    }

    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, dims));
    fp32_gpu.copyFromAsync(cpu, stream);
    if (dtype == DataType::FP32) {
        stream.synchronize();
        return fp32_gpu;
    }

    Tensor converted(gpu_placement, TensorDescriptor(dtype, dims));
    TypeConverter::convertType(fp32_gpu.getMemPtr<void>(),
                               converted.getMemPtr<void>(),
                               DataType::FP32,
                               dtype,
                               static_cast<long>(values.size()),
                               stream,
                               gpu_placement.getDeviceNum());
    stream.synchronize();
    return converted;
}

std::vector<float> copyGpuTensorAsFloat(Tensor gpu, Stream& stream) {
    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    if (gpu.getDataType() == DataType::FP32) {
        fp32_gpu.copyFromAsync(gpu, stream);
    } else {
        TypeConverter::convertType(gpu.getMemPtr<void>(),
                                   fp32_gpu.getMemPtr<void>(),
                                   gpu.getDataType(),
                                   DataType::FP32,
                                   static_cast<long>(gpu.getTotalNumElements()),
                                   stream,
                                   gpu.getPlacement().getDeviceNum());
    }

    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    cpu.copyFromAsync(fp32_gpu, stream);
    stream.synchronize();

    const auto* cpu_values = cpu.getMemPtr<float>();
    return std::vector<float>(cpu_values, cpu_values + gpu.getTotalNumElements());
}


void expectFp8DefaultBf16CancellationIsFiniteZero(DataType fp8_dtype, float value) {
    Stream stream(0);
    constexpr size_t num_values = 33;
    Tensor input = makeGpuTensorWithDType(std::vector<float>(num_values, value), fp8_dtype, stream);

    const Expression x = Expression::input("x", fp8_dtype, fp8_dtype);
    const Expression square = x * x;
    const Expression cancellation = square - square;
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", cancellation}}).physicalOutputs(), 0);

    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
    plan.run();
    const std::vector<float> result = copyGpuTensorAsFloat(plan.output("y"), stream);

    ASSERT_EQ(result.size(), num_values);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i])) << "index " << i;
        EXPECT_EQ(result[i], 0.0f) << "index " << i;
    }
}

void expectMixedFp8ProductPromotesToFiniteBf16() {
    Stream stream(0);
    constexpr size_t num_values = 33;
    constexpr float e4m3_value = 448.0f;
    constexpr float e5m2_value = 57344.0f;
    constexpr float expected_product = e4m3_value * e5m2_value;

    Tensor e4m3 = makeGpuTensorWithDType(std::vector<float>(num_values, e4m3_value), DataType::FP8_E4M3, stream);
    Tensor e5m2 = makeGpuTensorWithDType(std::vector<float>(num_values, e5m2_value), DataType::FP8_E5M2, stream);

    const Expression x = Expression::input("x", DataType::FP8_E4M3, DataType::FP8_E4M3);
    const Expression y = Expression::input("y", DataType::FP8_E5M2, DataType::FP8_E5M2);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"product", x * y}}).physicalOutputs(), 0);

    StampedExecutionPlan plan = equation.stamp({{"x", e4m3}, {"y", e5m2}}, stream);
    plan.run();
    Tensor output = plan.output("product");
    EXPECT_EQ(output.getDataType(), DataType::BF16);

    const std::vector<float> result = copyGpuTensorAsFloat(output, stream);
    ASSERT_EQ(result.size(), num_values);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i])) << "index " << i;
        EXPECT_EQ(result[i], expected_product) << "index " << i;
    }
}

void expectFp8ExplicitBf16CancellationIsFiniteZero(DataType fp8_dtype, float value) {
    Stream stream(0);
    constexpr size_t num_values = 33;
    Tensor input = makeGpuTensorWithDType(std::vector<float>(num_values, value), fp8_dtype, stream);

    const Expression x = Expression::input("x", fp8_dtype, fp8_dtype);
    const Expression square = (x * x).withComputeDType(DataType::BF16);
    const Expression cancellation = (square - square).withComputeDType(DataType::BF16);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", cancellation}}).physicalOutputs(), 0);

    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
    plan.run();
    const std::vector<float> result = copyGpuTensorAsFloat(plan.output("y"), stream);

    ASSERT_EQ(result.size(), num_values);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result[i])) << "index " << i;
        EXPECT_EQ(result[i], 0.0f) << "index " << i;
    }
}

}  // namespace

TEST(Fp8Bf16ExpressionExecution, ExplicitBf16ComputeAvoidsFp16RangeOverflowInVectorAndScalarTail) {
    REQUIRE_CUDA_DEVICE();

    // Both products exceed FP16's finite range, but remain comfortably finite
    // in BF16.  The subtraction therefore distinguishes a genuine BF16
    // intermediate path from FP8 -> half2 arithmetic hidden in vector codegen.
    expectFp8ExplicitBf16CancellationIsFiniteZero(DataType::FP8_E4M3, 448.0f);
    expectFp8ExplicitBf16CancellationIsFiniteZero(DataType::FP8_E5M2, 57344.0f);
}

TEST(Fp8Bf16ExpressionExecution, DefaultBf16ComputeAvoidsFp16RangeOverflowInVectorAndScalarTail) {
    REQUIRE_CUDA_DEVICE();

    // FP8D2 changes the default rather than requiring a per-node override.
    // Both products exceed FP16's finite range, so a finite zero proves that
    // the default fused arithmetic is genuinely using BF16 intermediates.
    expectFp8DefaultBf16CancellationIsFiniteZero(DataType::FP8_E4M3, 448.0f);
    expectFp8DefaultBf16CancellationIsFiniteZero(DataType::FP8_E5M2, 57344.0f);
}

TEST(Fp8Bf16ExpressionExecution, MixedE4M3E5M2PromotesToFiniteBf16StorageAndCompute) {
    REQUIRE_CUDA_DEVICE();

    // 448 * 57344 is exactly representable in BF16 but far outside FP16's
    // finite range. Under the pre-D4 mixed-FP8 -> FP16 promotion this product
    // overflowed; D4 must materialize a finite BF16 result instead. Length 33
    // exercises vector packs plus the scalar tail.
    expectMixedFp8ProductPromotesToFiniteBf16();
}

TEST(Fp8Bf16ExpressionExecution, Fp8OutputNarrowingUsesDestinationFormatOverflowSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr size_t numValues = 33;
    std::vector<float> values(numValues);
    for (size_t i = 0; i < numValues; ++i) {
        values[i] = (i % 2 == 0) ? 100000.0f : -100000.0f;
    }
    Tensor input = makeGpuTensorWithDType(values, DataType::FP32, stream);
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);

    FusedEquation e4Equation = FusedEquation::compile(
        Expression::outputs({{"y", x.cast(DataType::FP8_E4M3)}}).physicalOutputs(), 0);
    StampedExecutionPlan e4Plan = e4Equation.stamp({{"x", input}}, stream);
    e4Plan.run();
    const std::vector<float> e4Result = copyGpuTensorAsFloat(e4Plan.output("y"), stream);

    FusedEquation e5Equation = FusedEquation::compile(
        Expression::outputs({{"y", x.cast(DataType::FP8_E5M2)}}).physicalOutputs(), 0);
    StampedExecutionPlan e5Plan = e5Equation.stamp({{"x", input}}, stream);
    e5Plan.run();
    const std::vector<float> e5Result = copyGpuTensorAsFloat(e5Plan.output("y"), stream);

    ASSERT_EQ(e4Result.size(), numValues);
    ASSERT_EQ(e5Result.size(), numValues);
    for (size_t i = 0; i < numValues; ++i) {
        const bool positive = (i % 2 == 0);
        EXPECT_EQ(e4Result[i], positive ? 448.0f : -448.0f) << "index " << i;
        EXPECT_TRUE(std::isinf(e5Result[i])) << "index " << i;
        EXPECT_EQ(std::signbit(e5Result[i]), !positive) << "index " << i;
    }
}
